"""Embedding providers — three of them behind one interface.

* MeshEmbeddingProvider — the preferred path. Calls Mesh `/embeddings` and logs
  the call to `llm_calls`.
* LocalMiniLMEmbeddingProvider — **the default**. all-MiniLM-L6-v2 running
  locally through the ONNX runtime that ships with Chroma. Real semantic
  embeddings, no external API, so the "every AI call goes through Mesh" rule
  still holds: the only calls that leave the machine are chat completions, and
  they all go to Mesh. This is the default because Mesh serves no free embedding
  model (verified 4 Aug 2026 — 997 models, 3 free, none of them embeddings), so
  a zero-balance key gets HTTP 402 on /embeddings.
* HashingEmbeddingProvider — deterministic feature hashing over word tokens.
  *Lexical, not semantic*, labelled as such everywhere. Used by the test suite
  and by anyone who wants zero downloads. Never writes an `llm_calls` row.

`EMBEDDINGS=local` is the default. `EMBEDDINGS=auto` prefers Mesh and falls back
to local on the first failure, loudly — use it on a funded key. The active
provider name is printed by seed.py and shown on the admin page, so what produced
the index is never a guess.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Callable, Protocol

from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.mesh import get_mesh_client, log_llm_call

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
HASH_DIM = 384


class EmbeddingProvider(Protocol):
    """Anything that can turn texts into vectors."""

    name: str
    is_remote: bool

    def embed(self, texts: list[str], *, db: Session | None = None, purpose: str = "embed") -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Deterministic feature-hashing embeddings — no network, no model."""

    name = "local-hashing"
    is_remote = False

    def embed(self, texts: list[str], *, db: Session | None = None, purpose: str = "embed") -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        counts = Counter(_TOKEN_RE.findall(text.lower()))
        vec = [0.0] * HASH_DIM
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % HASH_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sublinear term frequency: a word repeated ten times is not ten times
            # as informative as a word seen once.
            vec[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An empty or punctuation-only text: return a valid zero-ish vector
            # rather than dividing by zero. Chroma needs a fixed-width vector.
            return vec
        return [v / norm for v in vec]


class MeshEmbeddingProvider:
    """Embeddings via Mesh API — the only remote embedding path."""

    name = "mesh"
    is_remote = True

    def __init__(self, client=None) -> None:
        self._client = client or get_mesh_client()
        if self._client is None:
            raise RuntimeError("MeshEmbeddingProvider requires MESH_API_KEY")

    def embed(self, texts: list[str], *, db: Session | None = None, purpose: str = "embed") -> list[list[float]]:
        result = self._client.embed(texts)
        if db is not None:
            log_llm_call(db, purpose=purpose, result=result)
        return result.embeddings


class LocalMiniLMEmbeddingProvider:
    """all-MiniLM-L6-v2 via Chroma's bundled ONNX runtime. Local, semantic, offline."""

    name = "local-minilm"
    is_remote = False

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        # First use downloads ~80 MB into ~/.cache/chroma, then runs offline.
        self._fn = embedding_functions.ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str], *, db: Session | None = None, purpose: str = "embed") -> list[list[float]]:
        # The ONNX function hands back numpy float32; Chroma's validator wants plain floats.
        return [[float(x) for x in vector] for vector in self._fn(texts)]


class FallbackEmbeddingProvider:
    """Try the primary provider; on its first failure switch to the fallback.

    A recommendation engine that cannot embed is dead, so an outage or an
    out-of-credit key degrades to local embeddings instead of a 500 — but it says
    so at WARNING level and changes `name`, which the admin page displays.
    """

    def __init__(self, primary: EmbeddingProvider, fallback_factory: Callable[[], EmbeddingProvider]) -> None:
        self._primary = primary
        self._factory = fallback_factory  # built lazily: no 80 MB download unless needed
        self._fallback: EmbeddingProvider | None = None

    @property
    def name(self) -> str:
        return self._fallback.name if self._fallback else self._primary.name

    @property
    def possible_names(self) -> set[str]:
        """Every embedder this provider might turn out to be.

        Before the first call we do not know whether Mesh will answer, so an
        index stamped with either name is legitimate and must not be reported as
        drift. Once we have fallen back, the answer is known.
        """
        if self._fallback is not None:
            return {self._fallback.name}
        return {self._primary.name, str(getattr(self._factory, "name", ""))} - {""}

    @property
    def is_remote(self) -> bool:
        return self._fallback.is_remote if self._fallback else self._primary.is_remote

    def embed(self, texts: list[str], *, db: Session | None = None, purpose: str = "embed") -> list[list[float]]:
        if self._fallback is None:
            try:
                return self._primary.embed(texts, db=db, purpose=purpose)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "%s embeddings failed (%s: %s) — falling back for the rest of this process. "
                    "The index must be rebuilt so it is not a mix of two embedding spaces.",
                    self._primary.name, type(exc).__name__, exc,
                )
                self._fallback = self._factory()
        return self._fallback.embed(texts, db=db, purpose=purpose)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Resolve the EMBEDDINGS setting into a provider."""
    settings = get_settings()
    mode = settings.embeddings

    if mode == "hashing":
        return HashingEmbeddingProvider()
    if mode == "local":
        return LocalMiniLMEmbeddingProvider()
    if mode == "mesh":
        return MeshEmbeddingProvider()  # raises if no key — explicit request, explicit failure
    if settings.mesh_configured:
        return FallbackEmbeddingProvider(MeshEmbeddingProvider(), LocalMiniLMEmbeddingProvider)
    return LocalMiniLMEmbeddingProvider()
