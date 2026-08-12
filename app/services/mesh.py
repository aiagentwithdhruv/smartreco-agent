"""The single door to Mesh API.

Nothing else in this project constructs an LLM client. Mesh is OpenAI-compatible,
so we use the `openai` SDK with `base_url` pointed at https://api.meshapi.ai/v1.

If `MESH_API_KEY` is unset, get_mesh_client() returns None and callers take a
documented no-LLM path. We never fabricate a response and call it a model output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import LLMCall


@dataclass
class MeshResult:
    """One Mesh response plus the numbers we log for observability."""

    text: str = ""
    embeddings: list[list[float]] = field(default_factory=list)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


def extract_text(response) -> str:
    """Pull the assistant's text out of a completion, whatever shape it arrives in.

    Mesh is OpenAI-compatible but the models behind it are not uniformly so, and
    both quirks below were seen live on 4 Aug 2026:

    * `minimax/m2-her` returns an extra `name` field on the message. Harmless —
      the SDK tolerates unknown fields — but it means the response is not the
      canonical shape and should not be treated as guaranteed.
    * `tencent/hy3` returns an empty `content` and puts the answer in
      `reasoning_content`.

    So: prefer `content`, fall back to `reasoning_content`, and return "" rather
    than raising if a model returns neither. An empty string makes the agent fall
    back to its rule-based narrative, which is the correct degradation.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    # `reasoning_content` is not part of the SDK's message model, but the SDK
    # allows extra fields and sets them as real attributes, so getattr finds it.
    for field in ("content", "reasoning_content"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class MeshClient:
    """Thin wrapper: chat completions and embeddings, both timed."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = OpenAI(
            api_key=self.settings.mesh_api_key,
            base_url=self.settings.mesh_base_url,
            timeout=60.0,
            max_retries=2,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.4,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> MeshResult:
        model = model or self.settings.mesh_chat_model
        max_tokens = self.settings.mesh_chat_max_tokens if max_tokens is None else max_tokens
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        return MeshResult(
            text=extract_text(response),
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> MeshResult:
        model = model or self.settings.mesh_embedding_model
        started = time.perf_counter()
        response = self._client.embeddings.create(model=model, input=texts)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        return MeshResult(
            embeddings=[item.embedding for item in response.data],
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


@lru_cache
def get_mesh_client() -> MeshClient | None:
    """Cached client, or None when no key is configured."""
    settings = get_settings()
    if not settings.mesh_configured:
        return None
    return MeshClient(settings)


def log_llm_call(
    db: Session,
    *,
    purpose: str,
    result: MeshResult | None = None,
    user_id: int | None = None,
    cache_hit: bool = False,
    model: str = "",
) -> LLMCall:
    """Record one model call — or one avoided call — in llm_calls.

    Cache hits are logged too, on purpose: the ratio of hits to real calls is the
    efficiency claim this project makes, and it should come from data, not prose.
    """
    call = LLMCall(
        user_id=user_id,
        purpose=purpose,
        model=(result.model if result else model),
        tokens_in=(result.tokens_in if result else 0),
        tokens_out=(result.tokens_out if result else 0),
        latency_ms=(result.latency_ms if result else 0),
        cache_hit=cache_hit,
    )
    db.add(call)
    return call
