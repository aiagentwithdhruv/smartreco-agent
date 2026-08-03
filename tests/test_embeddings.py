"""Embedding providers: determinism, the Mesh path, and the fallback contract."""

import pytest

from app.services.embeddings import (
    HASH_DIM,
    FallbackEmbeddingProvider,
    HashingEmbeddingProvider,
    MeshEmbeddingProvider,
)
from app.services.mesh import MeshResult


class FakeMeshClient:
    """Stands in for MeshClient — no network, and counts its calls."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def embed(self, texts, *, model=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("402 spend_limit_exceeded")
        return MeshResult(
            embeddings=[[0.1] * HASH_DIM for _ in texts],
            model="fake/embed",
            tokens_in=7,
            latency_ms=12,
        )


def test_hashing_embeddings_are_deterministic_and_normalised():
    provider = HashingEmbeddingProvider()
    a = provider.embed(["agentic ai with langgraph"])[0]
    b = provider.embed(["agentic ai with langgraph"])[0]
    assert a == b
    assert len(a) == HASH_DIM
    assert pytest.approx(sum(x * x for x in a), rel=1e-6) == 1.0


def test_hashing_embeddings_separate_unrelated_texts():
    provider = HashingEmbeddingProvider()
    agents, sql, agents2 = provider.embed(
        ["langgraph agents orchestration", "sql joins window functions", "langgraph agents workflow"]
    )
    dot = lambda x, y: sum(a * b for a, b in zip(x, y))  # noqa: E731
    assert dot(agents, agents2) > dot(agents, sql)


def test_mesh_provider_logs_the_call(db):
    from app.models import LLMCall

    provider = MeshEmbeddingProvider(client=FakeMeshClient())
    vectors = provider.embed(["a course about agents"], db=db, purpose="embed_product")
    db.commit()

    assert len(vectors) == 1
    call = db.query(LLMCall).one()
    assert call.purpose == "embed_product"
    assert call.model == "fake/embed"
    assert call.tokens_in == 7
    assert call.cache_hit is False


def test_mesh_provider_requires_a_key():
    with pytest.raises(RuntimeError):
        MeshEmbeddingProvider()  # conftest forces MESH_API_KEY=""


def test_fallback_prefers_the_primary_and_never_builds_the_fallback():
    mesh = FakeMeshClient()
    built = []

    def factory():
        built.append(1)
        return HashingEmbeddingProvider()

    provider = FallbackEmbeddingProvider(MeshEmbeddingProvider(client=mesh), factory)
    provider.embed(["hello"])

    assert mesh.calls == 1
    assert built == [], "the fallback must stay unbuilt while the primary works"
    assert provider.name == "mesh"


def test_fallback_switches_once_and_stays_switched():
    mesh = FakeMeshClient(fail=True)
    provider = FallbackEmbeddingProvider(MeshEmbeddingProvider(client=mesh), HashingEmbeddingProvider)

    first = provider.embed(["hello"])
    second = provider.embed(["hello"])

    assert first == second
    assert mesh.calls == 1, "a dead primary must not be retried on every embed"
    assert provider.name == "local-hashing"
    assert provider.is_remote is False
