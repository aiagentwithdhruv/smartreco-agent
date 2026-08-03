"""The SQLite ↔ Chroma dual-write, and what happens when half of it fails."""

import pytest
from sqlalchemy import select

from app.models import Product
from app.services.catalog import (
    ProductInput,
    create_product,
    delete_product,
    repair_sync,
    update_product,
)
from app.services.embeddings import HashingEmbeddingProvider
from app.services.vector_store import VectorStore

SAMPLE = ProductInput(
    title="Agentic Workflows with LangGraph",
    description="Stateful agents as explicit graphs, with checkpointing.",
    category="Agentic AI",
    price=4999,
    level="intermediate",
    tags=["langgraph", "agents"],
)


class BrokenStore:
    """A vector store whose writes always fail — stands in for a dead Chroma."""

    def __init__(self, real):
        self.provider = real.provider
        self.provider_mismatch = False
        self.indexed_provider = real.provider.name
        self._real = real

    def upsert_products(self, products, *, db=None):
        raise RuntimeError("chroma is down")

    def delete_products(self, product_ids):
        raise RuntimeError("chroma is down")

    def all_ids(self):
        return self._real.all_ids()

    def count(self):
        return self._real.count()


def test_create_writes_both_stores(db, store):
    product = create_product(db, SAMPLE, store=store)

    assert db.get(Product, product.id) is not None
    assert store.count() == 1
    assert store.all_ids() == {product.id}
    assert product.vector_synced is True


def test_created_product_is_retrievable_by_meaning_of_its_text(db, store):
    """A real query against the index — not just a row count."""
    create_product(db, SAMPLE, store=store)
    create_product(db, ProductInput("SQL for Data Analysis", "Joins and window functions.",
                                    "Data Analytics", 2999, "beginner", ["sql"]), store=store)

    hits = store.query("langgraph agents", top_k=1)
    assert [h.metadata["title"] for h in hits] == ["Agentic Workflows with LangGraph"]


def test_update_reembeds_so_search_follows_the_new_text(db, store):
    product = create_product(db, SAMPLE, store=store)
    update_product(
        db,
        product,
        ProductInput("Kubernetes for Data Teams", "Operators, autoscaling and storage classes.",
                     "MLOps", 5999, "advanced", ["kubernetes"]),
        store=store,
    )

    assert store.count() == 1, "an edit must update the vector, not add a second one"
    hits = store.query("kubernetes operators", top_k=1)
    assert hits[0].product_id == product.id
    assert hits[0].metadata["title"] == "Kubernetes for Data Teams"
    assert hits[0].metadata["category"] == "MLOps"

    # The old text must no longer be the best match for its own keywords.
    assert store.query("langgraph agents", top_k=1)[0].metadata["title"] == "Kubernetes for Data Teams"


def test_delete_removes_the_row_and_the_vector(db, store):
    product = create_product(db, SAMPLE, store=store)
    delete_product(db, product, store=store)

    assert db.scalar(select(Product)) is None
    assert store.count() == 0


def test_failed_vector_write_keeps_the_row_and_flags_the_drift(db, store):
    """A dead index must not lose the catalog write — but must be visible."""
    product = create_product(db, SAMPLE, store=BrokenStore(store))

    assert db.get(Product, product.id) is not None, "SQLite is the source of truth"
    assert product.vector_synced is False, "drift has to be detectable"
    assert store.count() == 0


def test_repair_reindexes_a_row_whose_vector_write_failed(db, store):
    product = create_product(db, SAMPLE, store=BrokenStore(store))
    assert store.count() == 0

    report = repair_sync(db, store=store)

    assert report.reindexed == 1
    assert report.still_broken == 0
    assert store.all_ids() == {product.id}
    assert db.get(Product, product.id).vector_synced is True


def test_repair_reindexes_a_row_that_claims_to_be_synced_but_is_not(db, store):
    """The nastier desync: the flag says fine, the index disagrees."""
    product = create_product(db, SAMPLE, store=store)
    store.delete_products([product.id])  # index lost behind the app's back
    assert product.vector_synced is True and store.count() == 0

    report = repair_sync(db, store=store)

    assert report.reindexed == 1
    assert store.all_ids() == {product.id}


def test_repair_removes_orphan_vectors(db, store):
    """A delete that never reached the index leaves a vector with no row."""
    product = create_product(db, SAMPLE, store=store)
    delete_product(db, product, store=BrokenStore(store))

    assert store.count() == 1, "the orphan is still indexed"
    report = repair_sync(db, store=store)

    assert report.orphans_removed == 1
    assert store.count() == 0


def test_repair_reports_rows_it_could_not_fix(db, store):
    create_product(db, SAMPLE, store=BrokenStore(store))
    report = repair_sync(db, store=BrokenStore(store))

    assert report.reindexed == 0
    assert report.still_broken == 1


def test_repair_is_a_no_op_on_a_healthy_catalog(db, store):
    create_product(db, SAMPLE, store=store)
    report = repair_sync(db, store=store)
    assert report.as_dict() == {"reindexed": 0, "orphans_removed": 0, "still_broken": 0}


def test_query_can_be_filtered_by_metadata(db, store):
    create_product(db, SAMPLE, store=store)
    create_product(db, ProductInput("Agentic AI Bootcamp", "Six weeks to a working agent.",
                                    "Agentic AI", 7999, "beginner", ["agents"]), store=store)

    hits = store.query("agents", top_k=5, filters={"level": "beginner"})
    assert [h.metadata["title"] for h in hits] == ["Agentic AI Bootcamp"]

    hits = store.query("agents", top_k=5, filters={"category": "Agentic AI", "level": "intermediate"})
    assert [h.metadata["title"] for h in hits] == ["Agentic Workflows with LangGraph"]


def test_query_excludes_ids_without_shrinking_the_result(db, store):
    first = create_product(db, SAMPLE, store=store)
    create_product(db, ProductInput("Agentic AI Bootcamp", "Six weeks to a working agent.",
                                    "Agentic AI", 7999, "beginner", ["agents"]), store=store)

    hits = store.query("agents", top_k=1, exclude_ids=[first.id])
    assert len(hits) == 1
    assert hits[0].product_id != first.id


def test_query_on_an_empty_index_returns_nothing(db, store):
    assert store.query("anything", top_k=5) == []


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_embedding_never_crashes_on_degenerate_text(store, text):
    assert len(store.provider.embed([text])[0]) == 384


class RenamedProvider(HashingEmbeddingProvider):
    """Same maths, different name — stands in for a swapped embedding model."""

    name = "some-other-model"


def test_index_records_which_embedder_built_it(db, store):
    create_product(db, SAMPLE, store=store)
    assert store.indexed_provider == "local-hashing"
    assert store.provider_mismatch is False


def test_index_is_stamped_with_the_embedder_that_actually_ran(db, tmp_path):
    """A provider that falls back mid-write must not stamp the index with the primary."""
    from app.services.embeddings import FallbackEmbeddingProvider, MeshEmbeddingProvider
    from tests.test_embeddings import FakeMeshClient

    provider = FallbackEmbeddingProvider(
        MeshEmbeddingProvider(client=FakeMeshClient(fail=True)), HashingEmbeddingProvider
    )
    # A fresh directory, so the collection really is created under the primary's
    # name and has to be restamped once the fallback runs.
    degraded = VectorStore(provider=provider, path=str(tmp_path / "chroma"))
    assert degraded.indexed_provider == "mesh"
    create_product(db, SAMPLE, store=degraded)

    assert degraded.indexed_provider == "local-hashing"
    assert degraded.provider_mismatch is False


def test_an_index_built_by_the_fallback_is_not_flagged_on_a_fresh_process(db, tmp_path):
    """A restarted process cannot know yet whether Mesh will answer — either
    stamp is acceptable until it has tried, so this must not read as drift."""
    from app.services.embeddings import FallbackEmbeddingProvider, MeshEmbeddingProvider
    from tests.test_embeddings import FakeMeshClient

    path = str(tmp_path / "chroma")
    degraded = VectorStore(
        provider=FallbackEmbeddingProvider(
            MeshEmbeddingProvider(client=FakeMeshClient(fail=True)), HashingEmbeddingProvider
        ),
        path=path,
    )
    create_product(db, SAMPLE, store=degraded)
    assert degraded.indexed_provider == "local-hashing"

    restarted = VectorStore(
        provider=FallbackEmbeddingProvider(
            MeshEmbeddingProvider(client=FakeMeshClient()), HashingEmbeddingProvider
        ),
        path=path,
    )
    assert restarted.provider.name == "mesh"
    assert restarted.provider_mismatch is False


def test_switching_embedding_model_is_detected_as_a_mismatch(db, store):
    create_product(db, SAMPLE, store=store)

    swapped = VectorStore(provider=RenamedProvider(), path=store.path)
    assert swapped.provider_mismatch is True, "vectors from two embedders are not comparable"


def test_repair_rebuilds_the_whole_index_after_a_model_switch(db, store):
    create_product(db, SAMPLE, store=store)
    create_product(db, ProductInput("SQL for Data Analysis", "Joins.", "Data Analytics",
                                    2999, "beginner", ["sql"]), store=store)

    swapped = VectorStore(provider=RenamedProvider(), path=store.path)
    report = repair_sync(db, store=swapped)

    assert report.reindexed == 2
    assert swapped.provider_mismatch is False
    assert swapped.indexed_provider == "some-other-model"
    assert swapped.count() == 2
