"""Chroma-backed vector index of the product catalog.

SQLite is the source of truth; this is a derived index. Every write goes through
app/services/catalog.py so the two never drift silently — and when they do drift
(a crashed process, a failed embed call) `vector_synced` on the product row is the
flag, and catalog.repair_sync() is the fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Product
from app.services.embeddings import EmbeddingProvider, get_embedding_provider

COLLECTION = "products"


@dataclass
class VectorHit:
    """One retrieved product: id, the distance Chroma reported, and metadata."""

    product_id: int
    distance: float
    metadata: dict[str, Any]

    @property
    def score(self) -> float:
        """Similarity in [0, 1]-ish terms — easier to reason about than distance."""
        return round(1.0 / (1.0 + max(self.distance, 0.0)), 4)


def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a Chroma `where` clause; it needs $and for more than one field."""
    if not filters:
        return None
    clauses = [{key: value} for key, value in filters.items() if value is not None]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class VectorStore:
    """The product collection. One instance per process (see get_vector_store)."""

    def __init__(self, provider: EmbeddingProvider | None = None, path: str | None = None) -> None:
        self.provider = provider or get_embedding_provider()
        self.path = path or get_settings().chroma_dir
        self._client = chromadb.PersistentClient(
            path=self.path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata=self._metadata()
        )

    def _metadata(self) -> dict[str, Any]:
        # The provider name is stamped into the collection: vectors from two
        # different embedders are not comparable, so we need to know which one
        # built the index. See provider_mismatch.
        return {"hnsw:space": "cosine", "provider": self.provider.name}

    @property
    def indexed_provider(self) -> str:
        return str((self._collection.metadata or {}).get("provider", ""))

    @property
    def provider_mismatch(self) -> bool:
        """True when the index was built by a different embedder than the live one.

        Happens when the embedding model is changed, or when the Mesh fallback
        kicks in mid-life. Retrieval is meaningless until the index is rebuilt.
        """
        return bool(self.count()) and self.indexed_provider not in self._acceptable_names()

    def _acceptable_names(self) -> set[str]:
        """Provider names the index may legitimately carry (see possible_names)."""
        names = set(getattr(self.provider, "possible_names", None) or {self.provider.name})
        return names | {""}

    # -- writes --------------------------------------------------------------

    def upsert_products(self, products: Sequence[Product], *, db: Session | None = None) -> int:
        """Embed and upsert products. Returns how many were written."""
        products = [p for p in products if p.id is not None]
        if not products:
            return 0
        empty = self.count() == 0
        vectors = self.provider.embed(
            [p.embedding_text() for p in products], db=db, purpose="embed_product"
        )
        # Stamp *after* embedding: a fallback provider only reveals which embedder
        # actually ran once the call has been made. Chroma refuses metadata edits
        # that touch hnsw:space, so an empty collection is recreated instead.
        if empty and self.indexed_provider != self.provider.name:
            self.reset()
        self._collection.upsert(
            ids=[str(p.id) for p in products],
            embeddings=vectors,
            documents=[p.embedding_text() for p in products],
            metadatas=[
                {
                    "product_id": p.id,
                    "title": p.title,
                    "category": p.category,
                    "level": p.level,
                    "price": float(p.price or 0.0),
                }
                for p in products
            ],
        )
        return len(products)

    def delete_products(self, product_ids: Iterable[int]) -> int:
        ids = [str(pid) for pid in product_ids]
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        """Drop and recreate the collection (tests, and full reindexes)."""
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata=self._metadata()
        )

    # -- reads ---------------------------------------------------------------

    def query(
        self,
        text: str,
        *,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
        exclude_ids: Iterable[int] = (),
        db: Session | None = None,
    ) -> list[VectorHit]:
        """Nearest products to `text`, optionally filtered by metadata.

        `exclude_ids` is over-fetched around rather than filtered server-side, so
        that excluding a product cannot shrink the result below top_k.
        """
        if self.count() == 0:
            return []
        exclude = {int(pid) for pid in exclude_ids}
        n_results = min(self.count(), top_k + len(exclude))
        vector = self.provider.embed([text], db=db, purpose="embed_query")[0]
        raw = self._collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=_where(filters),
        )
        hits: list[VectorHit] = []
        for cid, distance, metadata in zip(
            raw["ids"][0], raw["distances"][0], raw["metadatas"][0], strict=True
        ):
            product_id = int(cid)
            if product_id in exclude:
                continue
            hits.append(VectorHit(product_id=product_id, distance=float(distance), metadata=dict(metadata or {})))
        return hits[:top_k]

    def all_ids(self) -> set[int]:
        """Every product id currently in the index — used to find orphans."""
        return {int(cid) for cid in self._collection.get(include=[])["ids"]}

    def count(self) -> int:
        return self._collection.count()


@lru_cache
def get_vector_store() -> VectorStore:
    """Process-wide store. Chroma holds a file lock, so one instance only."""
    return VectorStore()
