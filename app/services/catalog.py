"""Catalog writes — the one code path that keeps SQLite and Chroma in step.

Rules of this module:

1. SQLite is the source of truth. It is written first and committed.
2. The vector index is written straight after, in the same call. Success sets
   `vector_synced = True` on the row.
3. A vector write that fails does not fail the request — it leaves
   `vector_synced = False`, which is a *detectable* drift. Deletes are
   row-first for the same reason: a leftover vector shows up as an orphan.
4. repair_sync() closes both gaps, and is exposed as a button in the admin UI.

Nothing else in the app is allowed to write to the vector store.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Product
from app.services.vector_store import VectorStore, get_vector_store

log = logging.getLogger(__name__)


@dataclass
class ProductInput:
    """Validated product fields coming from the admin form."""

    title: str
    description: str
    category: str
    price: float
    level: str
    tags: list[str] = field(default_factory=list)


@dataclass
class SyncReport:
    """What repair_sync() actually did — rendered back to the admin."""

    reindexed: int = 0
    orphans_removed: int = 0
    still_broken: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _store(store: VectorStore | None) -> VectorStore:
    return store or get_vector_store()


def sync_product(db: Session, product: Product, *, store: VectorStore | None = None) -> bool:
    """Push one product into the vector index and record whether it worked."""
    try:
        _store(store).upsert_products([product], db=db)
    except Exception:  # noqa: BLE001 — index failure must not lose the catalog write
        log.exception("vector upsert failed for product %s", product.id)
        product.vector_synced = False
        db.commit()
        return False
    product.vector_synced = True
    db.commit()
    return True


def create_product(db: Session, data: ProductInput, *, store: VectorStore | None = None) -> Product:
    product = Product(
        title=data.title,
        description=data.description,
        category=data.category,
        price=data.price,
        level=data.level,
        tags=data.tags,
        vector_synced=False,
    )
    db.add(product)
    db.commit()  # source of truth first — the row exists even if embedding fails
    sync_product(db, product, store=store)
    return product


def update_product(db: Session, product: Product, data: ProductInput, *, store: VectorStore | None = None) -> Product:
    """Edits re-embed: the indexed text is derived from these exact fields."""
    product.title = data.title
    product.description = data.description
    product.category = data.category
    product.price = data.price
    product.level = data.level
    product.tags = data.tags
    product.vector_synced = False
    db.commit()
    sync_product(db, product, store=store)
    return product


def delete_product(db: Session, product: Product, *, store: VectorStore | None = None) -> None:
    """Remove from SQLite, then from the index. A failed index delete becomes an orphan."""
    product_id = product.id
    db.delete(product)
    db.commit()
    try:
        _store(store).delete_products([product_id])
    except Exception:  # noqa: BLE001
        log.exception("vector delete failed for product %s — left as an orphan", product_id)


def reindex_all(db: Session, *, store: VectorStore | None = None) -> int:
    """Rebuild the whole index in one batched embed call (used by seed.py).

    repair_sync() deliberately stays row-by-row so it can attribute a failure to
    a specific product; this path trades that for one API round trip.
    """
    vs = _store(store)
    vs.reset()
    products = list(db.scalars(select(Product)).all())
    written = vs.upsert_products(products, db=db)
    for product in products:
        product.vector_synced = True
    db.commit()
    return written


def repair_sync(db: Session, *, store: VectorStore | None = None) -> SyncReport:
    """Reconcile SQLite and Chroma in both directions.

    Fixes three kinds of drift:
      * rows flagged `vector_synced = False` (a write that failed mid-flight)
      * rows flagged synced but missing from the index (index lost/rebuilt)
      * vectors with no matching row (a delete that never reached the index)
    """
    vs = _store(store)
    report = SyncReport()

    if vs.provider_mismatch:
        # Vectors from two embedders share no space; a partial fix would be worse
        # than none. Rebuild the whole index under the live provider.
        log.warning("index was built by %r, live provider is %r — full rebuild",
                    vs.indexed_provider, vs.provider.name)
        report.reindexed = reindex_all(db, store=vs)
        return report

    products = list(db.scalars(select(Product)).all())
    by_id = {p.id: p for p in products}
    indexed = vs.all_ids()

    stale_ids = {p.id for p in products if not p.vector_synced} | (set(by_id) - indexed)
    for product_id in sorted(stale_ids):
        if sync_product(db, by_id[product_id], store=vs):
            report.reindexed += 1
        else:
            report.still_broken += 1

    orphans = indexed - set(by_id)
    if orphans:
        vs.delete_products(orphans)
        report.orphans_removed = len(orphans)

    return report
