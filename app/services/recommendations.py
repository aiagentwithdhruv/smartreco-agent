"""Serving recommendations to pages.

The page render never waits for a model unless it has to. A user with no
recommendation yet gets one synchronously — an empty panel is worse than a
second of latency — and everyone else is served the stored recommendation while
the refresh runs after the response has been sent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Product, Recommendation
from app.services.agent import maybe_recommend, run_agent
from app.services.mesh import log_llm_call
from app.services.trigger import decide
from app.services.vector_store import VectorStore, get_vector_store

log = logging.getLogger(__name__)


@dataclass
class RecommendationView:
    """What the template renders."""

    narrative: str
    products: list[Product]
    version: int
    trigger_reason: str
    source: str


def _view(db: Session, recommendation: Recommendation | None) -> RecommendationView | None:
    if recommendation is None:
        return None
    ids = list(recommendation.product_ids or [])
    by_id = {p.id: p for p in db.scalars(select(Product).where(Product.id.in_(ids))).all()}
    return RecommendationView(
        narrative=recommendation.narrative,
        products=[by_id[pid] for pid in ids if pid in by_id],
        version=recommendation.version,
        trigger_reason=recommendation.trigger_reason,
        source=recommendation.source,
    )


def refresh_user(user_id: int) -> None:
    """Run the agent on its own session — used as a background task."""
    try:
        with SessionLocal() as db:
            maybe_recommend(db, user_id, store=get_vector_store())
    except Exception:  # noqa: BLE001 — a background failure must not affect the request
        log.exception("background recommendation refresh failed for user %s", user_id)


def current_for(
    db: Session,
    user_id: int,
    *,
    store: VectorStore,
    schedule=None,
) -> RecommendationView | None:
    """The recommendation to show this user, refreshing it when the trigger fires.

    `schedule` is FastAPI's BackgroundTasks.add_task (or anything callable) —
    when it is None the refresh runs inline, which is what the demo script and
    the tests want.
    """
    decision = decide(db, user_id)

    if decision.run:
        if decision.current is not None and schedule is not None:
            # Something to show already: refresh after the response goes out.
            schedule(refresh_user, user_id)
            return _view(db, decision.current)
        outcome = run_agent(db, user_id, decision.profile, decision.reason, store=store)
        return _view(db, outcome.recommendation or decision.current)

    if decision.cache_hit:
        # An avoided model call is still worth a row — it is the efficiency proof.
        log_llm_call(db, purpose="generate_rec", user_id=user_id, cache_hit=True)
        db.commit()

    return _view(db, decision.current)
