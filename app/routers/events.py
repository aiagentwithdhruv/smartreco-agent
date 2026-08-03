"""Bulk behavior-event ingestion — the endpoint static/tracker.js posts to.

Design notes that matter more than the code:

* One request carries a whole batch, so a browsing session costs a handful of
  requests instead of one per click.
* A batch is never rejected wholesale for one bad event. Junk is dropped and
  counted; the good events in the same batch are still stored. A tracker that
  loses a session because of one stale product id is worse than useless.
* Client clocks are not trusted: a timestamp in the future is clamped to server
  time, and events are never accepted for another user — `user_id` comes from
  the session cookie, never from the payload.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.deps import DbSession, require_user
from app.models import EVENT_TYPES, Product, User, utcnow
from app.services.tracking import InvalidEvent, record_event

router = APIRouter(prefix="/api", tags=["tracking"])

MAX_BATCH = 100  # tracker.js sends 20; the rest is headroom for a backed-up queue
MAX_DWELL_SECONDS = 3600.0
CLOCK_SKEW = timedelta(seconds=60)


class EventIn(BaseModel):
    """One behavior signal as the browser reports it."""

    type: str
    product_id: int | None = None
    query: str | None = Field(default=None, max_length=255)
    value: float | None = None
    ts: str | None = None  # ISO-8601 from the browser; parsed leniently below


class EventBatch(BaseModel):
    events: Annotated[list[EventIn], Field(max_length=MAX_BATCH)]


class BatchResult(BaseModel):
    accepted: int
    rejected: int


def _parse_ts(raw: str | None):
    """Parse a browser timestamp, clamping the future and defaulting to now."""
    from datetime import datetime, timezone

    now = utcnow()
    if not raw:
        return now
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return now
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return now if parsed > now + CLOCK_SKEW else parsed


@router.post("/events", response_model=BatchResult)
def ingest_events(
    batch: EventBatch,
    db: DbSession,
    user: Annotated[User, Depends(require_user)],
) -> BatchResult:
    """Store a batch of behavior events for the signed-in user."""
    referenced = {e.product_id for e in batch.events if e.product_id is not None}
    known: set[int] = set()
    if referenced:
        known = set(db.scalars(select(Product.id).where(Product.id.in_(referenced))).all())

    accepted = rejected = 0
    for event in batch.events:
        if event.type not in EVENT_TYPES:
            rejected += 1
            continue
        if event.product_id is not None and event.product_id not in known:
            rejected += 1  # a product deleted since the page was rendered
            continue
        value = event.value
        if value is not None:
            value = max(0.0, min(float(value), MAX_DWELL_SECONDS))
        try:
            record_event(
                db,
                user_id=user.id,
                type=event.type,
                product_id=event.product_id,
                query=event.query,
                value=value,
                ts=_parse_ts(event.ts),
            )
        except InvalidEvent:
            rejected += 1
            continue
        accepted += 1

    db.commit()
    return BatchResult(accepted=accepted, rejected=rejected)
