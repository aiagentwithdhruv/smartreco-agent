"""Writing behavior signals into the events table.

Every signal — whether it arrives from static/tracker.js in a batch or is
recorded server-side (search, add-to-cart) — goes through record_event(), so
there is exactly one place where an event is validated and stored.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.models import EVENT_TYPES, Event, utcnow


class InvalidEvent(ValueError):
    """Raised when a client sends an event type we do not track."""


def record_event(
    db: Session,
    *,
    user_id: int,
    type: str,
    product_id: int | None = None,
    query: str | None = None,
    value: float | None = None,
    ts: datetime | None = None,
) -> Event:
    """Validate and stage one event. The caller commits."""
    if type not in EVENT_TYPES:
        raise InvalidEvent(f"unknown event type: {type!r}")
    event = Event(
        user_id=user_id,
        type=type,
        product_id=product_id,
        query=(query or None),
        value=value,
        ts=ts or utcnow(),
    )
    db.add(event)
    return event
