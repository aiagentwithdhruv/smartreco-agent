"""The trigger engine — when is thinking worth paying for?

A recommendation agent that calls a model on every click is a bill, not a
product. This module is the gate: it decides, per request, whether the agent may
run. Everything it does is measurable, and the counts land in `llm_calls`.

Fire when any of these hold:
  * the user has no recommendation yet and has done something  (first_recommendation)
  * a search happened since the last recommendation            (search_intent — stated intent)
  * N meaningful events have piled up since the last one       (event_threshold)
  * the last one is older than the staleness window and there  (staleness)
    has been activity since

Never fire when:
  * the activity signature is unchanged                        (cache_hit — serve the stored one)
  * the last recommendation is younger than the cooldown       (cooldown)
  * nothing has happened at all                                (no_activity)

Cache is checked before cooldown on purpose: a cache hit is the cheapest and
most common answer, and it is the honest one — the input has not changed, so
neither should the output.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from weakref import WeakValueDictionary

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Event, Recommendation, utcnow
from app.services.behavior import BehaviorProfile, summarize

RUN_REASONS = ("first_recommendation", "search_intent", "event_threshold", "staleness")
SKIP_REASONS = ("no_activity", "cache_hit", "cooldown", "below_threshold")

# This registry deliberately scopes serialization to one user and one process.
# Weak values keep inactive user ids from accumulating forever. The registry
# guard protects only lock lookup/creation; it never serializes agent work.
_user_locks_guard = Lock()
_user_locks: WeakValueDictionary[int, Lock] = WeakValueDictionary()


@contextmanager
def user_trigger_lock(user_id: int):
    """Serialize trigger-authorized work for one user in this process."""
    with _user_locks_guard:
        lock = _user_locks.get(user_id)
        if lock is None:
            lock = Lock()
            _user_locks[user_id] = lock
    with lock:
        yield


@dataclass
class TriggerDecision:
    """Whether the agent runs, why, and what it would run on."""

    run: bool
    reason: str
    profile: BehaviorProfile
    current: Recommendation | None = None  # the stored recommendation, if any
    events_since: int = 0

    @property
    def cache_hit(self) -> bool:
        return self.reason == "cache_hit"


def _aware(value):
    """SQLite hands back naive datetimes; compare them in UTC."""
    if value is not None and value.tzinfo is None:
        from datetime import timezone

        return value.replace(tzinfo=timezone.utc)
    return value


def decide(
    db: Session,
    user_id: int,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> TriggerDecision:
    """Decide whether to run the agent for this user right now.

    `now` is injectable so a replayed session (scripts/demo.py) and the tests
    can exercise cooldown and staleness without sleeping.
    """
    settings = settings or get_settings()
    now = now or utcnow()
    profile = summarize(db, user_id)
    current = db.scalar(
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.version.desc())
        .limit(1)
    )

    if profile.is_empty:
        return TriggerDecision(False, "no_activity", profile, current)

    if current is None:
        return TriggerDecision(True, "first_recommendation", profile, None, profile.events_considered)

    # Same behavior in, same recommendation out — no model call needed.
    if current.behavior_hash == profile.signature_hash:
        return TriggerDecision(False, "cache_hit", profile, current)

    created = _aware(current.created_at)
    if now - created < timedelta(seconds=settings.trigger_cooldown_seconds):
        return TriggerDecision(False, "cooldown", profile, current)

    since = list(
        db.scalars(
            select(Event)
            .where(Event.user_id == user_id, Event.ts > created)
            .order_by(Event.ts)
        ).all()
    )
    events_since = len(since)

    if any(event.type == "search" for event in since):
        return TriggerDecision(True, "search_intent", profile, current, events_since)

    if events_since >= settings.trigger_min_events:
        return TriggerDecision(True, "event_threshold", profile, current, events_since)

    if events_since and now - created > timedelta(minutes=settings.trigger_staleness_minutes):
        return TriggerDecision(True, "staleness", profile, current, events_since)

    return TriggerDecision(False, "below_threshold", profile, current, events_since)


def efficiency_stats(db: Session) -> dict[str, int | float]:
    """Events in, model calls out — the number the README quotes."""
    from app.models import LLMCall

    events = db.scalar(select(func.count()).select_from(Event)) or 0
    calls = db.scalar(select(func.count()).select_from(LLMCall).where(LLMCall.cache_hit.is_(False))) or 0
    hits = db.scalar(select(func.count()).select_from(LLMCall).where(LLMCall.cache_hit.is_(True))) or 0
    return {
        "events": events,
        "llm_calls": calls,
        "cache_hits": hits,
        "events_per_llm_call": round(events / calls, 1) if calls else 0.0,
    }
