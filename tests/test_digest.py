"""The proactive digest sends stored work and never generates new work."""

from datetime import timedelta

from sqlalchemy import func, select

from app.config import Settings
from app.db import SessionLocal
from app.models import Event, LLMCall, Recommendation, User, utcnow
from app.security import hash_password
from app.services.behavior import summarize
from app.services.digest import build_digest_scheduler, run_digest


def _user(db, email: str) -> User:
    user = User(email=email, pw_hash=hash_password("password123"))
    db.add(user)
    db.commit()
    return user


def _recommendation(
    db,
    user: User,
    narrative: str = "Stored recommendation.",
    behavior_hash: str = "stored",
) -> None:
    db.add(Recommendation(
        user_id=user.id,
        narrative=narrative,
        product_ids=[],
        behavior_hash=behavior_hash,
        trigger_reason="event_threshold",
        source="rule-based",
    ))
    db.commit()


def test_job_selects_only_users_active_in_the_last_seven_days(db):
    now = utcnow()
    active = _user(db, "active@example.com")
    inactive = _user(db, "inactive@example.com")
    _recommendation(db, active)
    _recommendation(db, inactive)
    db.add_all([
        Event(user_id=active.id, type="search", query="agents", ts=now - timedelta(days=6)),
        Event(user_id=inactive.id, type="search", query="sql", ts=now - timedelta(days=8)),
    ])
    db.commit()
    sent: list[int] = []

    delivered = run_digest(
        session_factory=SessionLocal,
        settings=Settings(),
        now=now,
        delivery=lambda user, message, settings: sent.append(user.id) or "test",
    )

    assert delivered == [active.id]
    assert sent == [active.id]


def test_active_user_without_a_stored_recommendation_is_skipped(db):
    user = _user(db, "new@example.com")
    db.add(Event(user_id=user.id, type="view", ts=utcnow()))
    db.commit()

    delivered = run_digest(
        session_factory=SessionLocal,
        settings=Settings(),
        delivery=lambda user, message, settings: "test",
    )

    assert delivered == []


def test_digest_makes_zero_llm_calls(db):
    user = _user(db, "quiet@example.com")
    db.add(Event(user_id=user.id, type="search", query="langgraph", ts=utcnow()))
    db.commit()
    _recommendation(db, user, behavior_hash=summarize(db, user.id).signature_hash)
    before = db.scalar(select(func.count()).select_from(LLMCall))

    run_digest(
        session_factory=SessionLocal,
        settings=Settings(),
        delivery=lambda user, message, settings: "test",
    )

    db.expire_all()
    after = db.scalar(select(func.count()).select_from(LLMCall))
    assert after == before


def test_digest_disabled_schedules_nothing():
    assert build_digest_scheduler(settings=Settings(digest_enabled=False)) is None
