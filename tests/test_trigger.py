"""The trigger engine — the judged decision of when the agent may think.

Every rule in app/services/trigger.py has a test here, and each one is written
so that removing the rule makes it fail.
"""

from datetime import timedelta

import pytest

from app.config import Settings
from app.models import Recommendation, utcnow
from app.services.behavior import summarize
from app.services.trigger import decide, efficiency_stats
from tests.test_behavior import add

SETTINGS = Settings(
    trigger_min_events=8,
    trigger_cooldown_seconds=300,
    trigger_staleness_minutes=30,
    mesh_api_key="",
)


def store_recommendation(db, user, *, behavior_hash="", age_minutes=0, version=1):
    rec = Recommendation(
        user_id=user.id,
        narrative="stored",
        product_ids=[1],
        behavior_hash=behavior_hash,
        trigger_reason="event_threshold",
        version=version,
        created_at=utcnow() - timedelta(minutes=age_minutes),
    )
    db.add(rec)
    db.commit()
    return rec


def test_no_activity_never_triggers(db, user):
    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is False
    assert decision.reason == "no_activity"


def test_the_first_meaningful_event_triggers_a_cold_start(db, user, products):
    add(db, user, "view", products[0])

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is True
    assert decision.reason == "first_recommendation"


def test_unchanged_behavior_is_a_cache_hit_and_calls_nothing(db, user, products):
    add(db, user, "view", products[0])
    signature = summarize(db, user.id).signature_hash
    store_recommendation(db, user, behavior_hash=signature)

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is False
    assert decision.reason == "cache_hit"
    assert decision.cache_hit is True


def test_the_cache_is_checked_before_the_cooldown(db, user, products):
    """A page refresh inside the cooldown window is a cache hit, not a cooldown —
    the distinction is what makes the efficiency number meaningful."""
    add(db, user, "view", products[0])
    signature = summarize(db, user.id).signature_hash
    store_recommendation(db, user, behavior_hash=signature, age_minutes=1)

    assert decide(db, user.id, settings=SETTINGS).reason == "cache_hit"


def test_new_behavior_inside_the_cooldown_still_waits(db, user, products):
    add(db, user, "view", products[0], minutes_ago=10)
    store_recommendation(db, user, behavior_hash="stale", age_minutes=1)
    for _ in range(12):
        add(db, user, "click", products[1])

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is False
    assert decision.reason == "cooldown", "a burst of clicks must not outrun the cooldown"


def test_a_search_triggers_immediately_after_the_cooldown(db, user, products):
    store_recommendation(db, user, behavior_hash="stale", age_minutes=6)
    add(db, user, "search", query="langgraph agents")

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is True
    assert decision.reason == "search_intent", "one search is stronger than eight views"
    assert decision.events_since == 1


def test_eight_events_trigger_the_threshold(db, user, products):
    store_recommendation(db, user, behavior_hash="stale", age_minutes=6)
    for _ in range(7):
        add(db, user, "view", products[0])

    assert decide(db, user.id, settings=SETTINGS).reason == "below_threshold"

    add(db, user, "view", products[1])
    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is True
    assert decision.reason == "event_threshold"
    assert decision.events_since == 8


def test_staleness_triggers_on_light_activity(db, user, products):
    store_recommendation(db, user, behavior_hash="stale", age_minutes=45)
    add(db, user, "view", products[0])

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is True
    assert decision.reason == "staleness"


def test_staleness_alone_does_not_trigger_without_new_activity(db, user, products):
    add(db, user, "view", products[0], minutes_ago=90)
    store_recommendation(db, user, behavior_hash="stale", age_minutes=45)

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.run is False
    assert decision.reason == "below_threshold", "an idle user costs nothing"


def test_only_events_after_the_last_recommendation_count(db, user, products):
    for _ in range(20):
        add(db, user, "view", products[0], minutes_ago=30)
    store_recommendation(db, user, behavior_hash="stale", age_minutes=6)
    add(db, user, "view", products[1])

    decision = decide(db, user.id, settings=SETTINGS)
    assert decision.events_since == 1
    assert decision.reason == "below_threshold"


@pytest.mark.parametrize("threshold", [3, 15])
def test_the_threshold_is_configurable(db, user, products, threshold):
    settings = SETTINGS.model_copy(update={"trigger_min_events": threshold})
    store_recommendation(db, user, behavior_hash="stale", age_minutes=6)
    for _ in range(threshold):
        add(db, user, "click", products[0])

    assert decide(db, user.id, settings=settings).reason == "event_threshold"


def test_efficiency_stats_report_events_against_real_calls(db, user, products):
    from app.services.mesh import log_llm_call

    for _ in range(12):
        add(db, user, "view", products[0])
    log_llm_call(db, purpose="generate_rec", user_id=user.id, model="fake")
    log_llm_call(db, purpose="generate_rec", user_id=user.id, cache_hit=True)
    db.commit()

    stats = efficiency_stats(db)
    assert stats["events"] == 12
    assert stats["llm_calls"] == 1, "cache hits are not model calls"
    assert stats["cache_hits"] == 1
    assert stats["events_per_llm_call"] == 12.0
