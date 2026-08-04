"""Behavior summarisation and the activity signature the cache depends on."""

from datetime import timedelta

from app.models import utcnow
from app.services.behavior import summarize
from app.services.tracking import record_event


def add(db, user, etype, product=None, query=None, value=None, minutes_ago=0):
    record_event(
        db,
        user_id=user.id,
        type=etype,
        product_id=product.id if product else None,
        query=query,
        value=value,
        ts=utcnow() - timedelta(minutes=minutes_ago),
    )
    db.commit()


def test_an_empty_history_is_empty(db, user):
    profile = summarize(db, user.id)
    assert profile.is_empty
    assert profile.summary() == "No meaningful activity yet."


def test_the_profile_captures_what_the_user_did(db, user, products):
    langgraph, sql, mlops = products
    add(db, user, "search", query="langgraph agents", minutes_ago=5)
    add(db, user, "view", langgraph, minutes_ago=4)
    add(db, user, "dwell", langgraph, value=90.0, minutes_ago=3)
    add(db, user, "cart", langgraph, value=1.0, minutes_ago=2)

    profile = summarize(db, user.id)

    assert profile.events_considered == 4
    assert profile.counts == {"search": 1, "view": 1, "dwell": 1, "cart": 1}
    assert profile.searches == ["langgraph agents"]
    assert profile.viewed_titles == [langgraph.title]
    assert profile.carted_ids == [langgraph.id]
    assert profile.total_dwell_seconds == 90.0
    assert profile.top_categories[0] == langgraph.category
    assert "langgraph agents" in profile.summary()


def test_the_summary_hands_the_model_no_bare_numbers(db, user, products):
    """Interest weights are internal. Shown to a model they come back dressed as
    statistics — a live run turned a weight of 14.2 into "14.2% of your browsing
    time", which is a fact we never had."""
    import re

    add(db, user, "search", query="langgraph agents")
    add(db, user, "view", products[0])
    add(db, user, "dwell", products[0], value=142.0)

    summary = summarize(db, user.id).summary()

    assert "Interests, strongest first" in summary
    assert not re.search(r"\d+\.\d+", summary), f"no raw weights in: {summary}"
    assert "142" not in summary


def test_cart_outweighs_a_view(db, user, products):
    langgraph, sql, _ = products
    add(db, user, "view", sql)
    add(db, user, "view", sql)
    add(db, user, "cart", langgraph, value=1.0)

    profile = summarize(db, user.id)
    assert profile.top_categories[0] == langgraph.category, "intent beats idle browsing"


def test_long_dwells_count_more_than_short_ones(db, user, products):
    langgraph, sql, _ = products
    add(db, user, "dwell", sql, value=10.0)
    add(db, user, "dwell", langgraph, value=180.0)

    profile = summarize(db, user.id)
    assert profile.category_weights[langgraph.category] > profile.category_weights[sql.category]


def test_only_the_recent_window_is_considered(db, user, products):
    for i in range(50):
        add(db, user, "view", products[i % 3], minutes_ago=60 - i)

    profile = summarize(db, user.id)
    assert profile.events_considered == 40, "old behavior is not today's intent"


def test_the_signature_is_stable_for_identical_activity(db, user, products):
    add(db, user, "view", products[0])
    first = summarize(db, user.id).signature_hash
    second = summarize(db, user.id).signature_hash
    assert first == second


def test_the_signature_changes_when_something_new_happens(db, user, products):
    add(db, user, "view", products[0])
    before = summarize(db, user.id).signature_hash
    add(db, user, "click", products[1])
    assert summarize(db, user.id).signature_hash != before


def test_near_identical_dwells_share_a_signature(db, user, products):
    """41s and 43s of reading are the same behavior; the cache should not miss."""
    from sqlalchemy import select

    from app.models import Event

    add(db, user, "dwell", products[0], value=41.0)
    first = summarize(db, user.id).signature_hash

    db.scalar(select(Event)).value = 43.0
    db.commit()
    assert summarize(db, user.id).signature_hash == first

    db.scalar(select(Event)).value = 300.0
    db.commit()
    assert summarize(db, user.id).signature_hash != first, "five minutes is not forty seconds"


def test_the_retrieval_query_leads_with_stated_intent(db, user, products):
    add(db, user, "view", products[1])
    add(db, user, "search", query="multi agent supervisor")

    query = summarize(db, user.id).retrieval_query()
    assert query.startswith("multi agent supervisor")
    assert products[1].title in query
