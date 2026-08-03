"""Serving recommendations: the cold-start path, the background refresh, the page."""

from datetime import timedelta

from sqlalchemy import select

from app.models import LLMCall, Recommendation
from app.services.recommendations import current_for
from tests.conftest import login
from tests.test_agent import catalog  # noqa: F401  (fixture)
from tests.test_behavior import add


def test_a_new_user_gets_a_recommendation_on_the_first_render(db, user, store, catalog):
    """Cold start runs inline: an empty panel is worse than a moment of latency."""
    add(db, user, "search", query="langgraph agents")
    add(db, user, "view", catalog[0])

    view = current_for(db, user.id, store=store)

    assert view is not None
    assert view.version == 1
    assert view.products, "the panel shows real products"
    assert view.trigger_reason == "first_recommendation"


def test_a_refresh_is_pushed_into_the_background_when_there_is_something_to_show(db, user, store, catalog):
    add(db, user, "search", query="langgraph agents")
    current_for(db, user.id, store=store)

    rec = db.scalars(select(Recommendation)).one()
    rec.created_at = rec.created_at.replace(tzinfo=None) - timedelta(minutes=10)
    db.commit()
    add(db, user, "search", query="multi agent supervisor")

    scheduled: list[tuple] = []
    view = current_for(db, user.id, store=store, schedule=lambda fn, *a: scheduled.append((fn, a)))

    assert view.version == 1, "the page is served from the stored recommendation"
    assert len(scheduled) == 1, "the refresh runs after the response, not during it"
    assert scheduled[0][1] == (user.id,)


def test_an_unchanged_visit_records_a_cache_hit_and_shows_the_same_version(db, user, store, catalog):
    add(db, user, "view", catalog[0])
    first = current_for(db, user.id, store=store)

    again = current_for(db, user.id, store=store)

    assert again.version == first.version
    hits = db.scalars(select(LLMCall).where(LLMCall.cache_hit.is_(True))).all()
    assert len(hits) == 1


def test_the_home_page_renders_the_recommendation(client, db, user, store, catalog):
    login(client, user.email)
    body = client.get("/", params={"q": "langgraph agents"}).text

    rec = db.scalars(select(Recommendation)).one()
    assert "Picked for you" in body
    assert rec.narrative[:40] in body
    assert rec.trigger_reason in body
    assert rec.source in body, "the page says who wrote it"


def test_the_product_page_renders_the_recommendation(client, db, user, store, catalog):
    login(client, user.email)
    client.get("/", params={"q": "langgraph agents"})

    body = client.get(f"/products/{catalog[0].id}").text
    assert "Picked for you" in body


def test_anonymous_visitors_get_no_recommendation(client, db, store, catalog):
    body = client.get("/").text
    assert "Picked for you" not in body
    assert db.scalar(select(Recommendation)) is None


def test_a_quiet_user_sees_nothing_and_costs_nothing(client, db, user, store, catalog):
    login(client, user.email)
    body = client.get("/").text

    assert "Picked for you" not in body
    assert db.scalar(select(LLMCall)) is None
