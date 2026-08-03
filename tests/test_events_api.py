"""POST /api/events — the bulk ingestion endpoint tracker.js posts to."""

from datetime import timedelta

from sqlalchemy import func, select

from app.models import Event, utcnow
from tests.conftest import login


def batch(*events):
    return {"events": list(events)}


def test_a_batch_is_stored_in_one_request(client, db, user, products):
    login(client, user.email)
    response = client.post(
        "/api/events",
        json=batch(
            {"type": "view", "product_id": products[0].id},
            {"type": "dwell", "product_id": products[0].id, "value": 41.5},
            {"type": "click", "product_id": products[1].id},
            {"type": "search", "query": "langgraph"},
            {"type": "cart", "product_id": products[0].id, "value": 1},
        ),
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": 5, "rejected": 0}
    stored = db.scalars(select(Event).order_by(Event.id)).all()
    assert [e.type for e in stored] == ["view", "dwell", "click", "search", "cart"]
    assert stored[1].value == 41.5
    assert stored[3].query == "langgraph"
    assert {e.user_id for e in stored} == {user.id}


def test_events_are_rejected_without_a_session(client, db, products):
    response = client.post("/api/events", json=batch({"type": "view", "product_id": products[0].id}))
    assert response.status_code == 401
    assert db.scalar(select(func.count()).select_from(Event)) == 0


def test_the_user_comes_from_the_cookie_not_the_payload(client, db, user, admin, products):
    """A client cannot attribute its browsing to somebody else."""
    login(client, user.email)
    client.post("/api/events", json=batch({"type": "view", "product_id": products[0].id, "user_id": admin.id}))

    stored = db.scalar(select(Event))
    assert stored.user_id == user.id


def test_one_bad_event_does_not_lose_the_rest_of_the_batch(client, db, user, products):
    login(client, user.email)
    response = client.post(
        "/api/events",
        json=batch(
            {"type": "view", "product_id": products[0].id},
            {"type": "teleport", "product_id": products[0].id},
            {"type": "view", "product_id": 999_999},
            {"type": "click", "product_id": products[1].id},
        ),
    )

    assert response.json() == {"accepted": 2, "rejected": 2}
    assert db.scalar(select(func.count()).select_from(Event)) == 2


def test_a_client_timestamp_is_kept(client, db, user, products):
    login(client, user.email)
    earlier = (utcnow() - timedelta(minutes=3)).isoformat()
    client.post("/api/events", json=batch({"type": "view", "product_id": products[0].id, "ts": earlier}))

    stored = db.scalar(select(Event))
    assert abs((stored.ts.replace(tzinfo=None) - utcnow().replace(tzinfo=None)).total_seconds() + 180) < 5


def test_a_future_timestamp_is_clamped_to_server_time(client, db, user, products):
    """A skewed client clock must not park events in the future, where the
    trigger engine's staleness window would never see them."""
    login(client, user.email)
    future = (utcnow() + timedelta(days=2)).isoformat()
    client.post("/api/events", json=batch({"type": "view", "product_id": products[0].id, "ts": future}))

    stored = db.scalar(select(Event))
    assert stored.ts.replace(tzinfo=None) <= utcnow().replace(tzinfo=None) + timedelta(seconds=5)


def test_an_unparseable_timestamp_falls_back_to_now(client, db, user, products):
    login(client, user.email)
    client.post("/api/events", json=batch({"type": "view", "product_id": products[0].id, "ts": "yesterday-ish"}))

    stored = db.scalar(select(Event))
    assert stored is not None
    assert stored.ts is not None


def test_dwell_values_are_clamped(client, db, user, products):
    login(client, user.email)
    client.post(
        "/api/events",
        json=batch(
            {"type": "dwell", "product_id": products[0].id, "value": 999_999},
            {"type": "dwell", "product_id": products[1].id, "value": -5},
        ),
    )

    values = sorted(e.value for e in db.scalars(select(Event)).all())
    assert values == [0.0, 3600.0]


def test_an_oversized_batch_is_refused(client, db, user, products):
    login(client, user.email)
    response = client.post(
        "/api/events",
        json=batch(*[{"type": "view", "product_id": products[0].id} for _ in range(101)]),
    )

    assert response.status_code == 422
    assert db.scalar(select(func.count()).select_from(Event)) == 0


def test_an_empty_batch_is_harmless(client, db, user):
    login(client, user.email)
    assert client.post("/api/events", json={"events": []}).json() == {"accepted": 0, "rejected": 0}


def test_the_tracker_is_only_served_to_signed_in_visitors(client, user, products):
    anonymous = client.get(f"/products/{products[0].id}").text
    assert "tracker.js" not in anonymous

    login(client, user.email)
    signed_in = client.get(f"/products/{products[0].id}").text
    assert "/static/tracker.js" in signed_in
    assert f'"userId": {user.id}'.replace('"', "") in signed_in.replace('"', "")
