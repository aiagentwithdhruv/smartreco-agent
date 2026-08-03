"""Catalog pages, search, and the two server-recorded behavior signals."""

from sqlalchemy import func, select

from app.models import Event
from tests.conftest import login


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_home_lists_the_catalog(client, products):
    body = client.get("/").text
    for product in products:
        assert product.title in body


def test_search_filters_the_catalog(client, products):
    body = client.get("/", params={"q": "langgraph"}).text
    assert "Agentic Workflows with LangGraph" in body
    assert "SQL for Data Analysis" not in body


def test_category_filter(client, products):
    body = client.get("/", params={"category": "MLOps"}).text
    assert "MLOps: Deploying and Monitoring Models" in body
    assert "SQL for Data Analysis" not in body


def test_search_records_a_high_intent_event_for_signed_in_users(client, db, user, products):
    login(client, user.email)
    client.get("/", params={"q": "langgraph"})

    event = db.scalar(select(Event).where(Event.user_id == user.id))
    assert event is not None
    assert event.type == "search"
    assert event.query == "langgraph"


def test_search_records_nothing_for_anonymous_visitors(client, db, products):
    client.get("/", params={"q": "langgraph"})
    assert db.scalar(select(func.count()).select_from(Event)) == 0


def test_browsing_without_a_query_records_no_search(client, db, user, products):
    login(client, user.email)
    client.get("/")
    client.get("/", params={"q": "   "})
    assert db.scalar(select(func.count()).select_from(Event)) == 0


def test_product_detail_renders(client, products):
    product = products[0]
    body = client.get(f"/products/{product.id}").text
    assert product.title in body
    assert f'data-product-id="{product.id}"' in body


def test_missing_product_is_404(client):
    assert client.get("/products/9999").status_code == 404


def test_add_to_cart_records_a_cart_event(client, db, user, products):
    login(client, user.email)
    product = products[0]
    response = client.post("/cart/add", data={"product_id": product.id}, follow_redirects=False)
    assert response.status_code == 303

    event = db.scalar(select(Event).where(Event.type == "cart"))
    assert event is not None
    assert event.product_id == product.id
    assert event.user_id == user.id


def test_add_to_cart_requires_login(client, db, products):
    response = client.post("/cart/add", data={"product_id": products[0].id}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert db.scalar(select(func.count()).select_from(Event)) == 0
