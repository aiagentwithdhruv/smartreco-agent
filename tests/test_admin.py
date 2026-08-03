"""Admin CRUD routes — access control plus the dual-write reaching Chroma."""

from sqlalchemy import select

from app.models import Product
from tests.conftest import login

FORM = {
    "title": "Advanced RAG",
    "category": "LLM Engineering",
    "price": "6499",
    "level": "advanced",
    "tags": "rag, reranking",
    "description": "Retrieval grading, hybrid search and query rewriting.",
}


def test_admin_pages_require_login(client):
    response = client.get("/admin/products", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_admin_pages_reject_regular_users(client, user):
    login(client, user.email)
    assert client.get("/admin/products").status_code == 403
    assert client.post("/admin/products", data=FORM).status_code == 403


def test_admin_can_create_a_product_and_it_lands_in_both_stores(client, db, admin, store):
    login(client, admin.email)
    response = client.post("/admin/products", data=FORM, follow_redirects=False)
    assert response.status_code == 303

    product = db.scalar(select(Product).where(Product.title == "Advanced RAG"))
    assert product is not None
    assert product.tags == ["rag", "reranking"]
    assert product.vector_synced is True
    assert store.all_ids() == {product.id}


def test_admin_can_edit_a_product(client, db, admin, store):
    login(client, admin.email)
    client.post("/admin/products", data=FORM)
    product = db.scalar(select(Product))

    client.post(f"/admin/products/{product.id}", data={**FORM, "title": "Advanced RAG v2", "price": "7000"})
    db.refresh(product)

    assert product.title == "Advanced RAG v2"
    assert product.price == 7000
    assert store.count() == 1


def test_admin_can_delete_a_product(client, db, admin, store):
    login(client, admin.email)
    client.post("/admin/products", data=FORM)
    product = db.scalar(select(Product))

    client.post(f"/admin/products/{product.id}/delete")

    assert db.scalar(select(Product)) is None
    assert store.count() == 0


def test_invalid_level_is_rejected(client, db, admin):
    login(client, admin.email)
    response = client.post("/admin/products", data={**FORM, "level": "wizard"})
    assert response.status_code == 400
    assert db.scalar(select(Product)) is None


def test_missing_title_is_rejected(client, db, admin):
    login(client, admin.email)
    response = client.post("/admin/products", data={**FORM, "title": "   "})
    assert response.status_code == 400
    assert db.scalar(select(Product)) is None


def test_negative_price_is_rejected(client, db, admin):
    login(client, admin.email)
    assert client.post("/admin/products", data={**FORM, "price": "-5"}).status_code == 400


def test_repair_button_reconciles_and_reports(client, db, admin, store, products):
    """products fixture inserts rows directly — i.e. never indexed."""
    login(client, admin.email)
    assert store.count() == 0

    response = client.post("/admin/sync/repair", follow_redirects=False)
    assert response.status_code == 303
    assert "3+reindexed" in response.headers["location"]
    assert store.count() == 3


def test_admin_table_shows_the_sync_state(client, db, admin, products):
    login(client, admin.email)
    body = client.get("/admin/products").text
    assert "unsynced" in body
    assert "3 rows in SQLite" in body
    assert "local-hashing" in body
