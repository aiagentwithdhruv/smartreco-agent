"""Model-level invariants and the single event-writing path."""

import pytest
from sqlalchemy import select

from app.models import Event, Product
from app.services.tracking import InvalidEvent, record_event


def test_embedding_text_includes_the_fields_we_search_on(db):
    product = Product(
        title="Advanced RAG",
        description="Reranking and grading.",
        category="LLM Engineering",
        level="advanced",
        tags=["rag", "reranking"],
    )
    text = product.embedding_text()
    for fragment in ("Advanced RAG", "LLM Engineering", "advanced", "Reranking and grading.", "reranking"):
        assert fragment in text


def test_record_event_rejects_unknown_types(db, user):
    with pytest.raises(InvalidEvent):
        record_event(db, user_id=user.id, type="teleport")


def test_record_event_accepts_every_tracked_type(db, user, products):
    for etype in ("view", "search", "click", "dwell", "cart"):
        record_event(db, user_id=user.id, type=etype, product_id=products[0].id, query="q", value=1.0)
    db.commit()
    assert len(db.scalars(select(Event)).all()) == 5


def test_empty_query_is_stored_as_null(db, user):
    record_event(db, user_id=user.id, type="search", query="")
    db.commit()
    assert db.scalar(select(Event)).query is None


def test_events_are_indexed_by_user_and_time(db):
    """The agent's hot read is 'this user's recent events' — it must be indexed."""
    indexes = {idx.name for idx in Event.__table__.indexes}
    assert "ix_events_user_ts" in indexes
