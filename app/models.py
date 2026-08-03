"""SQLAlchemy models — the whole data model of SmartReco lives here.

Five tables:
  users            who is browsing (and who can administer the catalog)
  products         the course catalog, mirrored into Chroma
  events           raw behavior signals sent by static/tracker.js
  recommendations  versioned agent output per user
  llm_calls        every call made to Mesh — our own observability trail
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Behavior signal vocabulary. `cart` is the strongest purchase intent short of
# checkout; `search` is the strongest *stated* intent and triggers the agent.
EVENT_TYPES = ("view", "search", "click", "dwell", "cart")


def utcnow() -> datetime:
    """Timezone-aware UTC now (SQLite has no native tz support)."""
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    pw_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="user", nullable=False)  # user | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    events: Mapped[list["Event"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    level: Mapped[str] = mapped_column(String(32), default="beginner")  # beginner|intermediate|advanced
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    # False whenever SQLite and Chroma may disagree — see services/catalog.py.
    vector_synced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def embedding_text(self) -> str:
        """The text actually embedded into the vector store."""
        tags = ", ".join(self.tags or [])
        return f"{self.title}. Category: {self.category}. Level: {self.level}. {self.description} Tags: {tags}"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id", ondelete="SET NULL"), nullable=True)
    query: Mapped[str | None] = mapped_column(String(255), nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)  # dwell seconds, quantity, ...
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="events")


# The agent reads "this user's recent events, newest first" on every request.
Index("ix_events_user_ts", Event.user_id, Event.ts)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")
    product_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    # Signature of the behavior that produced this rec — identical signature ⇒ no LLM call.
    behavior_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="")
    trigger_reason: Mapped[str] = mapped_column(String(64), default="")
    # Which model wrote the narrative, or "rule-based" when no LLM was available.
    # Recorded so a recommendation never has to be taken on trust.
    source: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    purpose: Mapped[str] = mapped_column(String(64), nullable=False)  # embed_product | generate_rec | ...
    model: Mapped[str] = mapped_column(String(128), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    # True when we served a stored result instead of calling Mesh. The ratio of
    # cache_hit rows to total rows is the efficiency number the README reports.
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
