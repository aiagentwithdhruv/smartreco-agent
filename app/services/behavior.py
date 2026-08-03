"""Turning raw events into something an agent can reason about.

This is deliberately deterministic — no model call. Summarising behavior with an
LLM would double the cost of every recommendation for a job that weighted counts
do better. The LLM is spent on the one thing only it can do: writing the pitch.

The summary carries a `signature_hash`: a stable fingerprint of the recent
activity. Identical activity ⇒ identical hash ⇒ the trigger engine serves the
stored recommendation and calls no model at all.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Event, Product

# How much each signal says about intent. Add-to-cart is worth four views.
WEIGHTS = {"view": 1.0, "click": 1.5, "search": 2.0, "cart": 4.0}
DWELL_SECONDS_PER_POINT = 45.0
MAX_DWELL_POINTS = 3.0
WINDOW = 40  # events considered; older behavior is not what someone wants today


@dataclass
class BehaviorProfile:
    """What we know about one user's current intent."""

    user_id: int
    events_considered: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)
    viewed_titles: list[str] = field(default_factory=list)
    carted_titles: list[str] = field(default_factory=list)
    carted_ids: list[int] = field(default_factory=list)
    category_weights: dict[str, float] = field(default_factory=dict)
    level_hint: str | None = None
    total_dwell_seconds: float = 0.0
    last_event_at: datetime | None = None
    signature_hash: str = ""

    @property
    def is_empty(self) -> bool:
        return self.events_considered == 0

    @property
    def top_categories(self) -> list[str]:
        """Categories worth filtering retrieval by, strongest first."""
        ranked = sorted(self.category_weights.items(), key=lambda kv: -kv[1])
        return [name for name, _ in ranked[:2]]

    def summary(self) -> str:
        """A compact, human-readable brief — this is what the LLM is shown."""
        parts: list[str] = []
        if self.searches:
            parts.append("Searched for: " + "; ".join(self.searches[-4:]) + ".")
        if self.viewed_titles:
            parts.append("Recently opened: " + ", ".join(self.viewed_titles[-5:]) + ".")
        if self.carted_titles:
            parts.append("Added to cart: " + ", ".join(self.carted_titles) + ".")
        if self.category_weights:
            ranked = sorted(self.category_weights.items(), key=lambda kv: -kv[1])
            parts.append(
                "Strongest interest: "
                + ", ".join(f"{name} ({weight:.1f})" for name, weight in ranked[:3])
                + "."
            )
        if self.total_dwell_seconds:
            parts.append(f"Spent about {round(self.total_dwell_seconds)}s reading course pages.")
        if self.level_hint:
            parts.append(f"Browsing mostly {self.level_hint} material.")
        if not parts:
            return "No meaningful activity yet."
        return " ".join(parts)

    def retrieval_query(self) -> str:
        """The text embedded to search the catalog. Searches lead — they are stated intent."""
        bits: list[str] = []
        bits.extend(self.searches[-3:])
        bits.extend(self.viewed_titles[-4:])
        bits.extend(self.top_categories)
        if self.level_hint:
            bits.append(f"{self.level_hint} level")
        return " ".join(bits) if bits else "popular online courses"


def summarize(db: Session, user_id: int, *, window: int = WINDOW) -> BehaviorProfile:
    """Build a profile from this user's most recent events."""
    rows = list(
        db.execute(
            select(Event, Product)
            .join(Product, Event.product_id == Product.id, isouter=True)
            .where(Event.user_id == user_id)
            .order_by(Event.ts.desc(), Event.id.desc())
            .limit(window)
        ).all()
    )
    rows.reverse()  # back to chronological order

    profile = BehaviorProfile(user_id=user_id, events_considered=len(rows))
    if not rows:
        profile.signature_hash = _hash([])
        return profile

    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    signature: list[tuple] = []

    for event, product in rows:
        counts[event.type] += 1
        weight = WEIGHTS.get(event.type, 1.0)

        if event.type == "search" and event.query:
            if event.query not in profile.searches:
                profile.searches.append(event.query)
        if event.type == "dwell" and event.value:
            profile.total_dwell_seconds += event.value
            weight = min(event.value / DWELL_SECONDS_PER_POINT, MAX_DWELL_POINTS)
        if product is not None:
            categories[product.category] += weight
            levels[product.level] += weight
            if event.type == "view" and product.title not in profile.viewed_titles:
                profile.viewed_titles.append(product.title)
            if event.type == "cart":
                if product.title not in profile.carted_titles:
                    profile.carted_titles.append(product.title)
                if product.id not in profile.carted_ids:
                    profile.carted_ids.append(product.id)

        # The signature describes *what happened*, not when — two identical
        # sessions hash the same, and one new click changes the hash.
        signature.append((event.type, event.product_id, event.query, _bucket(event.value)))

    profile.counts = dict(counts)
    profile.category_weights = {name: round(weight, 2) for name, weight in categories.items()}
    profile.level_hint = levels.most_common(1)[0][0] if levels else None
    profile.last_event_at = rows[-1][0].ts
    profile.signature_hash = _hash(signature)
    return profile


def _bucket(value: float | None) -> int | None:
    """Round dwell into 15s buckets: 41s and 43s are the same behavior."""
    if value is None:
        return None
    return int(value // 15)


def _hash(signature: list[tuple]) -> str:
    return hashlib.sha256(json.dumps(signature, default=str).encode()).hexdigest()
