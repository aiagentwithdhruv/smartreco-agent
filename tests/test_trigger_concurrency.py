"""Concurrency guarantees at the trigger-to-agent boundary."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier, BrokenBarrierError, Event, Lock

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Recommendation, User, utcnow
from app.security import hash_password
from app.services.agent import maybe_recommend
from app.services.mesh import MeshResult
from app.services.vector_store import VectorHit
from tests.test_behavior import add
from tests.test_trigger import SETTINGS, store_recommendation


class HoldingMesh:
    """Count calls and briefly hold the first so contenders can reach it."""

    def __init__(self, product_id: int, expected_if_racy: int = 6):
        self.product_id = product_id
        self.expected_if_racy = expected_if_racy
        self.calls = 0
        self._lock = Lock()
        self._racy_calls_arrived = Event()

    def chat(self, messages, *, temperature=0.4, model=None):
        with self._lock:
            self.calls += 1
            if self.calls >= self.expected_if_racy:
                self._racy_calls_arrived.set()
        # With the guard, this times out and the sole call completes. Without
        # either half of the guard, all contenders arrive and release it early.
        self._racy_calls_arrived.wait(timeout=1.0)
        return _mesh_result(self.product_id)


class OverlapMesh:
    """Prove two users can be inside model calls at the same time."""

    def __init__(self, product_id: int):
        self.product_id = product_id
        self.calls = 0
        self.overlapped = True
        self._barrier = Barrier(2)
        self._lock = Lock()

    def chat(self, messages, *, temperature=0.4, model=None):
        with self._lock:
            self.calls += 1
        try:
            self._barrier.wait(timeout=2.0)
        except BrokenBarrierError:
            self.overlapped = False
        return _mesh_result(self.product_id)


def _mesh_result(product_id: int) -> MeshResult:
    return MeshResult(
        text=json.dumps({"narrative": "A grounded pick.", "product_ids": [product_id]}),
        model="fake/concurrency",
        tokens_in=10,
        tokens_out=5,
        latency_ms=1,
    )


def _fixed_confident_retrieval(monkeypatch, products):
    hits = [
        VectorHit(
            product_id=product.id,
            distance=0.25,
            metadata={"title": product.title, "category": product.category},
        )
        for product in products
    ]
    monkeypatch.setattr("app.services.agent.retrieve", lambda *args, **kwargs: (hits, False))


def _make_ready(db, user, products, now):
    store_recommendation(db, user, behavior_hash="old", age_minutes=0)
    rec = db.scalar(
        select(Recommendation)
        .where(Recommendation.user_id == user.id)
        .order_by(Recommendation.version.desc())
    )
    rec.created_at = (now - timedelta(seconds=301)).replace(tzinfo=None)
    db.commit()
    add(db, user, "search", query="concurrency-safe recommendations")


def _run_concurrently(user_ids, *, mesh, now):
    start = Barrier(len(user_ids))

    def invoke(user_id):
        with SessionLocal() as thread_db:
            start.wait(timeout=3.0)
            return maybe_recommend(
                thread_db,
                user_id,
                store=object(),
                mesh=mesh,
                settings=SETTINGS,
                now=now,
            )

    with ThreadPoolExecutor(max_workers=len(user_ids)) as pool:
        return list(pool.map(invoke, user_ids))


def test_same_user_concurrency_makes_exactly_one_model_call(
    db, user, products, monkeypatch
):
    _fixed_confident_retrieval(monkeypatch, products)
    now = utcnow()
    _make_ready(db, user, products, now)
    mesh = HoldingMesh(products[0].id, expected_if_racy=6)

    outcomes = _run_concurrently([user.id] * 6, mesh=mesh, now=now)

    assert mesh.calls == 1
    assert sum(outcome.ran for outcome in outcomes) == 1
    assert all(outcome.recommendation is not None for outcome in outcomes)


def test_different_users_generate_concurrently(db, user, products, monkeypatch):
    _fixed_confident_retrieval(monkeypatch, products)
    other = User(
        email="other@example.com",
        pw_hash=hash_password("password123"),
        role="user",
    )
    db.add(other)
    db.commit()
    now = utcnow()
    _make_ready(db, user, products, now)
    _make_ready(db, other, products, now)
    mesh = OverlapMesh(products[0].id)

    outcomes = _run_concurrently([user.id, other.id], mesh=mesh, now=now)

    assert mesh.calls == 2
    assert mesh.overlapped, "different users must not wait on one global generation lock"
    assert all(outcome.ran for outcome in outcomes)


def test_losing_same_user_caller_gets_cooldown_and_stored_recommendation(
    db, user, products, monkeypatch
):
    _fixed_confident_retrieval(monkeypatch, products)
    now = utcnow()
    _make_ready(db, user, products, now)
    mesh = HoldingMesh(products[0].id, expected_if_racy=2)

    outcomes = _run_concurrently([user.id, user.id], mesh=mesh, now=now)

    winner = next(outcome for outcome in outcomes if outcome.ran)
    loser = next(outcome for outcome in outcomes if not outcome.ran)
    assert mesh.calls == 1
    assert loser.reason == "cooldown"
    assert loser.recommendation is not None
    assert loser.recommendation.id == winner.recommendation.id
    assert loser.recommendation.version == 2
