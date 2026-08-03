"""The 60-second demo: seed, browse, watch the trigger engine, read the pitch.

    python scripts/demo.py            # offline, with a fake Mesh client
    python scripts/demo.py --live     # same run, calling Mesh for real

It replays each scripted journey one event at a time and asks the trigger engine
after every single event, so the interesting number is visible: how many events
went by, and how few model calls that cost.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timedelta

sys.path.insert(0, ".")

from sqlalchemy import delete, select  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.models import Event, LLMCall, Recommendation, utcnow  # noqa: E402
from app.services.agent import maybe_recommend  # noqa: E402
from app.services.catalog import reindex_all  # noqa: E402
from app.services.mesh import MeshResult, get_mesh_client  # noqa: E402
from app.services.trigger import efficiency_stats  # noqa: E402
from app.services.tracking import record_event  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402
from seed import JOURNEYS, load_catalog, upsert_accounts  # noqa: E402

BAR = "─" * 74
MINUTES_PER_STEP = 3  # how far apart the replayed events sit on the simulated clock


class FakeMesh:
    """Deterministic stand-in for Mesh, so the demo runs with no key and no cost.

    It reads the candidate ids out of the prompt it is given — which means it is
    still subject to the grounding validator, exactly like the real model.
    """

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, temperature=0.4, model=None) -> MeshResult:
        self.calls += 1
        prompt = messages[-1]["content"]
        ids = [int(m) for m in re.findall(r"id=(\d+)", prompt)][:2]
        titles = re.findall(r"id=\d+ \| ([^|]+) \|", prompt)[:2]
        narrative = (
            f"Your last few sessions point in one direction, so start with {titles[0].strip()}"
            + (f" and follow it with {titles[1].strip()}" if len(titles) > 1 else "")
            + ". Both pick up exactly where your recent reading stopped."
        )
        return MeshResult(
            text=json.dumps({"narrative": narrative, "product_ids": ids}),
            model="demo/fake-mesh",
            tokens_in=len(prompt) // 4,
            tokens_out=len(narrative) // 4,
            latency_ms=0,
        )


def reset(db) -> None:
    for table in (Event, Recommendation, LLMCall):
        db.execute(delete(table))
    db.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SmartReco demo.")
    parser.add_argument("--live", action="store_true", help="use the real Mesh client instead of the fake one")
    args = parser.parse_args()

    mesh = get_mesh_client() if args.live else FakeMesh()
    if args.live and mesh is None:
        print("--live needs MESH_API_KEY in .env")
        return 1

    init_db()
    store = get_vector_store()
    with SessionLocal() as db:
        reset(db)
        users = upsert_accounts(db)
        products = load_catalog(db)
        indexed = reindex_all(db, store=store)
        print(f"{BAR}\nCatalog: {len(products)} courses, {indexed} indexed with "
              f"{store.provider.name} embeddings\n{BAR}")

        start = utcnow() - timedelta(minutes=len(max(JOURNEYS.values(), key=len)) * MINUTES_PER_STEP)

        for email, steps in JOURNEYS.items():
            user = users[email]
            print(f"\n{email} — {len(steps)} events over "
                  f"{len(steps) * MINUTES_PER_STEP} simulated minutes")
            for index, (etype, target, value) in enumerate(steps):
                # A real session is spread over time; replaying it at machine speed
                # would make every decision a cooldown. The clock is injected.
                clock = start + timedelta(minutes=index * MINUTES_PER_STEP)
                if etype == "search":
                    record_event(db, user_id=user.id, type=etype, query=target, ts=clock)
                    label = f'search "{target}"'
                else:
                    record_event(db, user_id=user.id, type=etype,
                                 product_id=products[target].id, value=value, ts=clock)
                    label = f"{etype} {target}"
                db.commit()

                outcome = maybe_recommend(db, user.id, store=store, mesh=mesh, now=clock)
                stamp = clock.strftime("%H:%M")
                if outcome.ran:
                    dropped = f", dropped {outcome.dropped_ids}" if outcome.dropped_ids else ""
                    print(f"  {stamp}  {label:46.46s} AGENT RAN ({outcome.reason}{dropped})")
                else:
                    print(f"  {stamp}  {label:46.46s} skipped ({outcome.reason})")

            # Three page loads with no new activity: the behavior cache answers.
            end = start + timedelta(minutes=len(steps) * MINUTES_PER_STEP)
            for _ in range(3):
                repeat = maybe_recommend(db, user.id, store=store, mesh=mesh, now=end)
                print(f"  {end.strftime('%H:%M')}  {'(page reload, no new activity)':46.46s} "
                      f"skipped ({repeat.reason})")

            rec = db.scalar(
                select(Recommendation)
                .where(Recommendation.user_id == user.id)
                .order_by(Recommendation.version.desc())
            )
            if rec is None:
                print("  no recommendation stored")
                continue
            titles = [
                db.get(type(products[next(iter(products))]), pid).title for pid in rec.product_ids
            ]
            print(f"\n  ── recommendation v{rec.version} "
                  f"(trigger: {rec.trigger_reason}, written by: {rec.source}) ──")
            print(f"  {rec.narrative}")
            for title in titles:
                print(f"    · {title}")

        stats = efficiency_stats(db)
        print(f"\n{BAR}")
        print(f"{stats['events']} events → {stats['llm_calls']} model calls "
              f"({stats['cache_hits']} served from the behavior cache). "
              f"{stats['events_per_llm_call']} events per call.")
        print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
