"""The recommendation agent.

    behavior profile → retrieve from our own catalog → generate a pitch → ground it

Two properties matter more than the prose it writes:

* **It can only recommend what it retrieved.** The generator is handed a
  candidate list and asked for ids. Any id that was not in that list is dropped
  by ground() before anything is stored. The model cannot invent a course.
* **It is never called speculatively.** app/services/trigger.py decides whether
  this module runs at all, and every call — including every avoided call — is
  written to `llm_calls`.

Without a Mesh key the agent still produces a recommendation, from the retrieved
products and the behavior profile, and records `source="rule-based"`. It does not
pretend a model wrote it.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Product, Recommendation, utcnow
from app.services.behavior import BehaviorProfile
from app.services.mesh import MeshClient, get_mesh_client, log_llm_call
from app.services.trigger import TriggerDecision, decide
from app.services.vector_store import VectorHit, VectorStore, get_vector_store

log = logging.getLogger(__name__)

RULE_BASED = "rule-based"
MIN_CANDIDATES = 3  # below this, the category filter is dropped and retrieval widened

SYSTEM_PROMPT = (
    "You are the course advisor for SmartReco, an online course marketplace. "
    "You write short, specific, persuasive notes to a learner about what to study next. "
    "You may ONLY recommend courses from the CANDIDATES list you are given, referring to them by id. "
    "Never invent a course, a price or a claim about content. "
    "Two or three sentences, warm and concrete, referencing what the learner actually did. "
    "No bullet points, no hype, no exclamation marks."
)


@dataclass
class AgentOutcome:
    """What one pass of the agent did — the shape the tests and demo assert on."""

    ran: bool
    reason: str
    recommendation: Recommendation | None = None
    retrieved_ids: list[int] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)
    llm_used: bool = False
    widened: bool = False


# ---------------------------------------------------------------- retrieval --


def retrieve(
    store: VectorStore,
    profile: BehaviorProfile,
    *,
    top_k: int,
    db: Session | None = None,
) -> tuple[list[VectorHit], bool]:
    """Fetch candidates for this profile. Returns (hits, widened).

    First pass is filtered to the categories the behavior points at. If that
    leaves too little to choose from, the filter is dropped rather than handing
    the generator two options and calling it a recommendation.
    """
    query = profile.retrieval_query()
    filters = {"category": {"$in": profile.top_categories}} if profile.top_categories else None

    hits = store.query(query, top_k=top_k, filters=filters, exclude_ids=profile.carted_ids, db=db)
    if len(hits) >= MIN_CANDIDATES or filters is None:
        return hits, False

    wider = store.query(query, top_k=top_k, exclude_ids=profile.carted_ids, db=db)
    return wider, True


# --------------------------------------------------------------- generation --


def _candidate_block(products: list[Product]) -> str:
    return "\n".join(
        f"- id={p.id} | {p.title} | {p.category} | {p.level} | ₹{p.price:,.0f} | {p.description[:140]}"
        for p in products
    )


def _build_messages(profile: BehaviorProfile, products: list[Product]) -> list[dict[str, str]]:
    user_prompt = (
        f"LEARNER BEHAVIOR:\n{profile.summary()}\n\n"
        f"CANDIDATES (the only courses you may recommend):\n{_candidate_block(products)}\n\n"
        "Pick the 2-3 best fits and reply with JSON only, no code fence:\n"
        '{"narrative": "<2-3 sentences>", "product_ids": [<ids from CANDIDATES>]}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_generation(text: str) -> tuple[str, list[int]]:
    """Pull (narrative, product_ids) out of a model reply.

    Models wrap JSON in prose or code fences often enough that being strict here
    would mean throwing away good answers, so the first JSON object in the reply
    wins. A reply we cannot parse yields no ids, and the caller falls back.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return "", []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", []
    narrative = str(data.get("narrative") or "").strip()
    ids: list[int] = []
    for raw in data.get("product_ids") or []:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return narrative, ids


def ground(candidate_ids: list[int], allowed_ids: list[int]) -> tuple[list[int], list[int]]:
    """The grounding validator: only retrieved ids survive.

    Returns (kept, dropped), both in the order the model gave them, deduplicated.
    This is the guarantee that a hallucinated course id never reaches a user.
    """
    allowed = set(allowed_ids)
    kept: list[int] = []
    dropped: list[int] = []
    for product_id in candidate_ids:
        if product_id in kept or product_id in dropped:
            continue
        (kept if product_id in allowed else dropped).append(product_id)
    return kept, dropped


def _rule_based_narrative(profile: BehaviorProfile, products: list[Product]) -> str:
    """A deterministic fallback pitch. Labelled `rule-based`, never as a model."""
    if not products:
        return "Nothing to recommend yet — browse a few courses and this will fill in."
    interest = profile.top_categories[0] if profile.top_categories else products[0].category
    titles = ", ".join(p.title for p in products[:2])
    opener = (
        f"You have been looking at {interest.lower()} material"
        + (f", including searches for {profile.searches[-1]!r}" if profile.searches else "")
        + "."
    )
    return f"{opener} {titles} pick up exactly where that leaves off, and they are the closest match in the catalog to what you have been reading."


# ------------------------------------------------------------------- driver --


def run_agent(
    db: Session,
    user_id: int,
    profile: BehaviorProfile,
    reason: str,
    *,
    store: VectorStore | None = None,
    mesh: MeshClient | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> AgentOutcome:
    """Retrieve, generate, ground, store. Assumes the trigger already said yes."""
    settings = settings or get_settings()
    store = store or get_vector_store()
    mesh = mesh if mesh is not None else get_mesh_client()

    hits, widened = retrieve(store, profile, top_k=settings.retrieval_top_k, db=db)
    retrieved_ids = [hit.product_id for hit in hits]
    if not retrieved_ids:
        return AgentOutcome(False, "no_candidates", None, [], [], False, widened)

    by_id = {
        product.id: product
        for product in db.scalars(select(Product).where(Product.id.in_(retrieved_ids))).all()
    }
    candidates = [by_id[pid] for pid in retrieved_ids if pid in by_id]
    if not candidates:
        # Every retrieved id has since been deleted from SQLite — a desync the
        # admin repair tool exists for. Do not fabricate a recommendation.
        return AgentOutcome(False, "no_candidates", None, retrieved_ids, [], False, widened)

    narrative, proposed_ids, source, llm_used = "", [], RULE_BASED, False
    if mesh is not None:
        result = None
        try:
            result = mesh.chat(_build_messages(profile, candidates))
            narrative, proposed_ids = parse_generation(result.text)
            llm_used = True
            source = result.model
        except Exception:  # noqa: BLE001 — a model outage degrades, it does not 500
            log.exception("Mesh generation failed for user %s — falling back to rule-based", user_id)
        finally:
            if result is not None:
                log_llm_call(db, purpose="generate_rec", result=result, user_id=user_id)

    kept, dropped = ground(proposed_ids, retrieved_ids)
    if not kept:
        # No LLM, an unparseable reply, or every id it gave was invented. In the
        # last case the narrative describes courses we are not showing, so it is
        # discarded too — a pitch for products that were dropped is worse than no
        # pitch. Fall back to the top of retrieval.
        kept = retrieved_ids[:3]
        narrative = ""
    if not narrative or not llm_used:
        narrative = _rule_based_narrative(profile, [by_id[pid] for pid in kept if pid in by_id])
        source = RULE_BASED

    previous_version = db.scalar(
        select(Recommendation.version)
        .where(Recommendation.user_id == user_id)
        .order_by(Recommendation.version.desc())
        .limit(1)
    )
    recommendation = Recommendation(
        user_id=user_id,
        narrative=narrative,
        product_ids=kept,
        behavior_hash=profile.signature_hash,
        trigger_reason=reason,
        source=source,
        version=(previous_version or 0) + 1,
        created_at=now or utcnow(),
    )
    db.add(recommendation)
    db.commit()

    if dropped:
        log.warning("grounding dropped ids %s for user %s (not retrieved)", dropped, user_id)

    return AgentOutcome(True, reason, recommendation, retrieved_ids, dropped, llm_used, widened)


def maybe_recommend(
    db: Session,
    user_id: int,
    *,
    store: VectorStore | None = None,
    mesh: MeshClient | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> AgentOutcome:
    """Ask the trigger engine, then run the agent only if it said yes."""
    settings = settings or get_settings()
    decision: TriggerDecision = decide(db, user_id, settings=settings, now=now)

    if not decision.run:
        if decision.cache_hit:
            # Logged as a call that did not happen: this row is the efficiency proof.
            log_llm_call(db, purpose="generate_rec", user_id=user_id, cache_hit=True, model="")
            db.commit()
        return AgentOutcome(False, decision.reason, decision.current)

    return run_agent(
        db, user_id, decision.profile, decision.reason,
        store=store, mesh=mesh, settings=settings, now=now,
    )
