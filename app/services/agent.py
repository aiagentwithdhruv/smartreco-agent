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
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Product, Recommendation, utcnow
from app.services.behavior import BehaviorProfile
from app.services.mesh import MeshClient, get_mesh_client, log_llm_call
from app.services.trigger import TriggerDecision, decide, user_trigger_lock
from app.services.vector_store import VectorHit, VectorStore, get_vector_store

log = logging.getLogger(__name__)

RULE_BASED = "rule-based"
MIN_CANDIDATES = 3  # below this, the category filter is dropped and retrieval widened
RETRIEVAL_GRADE_THRESHOLD = 0.6
# Measured on this catalogue: clear matches score 0.68-0.76, vague queries
# score about 0.60, and off-catalogue nonsense scores 0.52-0.53.
RETRIEVAL_CONFIDENT_SCORE = 0.65

SYSTEM_PROMPT = (
    "You are the course advisor for SmartReco, an online course marketplace. "
    "You write short, specific, persuasive notes to a learner about what to study next. "
    "You may ONLY recommend courses from the CANDIDATES list you are given. "
    "In the narrative, name each course by its exact title — never write an id, "
    "the ids belong in the product_ids field only. "
    "Never invent a course, a price, a statistic or a claim about content — if a "
    "number was not given to you, do not write one. "
    "Two or three sentences, warm and concrete, referencing what the learner actually did. "
    "No bullet points, no hype, no exclamation marks. "
    "Return one compact, single-line JSON object and nothing else. "
    "No reasoning, no preamble, no commentary, and no code fence."
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
    visited_nodes: list[str] = field(default_factory=list)
    grade_mode: Literal["skipped_confident", "graded"] | None = None


# ---------------------------------------------------------------- retrieval --


def retrieve(
    store: VectorStore,
    profile: BehaviorProfile,
    *,
    top_k: int,
    db: Session | None = None,
    query: str | None = None,
) -> tuple[list[VectorHit], bool]:
    """Fetch candidates for this profile. Returns (hits, widened).

    First pass is filtered to the categories the behavior points at. If that
    leaves too little to choose from, the filter is dropped rather than handing
    the generator two options and calling it a recommendation.
    """
    query = query or profile.retrieval_query()
    filters = {"category": {"$in": profile.top_categories}} if profile.top_categories else None

    hits = store.query(query, top_k=top_k, filters=filters, exclude_ids=profile.carted_ids, db=db)
    if len(hits) >= MIN_CANDIDATES or filters is None:
        return hits, False

    wider = store.query(query, top_k=top_k, exclude_ids=profile.carted_ids, db=db)
    return wider, True


# --------------------------------------------------------------- generation --


def _candidate_block(products: list[Product]) -> str:
    """Candidates as the model sees them.

    Prices are deliberately left out. A persuasive pitch does not need them, and
    handing a model a column of numbers is handing it something to misquote —
    seen live: a ₹4,999 course pitched as "(₹2,999)", copied off a neighbouring
    row. The page renders the real price next to the card anyway.
    """
    return "\n".join(
        f"- id={p.id} | {p.title} | {p.category} | {p.level} | {p.description[:140]}"
        for p in products
    )


# ₹2,999 · Rs. 2999 · INR 2,999 — any figure the model might present as a price.
_MONEY_RE = re.compile(r"(?:₹|Rs\.?|INR)\s?([\d,]+)", re.IGNORECASE)


def unsupported_prices(narrative: str, products: list[Product]) -> list[str]:
    """Currency figures in the narrative that no recommended product charges.

    Grounding is not only about ids. A correct product with an invented price is
    still a false claim, and it is the kind users notice at checkout.
    """
    allowed = {int(p.price) for p in products}
    return [
        figure
        for figure in _MONEY_RE.findall(narrative or "")
        if int(figure.replace(",", "") or 0) not in allowed
    ]


def _build_messages(profile: BehaviorProfile, products: list[Product]) -> list[dict[str, str]]:
    user_prompt = (
        f"LEARNER BEHAVIOR:\n{profile.summary()}\n\n"
        f"CANDIDATES (the only courses you may recommend):\n{_candidate_block(products)}\n\n"
        "Pick the 2-3 best fits. Reply with one compact JSON object and nothing else; "
        "do not include reasoning or a preamble:\n"
        '{"narrative": "<2-3 sentences, courses named by title, no ids>", '
        '"product_ids": [<ids from CANDIDATES>]}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_generation(text: str) -> tuple[str, list[int]]:
    """Pull (narrative, product_ids) out of a model reply.

    Models wrap JSON in prose or code fences often enough that being strict here
    would mean throwing away good answers, so the first JSON object in the reply
    wins. If the provider truncates the object after both required values are
    complete, salvage them; a narrative without its closing quote or without a
    complete, non-empty id array remains a parse failure.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        else:
            if not isinstance(data, dict):
                return "", []
            return str(data.get("narrative") or "").strip(), _parse_product_ids(
                data.get("product_ids")
            )

    # A completion can end after `]` but before the object's final `}`. Match a
    # JSON string (including escapes) so an unterminated narrative never passes.
    object_start = (text or "").find("{")
    if object_start < 0:
        return "", []
    fragment = text[object_start:]
    narrative_match = re.search(
        r'"narrative"\s*:\s*("(?:\\.|[^"\\])*")', fragment, re.DOTALL
    )
    ids_match = re.search(r'"product_ids"\s*:\s*(\[[^\]]*\])', fragment, re.DOTALL)
    if not narrative_match or not ids_match:
        return "", []
    try:
        narrative = json.loads(narrative_match.group(1)).strip()
        raw_ids = json.loads(ids_match.group(1))
    except (AttributeError, json.JSONDecodeError):
        return "", []
    ids = _parse_product_ids(raw_ids)
    if not narrative or not ids:
        return "", []
    return narrative, ids


def _parse_product_ids(raw_ids: Any) -> list[int]:
    """Normalize model-provided ids while ignoring values that are not integers."""
    ids: list[int] = []
    for raw in raw_ids or []:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return ids


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


class AgentState(TypedDict, total=False):
    """Internal LangGraph state; the public API remains :class:`AgentOutcome`."""

    db: Session
    user_id: int
    profile: BehaviorProfile
    reason: str
    store: VectorStore
    mesh: MeshClient | None
    settings: Settings
    now: datetime | None
    profile_summary: str
    retrieval_query: str
    hits: list[VectorHit]
    retrieved_ids: list[int]
    widened: bool
    retry_count: int
    retry_requested: bool
    grade_score: float | None
    grade_mode: Literal["skipped_confident", "graded"]
    candidates: list[Product]
    narrative: str
    proposed_ids: list[int]
    source: str
    llm_used: bool
    outcome: AgentOutcome
    visited_nodes: list[str]


def _visited(state: AgentState, node: str) -> list[str]:
    return [*state.get("visited_nodes", []), node]


def _summarize_behavior_node(state: AgentState) -> AgentState:
    # The trigger has already called the existing deterministic summarize()
    # function. Carry its exact profile forward instead of recomputing behavior.
    return {"profile": state["profile"], "visited_nodes": _visited(state, "summarize_behavior")}


def _build_interest_profile_node(state: AgentState) -> AgentState:
    profile = state["profile"]
    return {
        "profile_summary": profile.summary(),
        "retrieval_query": profile.retrieval_query(),
        "visited_nodes": _visited(state, "build_interest_profile"),
    }


def _retrieve_node(state: AgentState) -> AgentState:
    hits, widened = retrieve(
        state["store"],
        state["profile"],
        top_k=state["settings"].retrieval_top_k,
        db=state["db"],
        query=state["retrieval_query"],
    )
    return {
        "hits": hits,
        "retrieved_ids": [hit.product_id for hit in hits],
        "widened": state.get("widened", False) or widened,
        "visited_nodes": _visited(state, "retrieve"),
    }


RETRIEVAL_GRADE_PROMPT = (
    "You are a retrieval judge. Score how well the candidate courses match the "
    "learner interest profile from 0.0 to 1.0. If the score is below 0.6, rewrite "
    "the search query to improve retrieval. Reply with JSON only: "
    '{"score": 0.0, "reason": "brief reason", "rewritten_query": "better query"}'
)


def _grade_messages(state: AgentState) -> list[dict[str, str]]:
    candidates = "\n".join(
        f"- {hit.metadata.get('title', '')} | {hit.metadata.get('category', '')} | similarity={hit.score}"
        for hit in state.get("hits", [])
    )
    return [
        {"role": "system", "content": RETRIEVAL_GRADE_PROMPT},
        {
            "role": "user",
            "content": (
                f"INTEREST PROFILE:\n{state['profile_summary']}\n\n"
                f"QUERY:\n{state['retrieval_query']}\n\n"
                f"CANDIDATES:\n{candidates or '(none)'}"
            ),
        },
    ]


def _parse_grade(text: str) -> tuple[float | None, str]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
        score = float(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, ""
    rewritten = str(data.get("rewritten_query") or "").strip()
    return max(0.0, min(score, 1.0)), rewritten


def _fallback_rewrite(profile: BehaviorProfile) -> str:
    pieces = [profile.retrieval_query(), *profile.top_categories]
    if profile.level_hint:
        pieces.append(f"{profile.level_hint} course")
    return " ".join(piece for piece in pieces if piece).strip()


def _grade_retrieval_node(state: AgentState) -> AgentState:
    hits = state.get("hits", [])
    confident = (
        bool(hits)
        and len(hits) >= MIN_CANDIDATES
        and not state.get("widened", False)
        and hits[0].score >= RETRIEVAL_CONFIDENT_SCORE
    )
    if confident:
        return {
            "grade_score": hits[0].score,
            "grade_mode": "skipped_confident",
            "retry_requested": False,
            "visited_nodes": _visited(state, "grade_retrieval"),
        }

    score: float | None = None
    rewritten_query = ""
    mesh = state.get("mesh")

    if mesh is not None and state.get("hits"):
        result = None
        try:
            result = mesh.chat(_grade_messages(state), temperature=0.0)
            score, rewritten_query = _parse_grade(result.text)
        except Exception:  # noqa: BLE001 — grading failure must not block a recommendation
            log.exception("retrieval grading failed for user %s — proceeding", state["user_id"])
        finally:
            if result is not None:
                log_llm_call(
                    state["db"],
                    purpose="grade_retrieval",
                    result=result,
                    user_id=state["user_id"],
                )
                state["db"].commit()

    weak = score is not None and score < RETRIEVAL_GRADE_THRESHOLD
    retry_count = state.get("retry_count", 0)
    retry_requested = weak and retry_count < 1
    update: AgentState = {
        "grade_score": score,
        "grade_mode": "graded",
        "retry_requested": retry_requested,
        "visited_nodes": _visited(state, "grade_retrieval"),
    }
    if retry_requested:
        update["retry_count"] = retry_count + 1
        update["retrieval_query"] = rewritten_query or _fallback_rewrite(state["profile"])
    return update


def _after_grade(state: AgentState) -> Literal["retry", "proceed"]:
    return "retry" if state.get("retry_requested", False) else "proceed"


def _generate_node(state: AgentState) -> AgentState:
    retrieved_ids = state.get("retrieved_ids", [])
    by_id = {
        product.id: product
        for product in state["db"].scalars(select(Product).where(Product.id.in_(retrieved_ids))).all()
    } if retrieved_ids else {}
    candidates = [by_id[product_id] for product_id in retrieved_ids if product_id in by_id]

    narrative, proposed_ids, source, llm_used = "", [], RULE_BASED, False
    mesh = state.get("mesh")
    if mesh is not None and candidates:
        messages = _build_messages(state["profile"], candidates)
        for attempt in range(2):  # initial generation plus exactly one parse-failure retry
            result = None
            try:
                result = mesh.chat(messages)
                narrative, proposed_ids = parse_generation(result.text)
                llm_used = True
                source = result.model
                if narrative and proposed_ids:
                    break
                log.warning(
                    "unparseable generation from %s for user %s "
                    "(attempt=%s/2, narrative=%s, ids=%s): %.300s",
                    result.model, state["user_id"], attempt + 1,
                    bool(narrative), proposed_ids, result.text,
                )
            except Exception:  # noqa: BLE001 — a model outage degrades, it does not 500
                log.exception(
                    "Mesh generation failed for user %s — falling back to rule-based",
                    state["user_id"],
                )
                break
            finally:
                if result is not None:
                    # Each attempt is a real Mesh call and gets its own row.
                    log_llm_call(
                        state["db"], purpose="generate_rec", result=result,
                        user_id=state["user_id"],
                    )

    return {
        "candidates": candidates,
        "narrative": narrative,
        "proposed_ids": proposed_ids,
        "source": source,
        "llm_used": llm_used,
        "visited_nodes": _visited(state, "generate"),
    }


def _validate_grounding_node(state: AgentState) -> AgentState:
    db = state["db"]
    user_id = state["user_id"]
    retrieved_ids = state.get("retrieved_ids", [])
    candidates = state.get("candidates", [])
    visited = _visited(state, "validate_grounding")

    if not retrieved_ids or not candidates:
        outcome = AgentOutcome(
            False, "no_candidates", None, retrieved_ids, [], False,
            state.get("widened", False), visited, state.get("grade_mode"),
        )
        return {"outcome": outcome, "visited_nodes": visited}

    by_id = {product.id: product for product in candidates}
    narrative = state.get("narrative", "")
    source = state.get("source", RULE_BASED)
    llm_used = state.get("llm_used", False)
    kept, dropped = ground(state.get("proposed_ids", []), retrieved_ids)
    if not kept:
        kept = retrieved_ids[:3]
        narrative = ""

    bad_prices = unsupported_prices(narrative, [by_id[pid] for pid in kept if pid in by_id])
    if bad_prices:
        log.warning(
            "dropping narrative for user %s — quoted prices %s match no recommended course",
            user_id, bad_prices,
        )
        narrative = ""

    if not narrative or not llm_used:
        narrative = _rule_based_narrative(
            state["profile"], [by_id[pid] for pid in kept if pid in by_id]
        )
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
        behavior_hash=state["profile"].signature_hash,
        trigger_reason=state["reason"],
        source=source,
        version=(previous_version or 0) + 1,
        created_at=state.get("now") or utcnow(),
    )
    db.add(recommendation)
    db.commit()

    if dropped:
        log.warning("grounding dropped ids %s for user %s (not retrieved)", dropped, user_id)

    outcome = AgentOutcome(
        True,
        state["reason"],
        recommendation,
        retrieved_ids,
        dropped,
        llm_used,
        state.get("widened", False),
        visited,
        state.get("grade_mode"),
    )
    return {"outcome": outcome, "visited_nodes": visited}


def _build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("summarize_behavior", _summarize_behavior_node)
    graph.add_node("build_interest_profile", _build_interest_profile_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("grade_retrieval", _grade_retrieval_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("validate_grounding", _validate_grounding_node)
    graph.add_edge(START, "summarize_behavior")
    graph.add_edge("summarize_behavior", "build_interest_profile")
    graph.add_edge("build_interest_profile", "retrieve")
    graph.add_edge("retrieve", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        _after_grade,
        {"retry": "retrieve", "proceed": "generate"},
    )
    graph.add_edge("generate", "validate_grounding")
    graph.add_edge("validate_grounding", END)
    return graph.compile(name="smartreco-recommendation-agent")


AGENT_GRAPH = _build_agent_graph()


def _invoke_agent_graph(initial: AgentState) -> AgentState:
    settings = initial["settings"]
    if not settings.langchain_api_key:
        return AGENT_GRAPH.invoke(initial)

    # Imported only in the opt-in path: no key means no client, background
    # worker, warning, or tracing side effect.
    from langsmith import Client, tracing_context

    client = Client(api_key=settings.langchain_api_key)
    with tracing_context(enabled=True, client=client, project_name="smartreco"):
        return AGENT_GRAPH.invoke(initial)


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
    """Run the existing recommendation steps through the explicit LangGraph."""
    settings = settings or get_settings()
    store = store or get_vector_store()
    mesh = mesh if mesh is not None else get_mesh_client()
    final = _invoke_agent_graph({
        "db": db,
        "user_id": user_id,
        "profile": profile,
        "reason": reason,
        "store": store,
        "mesh": mesh,
        "settings": settings,
        "now": now,
        "retry_count": 0,
        "visited_nodes": [],
    })
    return final["outcome"]


def maybe_recommend(
    db: Session,
    user_id: int,
    *,
    store: VectorStore | None = None,
    mesh: MeshClient | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> AgentOutcome:
    """Ask the trigger engine, then atomically authorize and run per user.

    The first decision keeps ordinary skip paths lock-free. A decision that
    wants to run must acquire this user's lock and re-read the database before
    generation. That closes the gap in which concurrent requests could all see
    the same old recommendation and all call the model.
    """
    settings = settings or get_settings()
    decision: TriggerDecision = decide(db, user_id, settings=settings, now=now)

    if not decision.run:
        if decision.cache_hit:
            # Logged as a call that did not happen: this row is the efficiency proof.
            log_llm_call(db, purpose="generate_rec", user_id=user_id, cache_hit=True, model="")
            db.commit()
        return AgentOutcome(False, decision.reason, decision.current)

    observed_recommendation_id = decision.current.id if decision.current is not None else None
    with user_trigger_lock(user_id):
        # TOCTOU guard: this must be a new DB-backed decision inside the lock.
        decision = decide(db, user_id, settings=settings, now=now)
        current_recommendation_id = decision.current.id if decision.current is not None else None

        if current_recommendation_id != observed_recommendation_id:
            # Another request won while this one waited. Report the cooldown
            # contract (rather than cache_hit) and serve the row it just wrote.
            return AgentOutcome(False, "cooldown", decision.current)

        if not decision.run:
            if decision.cache_hit:
                log_llm_call(
                    db, purpose="generate_rec", user_id=user_id, cache_hit=True, model=""
                )
                db.commit()
            return AgentOutcome(False, decision.reason, decision.current)

        return run_agent(
            db, user_id, decision.profile, decision.reason,
            store=store, mesh=mesh, settings=settings, now=now,
        )
