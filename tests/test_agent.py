"""The agent: retrieval, generation, and the grounding guarantee.

Mesh is always faked here. Nothing in this file touches the network.
"""

import json

import pytest
from sqlalchemy import select

from app.models import LLMCall, Recommendation
from app.services.agent import RULE_BASED, ground, maybe_recommend, parse_generation, run_agent
from app.services.behavior import summarize
from app.services.catalog import ProductInput, create_product
from app.services.mesh import MeshResult
from tests.test_behavior import add
from tests.test_trigger import SETTINGS, store_recommendation


class FakeMesh:
    """A Mesh chat client that returns whatever the test tells it to."""

    def __init__(self, reply: str = "", raises: bool = False):
        self.reply = reply
        self.raises = raises
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=0.4, model=None):
        self.calls.append(messages)
        if self.raises:
            raise RuntimeError("mesh is down")
        return MeshResult(text=self.reply, model="fake/chat", tokens_in=120, tokens_out=48, latency_ms=310)


def reply(narrative: str, ids: list[int], *, fence: bool = False) -> str:
    body = json.dumps({"narrative": narrative, "product_ids": ids})
    return f"```json\n{body}\n```" if fence else body


@pytest.fixture
def catalog(db, store):
    """A small but real indexed catalog, written through the dual-write path."""
    rows = [
        ProductInput("Agentic Workflows with LangGraph", "Stateful agents as graphs, checkpointing, routing.",
                     "Agentic AI", 4999, "intermediate", ["langgraph", "agents"]),
        ProductInput("Multi-Agent Systems in Production", "Supervisor patterns and agent-to-agent delegation.",
                     "Agentic AI", 8999, "advanced", ["multi-agent", "supervisor"]),
        ProductInput("Agentic AI Bootcamp", "Six weeks from zero to a working tool-calling agent.",
                     "Agentic AI", 7999, "beginner", ["agents", "bootcamp"]),
        ProductInput("SQL for Data Analysis", "Joins, window functions and CTEs on messy data.",
                     "Data Analytics", 2999, "beginner", ["sql", "analytics"]),
        ProductInput("Apache Spark for Large-Scale Data", "Partitions, shuffles and joins that survive scale.",
                     "Data Engineering", 6999, "intermediate", ["spark", "big-data"]),
    ]
    return [create_product(db, row, store=store) for row in rows]


def agent_behavior(db, user, catalog):
    """A learner clearly interested in agents."""
    add(db, user, "search", query="langgraph agents", minutes_ago=6)
    add(db, user, "view", catalog[0], minutes_ago=5)
    add(db, user, "dwell", catalog[0], value=95.0, minutes_ago=4)
    add(db, user, "view", catalog[1], minutes_ago=3)
    return summarize(db, user.id)


# ------------------------------------------------------------------ parsing --


def test_parse_generation_reads_plain_json():
    narrative, ids = parse_generation(reply("Two great picks.", [3, 7]))
    assert narrative == "Two great picks."
    assert ids == [3, 7]


def test_parse_generation_survives_a_code_fence_and_chatter():
    text = "Sure! Here you go:\n" + reply("Picks.", [1], fence=True) + "\nHope that helps."
    assert parse_generation(text) == ("Picks.", [1])


def test_parse_generation_ignores_junk_ids():
    assert parse_generation('{"narrative": "x", "product_ids": [1, "two", null, 3]}')[1] == [1, 3]


@pytest.mark.parametrize("text", ["", "no json here", "{not json}"])
def test_parse_generation_gives_up_cleanly(text):
    assert parse_generation(text) == ("", [])


# ----------------------------------------------------------------- grounding --


def test_grounding_keeps_only_retrieved_ids():
    kept, dropped = ground([1, 2, 3], [1, 3, 9])
    assert kept == [1, 3]
    assert dropped == [2]


def test_grounding_deduplicates():
    kept, dropped = ground([1, 1, 2, 2], [1])
    assert kept == [1]
    assert dropped == [2]


def test_a_hallucinated_product_id_never_reaches_the_user(db, user, store, catalog):
    """The headline guarantee: the model names a course that does not exist."""
    profile = agent_behavior(db, user, catalog)
    real_id = catalog[0].id
    mesh = FakeMesh(reply("Try these two.", [real_id, 999_999]))

    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)

    assert outcome.dropped_ids == [999_999]
    assert 999_999 not in outcome.recommendation.product_ids
    assert real_id in outcome.recommendation.product_ids


def test_an_id_that_exists_but_was_not_retrieved_is_dropped(db, user, store, catalog):
    """Grounding is about retrieval, not about existence — a real but unretrieved
    course is still an ungrounded claim."""
    profile = agent_behavior(db, user, catalog)
    mesh = FakeMesh("")
    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)
    retrieved = set(outcome.retrieved_ids)

    outsider = next(p.id for p in catalog if p.id not in retrieved)
    mesh = FakeMesh(reply("Pick this.", [outsider]))
    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)

    assert outsider in outcome.dropped_ids
    assert outsider not in outcome.recommendation.product_ids


def test_when_every_id_is_hallucinated_the_agent_falls_back_to_retrieval(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)
    mesh = FakeMesh(reply("Nonsense.", [900, 901]))

    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)

    assert outcome.recommendation.product_ids, "a user is never shown an empty recommendation"
    assert set(outcome.recommendation.product_ids) <= set(outcome.retrieved_ids)
    assert outcome.recommendation.source == RULE_BASED, "an ungrounded reply is not credited to the model"


# ----------------------------------------------------------------- retrieval --


def test_retrieval_is_driven_by_behavior(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)
    outcome = run_agent(db, user.id, profile, "search_intent", store=store,
                        mesh=FakeMesh(""), settings=SETTINGS)

    titles = {p.title for p in db.scalars(
        select(type(catalog[0])).where(type(catalog[0]).id.in_(outcome.retrieved_ids))
    ).all()}
    assert any("Agentic" in title or "Multi-Agent" in title for title in titles)
    assert "SQL for Data Analysis" not in titles, "category filtering should keep analytics out"


def test_carted_products_are_not_recommended_back(db, user, store, catalog):
    add(db, user, "search", query="langgraph agents")
    add(db, user, "cart", catalog[0], value=1.0)
    profile = summarize(db, user.id)

    outcome = run_agent(db, user.id, profile, "search_intent", store=store,
                        mesh=FakeMesh(""), settings=SETTINGS)

    assert catalog[0].id not in outcome.retrieved_ids


def test_retrieval_widens_when_the_filtered_set_is_too_small(db, user, store):
    """One product in the preferred category is not a choice — drop the filter."""
    only = create_product(db, ProductInput("Kafka Streaming", "Topics and partitions.",
                                           "Data Engineering", 6499, "advanced", ["kafka"]), store=store)
    for i in range(4):
        create_product(db, ProductInput(f"Analytics {i}", "Dashboards and metrics.",
                                        "Data Analytics", 2999, "beginner", ["analytics"]), store=store)
    add(db, user, "view", only)
    profile = summarize(db, user.id)

    outcome = run_agent(db, user.id, profile, "first_recommendation", store=store,
                        mesh=FakeMesh(""), settings=SETTINGS)

    assert outcome.widened is True
    assert len(outcome.retrieved_ids) > 1


def test_no_recommendation_is_stored_when_nothing_can_be_retrieved(db, user, store, products):
    """An empty index must produce silence, not an invented pitch."""
    add(db, user, "view", products[0])
    profile = summarize(db, user.id)

    outcome = run_agent(db, user.id, profile, "first_recommendation", store=store,
                        mesh=FakeMesh(reply("Buy something.", [1])), settings=SETTINGS)

    assert outcome.ran is False
    assert outcome.reason == "no_candidates"
    assert db.scalar(select(Recommendation)) is None


# ---------------------------------------------------------------- generation --


def test_the_model_is_only_shown_retrieved_candidates(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)
    mesh = FakeMesh(reply("Picks.", []))

    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)

    prompt = mesh.calls[0][1]["content"]
    for product_id in outcome.retrieved_ids:
        assert f"id={product_id}" in prompt
    assert "CANDIDATES" in prompt
    assert profile.summary()[:40] in prompt


def test_a_generated_recommendation_is_stored_and_versioned(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)
    mesh = FakeMesh(reply("You have been deep in agent orchestration.", [catalog[0].id]))

    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)
    rec = outcome.recommendation

    assert rec.narrative == "You have been deep in agent orchestration."
    assert rec.product_ids == [catalog[0].id]
    assert rec.behavior_hash == profile.signature_hash
    assert rec.trigger_reason == "search_intent"
    assert rec.source == "fake/chat"
    assert rec.version == 1

    second = run_agent(db, user.id, profile, "staleness", store=store, mesh=mesh, settings=SETTINGS)
    assert second.recommendation.version == 2, "recommendations are versioned, not overwritten"


def test_every_model_call_is_logged(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)
    mesh = FakeMesh(reply("Picks.", [catalog[0].id]))

    run_agent(db, user.id, profile, "search_intent", store=store, mesh=mesh, settings=SETTINGS)

    call = db.scalars(select(LLMCall).where(LLMCall.purpose == "generate_rec")).one()
    assert call.model == "fake/chat"
    assert call.tokens_in == 120 and call.tokens_out == 48
    assert call.latency_ms == 310
    assert call.cache_hit is False
    assert call.user_id == user.id


def test_without_a_mesh_client_the_narrative_is_labelled_rule_based(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)

    outcome = run_agent(db, user.id, profile, "search_intent", store=store, mesh=None, settings=SETTINGS)

    assert outcome.llm_used is False
    assert outcome.recommendation.source == RULE_BASED
    assert outcome.recommendation.narrative
    assert outcome.recommendation.product_ids
    assert db.scalar(select(LLMCall)) is None, "no model was called, so no call is claimed"


def test_a_model_outage_degrades_instead_of_failing(db, user, store, catalog):
    profile = agent_behavior(db, user, catalog)

    outcome = run_agent(db, user.id, profile, "search_intent", store=store,
                        mesh=FakeMesh(raises=True), settings=SETTINGS)

    assert outcome.recommendation is not None
    assert outcome.recommendation.source == RULE_BASED


# -------------------------------------------------------------------- driver --


def test_maybe_recommend_respects_the_trigger(db, user, store, catalog):
    mesh = FakeMesh(reply("Picks.", [catalog[0].id]))

    quiet = maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)
    assert quiet.ran is False and quiet.reason == "no_activity"
    assert mesh.calls == []

    agent_behavior(db, user, catalog)
    fired = maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)
    assert fired.ran is True and fired.reason == "first_recommendation"
    assert len(mesh.calls) == 1


def test_a_cache_hit_calls_no_model_but_leaves_a_row(db, user, store, catalog):
    mesh = FakeMesh(reply("Picks.", [catalog[0].id]))
    agent_behavior(db, user, catalog)
    maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)

    for _ in range(5):
        outcome = maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)
        assert outcome.reason == "cache_hit"

    assert len(mesh.calls) == 1, "five repeat visits, one model call"
    calls = db.scalars(select(LLMCall)).all()
    assert sum(1 for c in calls if c.cache_hit) == 5
    assert sum(1 for c in calls if not c.cache_hit) == 1


def test_a_burst_of_browsing_does_not_become_a_burst_of_model_calls(db, user, store, catalog):
    """The whole point of the trigger engine, measured end to end."""
    mesh = FakeMesh(reply("Picks.", [catalog[0].id]))
    add(db, user, "view", catalog[0], minutes_ago=20)
    maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)

    for i in range(40):
        add(db, user, "view", catalog[i % len(catalog)])
        maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)

    assert len(mesh.calls) == 1, f"41 events, {len(mesh.calls)} model calls"
    assert db.scalar(select(Recommendation.version).order_by(Recommendation.version.desc())) == 1


def test_a_search_after_the_cooldown_produces_a_second_version(db, user, store, catalog):
    mesh = FakeMesh(reply("Picks.", [catalog[0].id]))
    agent_behavior(db, user, catalog)
    maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)

    store_recommendation  # noqa: B018 — imported for the helper's side-effect-free use
    rec = db.scalars(select(Recommendation)).one()
    rec.created_at = rec.created_at.replace(tzinfo=None) - __import__("datetime").timedelta(minutes=10)
    db.commit()

    add(db, user, "search", query="multi agent supervisor")
    outcome = maybe_recommend(db, user.id, store=store, mesh=mesh, settings=SETTINGS)

    assert outcome.ran is True
    assert outcome.reason == "search_intent"
    assert outcome.recommendation.version == 2
    assert len(mesh.calls) == 2
