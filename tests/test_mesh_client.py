"""The Mesh client, pinned to a response Mesh actually returned.

tests/fixtures/mesh_chat_response.json is the raw wire JSON from a live call to
`minimax/m2-her` on 4 Aug 2026, captured by `python scripts/smoke_mesh.py
--capture`. Parsing is asserted against that, not against an idealised
OpenAI-shaped response — the two differ, and the difference is the point.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from app.config import Settings
from app.services.agent import parse_generation
from app.services.mesh import MeshClient, extract_text

FIXTURE = Path(__file__).parent / "fixtures" / "mesh_chat_response.json"


@pytest.fixture
def live_response() -> ChatCompletion:
    return ChatCompletion.model_validate(json.loads(FIXTURE.read_text()))


def message(**fields) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(**fields))])


def test_the_captured_response_carries_the_quirk_it_was_captured_for():
    """minimax/m2-her puts a `name` on the assistant message. If a future capture
    loses that, this test should be the thing that notices."""
    raw = json.loads(FIXTURE.read_text())
    assert raw["model"] == "minimax/m2-her"
    assert raw["choices"][0]["message"]["name"] == "MiniMax AI", "non-standard field, kept on purpose"


def test_text_is_extracted_from_a_real_mesh_response(live_response):
    assert extract_text(live_response).startswith("{")


def test_the_full_parse_chain_works_on_a_real_response(live_response):
    """Wire JSON → SDK model → extract_text → parse_generation → ids."""
    narrative, ids = parse_generation(extract_text(live_response))
    assert narrative == "The quick brown fox jumps over the lazy dog."
    assert ids == [1, 2]


def test_reasoning_content_is_used_when_content_is_empty():
    """tencent/hy3 returns an empty `content` and the answer in `reasoning_content`."""
    assert extract_text(message(content="", reasoning_content="the answer")) == "the answer"


def test_reasoning_content_is_found_through_the_sdk_model_too():
    """`reasoning_content` is not a field the SDK declares. It survives as an
    extra — this pins that, because if the SDK ever started dropping unknown
    fields, hy3 replies would silently become empty."""
    wire = json.loads(FIXTURE.read_text())
    wire["model"] = "tencent/hy3"
    wire["choices"][0]["message"] = {
        "role": "assistant",
        "content": "",
        "reasoning_content": '{"narrative": "thought it through.", "product_ids": [4]}',
    }
    response = ChatCompletion.model_validate(wire)

    message_ = response.choices[0].message
    assert message_.content == ""
    assert "reasoning_content" not in type(message_).model_fields, "still an undeclared field"
    assert (message_.model_extra or {}).get("reasoning_content"), "kept as an extra, not dropped"
    assert parse_generation(extract_text(response)) == ("thought it through.", [4])


def test_content_wins_over_reasoning_content():
    assert extract_text(message(content="answer", reasoning_content="thinking out loud")) == "answer"


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=None),
        SimpleNamespace(choices=[SimpleNamespace(message=None)]),
        message(content=None),
        message(content="   "),
        message(content=None, reasoning_content=None),
        message(),
    ],
)
def test_an_unusable_reply_yields_an_empty_string_not_an_exception(response):
    """Empty means the agent falls back to its rule-based narrative — a 500 would
    take the whole page down over a model quirk."""
    assert extract_text(response) == ""


def test_chat_maps_usage_and_model_onto_the_result(live_response):
    class FakeCompletions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return live_response

    client = MeshClient(Settings(mesh_api_key="rsk_test", mesh_chat_model="minimax/m2-her"))
    completions = FakeCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = client.chat([{"role": "user", "content": "hi"}])

    assert result.model == "minimax/m2-her"
    assert result.tokens_in == 226
    assert result.tokens_out == 26
    assert result.latency_ms >= 0
    assert parse_generation(result.text)[1] == [1, 2]
    assert completions.kwargs["temperature"] == 0.4


def test_the_model_is_never_hardcoded():
    """A model id comes from settings, so swapping it is an env change."""
    client = MeshClient(Settings(mesh_api_key="rsk_test", mesh_chat_model="some/other-model"))

    class Recorder:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return ChatCompletion.model_validate(json.loads(FIXTURE.read_text()))

    recorder = Recorder()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    client.chat([{"role": "user", "content": "hi"}])

    assert recorder.kwargs["model"] == "some/other-model"


def test_mesh_model_env_alias_is_accepted(monkeypatch):
    """The .env in this repo uses MESH_MODEL; MESH_CHAT_MODEL also works."""
    monkeypatch.setenv("MESH_MODEL", "minimax/m2-her")
    assert Settings().mesh_chat_model == "minimax/m2-her"

    monkeypatch.delenv("MESH_MODEL")
    monkeypatch.setenv("MESH_CHAT_MODEL", "tencent/hy3")
    assert Settings().mesh_chat_model == "tencent/hy3"


def test_embeddings_are_local_by_default():
    """Locked decision: Mesh has no free embedding model, so the index is built
    locally and the only external AI calls are chat completions.

    Reads the declared default rather than Settings(): conftest pins EMBEDDINGS
    to `hashing` for the test session.
    """
    assert Settings.model_fields["embeddings"].default == "local"


def test_the_default_chat_model_is_one_mesh_serves_free():
    assert Settings.model_fields["mesh_chat_model"].default == "minimax/m2-her"
