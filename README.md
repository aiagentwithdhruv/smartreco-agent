# SmartReco — a behavioral recommendation agent

Submission for the **SmartReco Build Challenge 2026** (Krish Naik Academy × Mesh API).

An online course marketplace where a backend agent watches how each user actually
behaves — what they search, view, linger on and add to cart — reasons over that
behavior, retrieves matching courses from a vector index of our own catalog, and
writes a short persuasive recommendation that refreshes as the behavior changes.

Every LLM call in this repo goes through **Mesh API**
(`https://api.meshapi.ai/v1`, OpenAI-compatible). There are no other model providers.

---

## Run it in three commands

```bash
pip install -r requirements.txt      # Python 3.11+
python seed.py                       # 31 courses, 4 accounts, 3 scripted journeys
uvicorn app.main:app --reload        # http://127.0.0.1:8000
```

Log in as `aditi@example.com` / `smartreco123` (admin: `admin@smartreco.dev`),
search for "langgraph agents", open a course, and the panel at the top of the
page is the agent's work.

For the 60-second version without a browser:

```bash
python scripts/demo.py               # replays three sessions, shows every trigger decision
```

```
aditi@example.com — 11 events over 33 simulated minutes
  20:28  search "langgraph agents"                      AGENT RAN (first_recommendation)
  20:31  click Agentic Workflows with LangGraph         skipped (cooldown)
  20:34  view Agentic Workflows with LangGraph          skipped (below_threshold)
  ...
  20:46  search "multi agent supervisor"                AGENT RAN (search_intent)
  ...
  21:01  (page reload, no new activity)                 skipped (cache_hit)

26 events → 5 model calls (3 served from the behavior cache). 5.2 events per call.
```

Tests: `pytest` — 154 tests, no network calls, Mesh always faked.

Copy `.env.example` to `.env` and add `MESH_API_KEY` for live generation. Without
a key the app still runs end to end; it just labels the narrative `rule-based`
instead of claiming a model wrote it.

---

## Architecture

```
Browser (Jinja2 pages + static/tracker.js)
  │  batched POST /api/events        view · dwell · click
  │  server-recorded signals          search (page load) · cart (POST)
  ▼
FastAPI ──────── SQLite (users, products, events, recommendations, llm_calls)
  │  └── dual-write on every product create/update/delete ──► Chroma (local, persistent)
  │
  ├── TRIGGER ENGINE — decides whether the agent is allowed to think at all
  │      fires on: first activity · a search · N events · staleness + new activity
  │      blocked by: a behavior-hash cache hit, then a per-user cooldown
  ▼
AGENT: behavior profile → Chroma retrieval (category-filtered, widens if thin)
       → Mesh generates narrative + product ids → grounding validator
  ▼
recommendations (versioned per user) rendered on home + product pages
llm_calls logs every call and every avoided call: purpose, model, tokens, latency, cache_hit
```

**Stack:** FastAPI · SQLAlchemy + SQLite · Chroma · Jinja2 + vanilla JS · `openai`
SDK pointed at Mesh · pytest.

## The four decisions worth reading the code for

**1. The trigger engine — `app/services/trigger.py`**
An agent that calls a model on every click is a bill, not a product. The agent
runs only when the first activity arrives, when a *search* happens (stated intent
beats inferred intent), when eight meaningful events have piled up, or when the
stored recommendation has gone stale and there has been activity since. A five
minute per-user cooldown sits over all of it. Every decision is a named reason,
stored on the recommendation and printed by the demo.

**2. The behavior-hash cache — `app/services/behavior.py`**
Each profile carries a fingerprint of the recent activity. Same fingerprint, same
recommendation, zero model calls — a page reload costs nothing. Dwell times are
bucketed into 15-second bands, so 41s and 43s of reading count as the same
behavior rather than busting the cache. Cache hits are written to `llm_calls`
too: the efficiency claim comes from rows, not from prose.

**3. The grounding validator — `app/services/agent.py`**
The generator is handed the retrieved candidates and asked for ids. `ground()`
drops every id that was not in that list before anything is stored, so a
hallucinated course can never reach a user. If *every* id it returned was
invented, the narrative is discarded too — a pitch for products we are not
showing is worse than no pitch — and the fallback is labelled `rule-based`.
`tests/test_agent.py` proves both cases.

**4. The dual-write and its repair — `app/services/catalog.py`**
SQLite is the source of truth and is written first; Chroma is a derived index
written in the same call. A failed index write leaves `vector_synced = False`, a
failed index delete leaves an orphan vector, and both are *detectable*.
`repair_sync()` fixes three drift cases and rebuilds the whole index if it finds
the vectors were produced by a different embedder than the live one. The admin
page shows row count vs vector count and has the repair button.

## Data model

| table | what it holds |
|---|---|
| `users` | email, bcrypt hash, role (`user` / `admin`) |
| `products` | the course catalog, plus `vector_synced` — the SQLite↔Chroma truth flag |
| `events` | behavior signals: `view`, `search`, `click`, `dwell`, `cart`; indexed on (user, ts) |
| `recommendations` | versioned output: narrative, product ids, behavior hash, trigger reason, source |
| `llm_calls` | one row per model call *and* per avoided call — our own observability trail |

## Models and embeddings — an honest note

**Generation goes through Mesh, always.** The default model is
`minimax/m2-her`, one of the three Mesh serves free, so the app works on a
zero-balance key. The model id is never hardcoded — it comes from `MESH_MODEL`
(or `MESH_CHAT_MODEL`), and swapping it is an env change.

Mesh is OpenAI-compatible but the models behind it are not uniformly so, and
`app/services/mesh.py:extract_text` handles two quirks seen live on 4 Aug 2026:
`minimax/m2-her` adds a non-standard `name` field to the assistant message, and
`tencent/hy3` returns an empty `content` with the answer in `reasoning_content`.
`tests/fixtures/mesh_chat_response.json` is a real captured wire response —
`scripts/smoke_mesh.py --capture` refreshes it — and the tests parse *that*,
so the parsing is pinned to what Mesh returns rather than to what the spec says.
`GET /models` is a third quirk: it returns a bare JSON array the `openai` SDK
cannot wrap, so the smoke script reads it with httpx.

**Embeddings run locally**, by decision. Mesh serves 997 models, three free, and
none of the free ones is an embedding model, so `/embeddings` returns HTTP 402 on
a zero-balance key. `EMBEDDINGS=local` (the default) uses **all-MiniLM-L6-v2**
through the ONNX runtime that ships with Chroma: a real semantic embedder that
runs offline. Nothing leaves the machine, so the only external AI calls this
project makes are chat completions to Mesh — the rule holds exactly. The active
embedder is printed by `seed.py`, shown on the admin page, and stamped into the
Chroma collection, so changing it is detected as drift rather than silently
corrupting retrieval. On a funded key, `EMBEDDINGS=mesh` moves embeddings to Mesh
too, and `EMBEDDINGS=auto` prefers Mesh with a loud fallback.

## What is not built yet

S5–S8 of the plan: LangGraph node graph, retrieval grading with query rewrite,
LangSmith tracing, the APScheduler evening digest, and frontend polish. The
recommendation pipeline is a plain function chain today, not a LangGraph graph.

## Testing

```
pytest                    # 154 tests
```

Every judged behavior has a test, and the critical ones have been mutation-checked:
disabling grounding, removing the cooldown, checking cooldown before the cache,
dropping `llm_calls` logging, making the behavior signature constant, unbounding
the behavior window, and four tracker/ingest mutations — 16 mutations run, 16
caught. `tracker.js` is executed in Node against a DOM/time/network stub
(`tests/js/tracker_harness.js`), so batching, dwell and the beacon path are
asserted rather than eyeballed. The Mesh parsing is tested against a captured
live response rather than a hand-written one.
