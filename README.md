# SmartReco — a behavioral recommendation agent

Submission for the **SmartReco Build Challenge 2026** (Krish Naik Academy × Mesh API).

An online course marketplace where a backend agent watches how each user actually
behaves — what they search, view, linger on and add to cart — reasons over that
behavior, retrieves matching courses from a vector index of our own catalog, and
writes a short persuasive recommendation that refreshes as the behavior changes.

Every LLM and embedding call in this repo goes through **Mesh API**
(`https://api.meshapi.ai/v1`, OpenAI-compatible). There are no other model providers.

---

## Run it in three commands

```bash
pip install -r requirements.txt      # Python 3.11+
python seed.py                       # 31 courses, 4 accounts, 3 scripted journeys
uvicorn app.main:app --reload        # http://127.0.0.1:8000
```

Log in as `aditi@example.com` / `smartreco123` (admin: `admin@smartreco.dev`).
Copy `.env.example` to `.env` and add `MESH_API_KEY` for live LLM calls — without a
key the app still runs end to end, it just does not pretend to have called a model.

Tests: `pytest`

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
  │      fires on: N meaningful events · a search (high intent) · staleness + new activity
  │      blocked by: a per-user cooldown, and a behavior-hash cache hit
  ▼
AGENT: summarise behavior → retrieve from Chroma (metadata-filtered)
       → generate narrative via Mesh → grounding validator (only retrieved ids survive)
  ▼
recommendations (versioned per user) rendered on home + product pages
llm_calls logs every call: purpose, model, tokens, latency, cache_hit
```

**Stack:** FastAPI · SQLAlchemy + SQLite · Chroma · Jinja2 + vanilla JS · `openai`
SDK pointed at Mesh · pytest.

## Data model

| table | what it holds |
|---|---|
| `users` | email, bcrypt hash, role (`user` / `admin`) |
| `products` | the course catalog, plus `vector_synced` — the SQLite↔Chroma truth flag |
| `events` | every behavior signal: `view`, `search`, `click`, `dwell`, `cart`; indexed on (user, ts) |
| `recommendations` | versioned agent output: narrative, product ids, behavior hash, trigger reason |
| `llm_calls` | one row per model call or cache hit — our own observability trail |

## Status

| stage | state |
|---|---|
| S1 skeleton — auth, models, catalog pages, seed | done |
| S2 admin CRUD + SQLite↔Chroma dual-write | in progress |
| S3 batched behavior tracker | in progress |
| S4 trigger engine + agent + grounding | in progress |
| S5–S8 bonuses, polish, digest, hardening | planned |

This README is rewritten with the measured efficiency numbers (events vs LLM calls)
once the agent lands.
