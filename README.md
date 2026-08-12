# SmartReco — the recommendation agent that knows when not to think
SmartReco turns searches, views, dwell time, clicks, and carts into grounded course recommendations.
Its trigger engine suppresses waste before LangGraph is allowed to call a model.
`26 events → 6 model calls (3 served from the behavior cache). 4.3 events per call.`
That is five agent runs—not 26—with four retrieval judges avoided and one honestly counted.

Built by [Dhruv Tomar](https://aiwithdhruv.com) — AI Builder.

## Run it in three commands

```bash
pip install -r requirements.txt
python seed.py
python scripts/demo.py
```

The demo is deterministic and prints every trigger decision. It runs without
credentials so the central efficiency claim can be verified from a clean clone —
but that is the demo path only. **The full application does need a Mesh API key
to generate real recommendations.** Everything required is listed below.

To use the browser app, run `uvicorn app.main:app --reload` and open
`http://127.0.0.1:8000`. The seeded learner login is
`aditi@example.com` / `smartreco123`; the admin login is
`admin@smartreco.dev` / `smartreco123`.

## Setup — everything you need

**Prerequisites**

| Requirement | Notes |
|---|---|
| Python **3.11+** | Required |
| `pip install -r requirements.txt` | FastAPI, LangGraph, LangChain, ChromaDB, APScheduler, OpenAI SDK, passlib, Jinja2 |
| Disk | SQLite file and a local Chroma index are created on first run |

**Environment variables** — `cp .env.example .env`. That file documents every one
of these inline; the table is the summary.

*LLM gateway — Mesh is the only one this project talks to*

| Variable | Default | What it does |
|---|---|---|
| `MESH_API_KEY` | *empty* | Key (`rsk_…`). **Without it the app still runs end to end** — embeddings go local and the agent writes a clearly-labelled rule-based narrative. Nothing ever fakes a model call. |
| `MESH_BASE_URL` | `https://api.meshapi.ai/v1` | Gateway endpoint. |
| `MESH_MODEL` | `minimax/m2-her` | Chat model for the narrative. Free on Mesh, so this works on a zero-balance account. `MESH_CHAT_MODEL` is accepted as an alias. |
| `MESH_EMBEDDING_MODEL` | `google/embeddinggemma-300m` | Only used when `EMBEDDINGS=mesh`. Mesh has no free embedding model, so this returns HTTP 402 without balance. |
| `EMBEDDINGS` | `local` | `local` = all-MiniLM-L6-v2 via Chroma's ONNX runtime, real semantic vectors, offline, ~80 MB downloaded once · `mesh` = force Mesh · `auto` = Mesh, falling back to local · `hashing` = deterministic lexical hashing, used by the tests. |

*App*

| Variable | Default | What it does |
|---|---|---|
| `SESSION_SECRET` | `dev-secret-change-me` | Signs the session cookie. **Change it in any real deployment.** |
| `DATABASE_URL` | `sqlite:///./smartreco.sqlite3` | Relational source of truth. |
| `CHROMA_DIR` | `./.chroma` | Local vector index. |

*Trigger engine — how often the agent is allowed to think*

| Variable | Default | What it does |
|---|---|---|
| `TRIGGER_MIN_EVENTS` | `8` | Meaningful events since the last recommendation before a recompute. |
| `TRIGGER_COOLDOWN_SECONDS` | `300` | Hard floor between two LLM-backed recomputes for one user. |
| `TRIGGER_STALENESS_MINUTES` | `30` | Age at which a recommendation counts as stale, given new activity. |
| `RETRIEVAL_TOP_K` | `6` | Products pulled from the vector store per recommendation. |

*Daily digest — reports stored recommendations, never regenerates them*

| Variable | Default | What it does |
|---|---|---|
| `DIGEST_ENABLED` | `false` | Off by default so a clone starts no background work unasked. |
| `DIGEST_HOUR` | `18` | Hour (IST) the APScheduler job fires. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | *empty* | Telegram delivery. **Wins when both are set.** |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USERNAME` / `SMTP_PASSWORD` / `SMTP_FROM` / `SMTP_USE_TLS` | `587`, `true` | Email delivery, used when `SMTP_HOST` and `SMTP_FROM` are set. Without either channel the digest is composed and logged, not sent. |

*Observability*

| Variable | Default | What it does |
|---|---|---|
| `LANGCHAIN_API_KEY` | *empty* | Enables LangSmith tracing. Empty means tracing is fully off — it is gated solely by this key. |

**Full run**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add MESH_API_KEY
python seed.py                # catalogue + demo users, dual-written to SQLite and Chroma
python scripts/demo.py        # deterministic trigger/efficiency proof
uvicorn app.main:app --reload # browser app at http://127.0.0.1:8000
pytest                        # full suite
```

**CI** runs a syntax and dependency check on every push
(`.github/workflows/smartreco-checks.yml`) and needs `MESH_API_KEY` as a
repository secret under Settings → Secrets and variables → Actions.

`.env` is gitignored and no key is committed.

## Architecture

```text
Browser
  │
  ▼
Batched tracker (view · dwell · click) + server events (search · cart)
  │
  ▼
FastAPI
  │
  ├──► SQLite source of truth ◄── dual-write/repair ──► Chroma vector index
  │
  ▼
Trigger engine — should the agent run at all?
  │ yes
  ▼
LangGraph agent
  summarize_behavior → build_interest_profile → retrieve → grade_retrieval
                                                   ▲              │
                                                   └── rewrite ───┘  once at most
                                                                  │
                                          generate → validate_grounding
                                                                  │
                                                                  ▼
                                     Versioned stored recommendations
```

The graph is a wrapper around the proven agent functions, not a behavioral
rewrite. Retrieval grading is the one new reasoning step: weak results cause a
query rewrite and exactly one retry. A second weak grade proceeds, preventing an
unbounded loop from erasing the efficiency gain.

The trigger engine decides whether the agent should run at all; the retrieval
confidence gate decides whether that run should spend a second call
second-guessing a decisive result. It skips the judge only when there are at
least three candidates, retrieval was not widened, and the measured top-hit
score is at least 0.65—the same cost discipline, one level down. On a
well-matched catalogue the judge rarely fires; it remains for genuinely
uncertain retrieval.

## The trigger engine is the product

An agent that calls a model on every click is a bill, not a product. SmartReco
builds behavior profiles deterministically, checks a behavior-hash cache before
cooldown, and gives every decision a named reason.

| Decision reason | What it does | Why it exists |
|---|---|---|
| `first_recommendation` | Runs after a new user produces their first meaningful event. | Avoids a cold-start model call before there is any evidence to use. |
| `search_intent` | Runs when a search arrives after cooldown. | Stated intent is stronger than inferred browsing interest. |
| `event_threshold` | Runs after eight new meaningful events by default. | Accumulates enough changed behavior to justify recomputation. |
| `staleness` | Runs when a stored result is older than 30 minutes and new activity exists. | Refreshes aging advice without polling idle users. |
| `cooldown` | Serves the stored recommendation inside the five-minute floor. | A burst of clicks must not become a burst of model calls. |
| `below_threshold` | Serves the stored result while changed behavior is still too sparse. | Small, noisy changes do not deserve paid reasoning yet. |
| `cache_hit` | Serves the stored result when the behavior signature is identical. | Same evidence means same answer and zero model calls. |
| `no_activity` | Does nothing when there is no behavior to interpret. | No evidence, no speculation, no cost. |

Cache hits are written to `llm_calls` as avoided calls (`cache_hit=true`), while
real grading and generation calls have separate purposes. The claim comes from
database rows, not a hand-maintained counter.

## Measured demo output

Current output from `python scripts/demo.py`, pasted verbatim:

```text
26 events → 6 model calls (3 served from the behavior cache). 4.3 events per call.
Retrieval graded 1 time, skipped 4 times — decisive vector scores avoided 4 judge calls.
```

The trigger engine authorizes exactly five agent runs. Four have decisive vector
scores and go straight to generation; one widened retrieval is deliberately
graded because dropping the category filter means confidence was not earned.
Only that actual judge call creates a `grade_retrieval` row in `llm_calls`.

## The LangGraph agent and its safety boundary

The explicit graph visits:

```text
summarize_behavior → build_interest_profile → retrieve → grade_retrieval
→ generate → validate_grounding
```

The generator sees only retrieved catalog candidates. `validate_grounding`
drops every proposed product ID that was not retrieved, even when that ID exists
elsewhere in the catalog. If all IDs are invalid, both the unsupported narrative
and its attribution are discarded, then a deterministic `rule-based` fallback
uses the top retrieved products. Currency claims are validated too.

LangSmith tracing is optional and gated solely by `LANGCHAIN_API_KEY`. With no
key, tracing creates no client, emits no warnings, and has zero runtime effect.

## Mesh compliance, directly stated

Every **external AI call**—retrieval grading and recommendation generation—goes
through the OpenAI-compatible Mesh client in `app/services/mesh.py`. There is no
second remote model provider and no hidden direct API path. Every successful
call is recorded with purpose, model, token counts, latency, user, and cache
status.

Embeddings run locally by default with `local-minilm` (all-MiniLM-L6-v2 through
Chroma's ONNX runtime). Local inference is not an external API call: no embedding
request leaves the machine, so no external AI call happens outside Mesh. This
satisfies the Mesh rule rather than routing around it. A funded account can opt
into Mesh embeddings with `EMBEDDINGS=mesh`.

With no `MESH_API_KEY`, the application still runs end to end and labels its
deterministic narrative `rule-based`; it never pretends the offline fallback was
model output. The demo uses a deterministic fake Mesh client so both graph calls
and grounding remain testable at zero cost.

## Proactive digest: report, never regenerate

An env-gated APScheduler job runs daily at `DIGEST_HOUR` (18:00 IST by default)
when `DIGEST_ENABLED=true`. It selects users with activity in the last seven
days, loads each user's latest stored recommendation, and delivers it through:

1. Telegram when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set.
2. SMTP when `SMTP_HOST` and `SMTP_FROM` are set.
3. Log-only delivery otherwise.

Log-only is the legitimate local default, not a stub. Most importantly, the
digest never calls the agent and never generates a recommendation; users without
a stored result are skipped.

## Persistence and recovery

SQLite is the source of truth. Catalog creates, updates, and deletes dual-write
to Chroma. `vector_synced` exposes incomplete writes, while the admin repair path
detects missing vectors, orphan vectors, and embedding-provider drift. The
recommendation and `llm_calls` tables preserve version history and the evidence
behind efficiency claims.

## Testing

```bash
python -m pytest -q
```

Tests run offline: Mesh is faked and embeddings use deterministic hashing. They
currently total **184 passing tests** and cover trigger reasons, graph node
order, the single retrieval retry, grounding,
digest activity selection and zero-call discipline, dual-write repair, auth,
pages, tracker batching, and ingestion. The critical retry, grounding, digest,
trigger, cache, logging, behavior-window, and tracker guarantees have also been
mutation-checked.

## What is not built

- No payments, checkout, order lifecycle, refunds, or billing integration.
- No real email infrastructure by default; SMTP and Telegram delivery require
  operator-provided credentials, otherwise the digest is intentionally log-only.
- SQLite is not Postgres and is not presented as a horizontally scalable store.
- Trigger generation is serialized per user with in-process locks. This closes
  concurrent-request races in a single application process, but a multi-worker
  deployment needs a database-level advisory lock or a distributed Redis lock.
- Chroma is local and single-node, not a distributed vector service.
- There is no deployment, worker cluster, or multi-region scheduler in this repo.

Those are productionization choices. The submitted scope is the behavior-driven
trigger, auditable agent graph, grounding boundary, recoverable catalog index,
and zero-cost reproducible demo.
