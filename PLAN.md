# SmartReco Build Challenge 2026 — BUILD PLAN
> Krish Naik Academy × Mesh API · **Submission window: 2–11 Aug 2026, 12:00 IST** · Results 14 Aug
> Prizes: ₹20k / ₹10k / ₹5k / 4th–10th subscription. Judged first by an automated AI reading the repo, then humans.
> **Plan written 4 Aug 01:50 IST by Angelina.** Repo = the submission. Public from day one.

---

## 0. THE ONE-PARAGRAPH READ
Build an online-course marketplace where a backend **agent watches each user's tracked behavior** (views, searches, clicks, dwell), reasons over it, **retrieves matching products from a vector DB (RAG over our own catalog)**, and generates a **persuasive, personalized recommendation narrative** that refreshes as behavior changes. Every LLM call through **Mesh API** (OpenAI-compatible, `rsk_` key, base_url `https://api.meshapi.ai/v1`). Backend **must be Python (FastAPI)**. What's actually judged: real dual-write to SQL+vector, real retrieval, real LLM calls, **efficient triggering** (not per-click LLM spam), non-blocking batched tracking, production thinking.

## 1. NON-NEGOTIABLE RULES (from the brief — violating any = invalid)
- Public GitHub repo, all code in it. One submission per team.
- Backend: **FastAPI** (chosen over Flask — typed, async, we know it).
- **Every LLM/AI call through Mesh API** — including embeddings if we embed via API. No direct OpenAI/Anthropic calls anywhere.
- No secrets committed — `.env` gitignored. GitHub Actions secrets: `MESH_API_KEY`, `SUBMISSION_TOKEN`.
- CI workflow file at `.github/workflows/smartreco-checks.yml` (download from their careerapi URL — re-download latest; token issue was fixed per the banner).
- No faked features: hardcoded recs, never-queried vector DB, never-called LLM client = score kill.

## 2. DHRUV'S CLICKS (only he can do these)
- [x] Register on dashboard → SUBMISSION_TOKEN (✅ 4 Aug, set as repo secret by Angelina)
- [ ] Create Mesh API account → key `rsk_…` → paste to Angelina (sets secret + smoke test)
- [ ] **LinkedIn post link — REQUIRED by the submission form** (participation post; doubles as a daily-post rep)
- [ ] **X post link — REQUIRED by the submission form**
- [ ] **Demo video (YouTube) — REQUIRED by the form** (rules page said optional; the form marks it *) — record near the end, 2-3 min screen walkthrough
- [ ] Final submission before **11 Aug 12:00 IST** — ⚠️ repo URL LOCKS on submit, cannot be changed

## 2.5 DEPLOYMENT (his call 4 Aug: use the aiwithdhruv subdomain)
Live URL optional but strong for finalists: deploy dockerized app (needs a real server — Chroma persistence + APScheduler rule out pure serverless) and attach **smartreco.aiwithdhruv.com** via CNAME from the aiwithdhruv.com DNS. Host candidate: any free-tier container host (Render/Railway/Fly) — decide at S8, not before the core is green.

## 3. ARCHITECTURE (locked)
```
Browser (Jinja2 pages + tracker.js)
  │  POST /api/events  (batched every 5s or 20 events, sendBeacon on unload)
  ▼
FastAPI ──── SQLite (users, products, events, recommendations, llm_calls)
  │   └── dual-write on product CUD ──► Chroma (persistent, local dir)
  │
  ├─ TRIGGER ENGINE (the judged brain): recompute recommendation ONLY when
  │    (a) ≥N meaningful events since last rec (default 8), OR
  │    (b) a search happened (high intent), OR
  │    (c) staleness > 30 min with new activity — AND cooldown ≥5 min between LLM calls/user.
  │    Behavior-hash cache: same activity signature ⇒ serve stored rec, zero LLM.
  ▼
AGENT (LangGraph — bonus ⭐, explicit nodes):
  summarize_behavior → build_interest_profile → retrieve (Chroma top-k,
  metadata-filtered by category signals) → grade_retrieval (LLM-as-judge;
  weak ⇒ rewrite query, retry once) → generate_persuasive_rec (narrative
  + exact product ids, grounded — refuses products not in retrieval)
  ▼
recommendations table (versioned per user) → shown on home + product pages
APScheduler (bonus ⭐): daily 18:00 digest per active user → email (SMTP) or Telegram
LangSmith tracing (bonus ⭐) wrapped around the graph; llm_calls table = own observability
```
**Stack:** FastAPI · SQLite (SQLAlchemy) · **Chroma** (zero-infra, file-persistent — judges can run it) · Jinja2 + vanilla JS tracker · `openai` SDK pointed at Mesh · LangGraph · APScheduler · pytest.
**Embeddings decision:** embed via Mesh (`/embeddings`, e.g. `openai/text-embedding-3-small`) — keeps the every-AI-call-through-Mesh rule airtight. If Mesh lacks an embeddings endpoint (verify Day 1!), fall back to Chroma's default local ONNX embedder and document that no external AI call occurs (rule still satisfied — it's local, not another API).

## 4. DATA MODEL
- `users` (id, email, pw_hash, role user|admin, created_at)
- `products` (id, title, description, category, price, level, tags, created/updated_at, vector_synced bool)
- `events` (id, user_id, type view|search|click|dwell, product_id?, query?, value?, ts) — indexed (user_id, ts)
- `recommendations` (id, user_id, narrative, product_ids json, behavior_hash, trigger_reason, created_at, version)
- `llm_calls` (id, purpose, model, tokens_in/out, latency_ms, cache_hit, ts) — proves efficiency thinking to judges

## 5. WHAT MAKES US WIN (differentiators, from everything today taught us)
1. **Trigger discipline documented + measured** — the README shows llm_calls count vs events count ("212 events → 9 LLM calls"). Judges explicitly score this.
2. **Grounding guarantee** — generator can only cite retrieved ids; a validator drops hallucinated ids; test proves it.
3. **Behavior-hash caching** — identical recent-activity signature never re-calls the LLM. TrainBot's cache lesson, reapplied.
4. **Honest README** — architecture diagram, tradeoffs, what's NOT built. Judges are humans + AI reading a repo: the README is the pitch.
5. **Tests + a seeded demo script** (`python seed.py` → 30 products, 3 fake user journeys → visible recs) so an evaluator sees it work in 60 seconds.
6. All four ⭐ bonuses: LangGraph, APScheduler proactive digest, LangSmith, retrieval polish (metadata filters + retrieval grading + query rewrite).

## 6. BUILD SEQUENCE (Opus builder, staged commits — same discipline as rail-agent tonight)
- **S1 Skeleton** (repo, FastAPI, auth user/admin, SQLAlchemy models, Jinja2 base, seed script, CI workflow file, README stub) — commit per piece
- **S2 Catalog + dual-write** (admin CRUD → SQLite + Chroma sync, sync-repair on edit/delete, tests incl. desync case)
- **S3 Tracking** (tracker.js: batching 5s/20-events, dwell via visibilitychange, sendBeacon; `/api/events` bulk insert; tests)
- **S4 Agent v1** (behavior summary → Chroma retrieve → Mesh generate → store; trigger engine + cooldown + behavior-hash cache; llm_calls logging; tests with Mesh mocked)
- **S5 Agent v2 bonuses** (LangGraph graph, retrieval grading + query rewrite, LangSmith env-gated)
- **S6 Frontend polish** (rec cards on home/product pages, "why this" expander, admin table, decent CSS — clean not fancy)
- **S7 Proactive digest** (APScheduler daily; email via SMTP env or Telegram bot fallback; real scheduler, no button)
- **S8 Hardening + README** (architecture diagram, metrics table, run-in-3-commands, demo GIF optional; final live Mesh smoke test)
**Timeline:** S1–S4 = one focused day (the core, submission-valid on its own). S5–S8 = day two. Buffer to Aug 11 for the video (optional) + submission.

## 7. RISKS
- **Mesh API unknowns** (rate limits, embeddings endpoint, model quirks) → Day-1 first task: 3-call smoke test, capture real responses as fixtures (TrainBot rule: capture before contract).
- Their CI workflow expectations unknown → add file + push early, watch Actions tab same day.
- **Never commit the rsk_ key** — pre-commit secret scan, same as tonight's repo.
- Judged by AI first → conventional structure, type hints, docstrings, requirements.txt exact.

## 8. STATUS LOG
- 4 Aug 01:50 — Folder + plan created. Videos transcribing (`notes/`). Repo scaffold next. Build not started.
