# SmartReco — THE EXPLANATION (for Dhruv: drills, interviews, demo video)
> Own the concept, not just the repo. Drill out loud until each layer is 4/5+.

## L1 — One sentence
"A course-selling site with an AI salesman in the backend: it watches each visitor's behavior, figures out what they want, and writes them a personal pitch recommending real courses from the catalog."

## L2 — The story of one user (demo-video spine)
1. Priya browses: opens LangGraph Bootcamp, searches "AI agents", 3 min on an advanced course. Tracker.js records every action as an EVENT (view/search/click/dwell/cart) — batched every 5s, sendBeacon on exit, never blocks the page.
2. Nothing fires yet — deliberate. The TRIGGER ENGINE waits for signal: ≥8 meaningful events, OR a search (high intent), OR staleness with new activity — with a 5-min cooldown and a behavior-hash cache (same recent activity ⇒ reuse stored rec, zero LLM cost). This is the judged skill: deciding WHEN thinking is worth paying for.
3. On trigger, the AGENT runs: summarize behavior → interest profile → SEMANTIC SEARCH over our own catalog in Chroma (RAG — the AI can only recommend what exists) → grade the retrieval, rewrite the query if weak → GENERATE the persuasive pitch naming exact courses.
4. Recommendation stored + versioned + rendered on the site; refreshes as behavior evolves. Bonus: scheduled evening digest email/Telegram via APScheduler — proactive, not button-triggered.

## L3 — Pieces and why
- FastAPI + SQLite → the shop (users, products, events, recommendations, llm_calls)
- Dual-write SQL + Chroma → products exist twice: rows for the site, VECTORS for meaning-search ("AI agents" finds LangGraph course without keyword match). Kept in sync on every add/edit/delete — judges test for fake vector DBs.
- tracker.js → the eyes; batching/throttling = production thinking
- Trigger engine → the wallet-guard; llm_calls table PROVES it ("212 events → 9 LLM calls")
- Agent (LangGraph nodes) → the brain; GROUNDING VALIDATOR drops any product id not in retrieval — the AI cannot invent a course
- Mesh API → the mandated gateway; one OpenAI-compatible key, model env-swappable (free minimax/m2-her default)

## L4 — Vocabulary you now own
agentic (decides when/what, not one prompt) · RAG (generation grounded in retrieved real data) · dual-write consistency · trigger discipline / cost-aware inference · behavioral signals · behavior-hash caching · grounding validation · LLM-as-judge (retrieval grading) · observability (llm_calls, LangSmith).

## L5 — The connective line (use in interviews)
"TrainBot and SmartReco are the same architecture in different clothes: deterministic events in, a gate deciding when intelligence runs, retrieval before generation so the AI can't lie, and everything measured. That's what production AI looks like."

## Drill questions (answer out loud, ≥4/5)
1. Why not call the LLM on every click? (cost amplification; trigger discipline; cache)
2. Why a vector DB at all — why not SQL LIKE? (semantic vs lexical; embeddings capture meaning)
3. What stops the AI recommending a course that doesn't exist? (retrieval-grounding + id validator + test)
4. What makes tracking "non-blocking"? (batch buffer, 5s flush, sendBeacon, no awaits in UI path)
5. Why is dual-write hard? (two stores, partial-failure sync; vector_synced flag + repair)
6. When does a recommendation refresh? (the three triggers + cooldown; hash short-circuit)
7. Where does Mesh API sit and why is a gateway useful? (one key, many models, observability, swap without code)

## Demo video skeleton (2-3 min, record at S8)
0:00 hook: "This site is watching me — watch what it does with that."
0:20 browse as a user, show signals accumulating (admin/events view)
0:50 trigger fires → show the generated pitch on the homepage, read one line aloud
1:20 show the agent trace (LangGraph nodes / logs), the llm_calls ratio — "212 events, 9 calls"
1:50 admin adds a product → appears in vector search instantly (dual-write)
2:20 the scheduled digest email arriving
2:40 close: repo + stack line.
