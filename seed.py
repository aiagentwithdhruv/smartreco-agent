"""Seed the demo database: 30 courses, 4 accounts, 3 scripted user journeys.

    python seed.py            # fresh catalog + accounts + journeys
    python seed.py --no-events  # catalog + accounts only

Idempotent: re-running wipes products/events/recommendations and rebuilds them.
Accounts are upserted, so passwords stay stable across runs.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.models import Event, LLMCall, Product, Recommendation, User, utcnow
from app.security import hash_password
from app.services.catalog import reindex_all
from app.services.embeddings import get_embedding_provider
from app.services.tracking import record_event
from app.services.vector_store import get_vector_store

DEMO_PASSWORD = "smartreco123"

ACCOUNTS: list[tuple[str, str]] = [
    ("admin@smartreco.dev", "admin"),
    ("aditi@example.com", "user"),  # journey: agentic builder
    ("rahul@example.com", "user"),  # journey: data analyst
    ("meera@example.com", "user"),  # journey: ML beginner
]

# (title, category, price, level, tags, description)
CATALOG: list[tuple[str, str, float, str, list[str], str]] = [
    # --- Agentic AI ---
    ("Agentic Workflows with LangGraph", "Agentic AI", 4999, "intermediate",
     ["langgraph", "agents", "orchestration", "python"],
     "Build stateful multi-step agents as explicit graphs: nodes, edges, conditional routing, checkpointing and human-in-the-loop pauses. Ships three production graphs you can adapt."),
    ("Agentic AI Bootcamp", "Agentic AI", 7999, "beginner",
     ["agents", "tool-calling", "openai", "bootcamp"],
     "Six weeks from zero to a working agent: tool calling, memory, planning loops, evaluation and a capstone research assistant that cites its sources."),
    ("Multi-Agent Systems in Production", "Agentic AI", 8999, "advanced",
     ["multi-agent", "supervisor", "routing", "architecture"],
     "Supervisor patterns, agent-to-agent protocols, task delegation and failure isolation. How to decide when a second agent helps and when it just doubles your bill."),
    ("Tool-Calling and Function Design for Agents", "Agentic AI", 3499, "intermediate",
     ["tools", "function-calling", "schemas", "mcp"],
     "The unglamorous half of agent work: designing tool schemas an LLM can actually use, validating arguments, and recovering from bad calls."),
    ("MCP: Model Context Protocol from Scratch", "Agentic AI", 3999, "intermediate",
     ["mcp", "protocol", "integrations", "servers"],
     "Write your own MCP server and client, expose resources and tools, and wire them into a coding agent. Includes auth and transport tradeoffs."),

    # --- LLM Engineering ---
    ("LLM Engineering for AI Engineers", "LLM Engineering", 6999, "intermediate",
     ["llm", "prompting", "structured-output", "evals"],
     "Prompt architecture, structured outputs, streaming, retries and cost control. Written for engineers shipping features, not for researchers."),
    ("Retrieval-Augmented Generation End to End", "LLM Engineering", 5999, "intermediate",
     ["rag", "vector-db", "chunking", "retrieval"],
     "Chunking strategies, hybrid search, metadata filters, reranking and grounding checks. Build a RAG service that answers only from your own documents."),
    ("Advanced RAG: Reranking, Hybrid Search and Grading", "LLM Engineering", 6499, "advanced",
     ["rag", "reranking", "hybrid-search", "evaluation"],
     "Take a demo RAG to production quality: retrieval grading, query rewriting, hybrid BM25 + vector fusion, and honest offline evaluation."),
    ("Prompt Engineering That Survives Production", "LLM Engineering", 2999, "beginner",
     ["prompting", "patterns", "testing"],
     "Prompt patterns that hold up under real traffic, plus a regression harness so a prompt edit cannot silently break yesterday's behavior."),
    ("Fine-Tuning Open Models with LoRA", "LLM Engineering", 7499, "advanced",
     ["fine-tuning", "lora", "peft", "training"],
     "When fine-tuning beats prompting and RAG — and when it does not. Data curation, LoRA/QLoRA training runs, evaluation and serving the adapter."),
    ("Guardrails and Safety Layers for LLM Apps", "LLM Engineering", 4499, "intermediate",
     ["guardrails", "safety", "validation", "pii"],
     "A six-layer guardrail stack: policy, input filtering, instruction hardening, execution limits, output validation and monitoring."),
    ("LLM Evaluation and Benchmarking", "LLM Engineering", 4999, "advanced",
     ["evals", "llm-as-judge", "metrics", "testing"],
     "Build the eval set before the feature. Golden datasets, LLM-as-judge with calibration, regression gates in CI, and reading noisy scores honestly."),

    # --- Machine Learning ---
    ("Machine Learning Foundations with Python", "Machine Learning", 3999, "beginner",
     ["python", "scikit-learn", "regression", "classification"],
     "Linear and logistic regression, trees, cross-validation and the bias-variance tradeoff, taught with scikit-learn on real tabular datasets."),
    ("Feature Engineering for Tabular Data", "Machine Learning", 3499, "intermediate",
     ["features", "tabular", "pandas", "encoding"],
     "Encoding, target leakage, time-aware splits and the feature checks that catch a broken pipeline before the model does."),
    ("Gradient Boosting with XGBoost and LightGBM", "Machine Learning", 4499, "intermediate",
     ["xgboost", "lightgbm", "boosting", "tabular"],
     "The models that still win on tabular data. Tuning that matters, early stopping, SHAP explanations and calibrated probabilities."),
    ("Deep Learning with PyTorch", "Machine Learning", 6999, "intermediate",
     ["pytorch", "neural-networks", "training", "gpu"],
     "Tensors to training loops: autograd, optimizers, schedulers, mixed precision and debugging a network that will not learn."),
    ("Time Series Forecasting in Practice", "Machine Learning", 4999, "intermediate",
     ["forecasting", "time-series", "prophet", "arima"],
     "Baselines first, then ARIMA, gradient boosting on lags and modern libraries. Backtesting that does not leak the future."),
    ("Recommendation Systems from Scratch", "Machine Learning", 5499, "advanced",
     ["recsys", "collaborative-filtering", "embeddings", "ranking"],
     "Collaborative filtering, content-based retrieval, two-tower embeddings and the ranking/evaluation loop behind every product feed."),

    # --- NLP & Vision ---
    ("Natural Language Processing Essentials", "NLP", 4499, "beginner",
     ["nlp", "tokenization", "embeddings", "text"],
     "Tokenization, embeddings, classification and sequence labelling — the vocabulary you need before transformers make sense."),
    ("Transformers Explained and Implemented", "NLP", 6499, "advanced",
     ["transformers", "attention", "pytorch", "architecture"],
     "Attention, positional encodings and a small transformer written line by line, then compared against the Hugging Face implementation."),
    ("Computer Vision with Deep Learning", "Computer Vision", 5999, "intermediate",
     ["vision", "cnn", "detection", "pytorch"],
     "Convolutional networks, transfer learning, object detection and segmentation, with a deployment-ready inference service at the end."),
    ("Multimodal AI: Text, Image and Audio", "Computer Vision", 6999, "advanced",
     ["multimodal", "clip", "whisper", "embeddings"],
     "One embedding space for many modalities. Build multimodal search and a document pipeline that reads scanned pages and charts."),

    # --- Data Engineering ---
    ("Data Engineering with Python and SQL", "Data Engineering", 5499, "beginner",
     ["sql", "python", "etl", "pipelines"],
     "Ingest, model and transform data with SQL and Python. Idempotent batch jobs, incremental loads and the schema decisions you cannot undo later."),
    ("Apache Spark for Large-Scale Data", "Data Engineering", 6999, "intermediate",
     ["spark", "pyspark", "big-data", "distributed"],
     "DataFrames, partitions, shuffles and joins that do not fall over at scale, plus reading a Spark UI to find the actual bottleneck."),
    ("Airflow: Orchestrating Data Pipelines", "Data Engineering", 4999, "intermediate",
     ["airflow", "orchestration", "dags", "scheduling"],
     "DAG design, sensors, backfills, retries and alerting — how a pipeline stays trustworthy when it runs unattended at 3am."),
    ("Streaming Data with Kafka", "Data Engineering", 6499, "advanced",
     ["kafka", "streaming", "events", "real-time"],
     "Topics, partitions, consumer groups and exactly-once semantics. Build a real-time behavior-event pipeline end to end."),

    # --- Data Analytics ---
    ("SQL for Data Analysis", "Data Analytics", 2999, "beginner",
     ["sql", "analytics", "joins", "window-functions"],
     "Joins, aggregations, window functions and CTEs, drilled on a messy real-world dataset until the syntax stops being the hard part."),
    ("Analytics Dashboards with Python", "Data Analytics", 3499, "beginner",
     ["dashboards", "streamlit", "visualization", "pandas"],
     "Turn a notebook into a dashboard people actually open: chart choice, layout, filters and caching so it loads fast."),
    ("Product Analytics and Experimentation", "Data Analytics", 4499, "intermediate",
     ["ab-testing", "metrics", "funnels", "statistics"],
     "Define metrics that mean something, design A/B tests, read funnels and know when a result is noise."),

    # --- MLOps ---
    ("MLOps: Deploying and Monitoring Models", "MLOps", 6999, "intermediate",
     ["mlops", "docker", "ci-cd", "monitoring"],
     "Package a model, ship it behind an API, version the data and watch for drift. Docker, CI/CD and rollback plans included."),
    ("LLMOps: Observability, Cost and Caching", "MLOps", 5999, "advanced",
     ["llmops", "observability", "caching", "cost"],
     "Trace every LLM call, attribute cost per feature, cache aggressively and set the triggers that stop an agent calling a model on every click."),
]

# Scripted journeys: (account email, [(event type, product title | search query, value)])
# Each is a plausible session that gives the trigger engine something to reason over.
JOURNEYS: dict[str, list[tuple[str, str | None, float | None]]] = {
    "aditi@example.com": [
        ("search", "langgraph agents", None),
        ("click", "Agentic Workflows with LangGraph", None),
        ("view", "Agentic Workflows with LangGraph", None),
        ("dwell", "Agentic Workflows with LangGraph", 96.0),
        ("view", "Multi-Agent Systems in Production", None),
        ("dwell", "Multi-Agent Systems in Production", 71.0),
        ("search", "multi agent supervisor", None),
        ("view", "Tool-Calling and Function Design for Agents", None),
        ("dwell", "Tool-Calling and Function Design for Agents", 44.0),
        ("cart", "Agentic Workflows with LangGraph", 1.0),
        ("view", "MCP: Model Context Protocol from Scratch", None),
    ],
    "rahul@example.com": [
        ("search", "sql analytics", None),
        ("click", "SQL for Data Analysis", None),
        ("view", "SQL for Data Analysis", None),
        ("dwell", "SQL for Data Analysis", 63.0),
        ("view", "Analytics Dashboards with Python", None),
        ("dwell", "Analytics Dashboards with Python", 38.0),
        ("view", "Product Analytics and Experimentation", None),
        ("dwell", "Product Analytics and Experimentation", 52.0),
        ("cart", "SQL for Data Analysis", 1.0),
    ],
    "meera@example.com": [
        ("view", "Machine Learning Foundations with Python", None),
        ("dwell", "Machine Learning Foundations with Python", 120.0),
        ("view", "Feature Engineering for Tabular Data", None),
        ("dwell", "Feature Engineering for Tabular Data", 41.0),
        ("view", "Deep Learning with PyTorch", None),
        ("dwell", "Deep Learning with PyTorch", 29.0),
    ],
}


def upsert_accounts(db: Session) -> dict[str, User]:
    """Create the demo accounts if missing; return them by email."""
    users: dict[str, User] = {}
    for email, role in ACCOUNTS:
        user = db.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(email=email, pw_hash=hash_password(DEMO_PASSWORD), role=role)
            db.add(user)
        user.role = role
        users[email] = user
    db.commit()
    return users


def load_catalog(db: Session) -> dict[str, Product]:
    """Replace the product catalog. Returns products keyed by title."""
    db.execute(delete(Product))
    db.commit()

    products: dict[str, Product] = {}
    for title, category, price, level, tags, description in CATALOG:
        product = Product(
            title=title,
            category=category,
            price=price,
            level=level,
            tags=tags,
            description=description,
            vector_synced=False,
        )
        db.add(product)
        products[title] = product
    db.commit()
    return products


def play_journeys(db: Session, users: dict[str, User], products: dict[str, Product]) -> int:
    """Replay the scripted sessions as timestamped events. Returns the event count."""
    now = utcnow()
    total = 0
    for email, steps in JOURNEYS.items():
        user = users[email]
        # Space the session out backwards from now so ordering and staleness look real.
        start = now - timedelta(minutes=len(steps) * 2)
        for i, (etype, target, value) in enumerate(steps):
            ts = start + timedelta(minutes=i * 2)
            if etype == "search":
                record_event(db, user_id=user.id, type=etype, query=target, ts=ts)
            else:
                product = products[target]
                record_event(db, user_id=user.id, type=etype, product_id=product.id, value=value, ts=ts)
            total += 1
    db.commit()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the SmartReco demo database.")
    parser.add_argument("--no-events", action="store_true", help="skip the scripted user journeys")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        # Behavior history and agent output belong to the old catalog — clear both.
        db.execute(delete(Event))
        db.execute(delete(Recommendation))
        db.execute(delete(LLMCall))
        db.commit()

        users = upsert_accounts(db)
        products = load_catalog(db)
        # The vector index is derived from the catalog, so rebuild it in the same run.
        indexed = reindex_all(db, store=get_vector_store())
        events = 0 if args.no_events else play_journeys(db, users, products)

    print(f"Seeded {len(products)} products, {len(users)} accounts, {events} events.")
    print(f"Indexed {indexed} products into Chroma via {get_embedding_provider().name} embeddings.")
    print(f"Log in as any of: {', '.join(e for e, _ in ACCOUNTS)}  (password: {DEMO_PASSWORD})")


if __name__ == "__main__":
    main()
