"""Live smoke test against Mesh API — three calls, run by hand.

    python scripts/smoke_mesh.py

Answers the three things the rest of the build depends on:
  1. does the key work at all (GET /models)
  2. does the configured chat model respond
  3. does /embeddings exist and return vectors (if not, the local hashing
     provider is the documented fallback — see app/services/embeddings.py)

Prints statuses only. Never prints the key.
"""

from __future__ import annotations

import sys

from openai import OpenAI

sys.path.insert(0, ".")

from app.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.mesh_configured:
        print("MESH_API_KEY is not set — copy .env.example to .env and add it.")
        return 1

    client = OpenAI(api_key=settings.mesh_api_key, base_url=settings.mesh_base_url, timeout=60.0, max_retries=0)
    print(f"base_url = {settings.mesh_base_url}")
    failures = 0

    # 1 — key check
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        print(f"[1/3] models  OK — {len(ids)} available, e.g. {', '.join(ids[:5])}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[1/3] models  FAILED — {type(exc).__name__}: {exc}")

    # 2 — chat
    try:
        response = client.chat.completions.create(
            model=settings.mesh_chat_model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            temperature=0.0,
        )
        usage = response.usage
        print(
            f"[2/3] chat    OK — model={settings.mesh_chat_model} "
            f"reply={response.choices[0].message.content!r} "
            f"tokens={getattr(usage, 'prompt_tokens', '?')}/{getattr(usage, 'completion_tokens', '?')}"
        )
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[2/3] chat    FAILED — model={settings.mesh_chat_model} {type(exc).__name__}: {exc}")

    # 3 — embeddings
    try:
        response = client.embeddings.create(
            model=settings.mesh_embedding_model, input=["agentic ai course"]
        )
        print(
            f"[3/3] embed   OK — model={settings.mesh_embedding_model} "
            f"dim={len(response.data[0].embedding)}"
        )
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[3/3] embed   FAILED — model={settings.mesh_embedding_model} {type(exc).__name__}: {exc}")

    print("\nAll three green." if not failures else f"\n{failures} of 3 failed — see above.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
