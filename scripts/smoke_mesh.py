"""Live smoke test against Mesh API — three calls, run by hand.

    python scripts/smoke_mesh.py
    python scripts/smoke_mesh.py --capture   # also refresh the response fixture

Answers the three things the rest of the build depends on:
  1. does the key work, and which models are free      (GET /models)
  2. does the configured chat model answer             (POST /chat/completions)
  3. is there any usable embedding model               (POST /embeddings)

Prints statuses only — never the key. `--capture` writes the raw chat response to
tests/fixtures/mesh_chat_response.json, which tests/test_mesh_client.py parses,
so our parsing is pinned to a shape Mesh actually returned rather than to one we
imagined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from openai import OpenAI

sys.path.insert(0, ".")

from app.config import get_settings  # noqa: E402
from app.services.mesh import extract_text  # noqa: E402

FIXTURE = Path("tests/fixtures/mesh_chat_response.json")


def list_models(settings) -> list[dict]:
    """Fetch /models with httpx, not the SDK.

    Mesh returns a bare JSON array here; the `openai` SDK expects
    `{"object": "list", "data": [...]}` and raises while wrapping it. A Mesh
    quirk, not a failure, so we read the endpoint directly.
    """
    response = httpx.get(
        f"{settings.mesh_base_url}/models",
        headers={"Authorization": f"Bearer {settings.mesh_api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("data", [])


def capture_raw(settings) -> dict:
    """One chat call read straight off the wire, for the test fixture.

    A nonce keeps Mesh's response cache (`x_cache: HIT`) from handing back an
    earlier answer, so the fixture is a genuine first response.
    """
    from uuid import uuid4

    response = httpx.post(
        f"{settings.mesh_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.mesh_api_key}"},
        json={
            "model": settings.mesh_chat_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Reply with JSON only: "
                        '{"narrative": "one short sentence", "product_ids": [1, 2]}. '
                        f"(request {uuid4().hex[:8]})"
                    ),
                }
            ],
            "temperature": 0.0,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the Mesh API key.")
    parser.add_argument("--capture", action="store_true", help="write the chat response fixture")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.mesh_configured:
        print("MESH_API_KEY is not set — copy .env.example to .env and add it.")
        return 1

    client = OpenAI(api_key=settings.mesh_api_key, base_url=settings.mesh_base_url, timeout=60.0, max_retries=0)
    print(f"base_url = {settings.mesh_base_url}")
    failures = 0

    # 1 — key check, and what this account can actually afford
    try:
        models = list_models(settings)
        free = [m["id"] for m in models if m.get("is_free")]
        free_embedders = [m["id"] for m in models if m.get("is_free") and m.get("supports_embeddings")]
        print(f"[1/3] models  OK — {len(models)} available, {len(free)} free: {', '.join(free)}")
        print(f"              free embedding models: {free_embedders or 'none — embeddings run locally'}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[1/3] models  FAILED — {type(exc).__name__}: {exc}")

    # 2 — chat on the configured (free by default) model
    try:
        response = client.chat.completions.create(
            model=settings.mesh_chat_model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            temperature=0.0,
        )
        usage = response.usage
        print(
            f"[2/3] chat    OK — model={settings.mesh_chat_model} "
            f"reply={extract_text(response)!r} "
            f"tokens={getattr(usage, 'prompt_tokens', '?')}/{getattr(usage, 'completion_tokens', '?')}"
        )
        if args.capture:
            # Capture the *wire* JSON, not response.model_dump(): the SDK's model
            # normalises away exactly the non-standard fields (e.g. m2-her's extra
            # `name` on the message) that the fixture exists to pin.
            raw = capture_raw(settings)
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(json.dumps(raw, indent=2) + "\n")
            print(f"              captured the raw wire response to {FIXTURE}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"[2/3] chat    FAILED — model={settings.mesh_chat_model} {type(exc).__name__}: {exc}")

    # 3 — embeddings (expected to fail on a zero-balance key; local is the default)
    try:
        response = client.embeddings.create(model=settings.mesh_embedding_model, input=["agentic ai course"])
        print(
            f"[3/3] embed   OK — model={settings.mesh_embedding_model} "
            f"dim={len(response.data[0].embedding)} — you can set EMBEDDINGS=mesh"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[3/3] embed   unavailable — model={settings.mesh_embedding_model} {type(exc).__name__}: {exc}")
        print("              expected on a zero-balance key; EMBEDDINGS=local is the default.")

    print("\nChat path green." if not failures else f"\n{failures} check(s) failed — see above.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
