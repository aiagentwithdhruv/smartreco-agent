"""Live counter for the demo video: events in, model calls out.

Read-only. Touches nothing the app owns. Run it in a second terminal beside the
browser so the efficiency claim is visible while you click, instead of asserted
afterwards.

    python scripts/watch.py
"""

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "smartreco.sqlite3"
CLEAR = "\033[2J\033[H"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def _local(ts: str) -> tuple[str, str]:
    """SQLite stores UTC. Show local time plus an age, so 'live' is obvious on camera."""
    try:
        stamp = datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
    except ValueError:
        return str(ts)[11:19], ""
    local = stamp.astimezone()
    seconds = int((datetime.now(timezone.utc) - stamp).total_seconds())
    if seconds < 60:
        age = f"  {DIM}{seconds}s ago{OFF}"
    elif seconds < 3600:
        age = f"  {DIM}{seconds // 60}m ago{OFF}"
    else:
        age = f"  {DIM}{seconds // 3600}h ago{OFF}"
    return local.strftime("%H:%M:%S"), age


def snapshot(conn):
    events = conn.execute("select count(*) from events").fetchone()[0]
    calls = conn.execute("select count(*) from llm_calls where cache_hit=0").fetchone()[0]
    cached = conn.execute("select count(*) from llm_calls where cache_hit=1").fetchone()[0]
    recent = conn.execute(
        "select model, tokens_in, tokens_out, latency_ms, cache_hit, ts"
        " from llm_calls order by id desc limit 8"
    ).fetchall()
    latest = conn.execute(
        "select version, trigger_reason, source from recommendations"
        " order by id desc limit 1"
    ).fetchone()
    return events, calls, cached, recent, latest


def main() -> None:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    while True:
        events, calls, cached, recent, latest = snapshot(conn)
        ratio = f"{events / calls:.1f}" if calls else "—"

        print(CLEAR, end="")
        print(f"{BOLD}SmartReco — live{OFF}   {DIM}read-only · refreshes every 2s{OFF}\n")
        print(f"  events tracked   {BOLD}{events}{OFF}")
        print(f"  model calls      {BOLD}{CYAN}{calls}{OFF}   {DIM}(cache hits: {cached}){OFF}")
        print(f"  events per call  {BOLD}{CYAN}{ratio}{OFF}\n")

        if latest:
            version, reason, source = latest
            print(f"  latest rec       {DIM}v{version} · {reason} ·{OFF} {CYAN}{source}{OFF}\n")

        print(f"  {DIM}recent LLM calls{OFF}   {DIM}(local time){OFF}")
        for model, tin, tout, ms, hit, ts in recent:
            when, age = _local(ts)
            if hit:
                print(f"    {when}{age}  {DIM}cache hit — no model call{OFF}")
            else:
                print(f"    {when}{age}  {CYAN}{model or '?'}{OFF}  {tin}→{tout} tok  {ms}ms")

        print(f"\n  {DIM}Ctrl-C to stop{OFF}")
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
