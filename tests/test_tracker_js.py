"""tracker.js, actually executed.

The tracker is run in Node against a DOM stub (tests/js/tracker_harness.js) with
fake time and a fake network, so batching, dwell and the beacon path are asserted
rather than eyeballed. Skipped when Node is not installed — the Python suite
still covers the endpoint those batches land in.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "js" / "tracker_harness.js"
ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run(scenario: str) -> dict:
    result = subprocess.run(
        [NODE, str(HARNESS), scenario], capture_output=True, text=True, cwd=ROOT, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_full_batch_flushes_without_waiting():
    out = run("batch_size_flush")
    assert out["max_batch"] == 20
    assert out["fetches_before_threshold"] == 0, "19 events must not trigger a send"
    assert out["pending_before_threshold"] == 19
    assert out["fetches_after_threshold"] == 1
    assert out["pending_after_threshold"] == 0
    assert out["batch_length"] == 20, "one request carries the whole batch"


def test_a_partial_batch_goes_out_on_the_timer():
    out = run("time_flush")
    assert out["flush_ms"] == 5000
    assert out["fetches_immediately"] == 0, "sending must not block the interaction"
    assert out["fetches_before_window"] == 0, "not before the window closes"
    assert out["fetches_after_window"] == 1
    assert [e["type"] for e in out["batch"]] == ["view", "click"]
    assert all(e["ts"] for e in out["batch"]), "every event carries a client timestamp"


def test_leaving_the_page_uses_sendbeacon():
    out = run("pagehide_uses_beacon")
    assert out["beacons"] == 1, "the tail of a session must survive unload"
    assert out["fetches"] == 0, "fetch does not survive an unloading document"
    assert [e["product_id"] for e in out["beacon_events"]] == [9]
    assert out["pending"] == 0


def test_dwell_is_measured_and_sent_when_the_tab_is_hidden():
    out = run("dwell_on_hide")
    assert [e["type"] for e in out["on_load"]] == ["view"], "a product page opens with a view"

    dwell = [e for e in out["beacon_events"] if e["type"] == "dwell"]
    assert len(dwell) == 1
    assert dwell[0]["product_id"] == 42
    assert dwell[0]["value"] == pytest.approx(37.0), "seconds spent, not milliseconds"


def test_a_glance_is_not_a_dwell():
    out = run("short_dwell_ignored")
    assert out["min_dwell_s"] == 2
    assert [e for e in out["beacon_events"] if e["type"] == "dwell"] == []


def test_a_failed_send_keeps_the_events():
    out = run("failed_send_requeues")
    assert out["fetches"] == 1
    assert out["pending"] == 1, "a dropped request must not silently lose behavior"
    assert out["requeued"][0]["product_id"] == 5


def test_a_server_error_also_keeps_the_events():
    out = run("server_error_requeues")
    assert out["fetches"] == 1
    assert out["pending"] == 1, "HTTP 500 is a failed send, not a delivered one"


def test_a_refused_beacon_keeps_the_events():
    out = run("beacon_refused_requeues")
    assert out["beacons"] == 1
    assert out["pending"] == 1


def test_clicks_are_delegated_and_the_product_page_tile_is_not_double_counted():
    out = run("click_delegation")
    assert [e["product_id"] for e in out["after_card"]] == [12]
    assert out["after_detail"] == out["after_card"], "the viewed product is not also a click"
    assert out["after_miss"] == out["after_card"], "clicks outside a card are ignored"


def test_anonymous_visitors_are_not_tracked():
    out = run("anonymous_is_not_tracked")
    assert out["document_listeners"] == []
    assert out["window_listeners"] == []
    assert out["pending"] == 0


def test_the_queue_is_capped_when_the_network_is_down():
    """260 events with every send failing: memory stays bounded, newest win."""
    out = run("queue_is_capped")
    assert out["pending"] == 100, "an offline session must not grow without bound"
    assert out["newest_kept"] == 260
    assert out["oldest_kept"] == 161
