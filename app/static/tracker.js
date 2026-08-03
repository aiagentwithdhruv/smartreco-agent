/**
 * SmartReco behavior tracker.
 *
 * Records what a signed-in visitor does and posts it in batches to /api/events.
 * Three rules it never breaks:
 *
 *   1. It never blocks the UI. No synchronous requests, passive listeners only,
 *      and the network call is fire-and-forget.
 *   2. It batches. A flush happens when the queue reaches MAX_BATCH events or
 *      FLUSH_MS after the first queued event — whichever comes first.
 *   3. It does not drop the last batch. On pagehide the queue goes out through
 *      navigator.sendBeacon, which survives the page being torn down.
 *
 * Anonymous visitors are not tracked at all: without window.SMARTRECO.userId
 * the script installs nothing.
 *
 * window.smartReco is exposed on purpose — open the console and inspect the
 * queue. The test harness in tests/js/ drives the same handle.
 */
(function (global) {
  "use strict";

  var CONFIG = global.SMARTRECO || {};
  var ENDPOINT = CONFIG.endpoint || "/api/events";
  var MAX_BATCH = 20;      // flush once this many events are queued
  var FLUSH_MS = 5000;     // ...or this long after the first one
  var MIN_DWELL_S = 2;     // a glance is not a dwell
  var MAX_QUEUE = 100;     // cap memory when the network is down

  var queue = [];
  var timer = null;
  var dwellProduct = null;
  var dwellStart = null;

  function enqueue(event) {
    event.ts = new Date().toISOString();
    queue.push(event);
    if (queue.length > MAX_QUEUE) {
      queue = queue.slice(-MAX_QUEUE);  // newest events win
    }
    if (queue.length >= MAX_BATCH) {
      flush(false);
    } else {
      schedule();
    }
  }

  function schedule() {
    if (timer !== null) return;
    timer = global.setTimeout(function () {
      timer = null;
      flush(false);
    }, FLUSH_MS);
  }

  function requeue(batch) {
    queue = batch.concat(queue).slice(-MAX_QUEUE);
    schedule();
  }

  function flush(useBeacon) {
    if (timer !== null) {
      global.clearTimeout(timer);
      timer = null;
    }
    if (queue.length === 0) return;

    var batch = queue;
    queue = [];
    var body = JSON.stringify({ events: batch });

    if (useBeacon && global.navigator && global.navigator.sendBeacon) {
      // The only transport guaranteed to survive an unloading document.
      var blob = new global.Blob([body], { type: "application/json" });
      if (!global.navigator.sendBeacon(ENDPOINT, blob)) requeue(batch);
      return;
    }

    global
      .fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        credentials: "same-origin",
        keepalive: true
      })
      .then(function (response) {
        if (!response || !response.ok) requeue(batch);
      })
      .catch(function () {
        requeue(batch);  // offline or server down: keep the signals, retry later
      });
  }

  // -- dwell -----------------------------------------------------------------

  function startDwell(productId) {
    dwellProduct = productId;
    dwellStart = Date.now();
  }

  function stopDwell() {
    if (dwellProduct === null || dwellStart === null) return;
    var seconds = (Date.now() - dwellStart) / 1000;
    dwellStart = null;
    if (seconds >= MIN_DWELL_S) {
      enqueue({ type: "dwell", product_id: dwellProduct, value: Math.round(seconds * 10) / 10 });
    }
  }

  // -- listeners -------------------------------------------------------------

  function onVisibilityChange() {
    if (global.document.visibilityState === "hidden") {
      stopDwell();          // time in a background tab is not attention
      flush(true);
    } else if (dwellProduct !== null) {
      dwellStart = Date.now();
    }
  }

  function onPageHide() {
    stopDwell();
    flush(true);
  }

  function onClick(event) {
    var target = event.target;
    if (!target || !target.closest) return;
    var card = target.closest("[data-product-id]");
    if (!card || card.hasAttribute("data-track-view")) return;  // that is a view, not a click
    var productId = parseInt(card.getAttribute("data-product-id"), 10);
    if (productId) enqueue({ type: "click", product_id: productId });
  }

  function init() {
    if (!CONFIG.userId) return;  // anonymous visitors are not tracked

    var viewed = global.document.querySelector("[data-track-view]");
    if (viewed) {
      var productId = parseInt(viewed.getAttribute("data-product-id"), 10);
      if (productId) {
        enqueue({ type: "view", product_id: productId });
        startDwell(productId);
      }
    }

    global.document.addEventListener("click", onClick, { passive: true, capture: true });
    global.document.addEventListener("visibilitychange", onVisibilityChange);
    global.addEventListener("pagehide", onPageHide);
  }

  global.smartReco = {
    enqueue: enqueue,
    flush: flush,
    init: init,
    peek: function () { return queue.slice(); },
    pending: function () { return queue.length; },
    startDwell: startDwell,
    stopDwell: stopDwell,
    config: { endpoint: ENDPOINT, maxBatch: MAX_BATCH, flushMs: FLUSH_MS, minDwellS: MIN_DWELL_S }
  };

  if (global.document.readyState === "loading") {
    global.document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(typeof window !== "undefined" ? window : globalThis);
