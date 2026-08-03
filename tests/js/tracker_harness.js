/**
 * Executes app/static/tracker.js inside a hand-rolled DOM stub and reports what
 * it did as JSON on stdout. Driven by tests/test_tracker_js.py.
 *
 *   node tests/js/tracker_harness.js <scenario>
 *
 * No npm dependencies on purpose: this stays a Python repo, and the stub is
 * small enough to read in one sitting. Time and the network are both faked, so
 * the batching thresholds are asserted rather than waited for.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const TRACKER = path.join(__dirname, "..", "..", "app", "static", "tracker.js");

function makeEnvironment(options = {}) {
  const state = {
    fetches: [],      // [{url, body, options}]
    beacons: [],      // [{url, body}]
    timers: [],       // pending setTimeout callbacks
    now: 1_000_000,
    listeners: { document: {}, window: {} },
    fetchOk: options.fetchOk !== false,
    fetchThrows: options.fetchThrows === true,
    beaconOk: options.beaconOk !== false
  };

  function addListener(bag, type, handler) {
    (bag[type] = bag[type] || []).push(handler);
  }

  const elements = options.elements || [];
  const findElement = (selector) => {
    if (selector === "[data-track-view]") {
      return elements.find((el) => el.attributes["data-track-view"] !== undefined) || null;
    }
    return null;
  };

  const document = {
    readyState: "complete",
    visibilityState: "visible",
    querySelector: findElement,
    addEventListener: (type, handler) => addListener(state.listeners.document, type, handler),
    dispatch: (type, event) => (state.listeners.document[type] || []).forEach((h) => h(event))
  };

  const sandbox = {
    document,
    navigator: {
      sendBeacon: (url, blob) => {
        state.beacons.push({ url, body: blob.parts.join("") });
        return state.beaconOk;
      }
    },
    Blob: class {
      constructor(parts, opts) {
        this.parts = parts;
        this.type = opts && opts.type;
      }
    },
    SMARTRECO: options.config === undefined ? { userId: 7 } : options.config,
    addEventListener: (type, handler) => addListener(state.listeners.window, type, handler),
    setTimeout: (fn, ms) => {
      state.timers.push({ fn, at: state.now + ms });
      return state.timers.length;
    },
    clearTimeout: (id) => {
      if (id) state.timers[id - 1] = null;
    },
    fetch: (url, opts) => {
      state.fetches.push({ url, body: opts.body, options: opts });
      if (state.fetchThrows) return Promise.reject(new Error("offline"));
      return Promise.resolve({ ok: state.fetchOk });
    },
    JSON,
    Math,
    console
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;

  // Date.now is the tracker's only clock; freeze and advance it by hand.
  sandbox.Date = class extends Date {
    constructor(...args) {
      super(...(args.length ? args : [state.now]));
    }
    static now() {
      return state.now;
    }
  };

  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(TRACKER, "utf8"), sandbox, { filename: "tracker.js" });

  return {
    state,
    sandbox,
    tracker: sandbox.smartReco,
    advance(ms) {
      state.now += ms;
      const due = state.timers.filter((t) => t && t.at <= state.now);
      state.timers = state.timers.map((t) => (t && t.at <= state.now ? null : t));
      due.forEach((t) => t.fn());
    },
    fireWindow(type, event) {
      (state.listeners.window[type] || []).forEach((h) => h(event));
    }
  };
}

const bodies = (entries) => entries.map((entry) => JSON.parse(entry.body).events);

/** Let queued promise callbacks (the fetch handlers) run. */
const settle = () => new Promise((resolve) => setImmediate(resolve));

const scenarios = {
  /** A full batch flushes immediately, without waiting for the timer. */
  batch_size_flush() {
    const env = makeEnvironment();
    for (let i = 0; i < 19; i += 1) env.tracker.enqueue({ type: "click", product_id: i + 1 });
    const beforeLast = { fetches: env.state.fetches.length, pending: env.tracker.pending() };
    env.tracker.enqueue({ type: "click", product_id: 20 });
    return {
      fetches_before_threshold: beforeLast.fetches,
      pending_before_threshold: beforeLast.pending,
      fetches_after_threshold: env.state.fetches.length,
      pending_after_threshold: env.tracker.pending(),
      batch_length: bodies(env.state.fetches)[0].length,
      max_batch: env.tracker.config.maxBatch
    };
  },

  /** A partial batch waits for the flush window, then goes out once. */
  time_flush() {
    const env = makeEnvironment();
    env.tracker.enqueue({ type: "view", product_id: 3 });
    env.tracker.enqueue({ type: "click", product_id: 4 });
    const early = env.state.fetches.length;
    env.advance(4000);
    const before = env.state.fetches.length;
    env.advance(1500);
    return {
      fetches_immediately: early,
      fetches_before_window: before,
      fetches_after_window: env.state.fetches.length,
      batch: bodies(env.state.fetches)[0],
      flush_ms: env.tracker.config.flushMs
    };
  },

  /** Leaving the page sends the tail through sendBeacon, not fetch. */
  pagehide_uses_beacon() {
    const env = makeEnvironment();
    env.tracker.enqueue({ type: "click", product_id: 9 });
    env.fireWindow("pagehide", {});
    return {
      fetches: env.state.fetches.length,
      beacons: env.state.beacons.length,
      beacon_events: bodies(env.state.beacons)[0],
      pending: env.tracker.pending()
    };
  },

  /** Hiding the tab closes the dwell and reports the seconds spent. */
  dwell_on_hide() {
    const env = makeEnvironment({
      elements: [{ attributes: { "data-track-view": "1", "data-product-id": "42" },
                   getAttribute(name) { return this.attributes[name]; },
                   hasAttribute(name) { return this.attributes[name] !== undefined; } }]
    });
    const afterLoad = env.tracker.peek();
    env.state.now += 37_000;
    env.sandbox.document.visibilityState = "hidden";
    env.sandbox.document.dispatch("visibilitychange", {});
    return {
      on_load: afterLoad,
      beacon_events: bodies(env.state.beacons)[0] || []
    };
  },

  /** A glance shorter than the floor is not reported as a dwell. */
  short_dwell_ignored() {
    const env = makeEnvironment({
      elements: [{ attributes: { "data-track-view": "1", "data-product-id": "42" },
                   getAttribute(name) { return this.attributes[name]; },
                   hasAttribute(name) { return this.attributes[name] !== undefined; } }]
    });
    env.state.now += 900;  // under MIN_DWELL_S
    env.sandbox.document.visibilityState = "hidden";
    env.sandbox.document.dispatch("visibilitychange", {});
    return {
      beacon_events: bodies(env.state.beacons)[0] || [],
      min_dwell_s: env.tracker.config.minDwellS
    };
  },

  /** A failed request keeps the events instead of dropping them. */
  async failed_send_requeues() {
    const env = makeEnvironment({ fetchThrows: true });
    env.tracker.enqueue({ type: "click", product_id: 5 });
    env.advance(6000);
    await settle();  // the retry happens in the fetch promise's catch
    return {
      fetches: env.state.fetches.length,
      pending: env.tracker.pending(),
      requeued: env.tracker.peek()
    };
  },

  /** A rejected-but-completed request (HTTP 500) is treated the same way. */
  async server_error_requeues() {
    const env = makeEnvironment({ fetchOk: false });
    env.tracker.enqueue({ type: "click", product_id: 5 });
    env.advance(6000);
    await settle();
    return { fetches: env.state.fetches.length, pending: env.tracker.pending() };
  },

  /** A refused beacon is kept too. */
  beacon_refused_requeues() {
    const env = makeEnvironment({ beaconOk: false });
    env.tracker.enqueue({ type: "click", product_id: 5 });
    env.fireWindow("pagehide", {});
    return { beacons: env.state.beacons.length, pending: env.tracker.pending() };
  },

  /** Clicking a catalog card is a click; the product page's own tile is not. */
  click_delegation() {
    const card = {
      attributes: { "data-product-id": "12" },
      getAttribute(name) { return this.attributes[name]; },
      hasAttribute(name) { return this.attributes[name] !== undefined; }
    };
    const detail = {
      attributes: { "data-product-id": "12", "data-track-view": "1" },
      getAttribute(name) { return this.attributes[name]; },
      hasAttribute(name) { return this.attributes[name] !== undefined; }
    };
    const env = makeEnvironment();
    env.sandbox.document.dispatch("click", { target: { closest: () => card } });
    const afterCard = env.tracker.peek();
    env.sandbox.document.dispatch("click", { target: { closest: () => detail } });
    const afterDetail = env.tracker.peek();
    env.sandbox.document.dispatch("click", { target: { closest: () => null } });
    return { after_card: afterCard, after_detail: afterDetail, after_miss: env.tracker.peek() };
  },

  /** No user id ⇒ no listeners, no events, nothing. */
  anonymous_is_not_tracked() {
    const env = makeEnvironment({ config: {} });
    env.sandbox.document.dispatch("click", {
      target: { closest: () => ({ attributes: { "data-product-id": "3" },
                                  getAttribute() { return "3"; },
                                  hasAttribute() { return false; } }) }
    });
    return {
      document_listeners: Object.keys(env.state.listeners.document),
      window_listeners: Object.keys(env.state.listeners.window),
      pending: env.tracker.pending()
    };
  },

  /** The queue is capped so a long offline session cannot eat memory. */
  async queue_is_capped() {
    const env = makeEnvironment({ fetchThrows: true });
    for (let i = 0; i < 260; i += 1) {
      env.tracker.enqueue({ type: "click", product_id: i + 1 });
      await settle();  // let each failed send requeue before the next event
    }
    const queued = env.tracker.peek();
    return {
      pending: queued.length,
      newest_kept: queued[queued.length - 1].product_id,
      oldest_kept: queued[0].product_id
    };
  }
};

const name = process.argv[2];
if (!scenarios[name]) {
  console.error(`unknown scenario: ${name}`);
  process.exit(2);
}
Promise.resolve(scenarios[name]()).then((result) => {
  process.stdout.write(JSON.stringify(result));
});
