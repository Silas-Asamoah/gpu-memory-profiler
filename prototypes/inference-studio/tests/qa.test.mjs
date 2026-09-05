import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
    parseArtifact,
    losslessJSON,
    summarize,
    filterRequests,
    exportSelection,
} from "../data.mjs";

const root = new URL("../", import.meta.url);
const fixtureText = readFileSync(
    new URL("fixtures/candidate.jsonl", root),
    "utf8",
);
const fixture = parseArtifact(fixtureText);
const original = fixture.raw.values().next().value;
const request = (changes) => ({ ...losslessJSON(original), ...changes });
const encode = (records) =>
    records.map((record) => JSON.stringify(record)).join("\n");

test("QA: imported workload numbers cannot carry HTML", () => {
    for (const field of [
        "concurrency",
        "target_input_tokens",
        "target_output_tokens",
    ]) {
        assert.throws(
            () =>
                parseArtifact(
                    encode([
                        request({ [field]: "<img src=x onerror=alert(1)>" }),
                    ]),
                ),
            undefined,
            field,
        );
    }
});

test("QA: request timestamps reject nulls, booleans and fractional values", () => {
    for (const started_at_ns of [null, false, true, 1.5]) {
        assert.throws(
            () => parseArtifact(encode([request({ started_at_ns })])),
            undefined,
            String(started_at_ns),
        );
    }
});

test("QA: token exactness flags cannot contradict between inspector and summary", () => {
    for (const field of ["prompt_token_exact", "output_token_exact"]) {
        assert.throws(() => parseArtifact(encode([request({ [field]: "false" })])), undefined, field);
    }
});

test("QA: malformed large JSON integer syntax is not silently repaired", () => {
    assert.throws(() => losslessJSON('{"timestamp_ns":0000000000000001}'));
});

test("QA: foreign-session samples cannot contaminate source measurements", () => {
    const own = request({});
    const samples = [
        {
            schema_version: 1,
            event_type: "infer.system_sample",
            session_id: "another-session",
            timestamp_ns: own.started_at_ns,
            sampler: "psutil",
            process_rss_bytes: 999999999,
        },
    ];
    try {
        const parsed = parseArtifact(encode([own, ...samples]));
        assert.ok(
            parsed.samples.every((s) => s.session_id === own.session_id),
            "foreign-session sample remains exposed",
        );
    } catch (error) {
        assert.match(error.message, /session|ownership/i);
    }
});

test("QA: all-failed population preserves denominators and missing latency", () => {
    const rows = filterRequests(fixture.requests, {
        phase: "measured",
        status: "error",
    });
    const s = summarize(rows);
    assert.equal(s.total, 14);
    assert.equal(s.failureRate, 1);
    assert.equal(s.ttftN, 0);
    assert.equal(s.e2e[1], null);
    assert.equal(s.tps, null);
});

test("QA: selected export is byte-faithful and contains only selected source records", () => {
    const selected = fixture.requests.slice(7, 12);
    assert.equal(
        exportSelection(fixture, selected),
        selected.map((r) => fixture.raw.get(r._key)).join("\n") + "\n",
    );
    const reparsed = parseArtifact(exportSelection(fixture, selected));
    assert.deepEqual(
        reparsed.requests.map((r) => r.request_id),
        selected.map((r) => r.request_id),
    );
    assert.equal(reparsed.samples.length, 0);
});

test("QA: import has a native keyboard-focusable action", () => {
    const html = readFileSync(new URL("index.html", root), "utf8");
    assert.match(
        html,
        /<button\b[^>]*>[\s\S]*?Open JSONL[\s\S]*?<\/button>/i,
        "hidden file input needs a keyboard-accessible button",
    );
});

// Small DOM boundary model: exercise actual app handlers and rendered HTML without
// controlling the shared browser. This does not substitute for visual/browser QA.
let appSequence = 0;
async function withApp(run) {
    class Element {
        constructor(id) {
            this.id = id;
            this.value = "";
            this.innerHTML = "";
            this.textContent = "";
            this.hidden = false;
            this.disabled = false;
            this.dataset = {};
            this.listeners = {};
            this.attributes = {};
        }
        addEventListener(event, handler) {
            this.listeners[event] = handler;
        }
        setAttribute(name, value) {
            this.attributes[name] = value;
        }
        removeAttribute(name) {
            delete this.attributes[name];
        }
        insertAdjacentHTML(_position, html) {
            this.innerHTML += html;
        }
        focus() {}
        click() {
            this.onclick?.({ target: this });
        }
        remove() {}
    }
    const nodes = new Map();
    const get = (id) => {
        if (!nodes.has(id)) nodes.set(id, new Element(id));
        return nodes.get(id);
    };
    get("phase-filter").value = "measured";
    get("sort").value = "e2e_latency_ms:desc";
    get("speed").value = "12";
    const views = ["requests", "compare", "sources"].map((view) => {
        const e = new Element(view);
        e.dataset.view = view;
        return e;
    });
    const listeners = {};
    const old = Object.fromEntries(
        [
            "document",
            "location",
            "history",
            "fetch",
            "setInterval",
            "clearInterval",
        ].map((k) => [k, globalThis[k]]),
    );
    let tick;
    globalThis.document = {
        getElementById: get,
        querySelectorAll: () => views,
        addEventListener: (event, handler) => {
            listeners[event] = handler;
        },
        createElement: (tag) => new Element(tag),
    };
    globalThis.location = { hash: "" };
    globalThis.history = {
        replaceState: (_state, _title, hash) => {
            globalThis.location.hash = hash;
        },
    };
    globalThis.fetch = async (path) => {
        const content = readFileSync(new URL(path, root), "utf8");
        return {
            ok: true,
            text: async () => content,
            json: async () => JSON.parse(content),
        };
    };
    globalThis.setInterval = (callback) => {
        tick = callback;
        return 1;
    };
    globalThis.clearInterval = () => {
        tick = null;
    };
    const click = (dataset, id = "") =>
        listeners.click({
            target: { id, closest: () => (dataset ? { dataset } : null) },
        });
    const change = (id) => get(id).listeners.change?.({ target: get(id) });
    const importText = async (text) =>
        get("import-file").onchange({
            target: {
                files: [
                    {
                        name: "qa-import.jsonl",
                        size: Buffer.byteLength(text),
                        text: async () => text,
                    },
                ],
                value: "",
            },
        });
    try {
        await import(new URL(`app.mjs?qa=${++appSequence}`, root));
        await run({ get, click, change, importText, tick: () => tick?.() });
    } finally {
        for (const [k, value] of Object.entries(old)) globalThis[k] = value;
    }
}

test("QA app: failed-request filter, empty state and pinned selection retain scope", async () =>
    withApp(async ({ get, change }) => {
        get("status-filter").value = "error";
        change("status-filter");
        assert.match(get("scope-count").innerHTML, /<strong>14<\/strong>/);
        assert.match(get("metrics").innerHTML, /100\.0/);
        assert.match(get("inspector").innerHTML, /Pinned selection is outside/);
        get("query").value = "not-a-real-request";
        get("query").listeners.input();
        assert.equal(get("empty").hidden, false);
        assert.equal(get("export").disabled, true);
        assert.equal(get("next").disabled, true);
    }));

test("QA app: replay gates completed records and pause preserves the frame", async () =>
    withApp(async ({ get, tick }) => {
        get("play").onclick();
        tick();
        assert.match(get("play").textContent, /Pause/);
        const before = get("scope-count").innerHTML;
        get("play").onclick();
        tick();
        assert.equal(get("scope-count").innerHTML, before);
        assert.match(get("play").textContent, /Replay/);
    }));

test("QA app: comparison includes fully failed case and explains separate scope", async () =>
    withApp(async ({ get, click, change }) => {
        get("status-filter").value = "error";
        change("status-filter");
        click({ view: "compare" });
        assert.match(
            get("compare-view").innerHTML,
            /independent of request filters/,
        );
        assert.match(get("compare-view").innerHTML, /c8_in8192_out128/);
        assert.match(get("compare-view").innerHTML, /All-failed cases/);
    }));

test("QA app: importing malicious string metadata never emits executable markup", async () =>
    withApp(async ({ get, click, importText }) => {
        const payload = "<img src=x onerror=alert(1)>";
        const own = request({
            case_id: payload,
            request_id: payload,
            model: payload,
            error_type: payload,
            error_message: payload,
            prompt_token_source: payload,
            output_token_source: payload,
        });
        await importText(encode([own]));
        click({ view: "compare" });
        for (const id of ["request-rows", "inspector", "compare-view"])
            assert.ok(!get(id).innerHTML.includes("<img"), id);
    }));

test("QA app: imported artifact never receives synthetic reference deltas", async () =>
    withApp(async ({ get, click, importText }) => {
        await importText(encode([request({ phase: "measured" })]));
        click({ view: "compare" });
        assert.ok(!get("compare-view").innerHTML.includes("+0.0%"));
        assert.match(
            get("compare-view").innerHTML,
            /workload equivalence is not established/,
        );
    }));

test("QA app: invalid import retains the existing usable artifact", async () =>
    withApp(async ({ get, importText }) => {
        const before = get("scope-count").innerHTML;
        await importText("{invalid");
        assert.match(get("notice").textContent, /invalid JSON/);
        assert.equal(get("scope-count").innerHTML, before);
        assert.equal(get("export").disabled, false);
    }));

test("QA app: a large valid chunk array imports without unbounded plot markup", async () =>
    withApp(async ({ get, importText }) => {
        const own = request({ phase: "measured", started_at_ns: "1788609600000000000", ended_at_ns: "1788609760000000000", e2e_latency_ms: 160000, ttft_ms: 100, first_chunk_latency_ms: 90, chunk_interarrival_ms: Array(150000).fill(1) });
        await importText(encode([own]));
        assert.match(get("notice").textContent, /Opened 1 requests/);
        assert.match(get("inspector").innerHTML, /150000|150,000/);
        assert.ok(get("inspector").innerHTML.length < 200000, "collapsed inspector emits unbounded per-gap markup");
    }));
