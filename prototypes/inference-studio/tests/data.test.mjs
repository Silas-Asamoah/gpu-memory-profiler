import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
    parseArtifact,
    losslessJSON,
    filterRequests,
    summarize,
    exportSelection,
    sortRequests,
} from "../data.mjs";
const read = (file) =>
    readFileSync(new URL(`../fixtures/${file}`, import.meta.url), "utf8");
const artifact = parseArtifact(read("candidate.jsonl"));
const golden = JSON.parse(read("candidate.report.json"));
test("aggregates match the real Python analyzer for each successful measured case", () => {
    const measured = filterRequests(artifact.requests, { phase: "measured" }),
        all = summarize(measured);
    assert.equal(all.total, golden.summary.total_requests);
    assert.equal(all.failed, golden.summary.failed_requests);
    assert.equal(all.failureRate, golden.summary.failure_rate);
    for (const [id, expected] of Object.entries(golden.cases)) {
        const actual = summarize(measured.filter((r) => r.case_id === id));
        assert.equal(actual.ok, expected.request_count);
        for (const [index, p] of [50, 95, 99].entries()) {
            assert.ok(
                Math.abs(actual.e2e[index] - expected.latency_ms[`e2e_p${p}`]) <
                    1e-8,
            );
            assert.ok(
                Math.abs(
                    actual.ttft[index] - expected.latency_ms[`ttft_p${p}`],
                ) < 1e-8,
            );
        }
        assert.ok(
            Math.abs(
                actual.tps - expected.throughput.output_tokens_per_second,
            ) < 1e-8,
        );
        assert.equal(actual.duration, expected.throughput.duration_seconds);
    }
});
test("all-failed case remains discoverable and never displays zero latency", () => {
    assert.ok(artifact.cases.includes("c8_in8192_out128"));
    const rows = filterRequests(artifact.requests, {
            phase: "measured",
            caseId: "c8_in8192_out128",
        }),
        s = summarize(rows);
    assert.equal(s.total, 8);
    assert.equal(s.failed, 8);
    assert.equal(s.failureRate, 1);
    assert.equal(s.ttft[1], null);
    assert.equal(s.tps, null);
});
test("combined filters and replay give the exact export population, preserving raw ns integers", () => {
    const rows = filterRequests(artifact.requests, {
        phase: "measured",
        caseId: "c8_in4096_out128",
        status: "ok",
        source: "server_usage",
        start: 80000,
        end: 100000,
        playhead: 100000,
        latencyMin: 750,
    });
    assert.ok(rows.length > 0);
    const output = exportSelection(artifact, rows),
        reloaded = parseArtifact(output);
    assert.deepEqual(
        reloaded.requests.map((r) => r.request_id),
        rows.map((r) => r.request_id),
    );
    assert.ok(
        rows.every(
            (r) =>
                r._start >= 80000 &&
                r._start <= 100000 &&
                r._end <= 100000 &&
                r.e2e_latency_ms >= 750,
        ),
    );
    assert.equal(typeof rows[0].started_at_ns, "string");
    assert.ok(output.includes(`"started_at_ns": ${rows[0].started_at_ns}`));
    assert.ok(!output.includes('"_start"'));
});
test("large integers and embedded JSON-like error strings remain intact", () => {
    const parsed = losslessJSON(
        '{"started_at_ns":1788609600000000001,"message":"a \\"timestamp_ns\\":1788609600000000003"}',
    );
    assert.equal(parsed.started_at_ns, "1788609600000000001");
    assert.equal(parsed.message, 'a "timestamp_ns":1788609600000000003');
});
test("empty filters, nonstreaming and unknown token counts keep their meaning", () => {
    const empty = summarize(
        filterRequests(artifact.requests, { query: "no-such-request" }),
    );
    assert.equal(empty.failureRate, null);
    assert.equal(empty.ttft[1], null);
    const nonstream = filterRequests(artifact.requests, {
        phase: "measured",
        stream: "false",
        status: "ok",
    });
    assert.ok(nonstream.length > 0);
    assert.equal(summarize(nonstream).ttftN, 0);
    const unknown = filterRequests(artifact.requests, {
        phase: "measured",
        source: "unknown",
        status: "ok",
    });
    assert.ok(unknown.length > 0);
    assert.equal(summarize(unknown).tps, null);
    assert.ok(
        summarize(filterRequests(artifact.requests, { phase: "measured" }))
            .tokenN < golden.summary.successful_requests,
    );
});
test("malformed import fails atomically; missing sort values are last", () => {
    for (const input of [
        "{broken",
        "[]",
        '{"schema_version":99}',
        '{"schema_version":1,"event_type":"other"}',
    ])
        assert.throws(() => parseArtifact(input));
    const sorted = sortRequests(artifact.requests, "ttft_ms", "desc");
    assert.equal(sorted.at(-1).ttft_ms, null);
    const line = artifact.raw.values().next().value;
    assert.throws(() => parseArtifact(`${line}\n${line}`), /duplicate/);
});
