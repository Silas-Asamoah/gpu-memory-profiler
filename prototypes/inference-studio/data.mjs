// Pure data contract. No DOM, network, narrative generation, or inferred server data.
export const numeric = (value) =>
    typeof value === "number" && Number.isFinite(value);
export function percentile(values, p) {
    const sorted = values.filter(numeric).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const rank = ((sorted.length - 1) * p) / 100,
        lo = Math.floor(rank),
        hi = Math.ceil(rank);
    return sorted[lo] * (1 - (rank - lo)) + sorted[hi] * (rank - lo);
}

// Quote large integer tokens before JSON.parse, preserving nanosecond timestamps.
// String tokens are copied whole, so an error message can never be rewritten.
export function losslessJSON(text) {
    JSON.parse(text); // Validate original JSON grammar before preserving integer tokens.
    let output = "",
        i = 0;
    while (i < text.length) {
        if (text[i] === '"') {
            const start = i++;
            while (i < text.length) {
                if (text[i] === "\\") {
                    i += 2;
                    continue;
                }
                if (text[i++] === '"') break;
            }
            output += text.slice(start, i);
        } else if (text[i] === "-" || /[0-9]/.test(text[i])) {
            const match = text
                .slice(i)
                .match(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
            if (!match) {
                output += text[i++];
                continue;
            }
            const token = match[0];
            output += /^-?\d{16,}$/.test(token) ? `"${token}"` : token;
            i += token.length;
        } else output += text[i++];
    }
    return JSON.parse(output);
}
const ns = (value) => BigInt(value);
export const elapsedMs = (value, origin) =>
    Number(ns(value) - ns(origin)) / 1e6;
export function parseArtifact(text) {
    if (text.length > 25 * 1024 * 1024)
        throw new Error("This prototype accepts artifacts up to 25 MB.");
    const records = [],
        raw = new Map();
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        let record;
        try {
            record = losslessJSON(lines[i]);
        } catch {
            throw new Error(`Line ${i + 1}: invalid JSON.`);
        }
        if (!record || Array.isArray(record) || typeof record !== "object")
            throw new Error(`Line ${i + 1}: expected an object.`);
        if (record.schema_version !== 1)
            throw new Error(
                `Line ${i + 1}: unsupported schema version ${record.schema_version}. Expected 1.`,
            );
        if (record.event_type === "infer.request") {
            for (const field of [
                "session_id",
                "request_id",
                "case_id",
                "phase",
                "status",
            ])
                if (typeof record[field] !== "string" || !record[field])
                    throw new Error(`Line ${i + 1}: missing ${field}.`);
            if (
                !["warmup", "measured"].includes(record.phase) ||
                !["ok", "error"].includes(record.status)
            )
                throw new Error(`Line ${i + 1}: unsupported phase or status.`);
            for (const field of ["prompt_token_exact", "output_token_exact"]) {
                if (
                    record[field] !== undefined &&
                    typeof record[field] !== "boolean"
                )
                    throw new Error(`Line ${i + 1}: invalid ${field}.`);
            }
            for (const field of [
                "concurrency",
                "target_input_tokens",
                "target_output_tokens",
            ])
                if (!Number.isSafeInteger(record[field]) || record[field] < 0)
                    throw new Error(`Line ${i + 1}: invalid ${field}.`);
            if (typeof record.stream !== "boolean")
                throw new Error(`Line ${i + 1}: invalid stream flag.`);
            for (const field of ["started_at_ns", "ended_at_ns"])
                if (!/^[0-9]+$/.test(String(record[field])))
                    throw new Error(`Line ${i + 1}: invalid ${field}.`);
            try {
                if (ns(record.ended_at_ns) < ns(record.started_at_ns))
                    throw Error();
            } catch {
                throw new Error(`Line ${i + 1}: invalid request timestamps.`);
            }
            for (const field of [
                "e2e_latency_ms",
                "ttft_ms",
                "first_chunk_latency_ms",
                "output_tokens",
                "prompt_tokens",
                "total_tokens",
            ])
                if (
                    record[field] != null &&
                    (!numeric(record[field]) || record[field] < 0)
                )
                    throw new Error(`Line ${i + 1}: invalid ${field}.`);
            if (
                !Array.isArray(record.chunk_interarrival_ms) ||
                record.chunk_interarrival_ms.some((x) => !numeric(x) || x < 0)
            )
                throw new Error(`Line ${i + 1}: invalid chunk intervals.`);
            const key = `${record.session_id}/${record.request_id}`;
            if (raw.has(key))
                throw new Error(`Line ${i + 1}: duplicate request identity.`);
            record._key = key;
            raw.set(key, lines[i]);
        }
        records.push(record);
    }
    const requests = records.filter((r) => r.event_type === "infer.request");
    if (!requests.length)
        throw new Error(
            "No infer.request records found. Export a schema-v1 inference run.",
        );
    if (new Set(requests.map((r) => r.session_id)).size !== 1)
        throw new Error(
            "Open one session at a time; this artifact contains multiple sessions.",
        );
    const origin = requests.reduce(
        (min, r) => (ns(r.started_at_ns) < ns(min) ? r.started_at_ns : min),
        requests[0].started_at_ns,
    );
    for (const r of requests) {
        r._start = elapsedMs(r.started_at_ns, origin);
        r._end = elapsedMs(r.ended_at_ns, origin);
    }
    const samples = records.filter(
        (r) => r.event_type === "infer.system_sample",
    );
    if (samples.some((r) => r.session_id !== requests[0].session_id))
        throw new Error("System samples must belong to the request session.");
    return {
        records,
        requests,
        raw,
        origin,
        duration: requests.reduce((max, r) => Math.max(max, r._end), 0),
        cases: [...new Set(requests.map((r) => r.case_id))],
        samples,
    };
}
export function filterRequests(requests, f = {}) {
    const query = (f.query || "").trim().toLowerCase();
    return requests.filter(
        (r) =>
            (!f.phase || f.phase === "all" || r.phase === f.phase) &&
            (!f.caseId || r.case_id === f.caseId) &&
            (!f.status || r.status === f.status) &&
            (!f.source || r.output_token_source === f.source) &&
            (!f.stream || String(r.stream) === f.stream) &&
            (!query ||
                [r.request_id, r.case_id, r.error_type, r.error_message]
                    .join(" ")
                    .toLowerCase()
                    .includes(query)) &&
            (f.playhead == null || r._end <= f.playhead) &&
            (f.start == null || r._start >= f.start) &&
            (f.end == null || r._start <= f.end) &&
            (f.latencyMin == null ||
                (numeric(r.e2e_latency_ms) &&
                    r.e2e_latency_ms >= f.latencyMin)) &&
            (f.latencyMax == null ||
                (numeric(r.e2e_latency_ms) && r.e2e_latency_ms < f.latencyMax)),
    );
}
export function summarize(requests) {
    const ok = requests.filter((r) => r.status === "ok");
    const values = (field) => ok.map((r) => r[field]).filter(numeric);
    const e2e = values("e2e_latency_ms"),
        ttft = values("ttft_ms");
    const start = ok.length
        ? ok.reduce(
              (a, r) => (ns(r.started_at_ns) < a ? ns(r.started_at_ns) : a),
              ns(ok[0].started_at_ns),
          )
        : 0n;
    const end = ok.length
        ? ok.reduce(
              (a, r) => (ns(r.ended_at_ns) > a ? ns(r.ended_at_ns) : a),
              ns(ok[0].ended_at_ns),
          )
        : 0n;
    const duration = Number(end - start) / 1e9;
    const output = values("output_tokens");
    return {
        total: requests.length,
        ok: ok.length,
        failed: requests.length - ok.length,
        failureRate: requests.length
            ? (requests.length - ok.length) / requests.length
            : null,
        e2e: [50, 95, 99].map((p) => percentile(e2e, p)),
        ttft: [50, 95, 99].map((p) => percentile(ttft, p)),
        e2eN: e2e.length,
        ttftN: ttft.length,
        tokenN: output.length,
        exactN: ok.filter(
            (r) => numeric(r.output_tokens) && r.output_token_exact === true,
        ).length,
        duration,
        outputTokens: output.reduce((a, b) => a + b, 0),
        tps:
            duration > 0 && output.length
                ? output.reduce((a, b) => a + b, 0) / duration
                : null,
    };
}
export function sortRequests(
    requests,
    field = "e2e_latency_ms",
    direction = "desc",
) {
    return [...requests].sort((a, b) => {
        if (a[field] == null && b[field] != null) return 1;
        if (b[field] == null && a[field] != null) return -1;
        const value =
            typeof a[field] === "number"
                ? a[field] - b[field]
                : String(a[field] ?? "").localeCompare(String(b[field] ?? ""));
        return (
            (direction === "desc" ? -value : value) ||
            a._key.localeCompare(b._key)
        );
    });
}
export function exportSelection(artifact, requests) {
    // Byte-preserve original request lines. Do not include unrelated session config or samples.
    return (
        requests.map((r) => artifact.raw.get(r._key)).join("\n") +
        (requests.length ? "\n" : "")
    );
}
