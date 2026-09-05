import {
    parseArtifact,
    filterRequests,
    summarize,
    sortRequests,
    exportSelection,
    percentile,
    numeric,
} from "./data.mjs";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
    String(value ?? "").replace(
        /[&<>"']/g,
        (c) =>
            ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
            })[c],
    );
const fmt = (value, digits = 1) =>
    numeric(value)
        ? value.toLocaleString("en-US", {
              minimumFractionDigits: digits,
              maximumFractionDigits: digits,
          })
        : "—";
const unit = (value, suffix, digits = 1) =>
    numeric(value) ? `${fmt(value, digits)} ${suffix}` : "Unavailable";
const sourceName = (s) =>
    ({
        server_usage: "Server usage",
        fallback_estimate: "Estimated",
        unknown: "Unknown",
    })[s] ||
    s ||
    "Unknown";
const issue = (n, label = `#${n}`) =>
    `<a href="https://github.com/Silas-Asamoah/stormlog/issues/${n}" target="_blank" rel="noreferrer">${label} ↗</a>`;
const state = {
    artifact: null,
    manifest: [],
    selected: null,
    view: "requests",
    page: 0,
    playhead: null,
    start: null,
    end: null,
    latencyMin: null,
    latencyMax: null,
    raw: false,
    gaps: false,
    playing: false,
    baseline: null,
};
let timer = null,
    loadVersion = 0;
function notice(message) {
    $("notice").textContent = message;
    $("notice").hidden = !message;
}
function filters(extra = {}) {
    return {
        query: $("query").value,
        caseId: $("case-filter").value,
        status: $("status-filter").value,
        phase: $("phase-filter").value,
        source: $("source-filter").value,
        playhead: state.playhead,
        start: state.start,
        end: state.end,
        latencyMin: state.latencyMin,
        latencyMax: state.latencyMax,
        ...extra,
    };
}
function current() {
    return filterRequests(state.artifact.requests, filters());
}
function reset() {
    for (const id of ["query", "case-filter", "status-filter", "source-filter"])
        $(id).value = "";
    $("phase-filter").value = "measured";
    state.start = state.end = state.latencyMin = state.latencyMax = null;
    state.playhead = null;
    $("time-start").value = $("time-end").value = "";
    $("playhead").value = 1000;
    state.page = 0;
    stop();
    notice("");
    render();
}
function download(content, name, type = "application/x-ndjson") {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function loadRun(id) {
    const version = ++loadVersion;
    stop();
    try {
        const item = state.manifest.find((x) => x.id === id);
        const response = await fetch(item.file);
        if (!response.ok) throw Error("The artifact could not be loaded.");
        const artifact = parseArtifact(await response.text());
        if (version !== loadVersion) return;
        activate(artifact, item);
        reset();
    } catch (error) {
        notice(error.message);
    }
}
function activate(artifact, item) {
    state.artifact = artifact;
    state.item = item;
    state.page = 0;
    state.raw = false;
    state.gaps = false;
    const hash = new URLSearchParams(location.hash.slice(1));
    const prior = artifact.requests.find(
        (r) => r.request_id === hash.get("request"),
    );
    state.selected = (
        prior ||
        sortRequests(
            artifact.requests.filter(
                (r) => r.status === "ok" && r.phase === "measured",
            ),
            "ttft_ms",
        )[0] ||
        artifact.requests[0]
    )._key;
    const option = (value, label = value) =>
        `<option value="${esc(value)}">${esc(label)}</option>`;
    $("case-filter").innerHTML =
        option("", "All cases") + artifact.cases.map((x) => option(x)).join("");
    $("source-filter").innerHTML =
        option("", "All sources") +
        [
            ...new Set(
                artifact.requests.map(
                    (r) => r.output_token_source || "unknown",
                ),
            ),
        ]
            .map((x) => option(x, sourceName(x)))
            .join("");
    $("artifact-kind").textContent = item.kind.toUpperCase();
    $("run-title").textContent = item.title;
    const first = artifact.requests[0];
    $("run-meta").textContent =
        `${first.model || "Unknown model"}  ·  ${artifact.cases.length} cases  ·  ${(artifact.duration / 1000).toFixed(1)} s artifact span  ·  ${first.session_id}`;
    $("fixture-note").textContent = item.description;
    $("export").disabled = false;
    updateHash();
}
function updateHash() {
    const r = state.artifact.requests.find((x) => x._key === state.selected);
    if (r)
        history.replaceState(
            null,
            "",
            `#request=${encodeURIComponent(r.request_id)}`,
        );
}
function render() {
    if (!state.artifact) return;
    const focusKey = document.activeElement?.dataset.request;
    const rows = current(),
        stats = summarize(rows),
        phase = $("phase-filter").value;
    $("scope-count").innerHTML =
        `<strong>${rows.length}</strong> of ${state.artifact.requests.length} recorded requests <span class="dot">·</span> ${esc(phase === "all" ? "all phases" : phase)} <span class="dot">·</span> ${stats.failed} failed`;
    $("clear-brush").hidden = [
        state.start,
        state.end,
        state.latencyMin,
        state.latencyMax,
    ].every((x) => x == null);
    const metric = (label, value, suffix, detail, cls = "") =>
        `<div class="metric"><div class="label">${label}</div><div class="value ${cls}">${value}<small>${suffix}</small></div><div class="detail">${detail}</div></div>`;
    $("metrics").innerHTML =
        metric(
            "Requests",
            fmt(stats.total, 0),
            "",
            `${stats.ok} successful · ${stats.failed} failed`,
        ) +
        metric(
            "Failure rate",
            fmt(stats.failureRate == null ? null : stats.failureRate * 100),
            "%",
            `${stats.failed} / ${stats.total} requests`,
            stats.failed ? "danger" : "",
        ) +
        metric(
            "TTFT · p95",
            fmt(stats.ttft[1]),
            "ms",
            `p50 ${fmt(stats.ttft[0])} · n=${stats.ttftN}`,
        ) +
        metric(
            "E2E · p95",
            fmt(stats.e2e[1]),
            "ms",
            `p99 ${fmt(stats.e2e[2])} · n=${stats.e2eN}`,
        ) +
        metric(
            "Reported output rate",
            fmt(stats.tps),
            "tok/s",
            `${stats.tokenN}/${stats.ok} counts · ${fmt(stats.duration)} s`,
        ) +
        metric(
            "Exact output counts",
            fmt(stats.exactN, 0),
            `/ ${stats.ok}`,
            `Source flag · ${stats.ok - stats.tokenN} missing`,
        );
    const base = filterRequests(
        state.artifact.requests,
        filters({ latencyMin: null, latencyMax: null }),
    );
    renderCharts(rows, base);
    const [field, direction] = $("sort").value.split(":");
    const sorted = sortRequests(rows, field, direction);
    state.page = Math.min(
        state.page,
        Math.max(0, Math.ceil(rows.length / 25) - 1),
    );
    const page = sorted.slice(state.page * 25, (state.page + 1) * 25);
    $("table-count").textContent = `/ ${rows.length}`;
    $("request-rows").innerHTML = page
        .map(
            (r) =>
                `<tr class="${r._key === state.selected ? "selected" : ""}"><td><button class="request-link" data-request="${esc(r._key)}" aria-label="Inspect ${esc(r.request_id)}" aria-pressed="${r._key === state.selected}">${esc(r.request_id.split("_").slice(-2).join("_"))}</button><div class="case-subline">${esc(r.case_id)}</div></td><td><span class="status ${r.status}">${r.status === "ok" ? "Success" : "Error"}</span></td><td class="number">${fmt(r._start / 1000, 2)}</td><td class="number">${fmt(r.ttft_ms)}</td><td class="number">${fmt(r.e2e_latency_ms)}</td><td class="number">${fmt(r.output_tokens, 0)}</td><td class="provenance ${r.output_token_source === "fallback_estimate" ? "estimated" : ""}">${esc(sourceName(r.output_token_source))}</td></tr>`,
        )
        .join("");
    $("empty").hidden = !!rows.length;
    $("page-label").textContent = rows.length
        ? `${state.page * 25 + 1}–${Math.min((state.page + 1) * 25, rows.length)} of ${rows.length} · 25 per page`
        : "0 requests";
    $("previous").disabled = state.page === 0;
    $("next").disabled = (state.page + 1) * 25 >= rows.length;
    $("replay-time").textContent =
        `${((state.playhead ?? state.artifact.duration) / 1000).toFixed(1)} s`;
    $("export").disabled = !rows.length;
    renderInspector(rows);
    if (state.view === "compare") renderComparison();
    if (state.view === "sources") renderSources(rows);
    if (focusKey)
        document
            .querySelector(`button[data-request="${CSS.escape(focusKey)}"]`)
            ?.focus({ preventScroll: true });
}
function renderCharts(rows, distributionRows) {
    const points = rows.filter((r) => numeric(r.e2e_latency_ms));
    const maxY = Math.max(250, ...points.map((r) => r.e2e_latency_ms)),
        end = state.artifact.duration || 1;
    const W = 740,
        H = 124,
        left = 49,
        top = 12,
        right = 12,
        bottom = 29;
    const x = (ms) => left + (ms / end) * (W - left - right),
        y = (ms) => H - bottom - (ms / maxY) * (H - top - bottom);
    const ticks = [0, 0.5, 1];
    // Cap plotting density while preserving the full table/summary population.
    const step = Math.max(1, Math.ceil(points.length / 800));
    const displayed = points.filter(
        (r, i) => i % step === 0 || r._key === state.selected,
    );
    $("timeline").innerHTML =
        `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="E2E latency against request start offset. ${points.length} observations; ${displayed.length} points drawn. Use the request table for keyboard selection."><text x="2" y="10">ms</text>${ticks.map((t) => `<line x1="${left}" x2="${W - right}" y1="${y(t * maxY)}" y2="${y(t * maxY)}" stroke="#29333f" stroke-dasharray="3 4"/><text x="${left - 7}" y="${y(t * maxY) + 3}" text-anchor="end">${Math.round(t * maxY)}</text>`).join("")}${[0, 0.25, 0.5, 0.75, 1].map((t) => `<text x="${x(t * end)}" y="${H - 7}" text-anchor="${t === 1 ? "end" : "middle"}">${fmt((t * end) / 1000, 0)}s</text>`).join("")}${displayed.map((r) => `<circle data-request="${esc(r._key)}" cx="${x(r._start)}" cy="${y(r.e2e_latency_ms)}" r="${r._key === state.selected ? 5 : 2.8}" fill="${r.status === "error" ? "#ff8096" : "#00e599"}" opacity="${r._key === state.selected ? 1 : 0.65}" stroke="${r._key === state.selected ? "#f0f3f6" : "none"}"><title>${esc(r.request_id)} · ${fmt(r.e2e_latency_ms)} ms · ${r.status}</title></circle>`).join("")}</svg>`;
    const distribution = distributionRows.filter((r) =>
        numeric(r.e2e_latency_ms),
    );
    const edges = [0, 250, 500, 750, 1000, 1500, Infinity],
        counts = edges
            .slice(0, -1)
            .map(
                (lo, i) =>
                    distribution.filter(
                        (r) =>
                            r.e2e_latency_ms >= lo &&
                            r.e2e_latency_ms < edges[i + 1],
                    ).length,
            ),
        max = Math.max(1, ...counts);
    $("histogram").innerHTML = counts
        .map(
            (count, i) =>
                `<button class="bin" data-bin="${i}" aria-pressed="${state.latencyMin === edges[i]}" aria-label="Filter ${edges[i]} to ${edges[i + 1] === Infinity ? "unbounded" : edges[i + 1]} milliseconds: ${count} requests"><b>${count}</b><i style="height:${(count / max) * 82}px"></i><span>${i === 5 ? "1500+" : `${edges[i]}–${edges[i + 1]}`}</span></button>`,
        )
        .join("");
    $("distribution-n").textContent =
        `${distribution.length} before latency filter · ms`;
}
function renderInspector(rows) {
    const r = state.artifact.requests.find((x) => x._key === state.selected);
    if (!r) {
        $("inspector").innerHTML =
            '<div class="empty">Select a request to inspect its measurements.</div>';
        return;
    }
    const kv = (entries) =>
        `<dl class="kv">${entries.map(([name, value, cls = ""]) => `<dt>${name}</dt><dd class="${cls}">${value}</dd>`).join("")}</dl>`;
    const gaps = r.chunk_interarrival_ms,
        gapMax = gaps.reduce((max, value) => Math.max(max, value), 0);
    const visibleGaps = gaps.slice(0, 128);
    const streaming = r.stream === true;
    const ttftRatio =
        numeric(r.ttft_ms) && r.e2e_latency_ms > 0
            ? Math.min(100, (r.ttft_ms / r.e2e_latency_ms) * 100)
            : 0;
    const tokens = (v) => (numeric(v) ? fmt(v, 0) : "Unavailable");
    $("inspector").innerHTML =
        `<div class="inspector-head"><div class="eyebrow">SELECTED REQUEST</div><div class="panel-heading" style="padding:0;min-height:0"><h2>${esc(r.request_id.split("_").slice(-2).join("_"))}</h2><span class="status ${r.status}">${r.status === "ok" ? "Success" : "Error"}</span></div><div class="request-full-id">${esc(r.request_id)}</div><span class="inspector-note">${streaming ? "Streaming" : "Non-streaming"} · ${esc(r.phase)} · concurrency ${esc(r.concurrency)}</span>${rows.some((x) => x._key === r._key) ? "" : '<div class="pinned">Pinned selection is outside the current filter / replay window.</div>'}</div>
  ${r.status === "error" ? `<div class="inspector-section"><div class="error-detail"><strong>${esc(r.error_type || "Unknown error")}</strong><br>${esc(r.error_message || "No error detail was recorded.")}</div></div>` : ""}
  <section class="inspector-section"><h3>Client timings <span class="inspector-note">milliseconds</span></h3>${kv(
      [
          ["End to end", unit(r.e2e_latency_ms, "ms"), "strong"],
          ["Time to first token", unit(r.ttft_ms, "ms"), "accent"],
          ["First response chunk", unit(r.first_chunk_latency_ms, "ms")],
          [
              "After first token",
              numeric(r.e2e_latency_ms) && numeric(r.ttft_ms)
                  ? unit(r.e2e_latency_ms - r.ttft_ms, "ms")
                  : "Unavailable",
          ],
          ["Start offset", unit(r._start / 1000, "s", 3)],
      ],
  )}<div class="timing-bar"><i style="width:${ttftRatio}%"></i></div><div class="timing-labels"><span>Before first token</span><span>Remainder of response</span></div><p class="inspector-note">${streaming ? "First content defines TTFT. First chunk may contain only role metadata." : "TTFT is unavailable for this non-streaming response."} These are client durations, not server phase attribution.</p></section>
  <section class="inspector-section"><h3>Tokens & provenance</h3>${kv([
      [
          "Prompt / target",
          `${tokens(r.prompt_tokens)} / ${tokens(r.target_input_tokens)}`,
      ],
      ["Prompt count source", esc(sourceName(r.prompt_token_source))],
      [
          "Output / target",
          `${tokens(r.output_tokens)} / ${tokens(r.target_output_tokens)}`,
      ],
      ["Output count source", esc(sourceName(r.output_token_source))],
      [
          "Exact output flag",
          r.output_token_exact === true
              ? "true"
              : r.output_token_exact === false
                ? "false"
                : "Unavailable",
      ],
      ["Total tokens", tokens(r.total_tokens)],
      ["Finish reason", esc(r.finish_reason || "Unavailable")],
  ])}<p class="inspector-note">“Exact” is the recorded source flag. Server usage is reported by the endpoint; estimates stay labeled.</p></section>
  <section class="inspector-section"><h3>Chunk arrival gaps <span class="inspector-note">n=${gaps.length}</span></h3>${gaps.length ? `${kv([["p50 / max", `${fmt(percentile(gaps, 50))} / ${fmt(gapMax)} ms`]])}<div class="gap-bars" aria-label="Chunk interarrival gaps in recorded order">${visibleGaps.map((v, i) => `<i style="height:${Math.max(2, (v / Math.max(1, gapMax)) * 38)}px" title="Gap ${i + 1}: ${v} ms"></i>`).join("")}</div><button class="text-button" id="toggle-gaps">${state.gaps ? "Hide" : "Show"} ${gaps.length > 128 ? "first 128 of" : "all"} ${gaps.length} gap values</button>${state.gaps ? `<div class="detail-gap">${visibleGaps.map((v, i) => `<span>${i + 1}: ${fmt(v, 3)}</span>`).join("")}</div>` : ""}` : '<p class="inspector-note">No chunk intervals were recorded.</p>'}<p class="inspector-note">${gaps.length > 128 ? "First 128 gaps shown; statistics and raw export include all gaps. " : ""}A chunk may contain multiple tokens. These gaps are not inter-token latency.</p></section>
  <div class="inspector-actions"><button id="toggle-raw">${state.raw ? "Hide" : "View"} raw record</button><button id="export-one">Export request</button></div>${state.raw ? `<pre class="raw" tabindex="0" aria-label="Original request JSON">${esc(state.artifact.raw.get(r._key))}</pre>` : ""}`;
}
function renderComparison() {
    const measured = state.artifact.requests.filter(
        (r) => r.phase === "measured",
    );
    const baseline =
        state.baseline?.requests.filter((r) => r.phase === "measured") || [];
    $("compare-view").innerHTML =
        `<div class="section-intro"><div><h2>Case comparison</h2><p>Full measured cases, independent of request filters. Successful-request percentiles match Stormlog’s analyzer. Failed cases remain visible.</p></div><div class="run-picker"><label>Reference artifact</label><span>Run 041 · synthetic reference</span></div></div><div class="comparison"><table><thead><tr><th>Case / configured workload</th><th class="number">Requests</th><th class="number">Failed</th><th class="number">TTFT p50</th><th class="number">TTFT p95</th><th class="number">E2E p95</th><th class="number">Δ TTFT p95</th><th class="number">Output tok/s</th><th class="number">Count coverage</th></tr></thead><tbody>${state.artifact.cases
            .map((id) => {
                const rows = measured.filter((r) => r.case_id === id),
                    s = summarize(rows),
                    b = summarize(baseline.filter((r) => r.case_id === id)),
                    r = rows[0];
                const comparable =
                    state.item.id === "candidate" ||
                    state.item.id === "baseline";
                return `<tr><td><button class="request-link" data-case="${esc(id)}">${esc(id)} ↗</button><div class="case-subline">${r ? `${esc(r.concurrency)} concurrent · ${esc(r.target_input_tokens)} in / ${esc(r.target_output_tokens)} out` : ""}</div></td><td class="number">${s.total}</td><td class="number ${s.failed ? "delta" : ""}">${s.failed}</td><td class="number">${fmt(s.ttft[0])}</td><td class="number">${fmt(s.ttft[1])}</td><td class="number">${fmt(s.e2e[1])}</td><td class="number delta">${comparable && numeric(s.ttft[1]) && numeric(b.ttft[1]) && b.ttft[1] > 0 ? `+${fmt((s.ttft[1] / b.ttft[1] - 1) * 100)}%` : "—"}</td><td class="number">${fmt(s.tps)}</td><td class="number">${s.tokenN} / ${s.ok}</td></tr>`;
            })
            .join(
                "",
            )}</tbody></table></div><p class="footnote">Latency values are milliseconds. Δ = (candidate / reference − 1) × 100. Synthetic runs use the same seed and authored timing scales; this is not evidence of an optimization. Imported artifacts show no baseline delta because workload equivalence is not established.</p><div class="availability-title"><h2>Before making a performance claim</h2></div><p class="footnote">Repetition, matched manifests, uncertainty intervals and overhead budgets belong to ${issue(213, "scientific baseline comparison")}. This prototype exposes arithmetic only. All-failed cases have no successful latency or token-rate value.</p>`;
}
function renderSources(rows) {
    const s = summarize(rows),
        samples = state.artifact.samples,
        rss = samples.map((r) => r.process_rss_bytes).filter(numeric),
        gpu = samples.map((r) => r.device_used_bytes).filter(numeric);
    const row = (name, value, source, scope, rule) =>
        `<tr><td>${name}</td><td class="number">${value}</td><td class="formula">${source}</td><td>${scope}</td><td>${rule}</td></tr>`;
    $("sources-view").innerHTML =
        `<div class="section-intro"><div><h2>Every number has a source</h2><p>Field names and calculations are inspectable. Missing telemetry remains unavailable. Labels and calculations are deterministic; no LLM is required.</p></div></div><div class="catalog"><table><thead><tr><th>Measurement</th><th class="number">Current value</th><th>Field / calculation</th><th>Scope</th><th>Availability rule</th></tr></thead><tbody>${row("Failure rate", unit(s.failureRate == null ? null : s.failureRate * 100, "%"), "errors / filtered requests", "Current filter", `n=${s.total}; no requests → unavailable`)}${row("TTFT p95", unit(s.ttft[1], "ms"), "ttft_ms · linear percentile", "Filtered successes", `n=${s.ttftN}; omit nulls, never use zero`)}${row("E2E p95", unit(s.e2e[1], "ms"), "e2e_latency_ms · linear percentile", "Filtered successes", `n=${s.e2eN}; errors retained in table`)}${row("Output rate", unit(s.tps, "tok/s"), "Σ output_tokens / success span", "Filtered successes", `${s.tokenN}/${s.ok} counts; partial if counts missing`)}${row("Client RSS peak", rss.length ? unit(rss.reduce((max, value) => Math.max(max, value), 0) / 1048576, "MiB") : "Unavailable", "process_rss_bytes", "Full artifact · profiler client", `${rss.length} numeric samples; not server RSS`)}${row("Local device memory peak", gpu.length ? unit(gpu.reduce((max, value) => Math.max(max, value), 0) / 1073741824, "GiB") : "Unavailable", "device_used_bytes", "Sampler host / local device", `${gpu.length} numeric samples; endpoint identity unverified`)}${row("Original timestamps", "Decimal integers", "started_at_ns / ended_at_ns", "Client record", "Preserved losslessly in raw view and export")}</tbody></table></div><h2 class="availability-title">Server and GPU evidence · collector required</h2><div class="catalog"><table><thead><tr><th>Measurement</th><th>Status</th><th>Required contract</th><th>Implementation work</th></tr></thead><tbody><tr><td>Server queue / prefill / decode</td><td class="unavailable">Unavailable</td><td>Request identity, clocks, server phase spans</td><td>${issue(214)} · ${issue(215)}</td></tr><tr><td>KV-cache usage / waiting requests</td><td class="unavailable">Unavailable</td><td>Versioned engine adapter metrics</td><td>${issue(215)} · ${issue(222)} · ${issue(223)}</td></tr><tr><td>PyTorch operator / GPU kernel timeline</td><td class="unavailable">Unavailable</td><td>Bounded capture with source and overhead</td><td>${issue(216)}</td></tr><tr><td>Request → batch → GPU attribution</td><td class="unavailable">Unavailable</td><td>Explicit iteration membership; shared execution scope</td><td>${issue(211)} · ${issue(217)}</td></tr><tr><td>Capture coverage / dropped events</td><td class="unavailable">Unavailable</td><td>Numerator, denominator, clock uncertainty, gaps</td><td>${issue(211)} · ${issue(219)}</td></tr></tbody></table></div><p class="footnote">Research and adapter references remain in the linked issues. The UI must not infer queue time from TTFT, treat chunk gaps as token gaps, or assign a shared kernel’s full duration to every request.</p>`;
}
function switchView(view) {
    state.view = view;
    for (const key of ["requests", "compare", "sources"])
        $(`${key === "requests" ? "request" : key}-view`).hidden = key !== view;
    document.querySelectorAll("[data-view]").forEach((b) => {
        if (b.dataset.view === view) b.setAttribute("aria-current", "page");
        else b.removeAttribute("aria-current");
    });
    render();
}
function stop() {
    clearInterval(timer);
    timer = null;
    state.playing = false;
    $("play").textContent = "▶ Replay";
    $("play").setAttribute("aria-label", "Play artifact replay");
}
function play() {
    if (state.playing) {
        stop();
        return;
    }
    if (state.playhead == null || state.playhead >= state.artifact.duration)
        state.playhead = 0;
    state.playing = true;
    $("play").textContent = "Ⅱ Pause";
    $("play").setAttribute("aria-label", "Pause artifact replay");
    timer = setInterval(() => {
        state.playhead = Math.min(
            state.artifact.duration,
            state.playhead + 250 * Number($("speed").value),
        );
        $("playhead").value = (state.playhead / state.artifact.duration) * 1000;
        render();
        if (state.playhead >= state.artifact.duration) stop();
    }, 250);
}
for (const id of [
    "case-filter",
    "status-filter",
    "phase-filter",
    "source-filter",
    "sort",
])
    $(id).addEventListener("change", () => {
        state.page = 0;
        render();
    });
$("query").addEventListener("input", () => {
    state.page = 0;
    render();
});
$("reset").onclick = reset;
$("empty-reset").onclick = reset;
$("clear-brush").onclick = () => {
    state.start = state.end = state.latencyMin = state.latencyMax = null;
    $("time-start").value = $("time-end").value = "";
    state.page = 0;
    render();
};
$("apply-window").onclick = () => {
    const start =
            $("time-start").value === ""
                ? null
                : Number($("time-start").value) * 1000,
        end =
            $("time-end").value === ""
                ? null
                : Number($("time-end").value) * 1000;
    if (
        (start != null && start < 0) ||
        (end != null && end < 0) ||
        (start != null && end != null && start > end)
    ) {
        notice("Choose a nonnegative time window with start at or before end.");
        return;
    }
    state.start = start;
    state.end = end;
    state.page = 0;
    notice("");
    render();
};
$("previous").onclick = () => {
    state.page--;
    render();
};
$("next").onclick = () => {
    state.page++;
    render();
};
$("play").onclick = play;
$("playhead").oninput = () => {
    stop();
    state.playhead =
        (Number($("playhead").value) / 1000) * state.artifact.duration;
    state.page = 0;
    render();
};
$("run-select").onchange = () => loadRun($("run-select").value);
$("export").onclick = () => {
    const rows = current();
    download(
        exportSelection(state.artifact, rows),
        `stormlog-filtered-${rows.length}-requests.jsonl`,
    );
    notice(
        `Exported ${rows.length} original request records from the current filter. Session configuration and unrelated samples are excluded. Error messages and endpoint fields remain in these records.`,
    );
};
$("open-file").onclick = () => $("import-file").click();
$("import-file").onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    stop();
    try {
        if (file.size > 25 * 1024 * 1024)
            throw Error("This prototype accepts artifacts up to 25 MB.");
        const artifact = parseArtifact(await file.text());
        ++loadVersion;
        activate(artifact, {
            id: "imported",
            title: file.name,
            kind: "Imported artifact",
            description:
                "Local file opened in this browser. Client observations only; no connection to an inference endpoint.",
        });
        $("run-select").insertAdjacentHTML(
            "beforeend",
            `<option value="imported" disabled>${esc(file.name)}</option>`,
        );
        $("run-select").value = "imported";
        reset();
        notice(
            `Opened ${artifact.requests.length} requests locally. No artifact was uploaded.`,
        );
    } catch (error) {
        notice(error.message);
    }
    event.target.value = "";
};
document.addEventListener("click", (event) => {
    const target = event.target.closest(
        "[data-request],[data-bin],[data-view],[data-case]",
    );
    if (target?.dataset.request) {
        state.selected = target.dataset.request;
        state.raw = state.gaps = false;
        updateHash();
        render();
    }
    if (target?.dataset.bin != null) {
        const i = Number(target.dataset.bin),
            edges = [0, 250, 500, 750, 1000, 1500, Infinity];
        state.latencyMin = edges[i];
        state.latencyMax = edges[i + 1];
        state.page = 0;
        render();
    }
    if (target?.dataset.view) switchView(target.dataset.view);
    if (target?.dataset.case) {
        reset();
        $("case-filter").value = target.dataset.case;
        switchView("requests");
    }
    if (event.target.id === "toggle-raw") {
        state.raw = !state.raw;
        renderInspector(current());
        $("toggle-raw").focus();
    }
    if (event.target.id === "toggle-gaps") {
        state.gaps = !state.gaps;
        renderInspector(current());
        $("toggle-gaps").focus();
    }
    if (event.target.id === "export-one") {
        const r = state.artifact.requests.find(
            (x) => x._key === state.selected,
        );
        download(
            exportSelection(state.artifact, [r]),
            "stormlog-request.jsonl",
        );
    }
});
try {
    state.manifest = await (await fetch("fixtures/manifest.json")).json();
    $("run-select").innerHTML = state.manifest
        .map((x) => `<option value="${esc(x.id)}">${esc(x.title)}</option>`)
        .join("");
    state.baseline = parseArtifact(
        await (await fetch("fixtures/baseline.jsonl")).text(),
    );
    await loadRun("candidate");
} catch (error) {
    notice(
        `Unable to load the demo: ${error.message}. Serve this directory over HTTP using the README command.`,
    );
}
