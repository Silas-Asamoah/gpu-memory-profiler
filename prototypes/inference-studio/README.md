# Stormlog Studio: request investigation prototype

Implementation companion to [Web UI #227](https://github.com/Silas-Asamoah/stormlog/issues/227) and [roadmap #210](https://github.com/Silas-Asamoah/stormlog/issues/210). This revision replaces narrative diagnosis with a working request population, linked measurements and raw-record inspection.

The revised [Figma page, “03 · Request investigation v2”](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-3) contains editable request, failed-case, comparison and sources screens, plus an [interaction-contract board](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-5795). These are visual states derived from this app, connected with native Figma prototype navigation. Figma does not run the parser, data filters, replay or exports; those controls run in the browser companion below. Exact node IDs are in `design-links.json`.

[Play the connected Figma walkthrough](https://www.figma.com/proto/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-3&scaling=min-zoom&content-scaling=fixed&page-id=28%3A2&starting-point-node-id=28%3A3). Start at Requests, then use the navigation to open Failed case, Case comparison, Sources & availability, or Developer handoff. Each destination has a Requests return path. Scroll to inspect the lower panels. Use this flow link rather than the archived v1 prototype.

## Run it

From this directory, run:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765. Stop the server with Ctrl-C. No build, API key, inference endpoint or JavaScript dependency install is required. The bundled fixtures are **synthetic**, generated with seed 42 through Stormlog's actual `InferenceRequestEvent` dataclass. They are UI test specimens, not measurements of a model or GPU. Replay plays saved completed records; it does not start or control a profiling job.

Use **Open JSONL** to inspect an existing schema-v1 Stormlog inference artifact. Parsing happens in the browser, with a 25 MB limit and one session per file. Nanosecond timestamps and original exported request lines are preserved. Files are not uploaded. Validation errors leave the previous artifact intact. Files can contain endpoint names and error messages; filtered export preserves those fields.

## Walk through the investigation

1. Open Run 042. The initial selected request is the slowest successful request by TTFT. Its source fields remain available in the inspector.
2. Select `c8_in4096_out128`, then the 750–1000 ms distribution bin. The count, percentiles, request table and exported population update together. Clear the time/latency selection to restore the wider population.
3. Select a request from the table or latency plot. Inspect client TTFT, first response chunk, end-to-end duration, token count source and every chunk gap. Open **View raw record** to see the original JSONL record. The URL retains request selection.
4. Filter to **Failed**, or select `c8_in8192_out128` to see a case where every measured request failed. Successful latency summaries are unavailable; errors and elapsed request durations remain in the table.
5. Replay, pause or scrub the artifact. Only records completed by the playhead enter the population. A selected record outside the current filter or playhead stays pinned with an explicit notice.
6. Open **Case comparison** for all measured cases and a fixed synthetic reference. Click a case to investigate it. **Sources & availability** explains formulas, missing values, memory scope and the collectors required for future evidence.

The core UI has no LLM dependency. Labels are static, field values are formatted, and summaries are calculated. An optional future explanation must be a separate consumer of evidence and cannot substitute for measurements.

## Data contract

| View | Source / calculation | Scope and missing behavior |
| --- | --- | --- |
| Request table / raw record | `infer.request`, `stormlog/infer/events.py` | All matching statuses and selected phases; default measured. Original request identity is session + request ID. |
| TTFT / E2E percentiles | `ttft_ms`, `e2e_latency_ms`; rank = `(n-1)*p/100`, linear interpolation | Filtered successful requests with numeric values. Each metric exposes n. Null is never zero. |
| Failure rate | errors / all matching requests | Includes failures in the denominator. Empty population is unavailable. |
| Reported output rate | sum of known successful `output_tokens` / `(max end - min start)` of successes | Partial when token counts are missing; coverage is adjacent. A selection with no token counts is unavailable, deliberately clearer than the analyzer's zero. This is not per-GPU throughput. |
| Token provenance | `*_token_source`, `*_token_exact` | The endpoint's reported usage and estimates remain distinct. “Exact” is the stored flag, not independent validation. |
| Chunk gaps | `chunk_interarrival_ms`, in original order | Chunks can contain several tokens. Do not label this ITL. |
| Client memory | `infer.system_sample.process_rss_bytes` | Full-artifact peak; psutil observes the profiler process, not a remote inference server. |
| Device memory | `infer.system_sample.device_used_bytes` | Full-artifact peak on the sampler host. Endpoint/device identity is not established by this field. |
| Comparison | measured cases; fixed bundled reference | Includes all-failed cases omitted from the current analyzer's successful case groups. Imported workload equivalence is unknown, so deltas are unavailable. No significance claim. |
| Export | original JSONL request lines for current filter / selected request | Does not add unrelated session settings, samples or summary records. No field redaction is implied. |

`data.mjs` owns parsing, filtering, ordering, percentiles and export. `app.mjs` binds every view to that model. UI rendering is capped at 25 table rows and at most ~801 plotted points, and 128 visible chunk gaps per request; summary calculations still cover the entire filtered population. The chart discloses drawn-point count in its accessible description. This is a bounded prototype, not a production streaming store.

## Roadmap boundaries

| Required production capability | Issue |
| --- | --- |
| Correlation, units, scope, dropped events, clock uncertainty | [#211](https://github.com/Silas-Asamoah/stormlog/issues/211) |
| Repeated, matched scientific comparisons and overhead budgets | [#213](https://github.com/Silas-Asamoah/stormlog/issues/213) |
| Client/server identity and versioned vLLM metrics | [#214](https://github.com/Silas-Asamoah/stormlog/issues/214), [#215](https://github.com/Silas-Asamoah/stormlog/issues/215) |
| Bounded PyTorch capture and iteration membership | [#216](https://github.com/Silas-Asamoah/stormlog/issues/216), [#217](https://github.com/Silas-Asamoah/stormlog/issues/217) |
| Evidence reports, recorder and production export | [#218](https://github.com/Silas-Asamoah/stormlog/issues/218), [#219](https://github.com/Silas-Asamoah/stormlog/issues/219), [#220](https://github.com/Silas-Asamoah/stormlog/issues/220) |
| SGLang / TensorRT-LLM / TensorRT | [#222](https://github.com/Silas-Asamoah/stormlog/issues/222), [#223](https://github.com/Silas-Asamoah/stormlog/issues/223), [#224](https://github.com/Silas-Asamoah/stormlog/issues/224) |

Server queueing, prefill/decode attribution, KV-cache telemetry, GPU trace and join coverage are explicitly unavailable here. Current client timings cannot establish them. A shared batch/kernel duration must never be counted once per request. Research links and their feasibility limits remain in those issues.

## Verify

From the repository root, regenerate the fixtures and the golden reports using the actual Python analyzer:

```sh
python3 prototypes/inference-studio/generate_fixtures.py
```

From this directory:

```sh
npm test
```

Tests cover Python/JavaScript aggregate parity, all-failed cases, compound filters, replay cutoffs, exact exported populations, nanosecond preservation, invalid imports and missing observations. The QA findings and disposition are in `review.md`.

This prototype does not ship an HTTP control API, live ingestion, adapter discovery, server authentication, redaction policy, full trace viewer or uncertainty estimates. Its purpose is to make the numeric UI contract and interactions reviewable before implementing #227.
