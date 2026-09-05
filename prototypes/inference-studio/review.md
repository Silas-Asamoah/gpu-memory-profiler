# Developer review: request investigation

Primary task: find a slow or failed request, establish its measured behavior, inspect the original record, and export a reproducible selection. Core labels, values, and calculations must work without an LLM.

## Defects in the first visual specification

- Narrative claims displaced the request population and source evidence. The queue/prefill hypothesis was unsupported by the shown measurements.
- A selected request appeared without search, filters, sorting, a latency distribution, sample counts, or a stable selection mechanism.
- Three kernel rows and an undefined coverage percentage did not specify a usable trace explorer.
- Existing endpoint fields and future adapter signals looked equally available.
- Large headings and repeated prose cards left too little room for numerical inspection.

## Chosen replacement

A compact run context and filter bar lead into latency plots, a request table, and a persistent request inspector. The inspector exposes timings, chunk gaps, token counts/provenance, errors, and the original JSON. Case comparison and a measurement/source catalog are secondary views. Synthetic replay is visibly identified as playback of saved records. Unimplemented server/GPU signals remain unavailable with roadmap links.

## Source contract caveats

- `stormlog/infer/events.py:14`: request timing and token/accounting fields are client observations. First streamed chunk differs from first content; chunk gaps are not token ITL.
- `stormlog/infer/analysis.py:24`: case summaries include successful requests only. Build case discovery from raw requests/configuration so fully failed cases remain visible.
- `stormlog/infer/analysis.py:93`: percentiles use linear interpolation; throughput divides successful output counts by the earliest-success-start to latest-success-end span. Label the population and disclose missing counts.
- `stormlog/infer/samplers.py:31`: psutil samples the Stormlog client process. Local device samples do not establish endpoint-server identity. Do not combine sessions or devices into an unqualified peak.
- `docs/telemetry_schema.md` and `docs/telemetry_projection.md`: the package also has allocator/device/phase/collector-health data, but endpoint artifacts do not automatically contain those signals.
- Preserve nanosecond timestamps and original JSONL request lines. Browser-number rounding must not alter exported evidence.
- [#211](https://github.com/Silas-Asamoah/stormlog/issues/211), [#214](https://github.com/Silas-Asamoah/stormlog/issues/214), [#215](https://github.com/Silas-Asamoah/stormlog/issues/215), [#216](https://github.com/Silas-Asamoah/stormlog/issues/216), and [#217](https://github.com/Silas-Asamoah/stormlog/issues/217) govern future engine identity, queue/KV telemetry, capture, and membership; [#213](https://github.com/Silas-Asamoah/stormlog/issues/213) governs scientific comparison. A preview must not claim those backends exist.
- [#218](https://github.com/Silas-Asamoah/stormlog/issues/218) provides future structured findings. The frontend must not invent causal conclusions. [#227](https://github.com/Silas-Asamoah/stormlog/issues/227) tracks the Web UI.

## Acceptance gates

1. Fixture case aggregates match the real Python analyzer; successful, failed, warmup, missing, and non-streaming populations retain their meanings.
2. Filters, replay position, count labels, plots, table rows, and export have explicit, consistent scopes. Case comparison explicitly discloses that it uses full measured cases.
3. Selected requests keep stable identity when sorting/filtering; a selection outside the population is marked as pinned rather than silently replaced.
4. Fully failed cases remain visible, with unavailable successful latency/throughput values. Zero is not used for missing metrics.
5. Units, calculation, source, scope, timestamp, and count coverage are inspectable. Source peaks cannot include another session or mix unrelated devices.
6. Imported artifacts cannot inject markup or scripts. Invalid records fail before replacing the usable artifact. Original request exports preserve source lines.
7. Core import, filter, sorting, selection, pagination, replay, raw view, and export controls are keyboard reachable. Tables provide the numeric alternative to charts.
8. Imported runs receive no reference delta without an established comparison contract. Synthetic arithmetic is not presented as performance evidence.

## QA method and result

`tests/qa.test.mjs` runs independent parser/scope/export checks and actual application handlers against a small DOM boundary model. It does not claim browser visual, focus-order, or assistive-technology coverage. Browser validation is a separate gate performed by the implementation owner.

**PASS — bounded companion-prototype QA, 2026-09-05.** Independent rerun of `node --test tests/*.test.mjs`: **21 passed, 0 failed**. Six checks verify numerical parity and artifact semantics; fifteen independent QA checks cover parser hardening, source ownership, export fidelity, keyboard-accessible import markup, actual filter/replay/comparison/import handlers, and bounded rendering of a 150,000-gap request.

Fixed findings include import-HTML injection through workload fields, keyboard-inaccessible import, foreign-session sample contamination, invalid JSON silently repaired by timestamp handling, contradictory token-exactness flags, and unbounded chunk rendering. No blocker remains in the audited companion scope.

The implementation owner separately confirmed browser rendering, numeric/filter behavior, and no desktop document overflow. The selected 120-request case has 5 errors, TTFT p95 616.4 ms, and E2E p95 880.4 ms; the 8-request failed case has 8 errors and unavailable successful latency summaries. These values derive from the synthetic fixture, not a measured engine run.

The owner confirmed five native editable Figma states on **03 · Request investigation v2**: [Requests](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-3), [Failed case](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-1765), [Comparison](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-5192), [Sources](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-4561), and [Interaction contract](https://www.figma.com/design/gyf50thoqYq3ZubXJPeQzv/Stormlog-Studio---Native-Product-Spec?node-id=28-5795). They replace narrative-first layouts with numerical investigation and implementation guidance. **Figma contains static visual states; functional interactions are in the companion.** This review agent could not independently inspect the in-app Figma browser because its Computer tool returned `Browser is not available: iab`; Figma visual verification is attributed to the owner.

This PASS is not certification of a production service, a GPU/engine adapter, full assistive-technology compatibility, mobile layout, arbitrary artifact sizes, or backend scientific-comparison features. Those remain implementation/qualification work under the linked roadmap issues.
