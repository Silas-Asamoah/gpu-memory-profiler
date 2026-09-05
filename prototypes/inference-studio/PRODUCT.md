# Inference Studio prototype

Stormlog developers and inference engineers need to investigate a slow or failed run through raw requests and measurements. This is the primary workflow selected by the user. Success means selecting a latency outlier or failure, inspecting its original record and token provenance, and exporting the same filtered population without guessing how a number was calculated.

This standalone prototype reads Stormlog schema-v1 JSONL locally. The bundled runs are explicitly synthetic replay fixtures, not GPU benchmarks. The current package records client timing; server scheduling, KV-cache and kernel attribution remain roadmap capabilities. The core interface uses fixed labels, field formatters and arithmetic; it does not use an LLM. The visual direction follows stormlog.dev: dark neutral surfaces, restrained emerald selection, numeric detail and compact operational controls.
