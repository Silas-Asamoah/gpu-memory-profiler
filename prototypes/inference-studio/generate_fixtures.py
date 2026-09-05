"""Seeded UI specimens, serialized and summarized by the real Stormlog contract.

Run from the repository root: python3 prototypes/inference-studio/generate_fixtures.py
These are synthetic measurements, never evidence about a model or GPU.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEST = Path(__file__).parent / "fixtures"
DEST.mkdir(exist_ok=True)
BASE = 1_788_609_600_000_000_000


def generate(name: str, latency_scale: float) -> dict[str, str]:
    from stormlog.infer.analysis import analyze_inference_events
    from stormlog.infer.events import InferenceRequestEvent, InferenceSystemSample

    rng = random.Random(42)
    session = f"synthetic-{name}-042"
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "event_type": "infer.session",
            "session_id": session,
            "timestamp_ns": BASE,
            "status": "started",
            "config": {
                "model": "fixture/model-8b",
                "endpoint": "http://example.invalid/v1",
                "concurrency": [1, 4, 8],
                "input_tokens": [1024, 4096, 8192],
                "output_tokens": [128],
                "system_sampler": "psutil",
            },
        }
    ]
    specs = [(1, 1024, 72), (4, 1024, 96), (8, 4096, 120), (8, 8192, 8)]
    for ci, (concurrency, input_tokens, count) in enumerate(specs):
        case = f"c{concurrency}_in{input_tokens}_out128"
        for i in range(-3, count):
            warmup = i < 0
            failed = (
                ci == 3
                or (ci == 2 and i in [18, 39, 60, 81, 102])
                or (ci == 1 and i == 63)
            ) and not warmup
            stream = i % 19 != 0
            slow = ci == 2 and 48 <= i <= 71
            ttft = round(
                (70 + ci * 60 + rng.uniform(0, 70) + (380 if slow else 0))
                * latency_scale,
                3,
            )
            gaps = [round(rng.uniform(7, 16) * latency_scale, 3) for _ in range(15)]
            if slow:
                gaps[7] = round(96 * latency_scale, 3)
            e2e = round(ttft + sum(gaps) + 6, 3)
            if failed:
                e2e = 2000.0 if ci == 3 else 850.0
            start = BASE + int((ci * 38 + (i + 3) * 0.245) * 1e9)
            source = (
                "unknown"
                if failed or i % 23 == 0
                else ("fallback_estimate" if i % 11 == 0 else "server_usage")
            )
            output = (
                None
                if source == "unknown"
                else (128 if source == "server_usage" else 124)
            )
            event = InferenceRequestEvent(
                session_id=session,
                request_id=f"{case}_{'warmup' if warmup else 'measured'}_{i+3:03d}",
                case_id=case,
                phase="warmup" if warmup else "measured",
                started_at_ns=start,
                ended_at_ns=start + int(e2e * 1e6),
                endpoint="http://example.invalid/v1",
                model="fixture/model-8b",
                concurrency=concurrency,
                target_input_tokens=input_tokens,
                target_output_tokens=128,
                stream=stream,
                status="error" if failed else "ok",
                e2e_latency_ms=e2e,
                ttft_ms=ttft if stream and not failed else None,
                first_chunk_latency_ms=(
                    round(ttft - 8, 3) if stream and not failed else None
                ),
                chunk_interarrival_ms=gaps if stream and not failed else [],
                prompt_tokens=input_tokens,
                prompt_token_source="fallback_estimate" if failed else "server_usage",
                prompt_token_exact=not failed,
                output_tokens=output,
                output_token_source=source,
                output_token_exact=source == "server_usage",
                total_tokens=input_tokens + output if output is not None else None,
                finish_reason=None if failed else "length",
                error_type=(
                    "TimeoutError"
                    if ci == 3 and failed
                    else ("HTTPError" if failed else None)
                ),
                error_message=(
                    (
                        "Synthetic timeout: no response within 2 s"
                        if ci == 3
                        else "Synthetic HTTP 503: service unavailable"
                    )
                    if failed
                    else None
                ),
            )
            records.append(event.to_record())
    for i in range(122):
        records.append(
            InferenceSystemSample(
                session,
                BASE + i * 1_000_000_000,
                "psutil",
                process_rss_bytes=(86 + i % 13) * 1024 * 1024,
                metadata={"scope": "profiler_client", "synthetic": True},
            ).to_record()
        )
    records.sort(key=lambda r: int(r["timestamp_ns"]))
    path = DEST / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = analyze_inference_events(path)
    (DEST / f"{name}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        "id": name,
        "title": (
            "Run 042 · latency burst" if name == "candidate" else "Run 041 · reference"
        ),
        "file": f"fixtures/{name}.jsonl",
        "kind": "Synthetic fixture",
        "description": "Seed 42. Authored timings; no inference server or GPU was measured.",
    }


if __name__ == "__main__":
    manifest = [generate("candidate", 1.0), generate("baseline", 0.74)]
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated two schema-v1 fixtures and Python analyzer reports.")
