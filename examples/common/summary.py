"""Utilities for printing profiler summaries."""

from __future__ import annotations

from typing import Any, Dict

from .formatting import print_section


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def print_profiler_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the dictionary returned by profiler.get_summary()."""
    if not summary:
        print("No profiling data available.")
        return

    if "message" in summary:
        print(summary["message"])
        return

    print_section("Profiling Summary")
    print(f"Total functions profiled: {summary.get('total_functions_profiled', 0)}")
    print(f"Total function calls: {summary.get('total_function_calls', 0)}")
    print(
        f"Total execution time: {summary.get('total_execution_time', 0.0):.3f} seconds"
    )
    peak = summary.get("peak_memory_usage", 0)
    baseline = summary.get("memory_change_from_baseline", 0)
    print(f"Peak memory usage: {_format_bytes(peak)}")
    print(f"Delta from baseline: {_format_bytes(baseline)}")

    function_stats = summary.get("function_summaries", {})
    if not function_stats:
        return

    print_section("Function Stats")
    for name, stats in function_stats.items():
        print(f"{name}:")
        print(f"  Calls: {stats.get('call_count', 0)}")
        print(f"  Avg time: {stats.get('avg_time', 0.0):.4f}s")
        avg_mem = stats.get("avg_memory_allocated", 0)
        print(f"  Avg memory: {_format_bytes(avg_mem)}")
        peak_mem = stats.get("peak_memory", 0)
        print(f"  Peak memory: {_format_bytes(peak_mem)}")
