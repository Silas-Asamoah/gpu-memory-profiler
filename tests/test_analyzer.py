"""Regression tests for PyTorch memory analysis."""

from __future__ import annotations

import pytest

from stormlog.analyzer import MemoryAnalyzer
from stormlog.profiler import MemorySnapshot, ProfileResult


def _make_snapshot(timestamp: float, allocated_memory: int) -> MemorySnapshot:
    return MemorySnapshot(
        timestamp=timestamp,
        allocated_memory=allocated_memory,
        reserved_memory=allocated_memory,
        max_memory_allocated=allocated_memory,
        max_memory_reserved=allocated_memory,
        active_memory=allocated_memory,
        inactive_memory=0,
        cpu_memory=0,
    )


def _make_result(
    execution_time: float,
    timestamp: float,
    memory_change: int = 0,
) -> ProfileResult:
    before_memory = max(0, -memory_change)
    after_memory = before_memory + memory_change
    before = _make_snapshot(timestamp, before_memory)
    after = _make_snapshot(timestamp + execution_time, after_memory)
    return ProfileResult(
        function_name="fast_function",
        execution_time=execution_time,
        memory_before=before,
        memory_after=after,
        memory_peak=after,
        memory_allocated=max(0, memory_change),
        memory_freed=max(0, -memory_change),
        tensors_created=0,
        tensors_deleted=0,
    )


@pytest.mark.parametrize(
    "method_name",
    ["generate_performance_insights", "generate_optimization_report"],
)  # type: ignore[misc]
def test_zero_execution_times_do_not_crash_performance_analysis(
    method_name: str,
) -> None:
    analyzer = MemoryAnalyzer()
    results = [_make_result(0.0, float(index)) for index in range(5)]

    method = getattr(analyzer, method_name)

    assert method(results) is not None


def test_memory_leak_detection_ignores_mostly_freeing_function() -> None:
    mebibyte = 1024**2
    changes = [1024 * mebibyte, 1024 * mebibyte] + [-10 * mebibyte] * 11
    results = [
        _make_result(0.1, float(index), memory_change)
        for index, memory_change in enumerate(changes)
    ]

    patterns = MemoryAnalyzer().analyze_memory_patterns(results)

    assert all(pattern.pattern_type != "memory_leak" for pattern in patterns)


def test_memory_leak_detection_keeps_consistent_growth_signal() -> None:
    results = [_make_result(0.1, float(index), 40 * 1024**2) for index in range(3)]

    patterns = MemoryAnalyzer().analyze_memory_patterns(results)

    assert any(pattern.pattern_type == "memory_leak" for pattern in patterns)
