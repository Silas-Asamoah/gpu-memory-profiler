"""Helpers for exposing profile summaries inside the Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..context_profiler import (
    get_profile_results as get_pt_results,
    clear_results as clear_pt_results,
)

try:  # TensorFlow is optional
    from tfmemprof.context_profiler import (
        get_profile_summaries as get_tf_summaries,
        clear_profiles as clear_tf_profiles,
    )
except Exception:  # pragma: no cover - optional dependency
    get_tf_summaries = None  # type: ignore
    clear_tf_profiles = None  # type: ignore


@dataclass
class ProfileRow:
    """Lightweight view model used by the TUI tables."""

    name: str
    peak_mb: float
    delta_mb: float
    duration_ms: float
    call_count: int
    recorded_at: float


def fetch_pytorch_profiles(limit: int = 15) -> List[ProfileRow]:
    """Return recent PyTorch profile rows."""
    if get_pt_results is None:
        return []

    try:
        results = get_pt_results(limit=limit)
    except Exception:
        return []

    rows: List[ProfileRow] = []
    for result in results:
        timestamp = getattr(result.memory_after, "timestamp", None) or getattr(
            result.memory_peak, "timestamp", 0.0
        )
        rows.append(
            ProfileRow(
                name=result.function_name,
                peak_mb=result.peak_memory_usage() / (1024**2),
                delta_mb=result.memory_diff() / (1024**2),
                duration_ms=result.execution_time * 1000.0,
                call_count=result.call_count,
                recorded_at=timestamp,
            )
        )

    rows.sort(key=lambda row: row.recorded_at or 0.0, reverse=True)
    if limit:
        return rows[:limit]
    return rows


def clear_pytorch_profiles() -> bool:
    """Clear global PyTorch profile results."""
    if clear_pt_results is None:
        return False

    try:
        clear_pt_results()
        return True
    except Exception:
        return False


def fetch_tensorflow_profiles(limit: int = 15) -> List[ProfileRow]:
    """Return aggregated TensorFlow profile summaries."""
    if get_tf_summaries is None:
        return []

    try:
        summaries = get_tf_summaries(limit=limit)
    except Exception:
        return []

    rows: List[ProfileRow] = []
    for summary in summaries:
        calls = max(int(summary.get("calls", 0)), 1)
        total_duration = float(summary.get("total_duration", 0.0))
        total_memory = float(summary.get("total_memory_used", 0.0))
        peak_memory = float(summary.get("peak_memory", 0.0))
        timestamp = float(summary.get("last_timestamp") or 0.0)

        rows.append(
            ProfileRow(
                name=str(summary.get("name", "context")),
                peak_mb=peak_memory,
                delta_mb=total_memory / calls if calls else 0.0,
                duration_ms=(total_duration / calls) * 1000.0 if calls else 0.0,
                call_count=int(summary.get("calls", 0)),
                recorded_at=timestamp,
            )
        )

    rows.sort(key=lambda row: row.recorded_at or 0.0, reverse=True)
    if limit:
        return rows[:limit]
    return rows


def clear_tensorflow_profiles() -> bool:
    """Clear TensorFlow profile summaries if available."""
    if clear_tf_profiles is None:
        return False

    try:
        clear_tf_profiles()
        return True
    except Exception:
        return False

