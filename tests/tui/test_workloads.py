from gpumemprof.tui.workloads import format_cpu_summary, format_pytorch_summary


def test_format_pytorch_summary_formats_negative_delta_in_gb():
    summary = {
        "total_functions_profiled": 1,
        "total_function_calls": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_pytorch_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted


def test_format_cpu_summary_formats_negative_delta_in_gb():
    summary = {
        "snapshots_collected": 1,
        "peak_memory_usage": 0,
        "memory_change_from_baseline": -(1024**3),
    }

    formatted = format_cpu_summary(summary)

    assert "from baseline: -1.00 GB" in formatted
    assert "-1073741824" not in formatted
