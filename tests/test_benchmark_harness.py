import json
from pathlib import Path

from examples.cli import benchmark_harness


def _write_budget_file(path: Path, artifact_growth_max: float) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "budgets": {
                    "runtime_overhead_pct_max": 500.0,
                    "cpu_overhead_pct_max": 500.0,
                    "sampling_impact_pct_max": 500.0,
                    "artifact_growth_bytes_max": artifact_growth_max,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_benchmark_harness_writes_report_with_expected_shape(tmp_path: Path) -> None:
    budgets_path = tmp_path / "budgets.json"
    _write_budget_file(budgets_path, artifact_growth_max=2_000_000.0)

    output_path = tmp_path / "report.json"
    artifact_root = tmp_path / "artifacts"

    report = benchmark_harness.run_benchmark_harness(
        iterations=8,
        allocation_kb=64,
        default_interval=0.05,
        lowfreq_interval=0.2,
        budgets_path=budgets_path,
        artifact_root=artifact_root,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["passed"] is True
    assert set(report["scenarios"].keys()) == {
        "unprofiled",
        "tracked_default",
        "tracked_lowfreq",
    }
    assert report["scenarios"]["tracked_default"]["event_count"] >= 2
    assert report["metrics"]["artifact_growth_bytes"] >= 0.0


def test_benchmark_harness_check_mode_fails_intentional_budget_violation(
    tmp_path: Path,
) -> None:
    strict_budgets_path = tmp_path / "strict_budgets.json"
    _write_budget_file(strict_budgets_path, artifact_growth_max=0.0)

    output_path = tmp_path / "strict_report.json"
    artifact_root = tmp_path / "strict_artifacts"

    exit_code = benchmark_harness.main(
        [
            "--iterations",
            "6",
            "--allocation-kb",
            "32",
            "--default-interval",
            "0.05",
            "--lowfreq-interval",
            "0.2",
            "--budgets",
            str(strict_budgets_path),
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output_path),
            "--check",
        ]
    )

    assert exit_code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["budget_checks"]["artifact_growth_bytes"]["passed"] is False
