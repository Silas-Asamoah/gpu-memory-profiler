"""Tests for gpumemprof diagnose command."""

import json
from pathlib import Path
from types import SimpleNamespace

import gpumemprof.cli as gpumemprof_cli
import gpumemprof.diagnose as diagnose_module


def _patch_diagnose_env(monkeypatch, cuda_available=False, risk_detected=False):
    """Patch diagnose module's env/summary dependencies for predictable output."""
    monkeypatch.setattr(
        diagnose_module,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "architecture": "x86_64",
            "python_version": "3.10",
            "cuda_available": cuda_available,
            "detected_backend": "cuda" if cuda_available else "cpu",
        },
    )
    if cuda_available and risk_detected:
        gpu_info = {
            "device_id": 0,
            "device_name": "Test GPU",
            "total_memory": 1024**3,
            "allocated_memory": int(0.9 * 1024**3),
            "reserved_memory": int(0.95 * 1024**3),
            "max_memory_allocated": int(0.9 * 1024**3),
            "memory_stats": {"num_ooms": 1},
        }
        frag_info = {
            "fragmentation_ratio": 0.35,
            "utilization_ratio": 0.9,
        }
    elif cuda_available:
        gpu_info = {
            "device_id": 0,
            "device_name": "Test GPU",
            "total_memory": 1024**3,
            "allocated_memory": 100 * 1024**2,
            "reserved_memory": 150 * 1024**2,
            "max_memory_allocated": 100 * 1024**2,
            "memory_stats": {"num_ooms": 0},
        }
        frag_info = {
            "fragmentation_ratio": 0.1,
            "utilization_ratio": 0.2,
        }
    else:
        gpu_info = {"error": "CUDA is not available"}
        frag_info = {"error": "CUDA is not available"}

    monkeypatch.setattr(diagnose_module, "get_gpu_info", lambda device: gpu_info)
    monkeypatch.setattr(
        diagnose_module,
        "check_memory_fragmentation",
        lambda device: frag_info,
    )
    monkeypatch.setattr(
        diagnose_module,
        "suggest_memory_optimization",
        lambda frag: ["Suggestion 1"] if risk_detected else [],
    )


def _patch_timeline_capture(monkeypatch):
    """Patch run_timeline_capture to return fixed data without starting a tracker."""

    def _fake_capture(device, duration_seconds, interval):
        if duration_seconds <= 0:
            return {"timestamps": [], "allocated": [], "reserved": []}
        return {
            "timestamps": [0.0, 0.5, 1.0],
            "allocated": [0, 1000000, 2000000],
            "reserved": [0, 1000000, 2000000],
        }

    monkeypatch.setattr(diagnose_module, "run_timeline_capture", _fake_capture)


def test_diagnose_produces_artifact_bundle_with_duration_zero(monkeypatch, tmp_path):
    """Invocation with duration=0 produces directory with required files."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)

    assert exit_code in (0, 2)
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1
    artifact_dir = dirs[0]
    assert artifact_dir.is_dir()
    assert "gpumemprof-diagnose-" in artifact_dir.name

    assert (artifact_dir / "environment.json").exists()
    assert (artifact_dir / "diagnostic_summary.json").exists()
    assert (artifact_dir / "telemetry_timeline.json").exists()
    assert (artifact_dir / "manifest.json").exists()

    with open(artifact_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert "files" in manifest
    assert "exit_code" in manifest
    assert "risk_detected" in manifest
    assert "environment.json" in manifest["files"]
    assert "manifest.json" in manifest["files"]
    for name in manifest["files"]:
        assert (artifact_dir / name).exists()

    with open(artifact_dir / "diagnostic_summary.json") as f:
        summary = json.load(f)
    assert "backend" in summary
    assert "risk_flags" in summary
    assert "suggestions" in summary


def test_diagnose_artifact_completeness_with_timeline(monkeypatch, tmp_path):
    """With duration > 0, telemetry_timeline.json has data (when capture is mocked)."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=5.0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)

    dirs = list(tmp_path.iterdir())
    artifact_dir = dirs[0]
    with open(artifact_dir / "telemetry_timeline.json") as f:
        timeline = json.load(f)
    assert "timestamps" in timeline
    assert "allocated" in timeline
    assert "reserved" in timeline
    assert len(timeline["timestamps"]) == 3
    assert len(timeline["allocated"]) == 3


def test_diagnose_invalid_duration_returns_one(monkeypatch, capsys):
    """Invalid --duration < 0 returns 1 and prints error."""
    args = SimpleNamespace(
        output=None,
        device=None,
        duration=-1,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "duration" in err.lower()


def test_diagnose_invalid_interval_returns_one(monkeypatch, capsys):
    """Invalid --interval <= 0 returns 1 and prints error."""
    args = SimpleNamespace(
        output=None,
        device=None,
        duration=0,
        interval=0,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "interval" in err.lower()


def test_diagnose_exit_code_zero_when_no_risk(monkeypatch, tmp_path):
    """When risk flags are false, cmd_diagnose returns 0."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)
    assert exit_code == 0


def test_diagnose_exit_code_two_when_risk_detected(monkeypatch, tmp_path):
    """When risk is detected (e.g. high utilization, OOM), returns 2."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=True)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)
    assert exit_code == 2


def test_diagnose_stdout_contains_artifact_path_and_status(monkeypatch, tmp_path, capsys):
    """Stdout summary contains artifact path and status line."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)
    out = capsys.readouterr().out

    assert "Artifact:" in out
    assert "Status:" in out
    assert "gpumemprof-diagnose-" in out
    assert "OK" in out or "MEMORY_RISK" in out or "FAILED" in out


def test_diagnose_stdout_findings_when_risk(monkeypatch, tmp_path, capsys):
    """When risk detected, stdout includes findings line."""
    _patch_diagnose_env(monkeypatch, cuda_available=True, risk_detected=True)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(tmp_path),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)
    out = capsys.readouterr().out

    assert "Findings:" in out
    assert "MEMORY_RISK" in out


def test_diagnose_default_output_creates_timestamped_dir_in_cwd(monkeypatch, tmp_path):
    """With no --output, artifact is created in cwd with timestamped name."""
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = SimpleNamespace(
        output=None,
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)

    dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert dirs[0].name.startswith("gpumemprof-diagnose-")


def test_diagnose_output_existing_dir_creates_timestamped_subdir(monkeypatch, tmp_path):
    """When --output is an existing directory, create timestamped subdir inside."""
    out_dir = tmp_path / "myout"
    out_dir.mkdir()
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(out_dir),
        device=None,
        duration=0,
        interval=0.5,
    )
    gpumemprof_cli.cmd_diagnose(args)

    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1
    assert subdirs[0].name.startswith("gpumemprof-diagnose-")
    assert (subdirs[0] / "manifest.json").exists()


def test_diagnose_invalid_output_returns_one(monkeypatch, tmp_path):
    """When --output is an existing file (cannot create dir), returns 1."""
    existing_file = tmp_path / "existing_file"
    existing_file.write_text("x")
    _patch_diagnose_env(monkeypatch, cuda_available=False)
    _patch_timeline_capture(monkeypatch)

    args = SimpleNamespace(
        output=str(existing_file),
        device=None,
        duration=0,
        interval=0.5,
    )
    exit_code = gpumemprof_cli.cmd_diagnose(args)
    assert exit_code == 1
