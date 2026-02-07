import contextlib
from types import SimpleNamespace

import gpumemprof.cli as gpumemprof_cli


def _patch_cpu_process(monkeypatch):
    class DummyProcess:
        def oneshot(self):
            return contextlib.nullcontext()

        def memory_info(self):
            return SimpleNamespace(rss=1024, vms=2048)

    monkeypatch.setattr(gpumemprof_cli.psutil, "Process", lambda: DummyProcess())
    monkeypatch.setattr(
        gpumemprof_cli.psutil,
        "cpu_count",
        lambda logical=None: 8 if logical else 4,
    )


def test_gpumemprof_info_reports_mps_without_cpu_only_message(monkeypatch, capsys):
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Darwin",
            "python_version": "3.10",
            "cuda_available": False,
            "mps_built": True,
            "mps_available": True,
            "detected_backend": "mps",
        },
    )
    _patch_cpu_process(monkeypatch)

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))
    output = capsys.readouterr().out

    assert "Detected Backend: mps" in output
    assert "MPS Built: True" in output
    assert "MPS Available: True" in output
    assert "MPS backend is available" in output
    assert "Falling back to CPU-only profiling." not in output


def test_gpumemprof_info_reports_cpu_fallback_when_mps_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Darwin",
            "python_version": "3.10",
            "cuda_available": False,
            "mps_built": False,
            "mps_available": False,
            "detected_backend": "cpu",
        },
    )
    _patch_cpu_process(monkeypatch)

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))
    output = capsys.readouterr().out

    assert "Detected Backend: cpu" in output
    assert "MPS Available: False" in output
    assert "CUDA is not available. Falling back to CPU-only profiling." in output


def test_gpumemprof_info_keeps_cuda_output_when_available(monkeypatch, capsys):
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "python_version": "3.10",
            "cuda_available": True,
            "cuda_version": "12.1",
            "cuda_device_count": 1,
            "current_device": 0,
            "detected_backend": "cuda",
        },
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_gpu_info",
        lambda device_id: {
            "device_name": "GPU0",
            "total_memory": 1024**3,
            "allocated_memory": 0,
            "reserved_memory": 0,
            "multiprocessor_count": 1,
        },
    )

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))
    output = capsys.readouterr().out

    assert "Detected Backend: cuda" in output
    assert "CUDA Version: 12.1" in output
    assert "GPU 0 Information:" in output
    assert "Falling back to CPU-only profiling." not in output
