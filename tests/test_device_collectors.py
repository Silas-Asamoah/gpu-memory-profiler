from types import SimpleNamespace

import pytest

import gpumemprof.device_collectors as collectors


def test_detect_torch_runtime_backend_reports_rocm(monkeypatch):
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(collectors.torch, "version", SimpleNamespace(hip="6.3.0"))

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "rocm"


def test_detect_torch_runtime_backend_reports_mps(monkeypatch):
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(collectors, "_is_mps_available", lambda: True)

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "mps"


def test_build_device_memory_collector_rejects_cpu_device():
    with pytest.raises(ValueError, match="Only CUDA/ROCm and MPS"):
        collectors.build_device_memory_collector("cpu")


def test_build_device_memory_collector_allows_mps_device(monkeypatch):
    monkeypatch.setattr(
        collectors,
        "_resolve_device",
        lambda _device: SimpleNamespace(type="mps"),
    )

    collector = collectors.build_device_memory_collector("mps")

    assert collector.name() == "mps"
