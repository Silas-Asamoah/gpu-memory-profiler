import types

import pytest

from gpumemprof.tui import monitor


class DummyCPUTracker:
    """Minimal CPUMemoryTracker stand-in for TUI unit tests."""

    def __init__(self, sampling_interval=0.5, max_events=10_000, enable_alerts=True):
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts
        self.is_tracking = False
        self.events = []

    def start_tracking(self):
        self.is_tracking = True

    def stop_tracking(self):
        self.is_tracking = False

    def get_statistics(self):
        return {"mode": "cpu"}

    def get_memory_timeline(self, interval=1.0):
        return {}

    def get_events(self, since=None):
        return []

    def clear_events(self):
        self.events.clear()

    def export_events(self, *args, **kwargs):
        return None


class BrokenGPUTracker:
    """GPU tracker stub that always fails to initialize."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("No CUDA available")


def _stub_torch(cuda_available: bool):
    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return types.SimpleNamespace(cuda=cuda)


def test_tracker_session_falls_back_to_cpu(monkeypatch):
    """Ensure we gracefully fall back to CPU tracking when GPU tracker fails."""

    monkeypatch.setattr(monitor, "MemoryTracker", BrokenGPUTracker)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", _stub_torch(False))

    session = monitor.TrackerSession()
    session.start()

    assert session.backend == "cpu"
    assert isinstance(session._tracker, DummyCPUTracker)
    assert session._tracker.max_events == session.max_events
    assert session.is_active

    session.stop()
    assert not session.is_active


def test_tracker_session_works_without_gpu_dependency(monkeypatch):
    """TrackerSession should still operate when the GPU tracker cannot import."""

    monkeypatch.setattr(monitor, "MemoryTracker", None)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", None)

    session = monitor.TrackerSession()
    session.start()

    assert session.backend == "cpu"
    assert isinstance(session._tracker, DummyCPUTracker)

    session.stop()


def test_tracker_session_requires_backend(monkeypatch):
    """Validate that we surface a helpful error when no backends exist."""

    monkeypatch.setattr(monitor, "MemoryTracker", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", None)

    with pytest.raises(monitor.TrackerUnavailableError):
        monitor.TrackerSession()

