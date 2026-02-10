from __future__ import annotations

from typing import Any, Callable

import pytest

pytest.importorskip("textual")

from textual.pilot import Pilot
from textual.widgets import Header as TextualHeader
from textual.widgets import TabPane, TabbedContent

from gpumemprof.tui import app as appmod
from gpumemprof.tui.app import GPUMemoryProfilerTUI

pytestmark = pytest.mark.tui_snapshot

TERMINAL_SIZE = (140, 44)


def _configure_snapshot_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appmod, "WELCOME_MESSAGES", ["GPU Memory Profiler"])
    monkeypatch.setattr(appmod, "Figlet", None)
    monkeypatch.setattr(
        appmod,
        "get_system_info",
        lambda: {
            "platform": "Darwin-23.6.0-arm64",
            "python_version": "3.10.17",
            "cuda_available": False,
            "cuda_version": "N/A",
            "cuda_device_count": 0,
        },
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_gpu_info",
        lambda: {
            "device_name": "Snapshot GPU",
            "total_memory": 16 * (1024**3),
            "allocated_memory": 2 * (1024**3),
            "reserved_memory": 3 * (1024**3),
            "max_memory_allocated": 4 * (1024**3),
        },
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_tf_system_info",
        lambda: {"tensorflow_version": "2.15.0"},
    )
    monkeypatch.setattr(
        appmod,
        "_safe_get_tf_gpu_info",
        lambda: {
            "devices": [
                {
                    "name": "TF Snapshot GPU",
                    "current_memory_mb": 512.0,
                    "peak_memory_mb": 1024.0,
                }
            ],
            "total_memory": 2048.0,
        },
    )
    monkeypatch.setattr(appmod, "fetch_pytorch_profiles", lambda limit=15: [])
    monkeypatch.setattr(appmod, "fetch_tensorflow_profiles", lambda limit=15: [])

    class SnapshotHeader(TextualHeader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["show_clock"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(appmod, "Header", SnapshotHeader)


@pytest.fixture(autouse=True)
def _deterministic_snapshot_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_snapshot_overrides(monkeypatch)


def _pane_title(pane: TabPane) -> str:
    title = getattr(pane, "title", None)
    if title is None:
        title = getattr(pane, "_title", "")
    if hasattr(title, "plain"):
        title = title.plain
    return str(title)


def _activate_tab_before(title: str) -> Callable[[Pilot], Any]:
    async def _run_before(pilot: Pilot) -> None:
        app = pilot.app
        tabbed = app.query_one(TabbedContent)
        target_id = None
        for pane in app.query(TabPane):
            if _pane_title(pane) == title:
                target_id = pane.id
                break
        assert target_id is not None
        tabbed.active = target_id
        await pilot.pause()

    return _run_before


def test_snapshot_overview_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("Overview"),
    )


def test_snapshot_pytorch_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("PyTorch"),
    )


def test_snapshot_tensorflow_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("TensorFlow"),
    )


def test_snapshot_monitoring_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("Monitoring"),
    )


def test_snapshot_visualizations_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("Visualizations"),
    )


def test_snapshot_cli_actions_tab(snap_compare) -> None:
    assert snap_compare(
        GPUMemoryProfilerTUI(),
        terminal_size=TERMINAL_SIZE,
        run_before=_activate_tab_before("CLI & Actions"),
    )
