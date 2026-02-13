"""Interactive Textual TUI for GPU Memory Profiler."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Iterable, Sequence, List, Optional, cast

# Suppress TensorFlow oneDNN warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

logger = logging.getLogger(__name__)

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    Markdown,
    RichLog,
    Rule,
    TabPane,
    TabbedContent,
    Static,
    Label,
)

from rich.text import Text

from .monitor import TrackerEventView, TrackerSession, TrackerUnavailableError
from .commands import CLICommandRunner
from .profiles import (
    ProfileRow,
    clear_pytorch_profiles,
    clear_tensorflow_profiles,
    fetch_pytorch_profiles,
    fetch_tensorflow_profiles,
)
from gpumemprof.utils import get_system_info, get_gpu_info, format_bytes
from tfmemprof.utils import get_system_info as get_tf_system_info
from tfmemprof.utils import get_gpu_info as get_tf_gpu_info

try:
    import torch as _torch
    torch: Any = _torch
except ImportError as e:
    raise ImportError(
        "torch is required for the TUI application. Install it with: pip install torch"
    ) from e

try:
    import tensorflow as _tf
    # Suppress TensorFlow INFO and WARNING messages
    _tf.get_logger().setLevel("ERROR")
    # Also suppress oneDNN warnings via environment
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf: Optional[Any] = _tf
except ImportError:
    tf = None

try:
    from pyfiglet import Figlet as _Figlet
    Figlet: Optional[Any] = _Figlet
except ImportError:
    Figlet = None

try:
    from gpumemprof import GPUMemoryProfiler as _GPUMemoryProfiler
    GPUMemoryProfiler: Optional[Any] = _GPUMemoryProfiler
except ImportError as e:
    raise ImportError(
        "GPUMemoryProfiler is required for the TUI application. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    from gpumemprof.cpu_profiler import CPUMemoryProfiler as _CPUMemoryProfiler
    CPUMemoryProfiler: Optional[Any] = _CPUMemoryProfiler
except ImportError as e:
    raise ImportError(
        "CPUMemoryProfiler is required for the TUI application. "
        "Ensure gpumemprof is properly installed."
    ) from e

try:
    from tfmemprof.profiler import TFMemoryProfiler as _TFMemoryProfiler
    TFMemoryProfiler: Optional[Any] = _TFMemoryProfiler
except ImportError:
    TFMemoryProfiler = None


WELCOME_MESSAGES = [
    "GPU Memory Profiler",
    "Live Monitoring & Watchdogs",
    "CLI · Docs · Examples",
]


def _safe_get_gpu_info() -> dict[str, Any]:
    try:
        return get_gpu_info()
    except Exception as exc:
        logger.debug("_safe_get_gpu_info failed: %s", exc)
        return {}


def _safe_get_tf_system_info() -> dict[str, Any]:
    try:
        return get_tf_system_info()
    except Exception as exc:
        logger.debug("_safe_get_tf_system_info failed: %s", exc)
        return {}


def _safe_get_tf_gpu_info() -> dict[str, Any]:
    try:
        return get_tf_gpu_info()
    except Exception as exc:
        logger.debug("_safe_get_tf_gpu_info failed: %s", exc)
        return {}


def _build_welcome_info() -> str:
    """Build welcome navigation guide text."""
    return dedent(
        """
        # Quick Start Guide

        ## Navigate the TUI

        Click on any tab above to explore different features:

        - **[bold cyan]PyTorch[/]** → View PyTorch GPU stats, run profiling samples, and see profile results
        - **[bold cyan]TensorFlow[/]** → View TensorFlow GPU stats, run profiling samples, and see profile results  
        - **[bold cyan]Monitoring[/]** → Start live memory tracking, set alert thresholds, export CSV/JSON data
        - **[bold cyan]Visualizations[/]** → Generate timeline plots (PNG/HTML) from tracking sessions
        - **[bold cyan]CLI & Actions[/]** → Run CLI commands interactively and execute sample workloads

        ## Keyboard Shortcuts

        - **[bold white]r[/bold white]** - Refresh overview tab
        - **[bold white]g[/bold white]** - Log gpumemprof command examples
        - **[bold white]t[/bold white]** - Log tfmemprof command examples
        - **[bold white]f[/bold white]** - Focus log area in CLI tab
        - **[bold white]q[/bold white]** - Quit application

        ## Getting Started

        1. **Check System Info** - Scroll down to see your platform, Python version, and GPU details
        2. **View GPU Stats** - Visit **PyTorch** or **TensorFlow** tabs to see real-time GPU memory statistics
        3. **Start Tracking** - Go to **Monitoring** tab and click "Start Live Tracking" to begin monitoring
        4. **Run Samples** - Use **CLI & Actions** tab to run sample workloads and see profiling results
        5. **Export Data** - After tracking, use "Export CSV" or "Export JSON" buttons in Monitoring tab

        ---
        """
    ).strip()


def _build_system_markdown() -> str:
    info = get_system_info()
    gpu = _safe_get_gpu_info()
    tf_info = _safe_get_tf_system_info()
    tf_gpu = _safe_get_tf_gpu_info()

    lines = [
        "# System Overview",
        "",
        f"- **Platform**: {info.get('platform', 'Unknown')}",
        f"- **Python**: {info.get('python_version', 'Unknown')}",
        f"- **TensorFlow (Python)**: {tf_info.get('tensorflow_version', 'N/A')}",
        f"- **CUDA Available**: {info.get('cuda_available', False)}",
    ]

    if info.get("cuda_available"):
        lines.extend(
            [
                f"- **CUDA Version**: {info.get('cuda_version', 'Unknown')}",
                f"- **GPU Count**: {info.get('cuda_device_count', 0)}",
            ]
        )

    if gpu:
        lines.append("")
        lines.append("## GPU Snapshot")
        lines.extend(
            [
                f"- **Device Name**: {gpu.get('device_name', 'Unknown')}",
                f"- **Total Memory**: {gpu.get('total_memory', 0) / (1024**3):.2f} GB",
                f"- **Allocated**: {gpu.get('allocated_memory', 0) / (1024**3):.2f} GB",
                f"- **Reserved**: {gpu.get('reserved_memory', 0) / (1024**3):.2f} GB",
            ]
        )
    else:
        lines.append("")
        lines.append(
            "> GPU metrics are unavailable on this system. You can still run the CLI "
            "and CPU guides."
        )

    lines.append("")
    if tf_gpu and tf_gpu.get("devices"):
        lines.append("")
        lines.append("## TensorFlow GPU Snapshot")
        device = tf_gpu["devices"][0]
        lines.extend(
            [
                f"- **TF Device Name**: {device.get('name', 'Unknown')}",
                f"- **Current Memory**: {device.get('current_memory_mb', 0):.2f} MB",
                f"- **Peak Memory**: {device.get('peak_memory_mb', 0):.2f} MB",
            ]
        )

    lines.append("")
    lines.append("## Getting Started")
    lines.append("")
    lines.append("- `python -m examples.basic.pytorch_demo`")
    lines.append("- `python -m examples.basic.tensorflow_demo`")
    lines.append("- `python -m examples.cli.quickstart`")
    lines.append("")
    lines.append(
        "Need more? Visit the [Example Test Guides](docs/examples/test_guides/README.md)."
    )
    return "\n".join(lines)


def _pytorch_stats_provider() -> list[dict]:
    info = _safe_get_gpu_info()
    if not info:
        return []
    return [
        {
            "device": info.get("device_name", "gpu0"),
            "current": info.get("allocated_memory", 0) / (1024**2),
            "peak": info.get("max_memory_allocated", info.get("allocated_memory", 0))
            / (1024**2),
            "reserved": info.get("reserved_memory", 0) / (1024**2),
        }
    ]


def _tensorflow_stats_provider() -> list[dict]:
    gpu_info = _safe_get_tf_gpu_info()
    devices = gpu_info.get("devices", []) if gpu_info else []
    rows = []
    for device in devices:
        rows.append(
            {
                "device": device.get("name", "tf-gpu"),
                "current": device.get("current_memory_mb", 0),
                "peak": device.get("peak_memory_mb", 0),
                "reserved": gpu_info.get("total_memory", 0),
            }
        )
    return rows


def _build_framework_markdown(framework: str) -> str:
    if framework == "pytorch":
        return dedent(
            """
            # PyTorch Playbook

            1. **Basic profiling**
               ```bash
               python -m examples.basic.pytorch_demo
               ```
            2. **Advanced tracking (alerts, watchdog)**
               ```bash
               python -m examples.advanced.tracking_demo
               ```
            3. **CLI helpers**
               ```bash
               gpumemprof info
               gpumemprof track --duration 60 --output tracking.json
               ```

            Check the [PyTorch Testing Guide](docs/pytorch_testing_guide.md) for
            full workflows and troubleshooting steps.
            """
        ).strip()

    return dedent(
        """
        # TensorFlow Playbook

        1. **Basic profiling**
           ```bash
           python -m examples.basic.tensorflow_demo
           ```
        2. **CLI helpers**
           ```bash
           tfmemprof info
           tfmemprof monitor --duration 30 --interval 0.5
           tfmemprof track --output tf_results.json
           ```

        The [TensorFlow Testing Guide](docs/tensorflow_testing_guide.md) includes
        deeper recipes, including mixed precision and multi-GPU notes.
        """
    ).strip()


def _build_cli_markdown() -> str:
    return dedent(
        """
        # CLI Quick Samples

        ```bash
        gpumemprof info
        gpumemprof monitor --duration 30 --interval 0.5
        gpumemprof track --duration 60 --output tracking.json

        tfmemprof info
        tfmemprof monitor --duration 30 --interval 0.5
        tfmemprof track --duration 60 --output tf_tracking.json
        
        # Optional: fuller dashboard
        gpu-profiler

        # Ensure pip shows progress
        pip install --progress-bar on "gpu-memory-profiler[tui]"
        ```

        Use the buttons below to log summaries or copy commands.
        """
    ).strip()


def _build_visual_markdown() -> str:
    return dedent(
        """
        # Visualization Tips

        - Start live tracking to collect timeline samples, then refresh the view.
        - Use `Generate PNG Plot` to save a Matplotlib graph (writes to ./visualizations).
        - Prefer `Generate HTML Plot` for an interactive Plotly view you can open in a browser.
        - A lightweight ASCII chart appears below so you can inspect trends without leaving the terminal.
        """
    ).strip()


class AsciiWelcome(Static):
    """Animated ASCII welcome banner, uses pyfiglet when available."""

    def __init__(
        self,
        messages: list[str],
        font: str = "Standard",
        interval: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__("", **kwargs)
        self.messages = messages or ["GPU Memory Profiler"]
        self.font_name = font
        self.interval = interval
        self._frame_index = 0
        self._figlet = None

        if Figlet:
            try:
                self._figlet = Figlet(font=self.font_name)
            except Exception as exc:
                logger.debug("Figlet initialization failed: %s", exc)
                self._figlet = None

    def on_mount(self) -> None:
        self._render_frame()
        if len(self.messages) > 1:
            self.set_interval(self.interval, self._advance_frame)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self.messages)
        self._render_frame()

    def _render_frame(self) -> None:
        message = self.messages[self._frame_index]
        ascii_text = self._render_ascii(message)
        self.update(ascii_text)

    def _render_ascii(self, message: str) -> Text:
        if self._figlet:
            try:
                rendered = self._figlet.renderText(message)
                return Text(rendered.rstrip(), style="bold cyan")
            except Exception as exc:
                logger.debug("Figlet render failed, using fallback: %s", exc)

        fallback = dedent(
            f"""
            ██████╗  ██████╗ ██╗   ██╗
            ██╔══██╗██╔═══██╗██║   ██║
            ██████╔╝██║   ██║██║   ██║
            ██╔═══╝ ██║   ██║╚██╗ ██╔╝
            ██║     ╚██████╔╝ ╚████╔╝ 
            ╚═╝      ╚═════╝   ╚═══╝  
            {message.center(30)}
            """
        ).strip("\n")
        return Text(fallback, style="bold cyan")


class GPUStatsTable(DataTable):
    """Live-updating table of GPU stats."""

    def __init__(
        self,
        title: str,
        provider: Callable[[], list[dict[str, Any]]],
        refresh_interval: float = 2.0,
    ) -> None:
        super().__init__(show_header=True, zebra_stripes=True, id=f"table-{title}")
        self.title_text = title
        self.provider = provider
        self.refresh_interval = refresh_interval

    def on_mount(self) -> None:
        self.add_columns("Device", "Current (MB)", "Peak (MB)", "Reserved (MB)")
        self.refresh_rows()
        self.set_interval(self.refresh_interval, self.refresh_rows)

    def refresh_rows(self) -> None:
        stats = self.provider() or []
        self.clear()
        if not stats:
            self.add_row("N/A", "-", "-", "-")
            return

        for row in stats:
            self.add_row(
                row.get("device", "N/A"),
                f"{row.get('current', 0):.2f}",
                f"{row.get('peak', 0):.2f}",
                f"{row.get('reserved', 0):.2f}",
            )


class MarkdownPanel(Markdown):
    """Reusable Markdown panel with refresh support."""

    def __init__(self, builder: Callable[[], str], **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.builder = builder

    def refresh_content(self) -> None:
        self.update(self.builder())

    def on_mount(self) -> None:
        self.refresh_content()


class KeyValueTable(DataTable):
    """Simple key/value table for monitoring stats."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns("Metric", "Value")


class TimelineCanvas(Static):
    """ASCII timeline renderer for quick visual feedback."""

    def __init__(self, width: int = 72, height: int = 10, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.canvas_width = width
        self.canvas_height = height

    def render_timeline(self, timeline: dict[str, Any]) -> None:
        allocated = timeline.get("allocated") if timeline else None
        reserved = timeline.get("reserved") if timeline else None
        if not allocated:
            self.render_placeholder(
                "No timeline data yet. Start live tracking and press Refresh."
            )
            return

        allocated_lines = self._build_chart_lines("Allocated", allocated)
        reserved_lines = (
            self._build_chart_lines("Reserved", reserved) if reserved else []
        )
        text = "\n".join(allocated_lines + [""] + reserved_lines) if reserved_lines else "\n".join(allocated_lines)
        self.update(text)

    def render_placeholder(self, message: str) -> None:
        self.update(message)

    def _build_chart_lines(self, label: str, values: Sequence[float]) -> list[str]:
        samples = self._resample(values)
        samples_mb = [v / (1024**2) for v in samples]
        if not samples_mb:
            return [f"{label}: no samples"]

        sparkline = self._generate_sparkline(samples_mb)
        max_val = max(samples_mb) if samples_mb else 0.0
        latest = samples_mb[-1] if samples_mb else 0.0

        return [
            f"{label} (max {max_val:.2f} MB, latest {latest:.2f} MB)",
            f"[{sparkline}]",
        ]

    def _resample(self, values: Sequence[float]) -> list[float]:
        if not values:
            return []
        if len(values) <= self.canvas_width:
            return list(values)

        step = len(values) / self.canvas_width
        sampled = []
        for i in range(self.canvas_width):
            idx = min(int(round(i * step)), len(values) - 1)
            sampled.append(values[idx])
        return sampled

    def _generate_sparkline(self, values: Sequence[float]) -> str:
        if not values:
            return ""
        max_val = max(values) or 1.0
        palette = " .:-=+*#%@"
        last_index = len(palette) - 1
        chars = []
        for value in values:
            ratio = min(value / max_val, 1.0)
            idx = int(ratio * last_index)
            chars.append(palette[idx])
        return "".join(chars)


class AlertHistoryTable(DataTable):
    """Table displaying recent alerts."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns("Time", "Type", "Message")

    def update_rows(self, events: List[dict]) -> None:
        self.clear()
        if not events:
            self.add_row("-", "-", "No alerts yet.")
            return
        for event in events:
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            else:
                timestamp_str = str(timestamp or "-")
            event_type = str(event.get("type", "-")).upper()
            message = event.get("message", "")
            self.add_row(timestamp_str, event_type, message)


class ProfileResultsTable(DataTable):
    """Reusable table for displaying profile summaries."""

    def on_mount(self) -> None:
        if not self.columns:
            self.add_columns(
                "Name",
                "Peak (MB)",
                "Δ Avg (MB)",
                "Duration (ms)",
                "Calls",
                "Recorded",
            )

    def update_rows(self, rows: List[ProfileRow]) -> None:
        self.clear()
        if not rows:
            self.add_row("No profiles", "-", "-", "-", "-", "-")
            return

        for row in rows:
            timestamp = (
                datetime.fromtimestamp(row.recorded_at).strftime("%H:%M:%S")
                if row.recorded_at
                else "-"
            )
            self.add_row(
                row.name,
                f"{row.peak_mb:.2f}",
                f"{row.delta_mb:.2f}",
                f"{row.duration_ms:.2f}",
                str(row.call_count),
                timestamp,
            )


class GPUMemoryProfilerTUI(App):
    """Main Textual application."""

    tracker_session: TrackerSession | None
    cli_runner: CLICommandRunner
    monitor_auto_cleanup: bool
    _last_monitor_stats: dict[str, Any]
    _last_timeline: dict[str, list[Any]]
    recent_alerts: List[dict[str, Any]]

    CSS = """
    TabbedContent {
        padding: 1;
    }

    RichLog {
        height: 1fr;
        border: solid gray;
    }

    Button {
        margin: 0 1 1 0;
        height: 5;
        width: auto;
        min-width: 16;
        max-width: 30;
        padding: 1 3;
        content-align: center middle;
        text-style: bold;
        color: #ffffff;
        background: $panel;
    }

    Button.-primary {
        color: #ffffff;
        background: $primary;
        border: solid $primary-lighten-1;
    }

    Button.-success {
        color: #ffffff;
        background: $success;
        border: solid $success-lighten-1;
    }

    Button.-warning {
        color: #000000;
        background: $warning;
        border: solid $warning-lighten-1;
    }

    Button.-error {
        color: #ffffff;
        background: $error;
        border: solid $error-lighten-1;
    }

    Button:hover {
        opacity: 0.9;
    }

    #table-pytorch,
    #table-tensorflow {
        height: 12;
        border: solid gray;
    }

    #pytorch-tab,
    #tensorflow-tab {
        layout: vertical;
    }

    #cli-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #cli-buttons {
        layout: horizontal;
        content-align: left middle;
    }

    #cli-runner {
        layout: horizontal;
        content-align: left middle;
        margin: 1 0;
    }

    #cli-command-input {
        width: 1fr;
        padding: 0 1;
        height: 5;
    }

    #cli-loader {
        height: 3;
    }

    #monitoring-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #monitor-status {
        margin-bottom: 1;
    }

    #monitor-controls-row1,
    #monitor-controls-row2,
    #monitor-controls-row3 {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
        height: auto;
        min-height: 6;
    }

    #monitor-thresholds {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
    }

    #monitor-thresholds Label {
        width: 12;
    }

    #monitor-thresholds Input {
        width: 12;
        margin-right: 1;
    }

    #monitor-alerts-table {
        height: 8;
        border: solid gray;
        margin-top: 1;
    }

    #monitor-stats {
        height: 10;
        border: solid gray;
    }

    #monitor-log {
        height: 1fr;
        border: solid gray;
        margin-top: 1;
    }

    #visualizations-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #visual-buttons {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
        height: auto;
        min-height: 6;
        overflow: hidden;
    }

    #timeline-stats {
        height: 8;
        border: solid gray;
        margin-bottom: 1;
    }

    #timeline-canvas {
        border: solid gray;
        padding: 1;
    }

    #visual-log {
        height: 8;
        border: solid gray;
        margin-top: 1;
    }

    #overview-welcome {
        border: round $primary;
        padding: 1;
        margin: 0 0 1 0;
        background: $panel;
        text-align: center;
        text-style: bold;
        color: $accent;
        min-height: 10;
        content-align: center middle;
    }

    #welcome-info {
        border: solid $primary;
        padding: 2;
        margin: 0 0 1 0;
        background: $surface;
        height: auto;
        min-height: 15;
    }

    #welcome-info Markdown {
        color: $text;
    }

    #pytorch-profile-controls,
    #tensorflow-profile-controls {
        layout: horizontal;
        content-align: left middle;
        margin-top: 1;
        height: auto;
        min-height: 6;
    }

    #pytorch-profile-table,
    #tensorflow-profile-table {
        height: 12;
        border: solid gray;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_overview", "Refresh Overview"),
        ("f", "focus_log", "Focus Log"),
        ("g", "log_gpumemprof_help", "gpumemprof info"),
        ("t", "log_tfmemprof_help", "tfmemprof info"),
    ]

    def compose(self) -> ComposeResult:
        self.overview_panel = MarkdownPanel(_build_system_markdown, id="overview")
        self.welcome_panel = AsciiWelcome(WELCOME_MESSAGES, id="overview-welcome")
        self.welcome_info = Markdown(_build_welcome_info(), id="welcome-info")
        self.pytorch_panel = MarkdownPanel(
            lambda: _build_framework_markdown("pytorch"), id="pytorch"
        )
        self.tensorflow_panel = MarkdownPanel(
            lambda: _build_framework_markdown("tensorflow"), id="tensorflow"
        )
        self.cli_panel = MarkdownPanel(_build_cli_markdown, id="cli-docs")
        self.visual_panel = MarkdownPanel(_build_visual_markdown, id="visual-docs")
        self.command_log = RichLog(highlight=True, markup=True, id="command-log")
        self.loader = LoadingIndicator(id="cli-loader")
        self.loader.display = False
        self.cli_command_input = Input(
            placeholder="gpumemprof info", id="cli-command-input"
        )
        self.monitor_status = Markdown("", id="monitor-status")
        self.monitor_stats_table = KeyValueTable(zebra_stripes=True, id="monitor-stats")
        self.monitor_log = RichLog(highlight=True, markup=True, id="monitor-log")
        self.watchdog_button = Button(
            "Auto Cleanup: OFF", id="btn-toggle-watchdog", variant="warning"
        )
        self.timeline_stats_table = KeyValueTable(zebra_stripes=True, id="timeline-stats")
        self.timeline_canvas = TimelineCanvas(id="timeline-canvas")
        self.visual_log = RichLog(highlight=True, markup=True, id="visual-log")
        self.pytorch_profile_table = ProfileResultsTable(id="pytorch-profile-table")
        self.tensorflow_profile_table = ProfileResultsTable(id="tensorflow-profile-table")
        self.alert_history_table = AlertHistoryTable(id="monitor-alerts-table")
        self.warning_input = Input(value="80", placeholder="80", id="input-warning")
        self.critical_input = Input(value="95", placeholder="95", id="input-critical")

        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Overview"):
                yield VerticalScroll(
                    self.welcome_panel,
                    self.welcome_info,
                    self.overview_panel,
                )

            with TabPane("PyTorch"):
                yield VerticalScroll(
                    self.pytorch_panel,
                    Horizontal(
                        Button(
                            "Refresh Profiles",
                            id="btn-refresh-pt-profiles",
                            variant="primary",
                        ),
                        Button(
                            "Clear Profiles",
                            id="btn-clear-pt-profiles",
                            variant="warning",
                        ),
                        id="pytorch-profile-controls",
                    ),
                    GPUStatsTable("pytorch", _pytorch_stats_provider),
                    self.pytorch_profile_table,
                )

            with TabPane("TensorFlow"):
                yield VerticalScroll(
                    self.tensorflow_panel,
                    Horizontal(
                        Button(
                            "Refresh Profiles",
                            id="btn-refresh-tf-profiles",
                            variant="primary",
                        ),
                        Button(
                            "Clear Profiles",
                            id="btn-clear-tf-profiles",
                            variant="warning",
                        ),
                        id="tensorflow-profile-controls",
                    ),
                    GPUStatsTable("tensorflow", _tensorflow_stats_provider),
                    self.tensorflow_profile_table,
                )

            with TabPane("Monitoring"):
                yield VerticalScroll(
                    self.monitor_status,
                    Horizontal(
                        Button(
                            "Start Live Tracking",
                            id="btn-start-tracking",
                            variant="primary",
                        ),
                        Button(
                            "Stop Tracking",
                            id="btn-stop-tracking",
                            variant="warning",
                        ),
                        self.watchdog_button,
                        Button("Apply Thresholds", id="btn-apply-thresholds", variant="primary"),
                        id="monitor-controls-row1",
                    ),
                    Horizontal(
                        Button(
                            "Force Cleanup",
                            id="btn-force-cleanup",
                            variant="success",
                        ),
                        Button(
                            "Aggressive Cleanup",
                            id="btn-force-cleanup-aggressive",
                            variant="error",
                        ),
                        Button(
                            "Export CSV",
                            id="btn-export-csv",
                            variant="success",
                        ),
                        Button(
                            "Export JSON",
                            id="btn-export-json",
                            variant="success",
                        ),
                        id="monitor-controls-row2",
                    ),
                    Horizontal(
                        Button(
                            "Clear Monitor Log",
                            id="btn-clear-monitor-log",
                        ),
                        id="monitor-controls-row3",
                    ),
                    Horizontal(
                        Label("Warning %"),
                        self.warning_input,
                        Label("Critical %"),
                        self.critical_input,
                        id="monitor-thresholds",
                    ),
                    self.monitor_stats_table,
                    self.alert_history_table,
                    self.monitor_log,
                )

            with TabPane("Visualizations"):
                yield VerticalScroll(
                    self.visual_panel,
                    Horizontal(
                        Button(
                            "Refresh Timeline",
                            id="btn-refresh-visual",
                            variant="primary",
                        ),
                        Button(
                            "Generate PNG Plot",
                            id="btn-visual-png",
                            variant="success",
                        ),
                        Button(
                            "Generate HTML Plot",
                            id="btn-visual-html",
                            variant="success",
                        ),
                        id="visual-buttons",
                    ),
                    self.timeline_stats_table,
                    self.timeline_canvas,
                    self.visual_log,
                )

            with TabPane("CLI & Actions"):
                yield VerticalScroll(
                    self.cli_panel,
                    Rule(),
                    Horizontal(
                        Button("gpumemprof info", id="btn-log-system", variant="primary"),
                        Button(
                            "gpumemprof monitor",
                            id="btn-log-pytorch",
                            variant="success",
                        ),
                        Button(
                            "tfmemprof monitor",
                            id="btn-log-tensorflow",
                            variant="success",
                        ),
                        Button("PyTorch Sample", id="btn-run-pytorch", variant="primary"),
                        Button("TensorFlow Sample", id="btn-run-tf", variant="primary"),
                        id="cli-buttons",
                    ),
                    Horizontal(
                        self.cli_command_input,
                        Button("Run Command", id="btn-cli-run", variant="primary"),
                        Button("Cancel Command", id="btn-cli-cancel", variant="warning"),
                        id="cli-runner",
                    ),
                    self.loader,
                    self.command_log,
                )
        yield Footer()

    async def action_quit(self) -> None:
        self.exit()

    def action_refresh_overview(self) -> None:
        self.overview_panel.refresh_content()
        self.log_message("Overview", "System overview refreshed.")

    def action_focus_log(self) -> None:
        self.set_focus(self.command_log)

    def action_log_gpumemprof_help(self) -> None:
        self.log_message("gpumemprof info", "Run: gpumemprof info\nRun: gpumemprof monitor --duration 30")

    def action_log_tfmemprof_help(self) -> None:
        self.log_message("tfmemprof info", "Run: tfmemprof info\nRun: tfmemprof monitor --duration 30")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "btn-refresh-overview":
            self.action_refresh_overview()
        elif button_id == "btn-log-system":
            await self.run_cli_command("gpumemprof info")
        elif button_id == "btn-log-pytorch":
            await self.run_cli_command("gpumemprof monitor --duration 30 --interval 0.5")
        elif button_id == "btn-log-tensorflow":
            await self.run_cli_command(
                "tfmemprof monitor --duration 30 --interval 0.5"
            )
        elif button_id == "btn-run-pytorch":
            await self.run_pytorch_sample()
        elif button_id == "btn-run-tf":
            await self.run_tensorflow_sample()
        elif button_id == "btn-cli-run":
            await self.run_cli_command(self.cli_command_input.value)
        elif button_id == "btn-cli-cancel":
            await self.cancel_cli_command()
        elif button_id == "btn-start-tracking":
            await self.start_live_tracking()
        elif button_id == "btn-stop-tracking":
            self.stop_live_tracking()
        elif button_id == "btn-toggle-watchdog":
            self.toggle_auto_cleanup()
        elif button_id == "btn-force-cleanup":
            self.force_cleanup()
        elif button_id == "btn-force-cleanup-aggressive":
            self.force_cleanup(aggressive=True)
        elif button_id == "btn-export-csv":
            await self.export_tracker_events("csv")
        elif button_id == "btn-export-json":
            await self.export_tracker_events("json")
        elif button_id == "btn-apply-thresholds":
            self.apply_thresholds()
        elif button_id == "btn-clear-monitor-log":
            self.clear_monitor_log()
        elif button_id == "btn-refresh-visual":
            await self.refresh_visualizations()
        elif button_id == "btn-visual-png":
            await self.generate_visual_plot("png")
        elif button_id == "btn-visual-html":
            await self.generate_visual_plot("html")
        elif button_id == "btn-refresh-pt-profiles":
            await self.refresh_pytorch_profiles()
        elif button_id == "btn-clear-pt-profiles":
            await self.clear_pytorch_profiles()
        elif button_id == "btn-refresh-tf-profiles":
            await self.refresh_tensorflow_profiles()
        elif button_id == "btn-clear-tf-profiles":
            await self.clear_tensorflow_profiles()

    async def run_pytorch_sample(self) -> None:
        if GPUMemoryProfiler is None or torch is None:
            self.log_message("PyTorch Sample", "PyTorch profiler is unavailable in this environment.")
            return
        if not torch.cuda.is_available():
            if CPUMemoryProfiler is None:
                self.log_message("PyTorch Sample", "CPU profiler is unavailable; install psutil.")
                return
            await self._execute_task(
                "PyTorch Sample (CPU)",
                self._cpu_sample_workload,
                self._format_cpu_summary,
            )
            return
        await self._execute_task(
            "PyTorch Sample",
            self._pytorch_sample_workload,
            self._format_pytorch_summary,
        )
        await self.refresh_pytorch_profiles()

    async def run_tensorflow_sample(self) -> None:
        if TFMemoryProfiler is None or tf is None:
            self.log_message(
                "TensorFlow Sample",
                "TensorFlow profiler is unavailable. Install tensorflow and tfmemprof: "
                "pip install tensorflow tfmemprof"
            )
            return
        await self._execute_task(
            "TensorFlow Sample",
            self._tensorflow_sample_workload,
            self._format_tensorflow_results,
        )
        await self.refresh_tensorflow_profiles()

    async def start_live_tracking(self) -> None:
        session = self._get_or_create_tracker_session()
        if not session:
            return
        if session.is_active:
            self.log_monitor_message("Tracker", "Live tracking already running.")
            return
        try:
            session.start()
        except TrackerUnavailableError as exc:
            self.log_monitor_message("Tracker", str(exc))
            return
        self.log_monitor_message("Tracker", "Live tracking started.")
        self._sync_threshold_inputs()
        self._update_monitor_status()

    def stop_live_tracking(self) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message("Tracker", "Tracker is not running.")
            return
        session.stop()
        self.log_monitor_message("Tracker", "Live tracking stopped.")
        self._update_monitor_status()

    def toggle_auto_cleanup(self) -> None:
        self.monitor_auto_cleanup = not getattr(self, "monitor_auto_cleanup", False)
        session = self.tracker_session
        if session:
            session.set_auto_cleanup(self.monitor_auto_cleanup)
        state = "enabled" if self.monitor_auto_cleanup else "disabled"
        self.log_monitor_message("Watchdog", f"Auto cleanup {state}.")
        self._update_watchdog_button_label()
        self._update_monitor_status()

    def force_cleanup(self, aggressive: bool = False) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message("Watchdog", "Start tracking before requesting cleanup.")
            return
        if not session.force_cleanup(aggressive=aggressive):
            self.log_monitor_message(
                "Watchdog",
                "Watchdog controls are unavailable in this environment.",
            )
            return
        label = "aggressive" if aggressive else "standard"
        self.log_monitor_message("Watchdog", f"Requested {label} cleanup.")

    def clear_monitor_log(self) -> None:
        self.monitor_log.clear()
        self.log_monitor_message("Monitor", "Cleared monitoring log.")

    async def run_cli_command(self, command: str) -> None:
        command = (command or "").strip()
        if not command:
            self.log_message("CLI Runner", "Enter a command to run.")
            return
        if self.cli_runner.is_running:
            self.log_message("CLI Runner", "A command is already running.")
            return

        self.cli_command_input.value = command
        self.command_log.write(f"[bold green]$ {command}[/bold green]\n")
        self._set_loader(True)
        try:
            exit_code = await self.cli_runner.run(command, self._handle_cli_output)
            self.log_message("CLI Runner", f"Command finished with exit code {exit_code}.")
        except Exception as exc:
            self.log_message("CLI Runner", f"Error running command: {exc}")
        finally:
            self._set_loader(False)

    async def cancel_cli_command(self) -> None:
        if not self.cli_runner.is_running:
            self.log_message("CLI Runner", "No running command to cancel.")
            return
        await self.cli_runner.cancel()
        self._set_loader(False)
        self.log_message("CLI Runner", "Command was cancelled.")

    async def _handle_cli_output(self, stream: str, line: str) -> None:
        color = "cyan" if stream == "stdout" else "yellow"
        self.command_log.write(f"[{color}]{stream}[/] {line}\n")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self.cli_command_input:
            await self.run_cli_command(event.value)

    async def export_tracker_events(self, format: str) -> None:
        session = self.tracker_session
        if not session or not session.is_active:
            self.log_monitor_message(
                "Export", "Start tracking before exporting events."
            )
            return

        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = exports_dir / f"tracker_events_{timestamp}.{format}"
        active_session = session

        def _export() -> bool:
            return bool(active_session.export_events(str(file_path), format=format))

        success = await asyncio.to_thread(_export)
        if not success:
            self.log_monitor_message(
                "Export", "No tracker events available to export yet."
            )
            return

        self.log_monitor_message(
            "Export",
            f"Saved tracker events to {file_path}",
        )

    def apply_thresholds(self) -> None:
        session = self.tracker_session
        if not session or session.backend != "gpu":
            self.log_monitor_message(
                "Thresholds", "Thresholds are only available when using a GPU tracker."
            )
            return

        warning_text = (self.warning_input.value or self.warning_input.placeholder or "").strip()
        critical_text = (self.critical_input.value or self.critical_input.placeholder or "").strip()

        try:
            warning = float(warning_text)
            critical = float(critical_text)
        except ValueError:
            self.log_monitor_message(
                "Thresholds", "Enter numeric warning and critical percentages."
            )
            return
        if warning >= critical:
            self.log_monitor_message(
                "Thresholds", "Warning threshold must be less than critical threshold."
            )
            return

        session.set_thresholds(warning, critical)
        self.log_monitor_message(
            "Thresholds",
            f"Updated warning={warning:.0f}% critical={critical:.0f}%.",
        )

    async def refresh_pytorch_profiles(self) -> None:
        rows = await asyncio.to_thread(fetch_pytorch_profiles)
        self.pytorch_profile_table.update_rows(rows)
        msg = "Loaded PyTorch profile results." if rows else "No PyTorch profiles captured yet."
        self.log_message("PyTorch Profiles", msg)

    async def clear_pytorch_profiles(self) -> None:
        success = await asyncio.to_thread(clear_pytorch_profiles)
        message = "Cleared PyTorch profile results." if success else "No PyTorch profiles to clear."
        self.log_message("PyTorch Profiles", message)
        await self.refresh_pytorch_profiles()

    async def refresh_tensorflow_profiles(self) -> None:
        rows = await asyncio.to_thread(fetch_tensorflow_profiles)
        self.tensorflow_profile_table.update_rows(rows)
        msg = "Loaded TensorFlow profile summaries." if rows else "No TensorFlow profiles captured yet."
        self.log_message("TensorFlow Profiles", msg)

    async def clear_tensorflow_profiles(self) -> None:
        success = await asyncio.to_thread(clear_tensorflow_profiles)
        message = "Cleared TensorFlow profiles." if success else "No TensorFlow profiles to clear."
        self.log_message("TensorFlow Profiles", message)
        await self.refresh_tensorflow_profiles()

    def refresh_monitoring_panel(self) -> None:
        session = self.tracker_session
        stats: dict[str, Any] = {}
        cleanup_stats: dict[str, Any] = {}

        if session:
            stats = session.get_statistics() or {}
            cleanup_stats = session.get_cleanup_stats() or {}
            if session.is_active:
                events = session.pull_events()
                if events:
                    self._append_monitor_events(events)

        if stats:
            self._last_monitor_stats = stats
        elif self._last_monitor_stats:
            stats = self._last_monitor_stats

        self._update_monitor_stats(stats, cleanup_stats)
        self._update_monitor_status()

    def _update_monitor_stats(
        self,
        stats: dict[str, Any],
        cleanup_stats: dict[str, Any],
    ) -> None:
        table = self.monitor_stats_table
        table.clear()
        session = self.tracker_session
        status_label = "Active" if session and session.is_active else "Idle"
        device_label = session.get_device_label() if session else "-"

        if not stats:
            table.add_row("Status", status_label)
            table.add_row("Device", device_label or "-")
            table.add_row("Current Allocated", "-")
            table.add_row("Peak Memory", "-")
            table.add_row("Alerts", "-")
            cleanup_count = cleanup_stats.get("cleanup_count", 0)
            table.add_row("Cleanups", str(cleanup_count))
            return

        cleanup_count = cleanup_stats.get("cleanup_count", 0)
        utilization = stats.get("memory_utilization_percent", 0.0)
        duration = stats.get("tracking_duration_seconds", 0.0)

        table.add_row("Status", status_label)
        table.add_row("Device", device_label or "-")
        table.add_row(
            "Current Allocated",
            self._format_bytes_metric(stats.get("current_memory_allocated")),
        )
        table.add_row(
            "Current Reserved",
            self._format_bytes_metric(stats.get("current_memory_reserved")),
        )
        table.add_row(
            "Peak Memory",
            self._format_bytes_metric(stats.get("peak_memory")),
        )
        table.add_row("Utilization", f"{utilization:.1f}%")
        table.add_row(
            "Alloc/sec",
            f"{stats.get('allocations_per_second', 0.0):.2f}",
        )
        table.add_row("Alert Count", str(stats.get("alert_count", 0)))
        table.add_row("Total Events", str(stats.get("total_events", 0)))
        table.add_row("Duration (s)", f"{duration:.1f}")
        table.add_row("Cleanups", str(cleanup_count))

    def _format_bytes_metric(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return format_bytes(int(value))
        except (TypeError, ValueError):
            return "-"

    def _append_monitor_events(self, events: list[TrackerEventView]) -> None:
        for event in events:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            color = self._event_color(event.event_type)
            summary = event.message or "No context provided."
            self.monitor_log.write(
                f"[{timestamp}] [{color}]{event.event_type.upper()}[/{color}] {summary}\n"
                f"Allocated: {event.allocated} | Reserved: {event.reserved} | Δ: {event.change}\n"
            )
        self._capture_alerts(events)
        self.alert_history_table.update_rows(self.recent_alerts)

    def _capture_alerts(self, events: list[TrackerEventView]) -> None:
        alert_types = {"warning", "critical", "error"}
        for event in events:
            if event.event_type in alert_types:
                self.recent_alerts.append(
                    {
                        "timestamp": event.timestamp,
                        "type": event.event_type,
                        "message": event.message or "",
                    }
                )
        self.recent_alerts = self.recent_alerts[-50:]

    def _event_color(self, event_type: str) -> str:
        return {
            "warning": "yellow",
            "critical": "red",
            "error": "red",
            "cleanup": "cyan",
            "peak": "magenta",
        }.get(event_type, "green")

    def _get_or_create_tracker_session(self) -> TrackerSession | None:
        if self.tracker_session is None:
            try:
                self.tracker_session = TrackerSession(
                    auto_cleanup=self.monitor_auto_cleanup
                )
            except TrackerUnavailableError as exc:
                self.log_monitor_message("Tracker", str(exc))
                return None
        else:
            self.tracker_session.set_auto_cleanup(self.monitor_auto_cleanup)
        return self.tracker_session

    def _update_monitor_status(self) -> None:
        session = self.tracker_session
        cleanup_state = "enabled" if self.monitor_auto_cleanup else "disabled"

        if session and session.is_active:
            device_label = session.get_device_label() or "current CUDA device"
            message = (
                f"Live tracking on **{device_label}**.\n"
                f"Auto cleanup is {cleanup_state}."
            )
        else:
            message = (
                "Tracker idle. Start a session to stream GPU allocation events.\n"
                f"Auto cleanup is currently {cleanup_state}."
            )

        self.monitor_status.update(message)

    def _update_watchdog_button_label(self) -> None:
        label = "Auto Cleanup: ON" if self.monitor_auto_cleanup else "Auto Cleanup: OFF"
        variant = "success" if self.monitor_auto_cleanup else "warning"
        self.watchdog_button.label = label
        self.watchdog_button.variant = variant
        self._sync_threshold_inputs()

    def _sync_threshold_inputs(self) -> None:
        session = self.tracker_session
        if not session:
            return
        thresholds = session.get_thresholds()
        warning = thresholds.get("memory_warning_percent")
        critical = thresholds.get("memory_critical_percent")
        if warning is not None:
            self.warning_input.value = f"{warning:.0f}"
        if critical is not None:
            self.critical_input.value = f"{critical:.0f}"

    async def refresh_visualizations(self) -> None:
        timeline = self._collect_timeline_data()
        if not timeline:
            self.timeline_canvas.render_placeholder(
                "No timeline samples found. Start live tracking and try again."
            )
            self._clear_timeline_stats_table()
            self.log_visual_message("Visualizations", "No timeline data yet.")
            return

        self._last_timeline = timeline
        self._update_timeline_view(timeline)
        self.log_visual_message("Visualizations", "Timeline refreshed.")

    async def generate_visual_plot(self, format: str) -> None:
        timeline = self._last_timeline or self._collect_timeline_data()
        if not timeline:
            self.log_visual_message(
                "Visualizations", "Need timeline samples before exporting plots."
            )
            return

        self.log_visual_message(
            "Visualizations", f"Generating {format.upper()} timeline plot..."
        )
        try:
            file_path = await asyncio.to_thread(
                self._save_timeline_plot, timeline, format
            )
        except ImportError as exc:
            self.log_visual_message("Visualizations", f"Error: {exc}")
            return
        except Exception as exc:
            self.log_visual_message("Visualizations", f"Export failed: {exc}")
            return

        self.log_visual_message(
            "Visualizations", f"Saved timeline plot to: {file_path}"
        )

    def _collect_timeline_data(self, interval: float = 1.0) -> dict[str, Any]:
        session = self.tracker_session
        if session:
            timeline = session.get_memory_timeline(interval=interval)
            if timeline and timeline.get("timestamps"):
                return timeline
        return self._last_timeline or {}

    def _update_timeline_view(self, timeline: dict) -> None:
        if not timeline or not timeline.get("timestamps"):
            self.timeline_canvas.render_placeholder(
                "Timeline is empty. Start tracking to capture samples."
            )
            self._clear_timeline_stats_table()
            return

        self.timeline_canvas.render_timeline(timeline)
        self._update_timeline_stats_table(timeline)

    def _update_timeline_stats_table(self, timeline: dict) -> None:
        table = self.timeline_stats_table
        table.clear()
        timestamps = timeline.get("timestamps") or []
        allocated = timeline.get("allocated") or []
        reserved = timeline.get("reserved") or []

        sample_count = len(allocated)
        if not sample_count or not timestamps:
            self._clear_timeline_stats_table()
            return

        duration = max(0.0, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        alloc_max = max(allocated) if allocated else 0
        reserv_max = max(reserved) if reserved else 0
        alloc_latest = allocated[-1] if allocated else 0
        reserv_latest = reserved[-1] if reserved else 0

        table.add_row("Samples", str(sample_count))
        table.add_row("Duration (s)", f"{duration:.1f}")
        table.add_row("Allocated Max", format_bytes(int(alloc_max)))
        table.add_row("Reserved Max", format_bytes(int(reserv_max)))
        table.add_row("Allocated Latest", format_bytes(int(alloc_latest)))
        table.add_row("Reserved Latest", format_bytes(int(reserv_latest)))

    def _clear_timeline_stats_table(self) -> None:
        table = self.timeline_stats_table
        table.clear()
        table.add_row("Samples", "0")
        table.add_row("Duration (s)", "-")
        table.add_row("Allocated Max", "-")
        table.add_row("Reserved Max", "-")
        table.add_row("Allocated Latest", "-")
        table.add_row("Reserved Latest", "-")

    def _save_timeline_plot(self, timeline: dict, format: str) -> str:
        timestamps = timeline.get("timestamps") or []
        allocated = timeline.get("allocated") or []
        reserved = timeline.get("reserved") or []

        if not timestamps or not allocated:
            raise ValueError("Timeline data is empty.")

        start = timestamps[0]
        rel_times = [t - start for t in timestamps]
        allocated_gb = [val / (1024**3) for val in allocated]
        reserved_gb = [val / (1024**3) for val in reserved] if reserved else []

        plots_dir = Path.cwd() / "visualizations"
        plots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "png":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(rel_times, allocated_gb, label="Allocated (GB)", color="tab:blue")
            if reserved_gb:
                ax.plot(rel_times, reserved_gb, label="Reserved (GB)", color="tab:red")
            ax.set_title("GPU Memory Timeline")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Memory (GB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()

            file_path = plots_dir / f"timeline_{stamp}.png"
            fig.savefig(file_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return str(file_path)

        if format == "html":
            try:
                import plotly.graph_objects as go
            except ImportError as exc:
                raise ImportError(
                    "Plotly is required for HTML output. Install gpu-memory-profiler[viz]."
                ) from exc

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=allocated_gb,
                    mode="lines",
                    name="Allocated (GB)",
                )
            )
            if reserved_gb:
                fig.add_trace(
                    go.Scatter(
                        x=rel_times,
                        y=reserved_gb,
                        mode="lines",
                        name="Reserved (GB)",
                    )
                )

            fig.update_layout(
                title="GPU Memory Timeline",
                xaxis_title="Time (s)",
                yaxis_title="Memory (GB)",
                hovermode="x unified",
            )

            file_path = plots_dir / f"timeline_{stamp}.html"
            fig.write_html(file_path)
            return str(file_path)

        raise ValueError(f"Unsupported format: {format}")

    async def _execute_task(
        self,
        title: str,
        func: Callable[[], Any],
        formatter: Optional[Callable[[Any], str]],
    ) -> None:
        formatter = formatter or (lambda value: str(value))
        self._set_loader(True)
        self.log_message(title, "Running sample workload...")
        try:
            result = await asyncio.to_thread(func)
            self.log_message(title, formatter(result))
        except Exception as exc:
            self.log_message(title, f"Error: {exc}")
        finally:
            self._set_loader(False)

    def _set_loader(self, visible: bool) -> None:
        self.loader.display = visible

    @staticmethod
    def _pytorch_sample_workload() -> dict[str, Any]:
        if GPUMemoryProfiler is None or torch is None:
            raise RuntimeError("PyTorch profiler is unavailable.")
        profiler = GPUMemoryProfiler()

        def workload() -> Any:
            x = torch.randn((3072, 3072), device="cuda")
            y = torch.matmul(x, x)
            return y.sum()

        profiler.profile_function(workload)
        return cast(dict[str, Any], profiler.get_summary())

    @staticmethod
    def _tensorflow_sample_workload() -> Any:
        if TFMemoryProfiler is None or tf is None:
            raise RuntimeError(
                "TensorFlow profiler is unavailable. Install tensorflow and tfmemprof: "
                "pip install tensorflow tfmemprof"
            )
        profiler = TFMemoryProfiler()
        with profiler.profile_context("tf_sample"):
            tensor = tf.random.normal((2048, 2048))
            product = tf.matmul(tensor, tensor)
            tf.reduce_sum(product)
        return profiler.get_results()

    @staticmethod
    def _cpu_sample_workload() -> dict[str, Any]:
        if CPUMemoryProfiler is None:
            raise RuntimeError("CPUMemoryProfiler is unavailable.")
        profiler = CPUMemoryProfiler()

        def workload() -> int:
            data = [i for i in range(500000)]
            return sum(data)

        profiler.profile_function(workload)
        return cast(dict[str, Any], profiler.get_summary())

    @staticmethod
    def _format_pytorch_summary(summary: dict) -> str:
        peak = summary.get("peak_memory_usage", 0)
        delta = summary.get("memory_change_from_baseline", 0)
        calls = summary.get("total_function_calls", "N/A")
        lines = [
            f"Functions profiled: {summary.get('total_functions_profiled', 'N/A')}",
            f"Total calls: {calls}",
            f"Peak memory: {format_bytes(peak)}",
            f"Δ from baseline: {format_bytes(delta)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_tensorflow_results(results: Any) -> str:
        lines = [
            f"Duration: {results.duration:.2f}s",
            f"Peak memory: {results.peak_memory_mb:.2f} MB",
            f"Average memory: {results.average_memory_mb:.2f} MB",
            f"Snapshots: {len(results.snapshots)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_cpu_summary(summary: dict) -> str:
        lines = [
            f"Snapshots collected: {summary.get('snapshots_collected', 0)}",
            f"Peak RSS: {format_bytes(summary.get('peak_memory_usage', 0))}",
            f"Δ from baseline: {format_bytes(summary.get('memory_change_from_baseline', 0))}",
        ]
        return "\n".join(lines)

    def log_monitor_message(self, title: str, content: str) -> None:
        self.monitor_log.write(f"[bold]{title}[/bold]\n{content}\n")

    def log_visual_message(self, title: str, content: str) -> None:
        self.visual_log.write(f"[bold]{title}[/bold]\n{content}\n")

    def log_message(self, title: str, content: str) -> None:
        self.command_log.write(f"[bold]{title}[/bold]\n{content}\n")

    async def on_mount(self) -> None:
        self.tracker_session = None
        self.cli_runner = CLICommandRunner()
        self.monitor_auto_cleanup = False
        self._last_monitor_stats = {}
        self._last_timeline = {}
        self.recent_alerts = []
        self.set_interval(1.0, self.refresh_monitoring_panel)
        self._update_watchdog_button_label()
        self._update_monitor_status()
        self.timeline_canvas.render_placeholder(
            "No timeline data yet. Start live tracking and refresh."
        )
        self._clear_timeline_stats_table()

        # Initial log entry
        await asyncio.sleep(0)
        self.log_message(
            "Welcome",
            "Use the tabs or press [b]r[/b] to refresh the overview. "
            "Buttons in the CLI tab will log summaries here.",
        )
        await self.refresh_pytorch_profiles()
        await self.refresh_tensorflow_profiles()


def run_app() -> None:
    """Entry-point to launch the Textual application."""
    GPUMemoryProfilerTUI().run()


__all__ = ["run_app", "GPUMemoryProfilerTUI"]
