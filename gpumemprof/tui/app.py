"""Interactive Textual TUI for GPU Memory Profiler."""

from __future__ import annotations

import asyncio
from textwrap import dedent

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    LoadingIndicator,
    Markdown,
    RichLog,
    Rule,
    TabPane,
    TabbedContent,
)

from gpumemprof.utils import get_system_info, get_gpu_info, format_bytes
from tfmemprof.utils import get_system_info as get_tf_system_info
from tfmemprof.utils import get_gpu_info as get_tf_gpu_info

try:
    import torch
except Exception:
    torch = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from gpumemprof import GPUMemoryProfiler
except Exception:
    GPUMemoryProfiler = None

try:
    from tfmemprof.profiler import TFMemoryProfiler
except Exception:
    TFMemoryProfiler = None


def _safe_get_gpu_info() -> dict:
    try:
        return get_gpu_info()
    except Exception:
        return {}


def _safe_get_tf_system_info() -> dict:
    try:
        return get_tf_system_info()
    except Exception:
        return {}


def _safe_get_tf_gpu_info() -> dict:
    try:
        return get_tf_gpu_info()
    except Exception:
        return {}


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


class GPUStatsTable(DataTable):
    """Live-updating table of GPU stats."""

    def __init__(self, title: str, provider, refresh_interval: float = 2.0):
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

    def __init__(self, builder, **kwargs):
        super().__init__("", **kwargs)
        self.builder = builder

    def refresh_content(self) -> None:
        self.update(self.builder())

    def on_mount(self) -> None:
        self.refresh_content()


class GPUMemoryProfilerTUI(App):
    """Main Textual application."""

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

    #cli-loader {
        height: 3;
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
        self.pytorch_panel = MarkdownPanel(
            lambda: _build_framework_markdown("pytorch"), id="pytorch"
        )
        self.tensorflow_panel = MarkdownPanel(
            lambda: _build_framework_markdown("tensorflow"), id="tensorflow"
        )
        self.cli_panel = MarkdownPanel(_build_cli_markdown, id="cli-docs")
        self.command_log = RichLog(highlight=True, markup=True, id="command-log")
        self.loader = LoadingIndicator(id="cli-loader")
        self.loader.display = False

        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Overview"):
                yield VerticalScroll(self.overview_panel)

            with TabPane("PyTorch"):
                yield Container(
                    VerticalScroll(self.pytorch_panel),
                    GPUStatsTable("pytorch", _pytorch_stats_provider),
                    id="pytorch-tab",
                )

            with TabPane("TensorFlow"):
                yield Container(
                    VerticalScroll(self.tensorflow_panel),
                    GPUStatsTable("tensorflow", _tensorflow_stats_provider),
                    id="tensorflow-tab",
                )

            with TabPane("CLI & Actions"):
                yield Container(
                    self.cli_panel,
                    Rule(),
                    Horizontal(
                        Button("Log System Info", id="btn-log-system", variant="primary"),
                        Button("Refresh Overview", id="btn-refresh-overview", variant="warning"),
                        Button("GP CLI Tips", id="btn-log-pytorch", variant="success"),
                        Button("TF CLI Tips", id="btn-log-tensorflow", variant="success"),
                        Button("Run PyTorch Sample", id="btn-run-pytorch", variant="primary"),
                        Button("Run TensorFlow Sample", id="btn-run-tf", variant="primary"),
                        id="cli-buttons",
                    ),
                    self.loader,
                    self.command_log,
                    id="cli-tab",
                )
        yield Footer()

    def action_quit(self) -> None:
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
            self.log_message("System Info", _build_system_markdown())
        elif button_id == "btn-log-pytorch":
            self.log_message("PyTorch Summary", _build_framework_markdown("pytorch"))
        elif button_id == "btn-log-tensorflow":
            self.log_message(
                "TensorFlow Summary", _build_framework_markdown("tensorflow")
            )
        elif button_id == "btn-run-pytorch":
            await self.run_pytorch_sample()
        elif button_id == "btn-run-tf":
            await self.run_tensorflow_sample()

    async def run_pytorch_sample(self) -> None:
        if GPUMemoryProfiler is None or torch is None:
            self.log_message("PyTorch Sample", "PyTorch profiler is unavailable in this environment.")
            return
        if not torch.cuda.is_available():
            self.log_message("PyTorch Sample", "CUDA is not available; skipping sample workload.")
            return
        await self._execute_task(
            "PyTorch Sample",
            self._pytorch_sample_workload,
            self._format_pytorch_summary,
        )

    async def run_tensorflow_sample(self) -> None:
        if TFMemoryProfiler is None or tf is None:
            self.log_message("TensorFlow Sample", "TensorFlow profiler is unavailable in this environment.")
            return
        await self._execute_task(
            "TensorFlow Sample",
            self._tensorflow_sample_workload,
            self._format_tensorflow_results,
        )

    async def _execute_task(self, title: str, func, formatter) -> None:
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
    def _pytorch_sample_workload():
        profiler = GPUMemoryProfiler()

        def workload():
            x = torch.randn((3072, 3072), device="cuda")
            y = torch.matmul(x, x)
            return y.sum()

        profiler.profile_function(workload)
        return profiler.get_summary()

    @staticmethod
    def _tensorflow_sample_workload():
        profiler = TFMemoryProfiler()
        with profiler.profile_context("tf_sample"):
            tensor = tf.random.normal((2048, 2048))
            product = tf.matmul(tensor, tensor)
            tf.reduce_sum(product)
        return profiler.get_results()

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
    def _format_tensorflow_results(results) -> str:
        lines = [
            f"Duration: {results.duration:.2f}s",
            f"Peak memory: {results.peak_memory_mb:.2f} MB",
            f"Average memory: {results.average_memory_mb:.2f} MB",
            f"Snapshots: {len(results.snapshots)}",
        ]
        return "\n".join(lines)

    def log_message(self, title: str, content: str) -> None:
        self.command_log.write(f"[bold]{title}[/bold]\n{content}\n")

    async def on_mount(self) -> None:
        # Initial log entry
        await asyncio.sleep(0)
        self.log_message(
            "Welcome",
            "Use the tabs or press [b]r[/b] to refresh the overview. "
            "Buttons in the CLI tab will log summaries here.",
        )


def run_app() -> None:
    """Entry-point to launch the Textual application."""
    GPUMemoryProfilerTUI().run()


__all__ = ["run_app", "GPUMemoryProfilerTUI"]

