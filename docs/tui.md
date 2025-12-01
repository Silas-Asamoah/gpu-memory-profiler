# Terminal UI (Textual) Guide

The GPU Memory Profiler now ships with a Textual-based TUI that bundles key
workflows—system discovery, PyTorch/TensorFlow checklists, and CLI helpers—into
an interactive terminal experience.

## Installation

Install the optional TUI dependencies:

```bash
pip install "gpu-memory-profiler[tui]"
```

During development you can also rely on `requirements-dev.txt`, which already
includes `textual`.

## Launching the TUI

```bash
gpu-profiler
```

### What You’ll See

- **Overview tab** – Live system summary (platform, Python/TensorFlow versions,
  GPU snapshot). Use `r` to refresh.
- **PyTorch tab** – Copy-ready commands for curated demos plus a live GPU
  memory table sourced from `gpumemprof.utils.get_gpu_info`.
- **TensorFlow tab** – Equivalent guidance with an auto-updating table that
  reads from `tfmemprof.utils.get_gpu_info`.
- **CLI & Actions tab** – Rich instructions plus buttons for logging tips,
  refreshing system info, and launching sample PyTorch/TensorFlow profiling
  runs (with an inline loader/log output).

Keyboard shortcuts:

| Key | Action                   |
| --- | ------------------------ |
| `r` | Refresh overview         |
| `f` | Focus the log panel      |
| `g` | Log `gpumemprof info`    |
| `t` | Log `tfmemprof info`     |
| `q` | Quit the TUI             |

## Prompt Toolkit Roadmap

For command-palette or multi-step form experiences, we plan to layer in
`prompt_toolkit` components (e.g., an interactive shell for running
`gpumemprof`/`tfmemprof` commands with auto-completion). The Textual layout is
designed to accommodate this future addition without breaking compatibility.

## Troubleshooting

- **Missing dependency** – Ensure you used `pip install "gpu-memory-profiler[tui]"`.
- **GPU-less environments** – The overview tab will fall back to CPU-only data
  and explicitly state when GPU metrics are unavailable.
- **Terminal too small** – Textual adapts to smaller windows, but a minimum of
  ~100x30 characters makes the tabs most readable.

For more sample commands, see the Markdown test guides under
`docs/examples/test_guides/README.md`.

