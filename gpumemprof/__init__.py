"""GPU Memory Profiler - A comprehensive memory profiling tool for PyTorch."""

__version__ = "0.1.0"
__author__ = "GPU Memory Profiler Team"

from .profiler import GPUMemoryProfiler, MemorySnapshot, ProfileResult
from .context_profiler import profile_context, profile_function
from .analyzer import MemoryAnalyzer
from .utils import get_gpu_info, format_bytes, convert_bytes
from .tracker import MemoryTracker
from .cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker

__all__ = [
    "GPUMemoryProfiler",
    "MemorySnapshot",
    "ProfileResult",
    "profile_context",
    "profile_function",
    "MemoryVisualizer",
    "MemoryAnalyzer",
    "MemoryTracker",
    "CPUMemoryProfiler",
    "CPUMemoryTracker",
    "get_gpu_info",
    "format_bytes",
    "convert_bytes",
]


def __getattr__(name):
    """Lazily import optional visualizer dependencies."""
    if name != "MemoryVisualizer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .visualizer import MemoryVisualizer
    except ImportError as exc:
        raise ImportError(
            "MemoryVisualizer requires optional visualization dependencies. "
            "Install with `pip install gpu-memory-profiler[viz]`."
        ) from exc

    globals()[name] = MemoryVisualizer
    return MemoryVisualizer
