"""GPU Memory Profiler - A comprehensive memory profiling tool for PyTorch."""

__version__ = "0.1.0"
__author__ = "GPU Memory Profiler Team"

from .profiler import GPUMemoryProfiler, MemorySnapshot, ProfileResult
from .context_profiler import profile_context, profile_function
from .analyzer import MemoryAnalyzer
from .utils import get_gpu_info, format_bytes, convert_bytes
from .tracker import MemoryTracker
from .cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker

try:
    from .visualizer import MemoryVisualizer
except ImportError as exc:
    _MEMORY_VISUALIZER_IMPORT_ERROR = exc

    class MemoryVisualizer:  # type: ignore[no-redef]
        """Fallback placeholder when optional visualization dependencies are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MemoryVisualizer requires optional visualization dependencies. "
                "Install with `pip install gpu-memory-profiler[viz]`."
            ) from _MEMORY_VISUALIZER_IMPORT_ERROR

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
