"""GPU Memory Profiler - A comprehensive memory profiling tool for PyTorch."""

from typing import Any

__version__ = "0.1.0"
__author__ = "GPU Memory Profiler Team"

from .profiler import GPUMemoryProfiler, MemorySnapshot, ProfileResult
from .context_profiler import profile_context, profile_function
from .analyzer import MemoryAnalyzer
from .utils import get_gpu_info, format_bytes, convert_bytes
from .tracker import MemoryTracker
from .oom_flight_recorder import (
    OOMExceptionClassification,
    OOMFlightRecorder,
    OOMFlightRecorderConfig,
    classify_oom_exception,
)
from .telemetry import (
    TelemetryEventV2,
    load_telemetry_events,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
from .device_collectors import (
    DeviceMemoryCollector,
    DeviceMemorySample,
    build_device_memory_collector,
    detect_torch_runtime_backend,
)
from .cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker

try:
    from .visualizer import MemoryVisualizer
except ImportError as exc:
    _MEMORY_VISUALIZER_IMPORT_ERROR = exc

    class MemoryVisualizer:  # type: ignore[no-redef]
        """Fallback placeholder when optional visualization dependencies are missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
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
    "OOMFlightRecorder",
    "OOMFlightRecorderConfig",
    "OOMExceptionClassification",
    "classify_oom_exception",
    "TelemetryEventV2",
    "DeviceMemoryCollector",
    "DeviceMemorySample",
    "build_device_memory_collector",
    "detect_torch_runtime_backend",
    "CPUMemoryProfiler",
    "CPUMemoryTracker",
    "telemetry_event_from_record",
    "telemetry_event_to_dict",
    "validate_telemetry_record",
    "load_telemetry_events",
    "get_gpu_info",
    "format_bytes",
    "convert_bytes",
]
