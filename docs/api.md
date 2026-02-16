[← Back to main docs](index.md)

# API Reference

This document provides the complete API reference for GPU Memory Profiler.

## PyTorch Profiler (`gpumemprof`)

### Core Classes

#### `GPUMemoryProfiler`

Main profiler class for PyTorch applications.

```python
class GPUMemoryProfiler:
    def __init__(self,
                 device: Union[str, int, torch.device, None] = None,
                 track_tensors: bool = True,
                 track_cpu_memory: bool = True,
                 collect_stack_traces: bool = False):
        """
        Initialize the GPU memory profiler.

        Args:
            device: CUDA device (None uses current CUDA device)
            track_tensors: Track tensor creation/deletion deltas
            track_cpu_memory: Include process CPU memory in snapshots
            collect_stack_traces: Capture short stack traces per operation
        """
```

**Methods:**

- `start_monitoring(interval: float = None)`: Start real-time monitoring
- `stop_monitoring()`: Stop monitoring
- `profile_function(func, *args, **kwargs)`: Profile one callable invocation
- `profile_context(name: str)`: Context manager for profiling
- `get_summary()`: Return aggregated profiling metrics
- `clear_results()`: Reset captured results/snapshots

#### `MemorySnapshot`

Represents a memory snapshot at a point in time.

```python
class MemorySnapshot:
    timestamp: float
    allocated_memory: int
    reserved_memory: int
    max_memory_allocated: int
    max_memory_reserved: int
    active_memory: int
    inactive_memory: int
    cpu_memory: int
    device_id: int
    operation: Optional[str]
```

#### `ProfileResult`

Contains profiling results and statistics.

```python
class ProfileResult:
    function_name: str
    execution_time: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    memory_peak: MemorySnapshot
    memory_allocated: int
    memory_freed: int
    tensors_created: int
    tensors_deleted: int
    call_count: int
```

### Context Profiling

#### `profile_function`

Decorator for profiling individual functions.

```python
@profile_function
def my_function():
    pass
```

#### `profile_context`

Context manager for profiling code blocks.

```python
with profile_context("my_context"):
    pass
```

### Utilities

#### `get_system_info()`

Get system and GPU information.

```python
def get_system_info() -> dict:
    """Get comprehensive system information."""
```

`get_system_info()` now reports backend diagnostics:

- `detected_backend`: one of `cuda`, `rocm`, `mps`, or `cpu`
- `rocm_available` / `rocm_version`
- `mps_built` / `mps_available`

#### Backend Collectors

Backend-agnostic collector interfaces are available in `gpumemprof.device_collectors`:

```python
from gpumemprof import (
    DeviceMemoryCollector,
    DeviceMemorySample,
    build_device_memory_collector,
)
```

Collector contract:

- `name() -> str`
- `is_available() -> bool`
- `sample() -> DeviceMemorySample`
- `capabilities() -> dict`

`capabilities()` includes:

- `backend`
- `supports_device_total`
- `supports_device_free`
- `sampling_source`
- `telemetry_collector`

#### OOM Flight Recorder

The PyTorch tracker supports an optional OOM flight recorder that stores a bounded
ring buffer of recent events and writes an artifact bundle when OOM is detected.

```python
from gpumemprof import MemoryTracker

tracker = MemoryTracker(
    enable_oom_flight_recorder=True,
    oom_dump_dir="oom_dumps",
    oom_buffer_size=5000,
    oom_max_dumps=10,
    oom_max_total_mb=1024,
)

with tracker.capture_oom(context="train_loop"):
    # training code
    ...
```

Tracker OOM APIs:

- `handle_exception(exc, context=None, metadata=None) -> Optional[str]`
- `capture_oom(context="runtime", metadata=None)` (context manager)

Helper exports:

- `OOMFlightRecorderConfig`
- `OOMExceptionClassification`
- `classify_oom_exception(exc)`

#### `format_memory(bytes_value: int) -> str`

Format memory values in human-readable format.

```python
def format_memory(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
```

## TensorFlow Profiler (`tfmemprof`)

### Core Classes

#### `TensorFlowProfiler`

Main profiler class for TensorFlow applications.

```python
class TensorFlowProfiler:
    def __init__(self, device: Optional[str] = None):
        """
        High-level TensorFlow profiling helper.

        Args:
            device: TensorFlow device name (for example '/GPU:0')
        """
```

**Methods:**

- `profile_training(model, dataset, epochs=1, steps_per_epoch=None)`: Profile training loops
- `profile_inference(model, data, batch_size=32)`: Profile inference workloads
- `get_results()`: Get profiling results
- `reset()`: Reset accumulated profiling data

Use `TFMemoryProfiler` when you need direct `profile_context(...)` / decorator-style profiling in TensorFlow snippets.

#### Context profiling

Use the high-level `TensorFlowProfiler` context manager to profile a training or inference block.

```python
from tfmemprof import TensorFlowProfiler

profiler = TensorFlowProfiler()
with profiler.profile_context("training"):
    model.fit(x_train, y_train)
```

### Utilities

#### `get_system_info()`

Get TensorFlow-specific system information.

```python
def get_system_info() -> dict:
    """Get TensorFlow system information."""
```

## CLI Commands

### PyTorch CLI (`gpumemprof`)

```bash
# System information
gpumemprof info

# Real-time monitoring
gpumemprof monitor [--duration SECONDS] [--interval SECONDS] [--output FILE] [--format csv|json]

# Background tracking
gpumemprof track [--duration SECONDS] [--interval SECONDS] [--output FILE]
gpumemprof track [--warning-threshold PERCENT] [--critical-threshold PERCENT] [--watchdog]

# Analysis
gpumemprof analyze FILE [--visualization] [--plot-dir DIR]
gpumemprof analyze FILE [--output FILE] [--format json|txt]
```

### TensorFlow CLI (`tfmemprof`)

```bash
# System information
tfmemprof info

# Real-time monitoring
tfmemprof monitor [--duration SECONDS] [--interval SECONDS] [--threshold MB] [--output FILE]

# Background tracking
tfmemprof track [--threshold MB] [--interval SECONDS] --output FILE

# Analysis
tfmemprof analyze --input FILE [--detect-leaks] [--optimize] [--visualize] [--report FILE]
```

## Data Structures

### Memory Snapshot

```python
{
    "timestamp": 1234567890.123,
    "memory_used_mb": 2048.5,
    "memory_total_mb": 8192.0,
    "memory_percent": 25.0,
    "gpu_id": 0
}
```

### Function Call

```python
{
    "name": "train_step",
    "start_time": 1234567890.123,
    "end_time": 1234567890.456,
    "duration": 0.333,
    "memory_start_mb": 1024.0,
    "memory_end_mb": 1536.0,
    "memory_peak_mb": 1792.0
}
```

### Leak Detection Result

```python
{
    "detected": True,
    "severity": "high",
    "memory_increase_mb": 512.0,
    "time_window_seconds": 60.0,
    "suggestions": ["Check for unreleased tensors", "Review model cleanup"]
}
```

## Error Handling

All profiler methods raise appropriate exceptions:

- `ProfilerError`: Base exception for profiler errors
- `MemoryError`: Memory-related errors
- `ConfigurationError`: Configuration errors
- `VisualizationError`: Visualization errors

## Examples

See the [examples directory](../examples/) for complete API usage examples.

---

[← Back to main docs](index.md)
