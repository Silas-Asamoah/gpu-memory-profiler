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
                 sampling_interval: float = 1.0,
                 alert_threshold: float = None,
                 enable_visualization: bool = True,
                 export_format: str = 'json'):
        """
        Initialize the GPU memory profiler.

        Args:
            sampling_interval: Time between memory samples (seconds)
            alert_threshold: Memory threshold for alerts (MB)
            enable_visualization: Enable visualization features
            export_format: Export format ('json', 'csv')
        """
```

**Methods:**

- `start_monitoring(interval: float = None)`: Start real-time monitoring
- `stop_monitoring()`: Stop monitoring
- `profile_function(func)`: Decorator for profiling functions
- `profile_context(name: str)`: Context manager for profiling
- `get_results()`: Get profiling results
- `detect_leaks()`: Detect memory leaks
- `plot_memory_timeline()`: Plot memory usage over time
- `plot_memory_heatmap()`: Create memory heatmap
- `create_dashboard()`: Create interactive dashboard
- `export_results(filename: str)`: Export results to file

#### `MemorySnapshot`

Represents a memory snapshot at a point in time.

```python
class MemorySnapshot:
    def __init__(self, timestamp: float, memory_used: float, memory_total: float):
        self.timestamp = timestamp
        self.memory_used = memory_used
        self.memory_total = memory_total
```

#### `ProfileResult`

Contains profiling results and statistics.

```python
class ProfileResult:
    def __init__(self):
        self.peak_memory_mb = 0.0
        self.average_memory_mb = 0.0
        self.memory_snapshots = []
        self.function_calls = []
        self.leaks_detected = []
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
    def __init__(self,
                 sampling_interval: float = 1.0,
                 alert_threshold: float = None,
                 enable_visualization: bool = True,
                 export_format: str = 'json'):
        """
        Initialize the TensorFlow GPU memory profiler.

        Args:
            sampling_interval: Time between memory samples (seconds)
            alert_threshold: Memory threshold for alerts (MB)
            enable_visualization: Enable visualization features
            export_format: Export format ('json', 'csv')
        """
```

**Methods:**

- `start_monitoring(interval: float = None)`: Start real-time monitoring
- `stop_monitoring()`: Stop monitoring
- `profile_context(name: str)`: Context manager for profiling
- `get_results()`: Get profiling results
- `detect_leaks()`: Detect memory leaks
- `plot_memory_timeline()`: Plot memory usage over time
- `plot_memory_heatmap()`: Create memory heatmap
- `create_dashboard()`: Create interactive dashboard
- `export_results(filename: str)`: Export results to file

#### `TensorFlowContextProfiler`

Context-aware profiler for TensorFlow.

```python
class TensorFlowContextProfiler:
    def __init__(self, profiler: TensorFlowProfiler):
        self.profiler = profiler
```

### Context Profiling

#### `tensorflow_profiler`

Context manager for TensorFlow profiling.

```python
with tensorflow_profiler("training"):
    model.fit(x_train, y_train)
```

### Utilities

#### `get_system_info()`

Get TensorFlow-specific system information.

```python
def get_system_info() -> dict:
    """Get TensorFlow system information."""
```

#### `validate_tensorflow_setup()`

Validate TensorFlow installation and GPU setup.

```python
def validate_tensorflow_setup() -> dict:
    """Validate TensorFlow setup and return status."""
```

## CLI Commands

### PyTorch CLI (`gpumemprof`)

```bash
# System information
gpumemprof info

# Real-time monitoring
gpumemprof monitor [--duration SECONDS] [--output FILE]

# Background tracking
gpumemprof track [--threshold MB] [--alert] [--output FILE]

# Analysis
gpumemprof analyze FILE [--visualization] [--report FILE]
```

### TensorFlow CLI (`tfmemprof`)

```bash
# System information
tfmemprof info

# Real-time monitoring
tfmemprof monitor [--duration SECONDS] [--output FILE]

# Background tracking
tfmemprof track [--threshold MB] [--alert] [--output FILE]

# Analysis
tfmemprof analyze FILE [--visualization] [--report FILE]
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
