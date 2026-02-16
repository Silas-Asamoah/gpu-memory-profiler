[← Back to main docs](index.md)

# Architecture Guide

This document describes the architecture and design principles of GPU Memory Profiler.

## Overview

GPU Memory Profiler is designed with a modular, extensible architecture that supports both PyTorch and TensorFlow while maintaining clean separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory Profiler                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   PyTorch   │  │ TensorFlow  │  │     CLI     │         │
│  │  Profiler   │  │  Profiler   │  │   Tools     │         │
│  │ (gpumemprof)│  │(tfmemprof)  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Core Components                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Profiler  │  │  Tracker    │  │ Visualizer  │         │
│  │             │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Analyzer   │  │   Utils     │  │   Context   │         │
│  │             │  │             │  │  Profiler   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Framework Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   PyTorch   │  │ TensorFlow  │  │    CPU      │         │
│  │   Memory    │  │   Memory    │  │   Memory    │         │
│  │  Interface  │  │  Interface  │  │  Interface  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Profiler (`profiler.py`)

The main profiling engine that coordinates memory monitoring and data collection.

**Responsibilities:**

- Initialize profiling sessions
- Coordinate data collection from framework layers
- Manage profiling state and configuration
- Provide high-level API for users

**Key Classes:**

- `GPUMemoryProfiler` (PyTorch)
- `TensorFlowProfiler` (TensorFlow)

### 2. Tracker (`tracker.py`)

Real-time memory tracking with background monitoring capabilities.

**Responsibilities:**

- Continuous memory monitoring
- Alert system for memory thresholds
- Background data collection
- Memory leak detection

**Key Classes:**

- `MemoryTracker`
- `MemoryWatchdog`
- `TrackingResult`

### 3. Visualizer (`visualizer.py`)

Data visualization and reporting capabilities.

**Responsibilities:**

- Generate memory timeline plots
- Create heatmaps and charts
- Interactive dashboards
- Export visualizations

**Key Classes:**

- `MemoryVisualizer`
- `PlotlyVisualizer`
- `MatplotlibVisualizer`

### 4. Analyzer (`analyzer.py`)

Advanced analysis and optimization recommendations.

**Responsibilities:**

- Memory leak detection algorithms
- Performance analysis
- Optimization suggestions
- Pattern recognition

**Key Classes:**

- `MemoryAnalyzer`
- `LeakDetector`
- `OptimizationAdvisor`

### 5. Context Profiler (`context_profiler.py`)

Context-aware profiling with decorators and context managers.

**Responsibilities:**

- Function-level profiling
- Context manager support
- Decorator implementations
- Scope-based memory tracking

**Key Classes:**

- `ContextProfiler`
- `profile_function` (decorator)
- `profile_context` (context manager)

### 6. Utils (`utils.py`)

Utility functions and system information gathering.

**Responsibilities:**

- System information collection
- Memory formatting
- Framework detection
- Error handling

**Key Functions:**

- `get_system_info()`
- `format_memory()`
- `validate_setup()`

### 7. CLI (`cli.py`)

Command-line interface for standalone usage.

**Responsibilities:**

- Command-line argument parsing
- Real-time monitoring interface
- Data export and analysis
- System information display

**Key Commands:**

- `info` - System information
- `monitor` - Real-time monitoring
- `track` - Background tracking
- `analyze` - Results analysis

## Framework-Specific Architecture

### PyTorch Profiler (`gpumemprof`)

```
┌─────────────────────────────────────────┐
│              gpumemprof                 │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Profiler  │  │  Context    │      │
│  │             │  │  Profiler   │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Tracker   │  │ Visualizer  │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  Analyzer   │  │    Utils    │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────┤
│              PyTorch Layer              │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ torch.cuda  │  │   Memory    │      │
│  │   Memory    │  │  Allocator  │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

**PyTorch-Specific Features:**

- Tensor lifecycle tracking
- CUDA memory management integration
- PyTorch-specific optimizations
- Autograd memory profiling

### TensorFlow Profiler (`tfmemprof`)

```
┌─────────────────────────────────────────┐
│              tfmemprof                  │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Profiler  │  │  Context    │      │
│  │             │  │  Profiler   │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Tracker   │  │ Visualizer  │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐      │
│  │  Analyzer   │  │    Utils    │      │
│  │             │  │             │      │
│  └─────────────┘  └─────────────┘      │
├─────────────────────────────────────────┤
│            TensorFlow Layer             │
│  ┌─────────────┐  ┌─────────────┐      │
│  │   Session   │  │   Graph     │      │
│  │  Memory     │  │ Execution   │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘
```

**TensorFlow-Specific Features:**

- Session-based memory tracking
- Graph execution monitoring
- Keras model profiling
- Mixed precision support

## Data Flow

### 1. Initialization Flow

```
User Code → Profiler Init → Framework Detection → System Info → Ready
```

### 2. Profiling Flow

```
User Code → Context/Decorator → Memory Snapshot → Data Collection → Analysis
```

### 3. Monitoring Flow

```
Background Thread → Memory Sampling → Alert Check → Data Storage → Visualization
```

### 4. Analysis Flow

```
Collected Data → Pattern Detection → Leak Analysis → Optimization Suggestions → Reports
```

## Design Principles

### 1. Modularity

Each component has a single responsibility and can be used independently:

```python
# Use only the profiler
from gpumemprof import GPUMemoryProfiler
profiler = GPUMemoryProfiler()

# Use only the tracker
from gpumemprof import MemoryTracker
tracker = MemoryTracker()

# Use only the visualizer
from gpumemprof import MemoryVisualizer
visualizer = MemoryVisualizer()
```

### 2. Extensibility

The architecture supports easy extension for new frameworks:

```python
class NewFrameworkProfiler(BaseProfiler):
    def __init__(self):
        super().__init__()

    def get_memory_info(self):
        # Framework-specific implementation
        pass
```

### 3. Thread Safety

All components are designed to be thread-safe for concurrent usage:

```python
# Safe to use in multi-threaded environments
profiler = GPUMemoryProfiler()
profiler.start_monitoring()  # Background thread
# Main thread continues...
```

### 4. Performance

Minimal overhead design with configurable sampling:

```python
# Low overhead mode
profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=5.0)

# High precision mode
profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.1)
```

## Configuration Management

### Environment Variables

```bash
export GPU_MEMORY_PROFILER_LOG_LEVEL=DEBUG
export GPU_MEMORY_PROFILER_SAMPLING_INTERVAL=1.0
export GPU_MEMORY_PROFILER_ALERT_THRESHOLD=4000
```

### Configuration Files

```python
# config.yaml
profiler:
  sampling_interval: 1.0
  alert_threshold: 4000
  enable_visualization: true
  export_format: json

tracking:
  background_monitoring: true
  leak_detection: true
  threshold: 100
  window_size: 10
```

## Error Handling

### Exception Hierarchy

```
ProfilerError (base)
├── MemoryError
├── ConfigurationError
├── VisualizationError
├── FrameworkError
└── CLIError
```

### Graceful Degradation

```python
try:
    profiler = GPUMemoryProfiler()
except CUDAError:
    # Fall back to CPU mode
    from gpumemprof import CPUMemoryProfiler

    profiler = CPUMemoryProfiler()
```

## Testing Architecture

### Test Structure

```
tests/
├── unit/           # Unit tests for each component
├── integration/    # Integration tests
├── performance/    # Performance benchmarks
├── framework/      # Framework-specific tests
└── cli/           # CLI tests
```

### Mock Strategy

```python
# Mock CUDA for testing
@pytest.fixture
def mock_cuda():
    with patch('torch.cuda.is_available', return_value=True):
        yield
```

## Future Extensibility

### Plugin System

```python
class ProfilerPlugin:
    def on_memory_snapshot(self, snapshot):
        pass

    def on_leak_detected(self, leak):
        pass
```

### Custom Visualizations

```python
class CustomVisualizer(MemoryVisualizer):
    def create_custom_plot(self, data):
        # Custom visualization logic
        pass
```

### Framework Support

```python
class JAXProfiler(BaseProfiler):
    # JAX-specific implementation
    pass
```

---

[← Back to main docs](index.md)
