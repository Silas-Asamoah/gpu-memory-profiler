[← Back to main docs](index.md)

# Usage Guide

This guide covers how to use GPU Memory Profiler for both PyTorch and TensorFlow applications.

## Quick Start

### PyTorch Usage

```python
from gpumemprof import GPUMemoryProfiler

# Create profiler instance
profiler = GPUMemoryProfiler()

# Profile a function
@profiler.profile_function
def train_step(model, data, target):
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return loss

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### TensorFlow Usage

```python
from tfmemprof import TensorFlowProfiler

# Create profiler instance
profiler = TensorFlowProfiler()

# Profile training
with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=5)

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

## Advanced Usage

### Real-time Monitoring

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Start monitoring
profiler.start_monitoring(interval=1.0)  # Sample every second

# Your training code here
for epoch in range(10):
    for batch in dataloader:
        train_step(model, batch)

# Stop and get results
profiler.stop_monitoring()
results = profiler.get_results()
```

### Memory Leak Detection

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Enable leak detection
profiler.enable_leak_detection(
    threshold=100,  # MB
    window_size=10  # samples
)

# Run your code
for i in range(100):
    train_step(model, data)

# Check for leaks
leaks = profiler.detect_leaks()
if leaks:
    print(f"Potential memory leak detected: {leaks}")
```

### Visualization

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Profile your code
with profiler.profile_context("training"):
    train_model()

# Generate visualizations
profiler.plot_memory_timeline()
profiler.plot_memory_heatmap()
profiler.create_dashboard()
```

## CLI Usage

### Basic Monitoring

```bash
# Monitor for 60 seconds
gpumemprof monitor --duration 60 --output monitoring.csv

# Monitor with alerts
gpumemprof track --threshold 4000 --alert
```

### Analysis

```bash
# Analyze results
gpumemprof analyze monitoring.csv --visualization

# Generate report
gpumemprof analyze monitoring.csv --report report.html
```

## Configuration

### Profiler Settings

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler(
    sampling_interval=0.5,  # seconds
    alert_threshold=4000,   # MB
    enable_visualization=True,
    export_format='json'
)
```

### Context Profiling

```python
from gpumemprof import profile_function, profile_context

# Function decorator
@profile_function
def my_function():
    pass

# Context manager
with profile_context("my_context"):
    pass
```

## Best Practices

1. **Start Early**: Begin profiling early in development
2. **Use Contexts**: Use context managers for better organization
3. **Monitor Regularly**: Set up continuous monitoring in production
4. **Set Alerts**: Configure appropriate thresholds
5. **Export Data**: Save results for later analysis

## Examples

See the [examples directory](../examples/) for complete working examples:

- [Basic PyTorch profiling](../examples/basic_profiling.py)
- [Advanced tracking](../examples/advanced_tracking.py)
- [TensorFlow profiling](../examples/tensorflow_basic_profiling.py)

---

[← Back to main docs](index.md)
