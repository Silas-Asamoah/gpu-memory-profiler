[← Back to main docs](index.md)

# Examples Guide

This guide provides comprehensive examples for using GPU Memory Profiler with both PyTorch and TensorFlow.

## PyTorch Examples

### Basic Profiling

```python
import torch
import torch.nn as nn
from gpumemprof import GPUMemoryProfiler

# Create a simple model
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100)
)

# Initialize profiler
profiler = GPUMemoryProfiler()

# Profile training step
@profiler.profile_function
def train_step(model, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return loss

# Run training
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(dataloader):
        loss = train_step(model, data, target)

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### Advanced Monitoring

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler(
    sampling_interval=0.5,
    alert_threshold=4000,  # MB
    enable_visualization=True
)

# Start monitoring
profiler.start_monitoring()

# Your training loop
for epoch in range(10):
    for batch in dataloader:
        train_step(model, batch)

    # Check for memory leaks
    leaks = profiler.detect_leaks()
    if leaks:
        print(f"Memory leak detected: {leaks}")

# Stop monitoring
profiler.stop_monitoring()

# Generate visualizations
profiler.plot_memory_timeline()
profiler.plot_memory_heatmap()
profiler.create_dashboard()
```

### Context Profiling

```python
from gpumemprof import profile_context

# Profile different phases
with profile_context("data_loading"):
    train_data = load_dataset()
    val_data = load_validation_data()

with profile_context("model_creation"):
    model = create_model()

with profile_context("training"):
    for epoch in range(10):
        train_epoch(model, train_data)

with profile_context("validation"):
    validate_model(model, val_data)
```

## TensorFlow Examples

### Basic TensorFlow Profiling

```python
import tensorflow as tf
from tfmemprof import TensorFlowProfiler

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize profiler
profiler = TensorFlowProfiler()

# Profile training
with profiler.profile_context("training"):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

# Get results
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### Keras Model Profiling

```python
from tfmemprof import TensorFlowProfiler

profiler = TensorFlowProfiler()

# Profile model creation
with profiler.profile_context("model_creation"):
    model = create_complex_model()

# Profile data preprocessing
with profiler.profile_context("data_preprocessing"):
    x_train, y_train = preprocess_data()

# Profile training
with profiler.profile_context("training"):
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )

# Profile evaluation
with profiler.profile_context("evaluation"):
    test_loss, test_acc = model.evaluate(x_test, y_test)

# Analyze results
results = profiler.get_results()
profiler.plot_memory_timeline()
```

### Custom Training Loop

```python
import tensorflow as tf
from tfmemprof import TensorFlowProfiler

profiler = TensorFlowProfiler()

@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Profile custom training
with profiler.profile_context("custom_training"):
    for epoch in range(10):
        for batch_x, batch_y in train_dataset:
            loss = train_step(model, optimizer, batch_x, batch_y)
```

## CLI Examples

### Real-time Monitoring

```bash
# Monitor for 5 minutes
gpumemprof monitor --duration 300 --output monitoring.json

# Monitor with alerts
gpumemprof track --threshold 4000 --alert --output tracking.json

# Analyze results
gpumemprof analyze monitoring.json --visualization
```

### TensorFlow CLI

```bash
# Get system info
tfmemprof info

# Monitor TensorFlow training
tfmemprof monitor --duration 600 --output tf_monitoring.json

# Track with alerts
tfmemprof track --threshold 3000 --alert --output tf_tracking.json
```

## Complete Working Examples

### PyTorch Training Example

See [basic_profiling.py](../examples/basic_profiling.py) for a complete PyTorch training example with profiling.

### Advanced Tracking Example

See [advanced_tracking.py](../examples/advanced_tracking.py) for advanced memory tracking with alerts and visualization.

### TensorFlow Example

See [tensorflow_basic_profiling.py](../examples/tensorflow_basic_profiling.py) for a complete TensorFlow example.

## Best Practices

1. **Profile Early**: Start profiling during development, not just in production
2. **Use Contexts**: Organize profiling with meaningful context names
3. **Set Thresholds**: Configure appropriate memory thresholds for your hardware
4. **Monitor Continuously**: Use CLI tools for continuous monitoring
5. **Export Data**: Save results for later analysis and comparison
6. **Visualize**: Use built-in visualization tools to understand patterns

## Troubleshooting Examples

### Memory Leak Detection

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Enable leak detection
profiler.enable_leak_detection(threshold=100, window_size=10)

# Run your code
for i in range(100):
    train_step(model, data)

# Check for leaks
leaks = profiler.detect_leaks()
if leaks:
    print("Potential memory leaks detected:")
    for leak in leaks:
        print(f"  - {leak}")
```

### Performance Optimization

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()

# Profile different batch sizes
for batch_size in [16, 32, 64, 128]:
    with profiler.profile_context(f"batch_size_{batch_size}"):
        train_with_batch_size(model, dataloader, batch_size)

    results = profiler.get_results()
    print(f"Batch size {batch_size}: Peak memory {results.peak_memory_mb:.2f} MB")
```

---

[← Back to main docs](index.md)
