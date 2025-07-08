[← Back to main docs](index.md)

# TensorFlow Testing & Usage Guide

**Authors:** Prince Agyei Tuffour and Silas Bempong  
**Last Updated:** June 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Quick Start](#quick-start)
5. [GPU Profiling (CUDA Available)](#gpu-profiling-cuda-available)
6. [CPU Profiling (No GPU Required)](#cpu-profiling-no-gpu-required)
7. [Testing Framework](#testing-framework)
8. [Usage Examples](#usage-examples)
9. [Command Line Interface](#command-line-interface)
10. [TensorFlow-Specific Features](#tensorflow-specific-features)
11. [Troubleshooting](#troubleshooting)
12. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

The TensorFlow GPU Memory Profiler is a comprehensive tool for monitoring and optimizing memory usage during TensorFlow model training and inference. This guide provides complete instructions for testing and using the profiler in both GPU and CPU environments with TensorFlow-specific optimizations.

### Key Features

✅ **TensorFlow GPU Profiling** (CUDA required)

-   Real-time GPU memory tracking with `tf.config.experimental.get_memory_info()`
-   TensorFlow session management
-   Graph execution monitoring
-   Mixed precision profiling
-   Keras integration

✅ **TensorFlow CPU Profiling** (No GPU required)

-   System RAM usage monitoring
-   TensorFlow CPU operation tracking
-   Keras model profiling
-   Cross-platform compatibility

✅ **TensorFlow-Specific Features**

-   Automatic memory growth configuration
-   Graph vs Eager execution analysis
-   tf.function profiling
-   TensorFlow Lite optimization
-   Multi-GPU strategy profiling

---

## System Requirements

### Minimum Requirements

-   **Python**: 3.8 or higher
-   **TensorFlow**: 2.4.0 or higher
-   **System Memory**: 4GB RAM
-   **Storage**: 200MB free space

### For GPU Profiling

-   **CUDA**: 11.2 or higher
-   **cuDNN**: 8.1 or higher
-   **GPU**: NVIDIA GPU with CUDA support
-   **GPU Memory**: 4GB or higher recommended
-   **TensorFlow-GPU**: Properly configured

### For CPU-Only Profiling

-   **CPU**: Any modern CPU (4+ cores recommended)
-   **RAM**: 8GB or higher
-   **TensorFlow-CPU**: Any installation

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/nanaagyei/gpu-memory-profiler.git
cd gpu-memory-profiler
```

### Step 2: Choose Your TensorFlow Installation

#### For GPU Systems (CUDA Available)

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Or specific version
pip install tensorflow-gpu==2.12.0

# Install profiler dependencies
pip install -r requirements.txt

# Install the profiler package
pip install -e .
```

#### For CPU-Only Systems

```bash
# Install CPU-only TensorFlow
pip install tensorflow-cpu

# Install profiler dependencies
pip install psutil matplotlib numpy

# Install the profiler package
pip install -e .
```

### Step 3: Verify TensorFlow Installation

```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Check profiler installation
python -c "from tfmemprof import TFMemoryProfiler; print('TensorFlow Profiler installed successfully!')"

# Test GPU setup (if available)
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

---

## Quick Start

### 🚀 Ultra-Quick Test (30 seconds)

Choose your scenario:

#### With GPU:

```bash
python tensorflow_profiler_test_guide.py --quick
```

#### CPU Only:

```bash
python cpu_tensorflow_test_guide.py --quick
```

### Expected Output:

```
⚡ Quick TensorFlow Test - Basic Profiler Functionality
✅ Quick TensorFlow test passed!
✅ Result: 0.15
✅ Peak memory: 128.45 MB
✅ Functions profiled: 1
```

If you see ✅ indicators, you're ready to go!

---

## GPU Profiling (CUDA Available)

### Prerequisites Check

First, verify your TensorFlow GPU setup:

```bash
# Check TensorFlow GPU availability
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('CUDA built:', tf.test.is_built_with_cuda())
print('GPU support:', tf.test.is_built_with_gpu_support())
"

# Check CUDA compatibility
nvidia-smi
```

### Testing the TensorFlow GPU Profiler

#### 1. Run Individual Tests

```bash
# Test 1: Basic TensorFlow function profiling
python tensorflow_profiler_test_guide.py --test 1

# Test 2: Context-based profiling
python tensorflow_profiler_test_guide.py --test 2

# Test 3: Real-time memory tracking
python tensorflow_profiler_test_guide.py --test 3

# Test 4: Memory leak detection
python tensorflow_profiler_test_guide.py --test 4

# Test 5: Complete model profiling
python tensorflow_profiler_test_guide.py --test 5

# Test 6: Keras integration
python tensorflow_profiler_test_guide.py --test 6

# Test 7: Visualization
python tensorflow_profiler_test_guide.py --test 7

# Test 8: Command line tools
python tensorflow_profiler_test_guide.py --test 8

# Test 9: Mixed precision profiling
python tensorflow_profiler_test_guide.py --test 9
```

#### 2. Run Full Test Suite

```bash
# Complete TensorFlow test suite (3-4 minutes)
python tensorflow_profiler_test_guide.py
```

### Basic TensorFlow GPU Profiling Usage

#### Function Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

# Initialize profiler
profiler = TFMemoryProfiler()

@profiler.profile_function
def tf_gpu_operation():
    with tf.device('/GPU:0'):
        x = tf.random.normal((2000, 2000))
        y = tf.linalg.matmul(x, x, transpose_b=True)
        return tf.reduce_mean(y)

# Run profiled function
result = tf_gpu_operation()

# Get results
results = profiler.get_results()
print(f"Peak GPU memory: {results.peak_memory_mb:.2f} MB")
print(f"Execution time: {results.duration:.4f} seconds")
```

#### Context Profiling with Keras

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler()

with profiler.profile_context("model_training"):
    # Create Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Training data
    x_train = tf.random.normal((1000, 784))
    y_train = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

    # Train model
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=1)

# View results
results = profiler.get_results()
for context, stats in results.function_profiles.items():
    print(f"{context}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")
```

#### Real-Time Monitoring

```python
from tfmemprof import MemoryTracker

# Setup tracker with TensorFlow-specific alerts
tracker = MemoryTracker(
    sampling_interval=0.1,     # Sample every 100ms
    alert_threshold_mb=3000,   # Alert at 3GB
    enable_alerts=True
)

# Start tracking
tracker.start_tracking()

try:
    # Your TensorFlow operations here
    for i in range(10):
        with tf.device('/GPU:0'):
            x = tf.random.normal((1000, 1000))
            y = tf.nn.conv2d(
                tf.reshape(x, (1, 1000, 1000, 1)),
                tf.random.normal((3, 3, 1, 32)),
                strides=1, padding='SAME'
            )
        print(f"TensorFlow iteration {i+1}/10")
finally:
    tracker.stop_tracking()

# Analyze results
results = tracker.get_tracking_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### TensorFlow Command Line Tools

```bash
# Show TensorFlow GPU information
python -m tfmemprof.cli info

# Real-time TensorFlow monitoring
python -m tfmemprof.cli monitor --interval 1.0 --duration 30

# Background TensorFlow tracking
python -m tfmemprof.cli track --output tf_results.json --duration 60

# Analyze TensorFlow results
python -m tfmemprof.cli analyze --input tf_results.json --detect-leaks --visualize
```

---

## CPU Profiling (No GPU Required)

### When to Use TensorFlow CPU Profiling

-   GPU not available
-   Development and testing
-   TensorFlow Lite optimization
-   CPU inference optimization
-   Educational purposes
-   CI/CD pipelines

### Testing the TensorFlow CPU Profiler

#### 1. Run Individual CPU Tests

```bash
# Test 1: Basic TensorFlow CPU profiling
python cpu_tensorflow_test_guide.py --test 1

# Test 2: CPU model training
python cpu_tensorflow_test_guide.py --test 2

# Test 3: CPU memory tracking
python cpu_tensorflow_test_guide.py --test 3

# Test 4: Memory leak simulation
python cpu_tensorflow_test_guide.py --test 4

# Test 5: TensorFlow CPU operations
python cpu_tensorflow_test_guide.py --test 5

# Test 6: Keras CPU training
python cpu_tensorflow_test_guide.py --test 6

# Test 7: Memory analysis
python cpu_tensorflow_test_guide.py --test 7
```

#### 2. Run Full CPU Test Suite

```bash
# Complete TensorFlow CPU test suite (3-4 minutes)
python cpu_tensorflow_test_guide.py
```

### Basic TensorFlow CPU Profiling Usage

#### Simple CPU Profiler

```python
import tensorflow as tf
import psutil
import time

class TFCPUMemoryProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.results = []

        # Configure TensorFlow for CPU
        tf.config.set_visible_devices([], 'GPU')

    def profile_function(self, func):
        def wrapper(*args, **kwargs):
            before_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
            start_time = time.time()

            result = func(*args, **kwargs)

            after_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
            duration = time.time() - start_time

            self.results.append({
                'function': func.__name__,
                'memory_used_mb': after_mem - before_mem,
                'duration_s': duration
            })

            return result
        return wrapper

    def get_summary(self):
        for result in self.results:
            print(f"{result['function']}: {result['duration_s']:.4f}s, {result['memory_used_mb']:.2f}MB")

# Usage
profiler = TFCPUMemoryProfiler()

@profiler.profile_function
def tf_cpu_operation():
    with tf.device('/CPU:0'):
        x = tf.random.normal((1000, 1000))
        y = tf.linalg.matmul(x, x, transpose_b=True)
        return tf.reduce_mean(y)

result = tf_cpu_operation()
profiler.get_summary()
```

#### TensorFlow CPU Model Training

```python
import tensorflow as tf

def train_tf_cpu_model():
    # Configure for CPU
    tf.config.set_visible_devices([], 'GPU')

    # Create CPU model
    with tf.device('/CPU:0'):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    print("Training TensorFlow model on CPU...")
    for epoch in range(5):
        # Generate batch data (CPU)
        x_batch = tf.random.normal((64, 784))
        y_batch = tf.random.uniform((64,), maxval=10, dtype=tf.int32)

        # Training step
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch+1}/5, Loss: {loss.numpy():.4f}")

train_tf_cpu_model()
```

---

## Testing Framework

### Automated Test Execution

Both test suites provide comprehensive TensorFlow validation:

#### GPU Test Suite Structure

```
tensorflow_profiler_test_guide.py
├── check_requirements()           # Validate TensorFlow GPU setup
├── test_1_basic_tf_profiling()   # Function decorators
├── test_2_tf_context_profiling() # Context managers
├── test_3_tf_real_time_tracking() # Background monitoring
├── test_4_tf_memory_leak_detection() # Leak simulation
├── test_5_tf_model_profiling()   # Complete training
├── test_6_tf_keras_integration() # Keras integration
├── test_7_tf_visualization()     # Charts and exports
├── test_8_tf_command_line_tools() # CLI interface
└── test_9_tf_mixed_precision()   # Mixed precision
```

#### CPU Test Suite Structure

```
cpu_tensorflow_test_guide.py
├── test_1_basic_tf_cpu_profiling()    # Function profiling
├── test_2_tf_cpu_model_training()     # Model training
├── test_3_tf_cpu_memory_tracking()    # Memory monitoring
├── test_4_tf_cpu_memory_leak_simulation() # Leak detection
├── test_5_tf_cpu_operations()         # TensorFlow ops
├── test_6_tf_keras_cpu_training()     # Keras training
└── test_7_tf_cpu_memory_analysis()    # Analysis tools
```

### Test Execution Options

```bash
# Quick validation (30 seconds)
python tensorflow_profiler_test_guide.py --quick
python cpu_tensorflow_test_guide.py --quick

# Individual tests
python tensorflow_profiler_test_guide.py --test 1
python cpu_tensorflow_test_guide.py --test 1

# Full test suites
python tensorflow_profiler_test_guide.py
python cpu_tensorflow_test_guide.py

# Help and options
python tensorflow_profiler_test_guide.py --help
```

---

## Usage Examples

### Example 1: TensorFlow Model Training Profiling

#### GPU Version

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler, MemoryVisualizer

# Initialize profiler
profiler = TFMemoryProfiler(enable_tensor_tracking=True)

# Create model
with profiler.profile_context("model_setup"):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training loop
for epoch in range(5):
    with profiler.profile_context(f"epoch_{epoch}"):

        # Data generation
        with profiler.profile_context("data_gen"):
            x_batch = tf.random.normal((32, 224, 224, 3))
            y_batch = tf.random.uniform((32,), maxval=10, dtype=tf.int32)

        # Training step
        with profiler.profile_context("train_step"):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam()
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch+1}/5, Loss: {loss.numpy():.4f}")

# Generate report
results = profiler.get_results()
print(f"\nTensorFlow training completed!")
print(f"Peak GPU memory: {results.peak_memory_mb:.2f} MB")

# Create visualizations
visualizer = MemoryVisualizer()
visualizer.plot_memory_timeline(results, save_path='tf_training_memory.png')
```

### Example 2: TensorFlow Mixed Precision Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

profiler = TFMemoryProfiler()

with profiler.profile_context("mixed_precision_training"):
    # Create model with mixed precision
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # Output in float32
    ])

    optimizer = tf.keras.optimizers.Adam()

    # Training with loss scaling
    x = tf.random.normal((128, 784))
    y = tf.random.uniform((128,), maxval=10, dtype=tf.int32)

    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)

        # Scale loss for mixed precision
        scaled_loss = optimizer.get_scaled_loss(loss)

    # Get scaled gradients and unscale
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

results = profiler.get_results()
print(f"Mixed precision memory usage: {results.peak_memory_mb:.2f} MB")

# Reset policy
tf.keras.mixed_precision.set_global_policy('float32')
```

---

## TensorFlow-Specific Features

### 1. Graph vs Eager Execution Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler()

# Eager execution (default)
with profiler.profile_context("eager_execution"):
    @tf.function
    def eager_operation():
        x = tf.random.normal((1000, 1000))
        return tf.reduce_mean(tf.square(x))

    result = eager_operation()

# Graph execution
with profiler.profile_context("graph_execution"):
    @tf.function
    def graph_operation():
        x = tf.random.normal((1000, 1000))
        return tf.reduce_mean(tf.square(x))

    result = graph_operation()

# Compare results
results = profiler.get_results()
for context, stats in results.function_profiles.items():
    print(f"{context}: {stats['total_duration']:.4f}s, {stats['total_memory_used']:.2f}MB")
```

### 2. TensorFlow Data Pipeline Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

profiler = TFMemoryProfiler()

with profiler.profile_context("data_pipeline"):
    # Create efficient data pipeline
    def create_dataset():
        dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal((10000, 784)))
        dataset = dataset.batch(64)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.cache()
        return dataset

    train_dataset = create_dataset()

    # Consume dataset
    for batch in train_dataset.take(10):
        processed = tf.nn.relu(batch)

results = profiler.get_results()
print(f"Data pipeline memory: {results.peak_memory_mb:.2f} MB")
```

### 3. Multi-GPU Strategy Profiling

```python
import tensorflow as tf
from tfmemprof import TFMemoryProfiler

if len(tf.config.list_physical_devices('GPU')) > 1:
    # Multi-GPU strategy
    strategy = tf.distribute.MirroredStrategy()

    profiler = TFMemoryProfiler()

    with profiler.profile_context("multi_gpu_training"):
        with strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

            # Distributed training
            x = tf.random.normal((128, 784))
            y = tf.random.uniform((128,), maxval=10, dtype=tf.int32)

            model.fit(x, y, epochs=1, batch_size=64)

    results = profiler.get_results()
    print(f"Multi-GPU memory: {results.peak_memory_mb:.2f} MB")
```

---

## Command Line Interface

### TensorFlow CLI Commands

```bash
# System information
tfmemprof info
# Output: TensorFlow version, GPU details, CUDA info

# Real-time monitoring
tfmemprof monitor --interval 1.0 --duration 30 --device /GPU:0
# Monitor specific GPU for 30 seconds

# Background tracking
tfmemprof track --output tf_results.json --threshold 4096
# Track with 4GB alert threshold

# Analysis with TensorFlow optimizations
tfmemprof analyze --input tf_results.json --detect-leaks --optimize --report tf_report.md
# Analyze with TensorFlow-specific optimizations
```

### CPU CLI Alternative

```bash
# For TensorFlow CPU monitoring:

# Monitor TensorFlow CPU usage
python -c "
import tensorflow as tf
import psutil
import time
tf.config.set_visible_devices([], 'GPU')
for i in range(30):
    mem = psutil.virtual_memory()
    print(f'TensorFlow CPU Memory: {mem.used/1e9:.1f}GB ({mem.percent:.1f}%)')
    time.sleep(1)
"

# Profile TensorFlow script
python -m memory_profiler your_tf_script.py
```

---

## Troubleshooting

### Common TensorFlow Issues

#### Issue 1: TensorFlow GPU Out of Memory

**Symptoms:**

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

```python
# 1. Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 2. Limit GPU memory
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)

# 3. Use mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 4. Clear session regularly
tf.keras.backend.clear_session()
```

#### Issue 2: TensorFlow Import Errors

**Symptoms:**

```
ImportError: No module named 'tfmemprof'
```

**Solutions:**

```bash
# Reinstall package
pip install -e . --force-reinstall

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Verify profiler installation
pip list | grep tfmemprof
```

#### Issue 3: TensorFlow CUDA Compatibility

**Symptoms:**

```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solutions:**

```bash
# Check CUDA version compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Install compatible TensorFlow version
pip install tensorflow==2.12.0  # For CUDA 11.8

# Check cuDNN
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Performance Benchmarks

### Expected TensorFlow Performance

#### GPU Profiling Overhead

| Operation                   | Baseline | With Profiling | Overhead |
| --------------------------- | -------- | -------------- | -------- |
| Matrix Multiply (2000x2000) | 0.008s   | 0.009s         | 12%      |
| CNN Forward Pass            | 0.025s   | 0.028s         | 12%      |
| Keras Model Training        | 1.250s   | 1.350s         | 8%       |
| Memory Tracking             | -        | 0.001s         | Minimal  |

#### TensorFlow CPU Profiling Overhead

| Operation                   | Baseline | With Profiling | Overhead |
| --------------------------- | -------- | -------------- | -------- |
| Matrix Multiply (1000x1000) | 0.150s   | 0.153s         | 2%       |
| Model Forward Pass          | 0.800s   | 0.816s         | 2%       |
| Memory Tracking             | -        | 0.001s         | Minimal  |

### TensorFlow Optimization Tips

#### For GPU:

```python
# 1. Use tf.function for graph optimization
@tf.function
def optimized_training_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 2. Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 3. Use efficient data loading
dataset = dataset.prefetch(tf.data.AUTOTUNE).cache()

# 4. Reduce profiling overhead
profiler = TFMemoryProfiler(enable_tensor_tracking=False)
```

#### For CPU:

```python
# 1. Set CPU threads
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# 2. Use smaller models for CPU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),  # Smaller layers
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Batch size optimization for CPU
batch_size = 16  # Smaller batches for CPU

# 4. Increase sampling interval
tracker = TFCPUMemoryTracker(interval=1.0)  # Sample every second
```

---

## Conclusion

This comprehensive guide provides everything needed to test and use the TensorFlow GPU Memory Profiler in both GPU and CPU environments. The profiler offers TensorFlow-specific optimizations and insights that help optimize deep learning workflows.

### Quick Reference

**TensorFlow GPU Users:**

```bash
python tensorflow_profiler_test_guide.py --quick  # Quick test
python tensorflow_profiler_test_guide.py          # Full test suite
```

**TensorFlow CPU Users:**

```bash
python cpu_tensorflow_test_guide.py --quick       # Quick test
python cpu_tensorflow_test_guide.py               # Full test suite
```

**TensorFlow-Specific Features:**

-   Mixed precision profiling
-   Graph vs Eager execution analysis
-   Keras integration
-   Multi-GPU strategy profiling
-   TensorFlow Lite optimization

### Next Steps

1. **Start Testing**: Run the appropriate test suite for your TensorFlow setup
2. **Explore Features**: Try TensorFlow-specific profiling features
3. **Integrate**: Add profiling to your existing TensorFlow projects
4. **Optimize**: Use insights to improve TensorFlow memory efficiency
5. **Contribute**: Share improvements with the TensorFlow community

For questions, issues, or contributions, visit our [GitHub repository](https://github.com/nanaagyei/gpu-memory-profiler).

---

**Happy TensorFlow Profiling!** 🚀

_Authors: Prince Agyei Tuffour and Silas Bempong_
