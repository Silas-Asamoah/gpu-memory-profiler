#!/usr/bin/env python3
"""
TensorFlow CPU Memory Profiler - Testing Guide (No GPU Required)

This script demonstrates how to run and test the TensorFlow memory profiler on CPU-only systems.
It focuses on CPU memory tracking and general TensorFlow profiling functionality.
"""

import time
import sys
import os
import numpy as np
import psutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print(f"✅ TensorFlow Version: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow not available. Please install: pip install tensorflow")
    sys.exit(1)

print("🔍 Checking TensorFlow System Configuration...")
print("-" * 50)
print(f"✅ TensorFlow Version: {tf.__version__}")
print(f"✅ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"✅ CPU Count: {tf.config.threading.get_inter_op_parallelism_threads()}")
print(f"✅ System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# TensorFlow CPU-compatible profiler components


class TFCPUMemoryProfiler:
    """TensorFlow CPU Memory Profiler - works without GPU."""

    def __init__(self, track_tensors=True):
        self.track_tensors = track_tensors
        self.results = []
        self.function_profiles = {}
        self.snapshots = []
        self.baseline_memory = self._get_cpu_memory()

        # Configure TensorFlow for CPU
        tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices

    def _get_cpu_memory(self):
        """Get current CPU memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def _take_snapshot(self, name="snapshot"):
        """Take a CPU memory snapshot."""
        return {
            'name': name,
            'timestamp': time.time(),
            'cpu_memory_mb': self._get_cpu_memory(),
            'system_memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_percent': psutil.virtual_memory().percent
        }

    def profile_function(self, func):
        """Decorator to profile function CPU memory usage."""
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Before execution
            before_snapshot = self._take_snapshot(f"{func_name}_before")
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # After execution
                end_time = time.time()
                after_snapshot = self._take_snapshot(f"{func_name}_after")

                # Calculate metrics
                duration = end_time - start_time
                memory_used = after_snapshot['cpu_memory_mb'] - \
                    before_snapshot['cpu_memory_mb']

                # Store results
                if func_name not in self.function_profiles:
                    self.function_profiles[func_name] = {
                        'calls': 0,
                        'total_duration': 0.0,
                        'total_memory_used': 0.0,
                        'snapshots': []
                    }

                profile = self.function_profiles[func_name]
                profile['calls'] += 1
                profile['total_duration'] += duration
                profile['total_memory_used'] += memory_used
                profile['snapshots'].extend([before_snapshot, after_snapshot])

                self.snapshots.extend([before_snapshot, after_snapshot])

                return result

            except Exception as e:
                error_snapshot = self._take_snapshot(f"{func_name}_error")
                self.snapshots.append(error_snapshot)
                raise

        return wrapper

    def profile_context(self, name="context"):
        """Context manager for profiling code blocks."""
        class ProfileContext:
            def __init__(self, profiler, context_name):
                self.profiler = profiler
                self.name = context_name

            def __enter__(self):
                self.before_snapshot = self.profiler._take_snapshot(
                    f"{self.name}_start")
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                after_snapshot = self.profiler._take_snapshot(
                    f"{self.name}_end")

                duration = end_time - self.start_time
                memory_used = after_snapshot['cpu_memory_mb'] - \
                    self.before_snapshot['cpu_memory_mb']

                if self.name not in self.profiler.function_profiles:
                    self.profiler.function_profiles[self.name] = {
                        'calls': 0,
                        'total_duration': 0.0,
                        'total_memory_used': 0.0,
                        'snapshots': []
                    }

                profile = self.profiler.function_profiles[self.name]
                profile['calls'] += 1
                profile['total_duration'] += duration
                profile['total_memory_used'] += memory_used
                profile['snapshots'].extend(
                    [self.before_snapshot, after_snapshot])

                self.profiler.snapshots.extend(
                    [self.before_snapshot, after_snapshot])

        return ProfileContext(self, name)

    def get_results(self):
        """Get profiling results."""
        if not self.snapshots:
            return {
                'peak_memory_mb': 0.0,
                'average_memory_mb': 0.0,
                'min_memory_mb': 0.0,
                'function_profiles': {},
                'snapshots': []
            }

        cpu_memories = [s['cpu_memory_mb'] for s in self.snapshots]

        return {
            'peak_memory_mb': max(cpu_memories),
            'average_memory_mb': sum(cpu_memories) / len(cpu_memories),
            'min_memory_mb': min(cpu_memories),
            'function_profiles': self.function_profiles,
            'snapshots': self.snapshots
        }


class TFCPUMemoryTracker:
    """TensorFlow CPU Memory Tracker for real-time monitoring."""

    def __init__(self, sampling_interval=0.1, alert_threshold_mb=1000):
        self.sampling_interval = sampling_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.tracking = False
        self.samples = []
        self.alerts = []

    def start_tracking(self):
        """Start background memory tracking."""
        self.tracking = True
        self.samples = []
        self.alerts = []

        def track_loop():
            while self.tracking:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                sample = {
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'system_memory_percent': psutil.virtual_memory().percent
                }

                self.samples.append(sample)

                # Check for alerts
                if memory_mb > self.alert_threshold_mb:
                    alert = {
                        'timestamp': time.time(),
                        'message': f"Memory usage exceeded {self.alert_threshold_mb}MB: {memory_mb:.1f}MB"
                    }
                    self.alerts.append(alert)
                    print(f"⚠️  {alert['message']}")

                time.sleep(self.sampling_interval)

        import threading
        self.track_thread = threading.Thread(target=track_loop, daemon=True)
        self.track_thread.start()

    def stop_tracking(self):
        """Stop background memory tracking."""
        self.tracking = False
        if hasattr(self, 'track_thread'):
            self.track_thread.join(timeout=1.0)

    def get_tracking_results(self):
        """Get tracking results."""
        if not self.samples:
            return {
                'memory_samples': [],
                'peak_memory_mb': 0.0,
                'alerts': []
            }

        memories = [s['memory_mb'] for s in self.samples]

        return {
            'memory_samples': self.samples,
            'peak_memory_mb': max(memories),
            'average_memory_mb': sum(memories) / len(memories),
            'alerts': self.alerts
        }


def test_1_basic_tf_cpu_profiling():
    """Test 1: Basic TensorFlow CPU function profiling."""
    print("\n🧪 Test 1: Basic TensorFlow CPU Function Profiling")
    print("-" * 40)

    profiler = TFCPUMemoryProfiler()

    @profiler.profile_function
    def create_tf_tensor(shape=(1000, 1000)):
        """Create a TensorFlow tensor on CPU."""
        with tf.device('/CPU:0'):
            tensor = tf.random.normal(shape, dtype=tf.float32)
            return tensor

    @profiler.profile_function
    def process_tf_tensor(tensor):
        """Process the tensor with TensorFlow operations."""
        with tf.device('/CPU:0'):
            result = tf.multiply(tensor, 2.0)
            result = tf.nn.relu(result)
            result = tf.reduce_mean(result)
            return result

    print("Creating TensorFlow tensor...")
    tensor = create_tf_tensor((1500, 1500))  # ~9MB tensor

    print("Processing tensor...")
    result = process_tf_tensor(tensor)

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Functions profiled: {len(results['function_profiles'])}")
    print(f"✅ Processing result: {result.numpy():.6f}")

    # Clean up
    del tensor, result
    tf.keras.backend.clear_session()

    return profiler


def test_2_tf_cpu_model_training():
    """Test 2: TensorFlow CPU model training profiling."""
    print("\n🧪 Test 2: TensorFlow CPU Model Training Profiling")
    print("-" * 40)

    profiler = TFCPUMemoryProfiler()

    # Create model (CPU only)
    with profiler.profile_context("model_creation"):
        with tf.device('/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    128, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        print("TensorFlow CPU model created")

    # Training loop
    for epoch in range(3):
        with profiler.profile_context(f"epoch_{epoch}"):

            with profiler.profile_context("data_generation"):
                batch_size = 32  # Smaller batch for CPU
                with tf.device('/CPU:0'):
                    inputs = tf.random.normal((batch_size, 784))
                    targets = tf.random.uniform(
                        (batch_size,), maxval=10, dtype=tf.int32)

            with profiler.profile_context("forward_pass"):
                with tf.device('/CPU:0'):
                    with tf.GradientTape() as tape:
                        predictions = model(inputs, training=True)
                        loss = tf.keras.losses.sparse_categorical_crossentropy(
                            targets, predictions)
                        loss = tf.reduce_mean(loss)

            with profiler.profile_context("backward_pass"):
                with tf.device('/CPU:0'):
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer = tf.keras.optimizers.Adam()
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables))

            print(f"Epoch {epoch+1}/3 - Loss: {loss.numpy():.4f}")

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Context blocks profiled: {len(results['function_profiles'])}")

    for context_name, stats in results['function_profiles'].items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_3_tf_cpu_memory_tracking():
    """Test 3: TensorFlow CPU memory tracking."""
    print("\n🧪 Test 3: TensorFlow CPU Memory Tracking")
    print("-" * 40)

    tracker = TFCPUMemoryTracker(
        sampling_interval=0.1,
        alert_threshold_mb=500  # Lower threshold for CPU
    )

    print("Starting TensorFlow CPU memory tracking...")
    tracker.start_tracking()

    try:
        # Simulate memory-intensive TensorFlow operations
        tensors = []
        for i in range(6):
            print(f"Creating TensorFlow tensor {i+1}/6...")
            with tf.device('/CPU:0'):
                # Create ~8MB tensors
                tensor = tf.random.normal((1000, 1000), dtype=tf.float32)
                tensors.append(tensor)
            time.sleep(0.3)

        print("Processing TensorFlow tensors...")
        for i, tensor in enumerate(tensors):
            with tf.device('/CPU:0'):
                result = tf.linalg.matmul(tensor, tensor, transpose_b=True)
                result = tf.reduce_mean(result)  # Reduce to scalar
                tensors[i] = result
            time.sleep(0.2)

    finally:
        print("Stopping tracking...")
        tracker.stop_tracking()

    # Get results
    tracking_results = tracker.get_tracking_results()
    print(f"✅ Samples collected: {len(tracking_results['memory_samples'])}")
    print(f"✅ Peak CPU memory: {tracking_results['peak_memory_mb']:.2f} MB")
    print(f"✅ Alerts triggered: {len(tracking_results['alerts'])}")

    for alert in tracking_results['alerts']:
        print(f"   ⚠️  Alert: {alert['message']}")

    # Cleanup
    del tensors
    tf.keras.backend.clear_session()

    return tracker


def test_4_tf_cpu_memory_leak_simulation():
    """Test 4: TensorFlow CPU memory leak simulation."""
    print("\n🧪 Test 4: TensorFlow CPU Memory Leak Simulation")
    print("-" * 40)

    tracker = TFCPUMemoryTracker(sampling_interval=0.05)
    tracker.start_tracking()

    try:
        # Simulate TensorFlow memory leak
        leak_tensors = []

        print("Simulating TensorFlow CPU memory leak...")
        for i in range(12):
            # Create tensors and don't clean up properly
            with tf.device('/CPU:0'):
                tensor = tf.random.normal(
                    (700, 700), dtype=tf.float32)  # ~2MB each
                leak_tensors.append(tensor)

                # Do some work
                _ = tf.reduce_sum(tf.square(tensor))

            time.sleep(0.1)

            if i % 3 == 0:
                print(f"Iteration {i+1}/12 - Memory accumulating...")

    finally:
        tracker.stop_tracking()

    # Analyze results
    tracking_results = tracker.get_tracking_results()
    samples = tracking_results['memory_samples']

    if len(samples) > 1:
        initial_memory = samples[0]['memory_mb']
        final_memory = samples[-1]['memory_mb']
        memory_growth = final_memory - initial_memory

        print(f"✅ Memory growth detected: {memory_growth:.2f} MB")
        print(f"✅ Initial memory: {initial_memory:.2f} MB")
        print(f"✅ Final memory: {final_memory:.2f} MB")

        # Simple leak detection
        if memory_growth > 20:  # 20MB growth indicates potential leak
            print("🔴 Potential TensorFlow memory leak detected!")
        else:
            print("✅ No significant memory leak detected")

    # Cleanup (fix the "leak")
    del leak_tensors
    tf.keras.backend.clear_session()

    return tracker


def test_5_tf_cpu_operations():
    """Test 5: TensorFlow CPU operations profiling."""
    print("\n🧪 Test 5: TensorFlow CPU Operations")
    print("-" * 40)

    profiler = TFCPUMemoryProfiler()

    @profiler.profile_function
    def tf_matrix_operations():
        """Perform memory-intensive TensorFlow matrix operations."""
        with tf.device('/CPU:0'):
            # Large matrix multiplication
            a = tf.random.normal((1500, 1500))
            b = tf.random.normal((1500, 1500))
            c = tf.linalg.matmul(a, b)

            # More operations
            d = tf.nn.relu(c)
            e = tf.reduce_mean(d)

            return e

    @profiler.profile_function
    def tf_convolution_operations():
        """Perform TensorFlow convolution operations."""
        with tf.device('/CPU:0'):
            # Simulate image batch
            batch_size = 8  # Smaller for CPU
            channels = 3
            height = width = 64  # Smaller images for CPU

            images = tf.random.normal((batch_size, height, width, channels))

            # Convolution layers
            conv1 = tf.keras.layers.Conv2D(
                32, kernel_size=3, padding='same', activation='relu')
            conv2 = tf.keras.layers.Conv2D(
                64, kernel_size=3, padding='same', activation='relu')
            pool = tf.keras.layers.MaxPool2D(2, 2)

            x = pool(conv1(images))
            x = pool(conv2(x))

            return tf.reduce_mean(x)

    @profiler.profile_function
    def tf_text_processing():
        """Perform TensorFlow text processing operations."""
        with tf.device('/CPU:0'):
            # Simulate text data
            vocab_size = 10000
            sequence_length = 100
            batch_size = 16

            # Create embedding layer
            embedding = tf.keras.layers.Embedding(
                vocab_size, 128, input_length=sequence_length)

            # Generate random token sequences
            sequences = tf.random.uniform(
                (batch_size, sequence_length), maxval=vocab_size, dtype=tf.int32)

            # Process through embedding
            embedded = embedding(sequences)

            # LSTM processing
            lstm = tf.keras.layers.LSTM(64)
            output = lstm(embedded)

            return tf.reduce_mean(output)

    print("Running TensorFlow matrix operations...")
    result1 = tf_matrix_operations()

    print("Running TensorFlow convolution operations...")
    result2 = tf_convolution_operations()

    print("Running TensorFlow text processing...")
    result3 = tf_text_processing()

    # Get results
    results = profiler.get_results()
    print(f"✅ Matrix result: {result1.numpy():.6f}")
    print(f"✅ Convolution result: {result2.numpy():.6f}")
    print(f"✅ Text processing result: {result3.numpy():.6f}")
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Functions profiled: {len(results['function_profiles'])}")

    for func_name, stats in results['function_profiles'].items():
        print(
            f"   - {func_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_6_tf_keras_cpu_training():
    """Test 6: TensorFlow Keras CPU training."""
    print("\n🧪 Test 6: TensorFlow Keras CPU Training")
    print("-" * 40)

    profiler = TFCPUMemoryProfiler()

    with profiler.profile_context("dataset_creation"):
        # Create a smaller synthetic dataset for CPU
        def create_cpu_dataset():
            with tf.device('/CPU:0'):
                x = tf.random.normal((500, 28, 28, 1))  # Smaller dataset
                y = tf.random.uniform((500,), maxval=10, dtype=tf.int32)
                dataset = tf.data.Dataset.from_tensor_slices((x, y))
                dataset = dataset.batch(16).prefetch(
                    tf.data.AUTOTUNE)  # Smaller batch
                return dataset

        train_dataset = create_cpu_dataset()
        print("CPU dataset created")

    with profiler.profile_context("model_build"):
        # Create smaller CNN model for CPU
        with tf.device('/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        print("CPU CNN model built")

    with profiler.profile_context("model_training"):
        # Train model using Keras fit on CPU
        with tf.device('/CPU:0'):
            history = model.fit(
                train_dataset,
                epochs=2,
                verbose=1
            )
        print("CPU model training completed")

    # Get results
    results = profiler.get_results()
    print(f"✅ Keras CPU training completed")
    print(f"✅ Peak memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Final accuracy: {history.history['accuracy'][-1]:.4f}")

    for context_name, stats in results['function_profiles'].items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_7_tf_cpu_memory_analysis():
    """Test 7: TensorFlow CPU memory usage analysis."""
    print("\n🧪 Test 7: TensorFlow CPU Memory Usage Analysis")
    print("-" * 40)

    # Run a comprehensive test
    profiler = TFCPUMemoryProfiler()

    # Multiple TensorFlow operations
    operations = [
        ("small_tensor", lambda: tf.random.normal((100, 100))),
        ("medium_tensor", lambda: tf.random.normal((500, 500))),
        ("large_tensor", lambda: tf.random.normal((1000, 1000))),
        ("matrix_multiply", lambda: tf.linalg.matmul(
            tf.random.normal((800, 800)), tf.random.normal((800, 800)))),
        ("convolution", lambda: tf.nn.conv2d(tf.random.normal((1, 64, 64, 3)),
         tf.random.normal((3, 3, 3, 32)), strides=1, padding='SAME')),
        ("dense_layer", lambda: tf.keras.layers.Dense(
            128, activation='relu')(tf.random.normal((32, 784))))
    ]

    results = {}

    for name, operation in operations:
        @profiler.profile_function
        def wrapped_operation():
            with tf.device('/CPU:0'):
                return operation()

        wrapped_operation.__name__ = name
        print(f"Running TensorFlow {name}...")
        result = wrapped_operation()

    # Analyze results
    profiler_results = profiler.get_results()

    print("\n📊 TensorFlow CPU Memory Usage Analysis:")
    print("-" * 40)

    for func_name, stats in profiler_results['function_profiles'].items():
        avg_memory = stats['total_memory_used'] / stats['calls']
        avg_time = stats['total_duration'] / stats['calls']

        print(f"  {func_name}:")
        print(f"    Memory: {avg_memory:.2f} MB")
        print(f"    Time: {avg_time:.4f}s")
        if avg_time > 0:
            print(f"    Efficiency: {avg_memory/avg_time:.2f} MB/s")

    print(
        f"\n✅ Overall peak memory: {profiler_results['peak_memory_mb']:.2f} MB")
    print(f"✅ Average memory: {profiler_results['average_memory_mb']:.2f} MB")

    return profiler


def run_tf_cpu_tests():
    """Run all TensorFlow CPU-compatible tests."""
    print("🚀 TensorFlow CPU Memory Profiler - Complete Test Suite")
    print("=" * 60)
    print("💡 Running TensorFlow CPU-only tests (no GPU required)")
    print()

    try:
        # Run all CPU tests
        test_1_basic_tf_cpu_profiling()
        test_2_tf_cpu_model_training()
        test_3_tf_cpu_memory_tracking()
        test_4_tf_cpu_memory_leak_simulation()
        test_5_tf_cpu_operations()
        test_6_tf_keras_cpu_training()
        test_7_tf_cpu_memory_analysis()

        print("\n🎉 All TensorFlow CPU Tests Completed Successfully!")
        print("=" * 60)

        # Final system memory summary
        print(f"\n📊 Final TensorFlow System Memory Summary:")
        memory = psutil.virtual_memory()
        print(f"Total System Memory: {memory.total / (1024**3):.2f} GB")
        print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
        print(f"Used Memory: {memory.used / (1024**3):.2f} GB")
        print(f"Memory Usage: {memory.percent:.1f}%")

        process = psutil.Process()
        print(
            f"Process Memory: {process.memory_info().rss / (1024**2):.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ TensorFlow test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_tf_cpu_test():
    """Quick TensorFlow CPU test to verify basic functionality."""
    print("⚡ Quick TensorFlow CPU Test - Basic Profiler Functionality")
    print("-" * 50)

    try:
        profiler = TFCPUMemoryProfiler()

        @profiler.profile_function
        def quick_tf_cpu_test():
            with tf.device('/CPU:0'):
                x = tf.random.normal((500, 500))
                y = tf.linalg.matmul(x, x, transpose_b=True)
                return tf.reduce_mean(y)

        result = quick_tf_cpu_test()
        results = profiler.get_results()

        print(f"✅ Quick TensorFlow CPU test passed!")
        print(f"✅ Result: {result.numpy():.2f}")
        print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
        print(f"✅ Functions profiled: {len(results['function_profiles'])}")

        return True

    except Exception as e:
        print(f"❌ Quick TensorFlow CPU test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test TensorFlow CPU Memory Profiler (No GPU required)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test only")
    parser.add_argument("--test", type=int, help="Run specific test (1-7)")

    args = parser.parse_args()

    if args.quick:
        success = run_quick_tf_cpu_test()
    elif args.test:
        test_functions = {
            1: test_1_basic_tf_cpu_profiling,
            2: test_2_tf_cpu_model_training,
            3: test_3_tf_cpu_memory_tracking,
            4: test_4_tf_cpu_memory_leak_simulation,
            5: test_5_tf_cpu_operations,
            6: test_6_tf_keras_cpu_training,
            7: test_7_tf_cpu_memory_analysis
        }

        if args.test in test_functions:
            test_functions[args.test]()
            success = True
        else:
            print(f"❌ Invalid test number: {args.test}. Choose 1-7.")
            success = False
    else:
        success = run_tf_cpu_tests()

    sys.exit(0 if success else 1)
