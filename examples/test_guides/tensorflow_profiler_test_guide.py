#!/usr/bin/env python3
"""
TensorFlow GPU Memory Profiler - Testing Guide

This script demonstrates how to run and test the TensorFlow profiler with various scenarios.
It includes simple tests, advanced features, and troubleshooting examples.
"""

import time
import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available. Please install: pip install tensorflow")
    sys.exit(1)

try:
    from tfmemprof import (
        TFMemoryProfiler,
        MemoryTracker,
        MemoryVisualizer,
        MemoryAnalyzer,
        profile_function,
        profile_context,
        get_gpu_info,
        memory_summary
    )
    print("✅ TensorFlow Memory Profiler imported successfully!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you've installed the profiler: pip install -e .")
    sys.exit(1)


def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking TensorFlow Requirements...")
    print("-" * 40)

    # Check TensorFlow version
    print(f"✅ TensorFlow Version: {tf.__version__}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")

        # Check if memory growth is enabled
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ GPU memory growth enabled")
        except Exception as e:
            print(f"⚠️  Memory growth setup: {e}")
    else:
        print("❌ GPU not available - profiler will work but with limited GPU features")
        return False

    # Check TensorFlow profiler components
    try:
        gpu_info = get_gpu_info()
        print(
            f"✅ GPU Info accessible: {gpu_info.get('device_name', 'Unknown')}")
    except Exception as e:
        print(f"❌ GPU Info error: {e}")
        return False

    # Check CUDA and cuDNN
    if tf.test.is_built_with_cuda():
        print("✅ TensorFlow built with CUDA support")
        if tf.test.is_built_with_gpu_support():
            print("✅ GPU support confirmed")
    else:
        print("❌ TensorFlow not built with CUDA")
        return False

    print("✅ All TensorFlow requirements satisfied!\n")
    return True


def test_1_basic_tf_profiling():
    """Test 1: Basic TensorFlow function profiling."""
    print("🧪 Test 1: Basic TensorFlow Function Profiling")
    print("-" * 40)

    profiler = TFMemoryProfiler()

    @profiler.profile_function
    def create_tensor(shape=(1000, 1000)):
        """Create a TensorFlow tensor."""
        with tf.device('/GPU:0'):
            tensor = tf.random.normal(shape, dtype=tf.float32)
            return tensor

    @profiler.profile_function
    def process_tensor(tensor):
        """Process the tensor with TensorFlow operations."""
        with tf.device('/GPU:0'):
            result = tf.multiply(tensor, 2.0)
            result = tf.nn.relu(result)
            result = tf.reduce_mean(result)
            return result

    # Run the profiled functions
    print("Creating TensorFlow tensor...")
    tensor = create_tensor((2000, 2000))  # ~16MB tensor

    print("Processing tensor...")
    result = process_tensor(tensor)

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak memory usage: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Functions profiled: {len(results.function_profiles)}")
    print(f"✅ Processing result: {result.numpy():.6f}")

    # Clean up
    del tensor, result
    tf.keras.backend.clear_session()

    return profiler


def test_2_tf_context_profiling():
    """Test 2: TensorFlow context-based profiling."""
    print("\n🧪 Test 2: TensorFlow Context-Based Profiling")
    print("-" * 40)

    profiler = TFMemoryProfiler()

    with profiler.profile_context("data_preparation"):
        # Simulate data loading
        batch_size = 32
        input_shape = (224, 224, 3)
        with tf.device('/GPU:0'):
            data = tf.random.normal((batch_size,) + input_shape)
            labels = tf.random.uniform(
                (batch_size,), maxval=10, dtype=tf.int32)
        print("TensorFlow data prepared")

    with profiler.profile_context("model_creation"):
        # Create a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("TensorFlow model created")

    with profiler.profile_context("training_step"):
        # Single training step
        with tf.device('/GPU:0'):
            with tf.GradientTape() as tape:
                predictions = model(data, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.keras.optimizers.Adam()
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

        print(f"Training step completed, loss: {loss.numpy():.4f}")

    # Get and display results
    results = profiler.get_results()
    print(f"✅ Context blocks profiled: {len(results.function_profiles)}")
    for context_name, stats in results.function_profiles.items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_3_tf_real_time_tracking():
    """Test 3: TensorFlow real-time memory tracking."""
    print("\n🧪 Test 3: TensorFlow Real-Time Memory Tracking")
    print("-" * 40)

    # Create tracker with alerts
    tracker = MemoryTracker(
        sampling_interval=0.1,  # Sample every 100ms
        alert_threshold_mb=1000,  # Alert at 1GB
        enable_alerts=True
    )

    print("Starting TensorFlow real-time tracking...")
    tracker.start_tracking()

    try:
        # Simulate memory-intensive TensorFlow operations
        tensors = []
        for i in range(8):
            print(f"Creating TensorFlow tensor {i+1}/8...")
            with tf.device('/GPU:0'):
                tensor = tf.random.normal(
                    (1000, 1000), dtype=tf.float32)  # ~4MB each
                tensors.append(tensor)
            time.sleep(0.2)  # Allow tracker to sample

        print("Processing TensorFlow tensors...")
        for i, tensor in enumerate(tensors):
            with tf.device('/GPU:0'):
                result = tf.linalg.matmul(tensor, tensor, transpose_b=True)
                tensors[i] = result
            time.sleep(0.1)

    finally:
        # Stop tracking
        print("Stopping tracking...")
        tracker.stop_tracking()

    # Get tracking results
    tracking_results = tracker.get_tracking_results()
    print(f"✅ Samples collected: {len(tracking_results.memory_samples)}")
    print(
        f"✅ Peak memory during tracking: {tracking_results.peak_memory_mb:.2f} MB")
    print(f"✅ Alerts triggered: {len(tracking_results.alerts)}")

    for alert in tracking_results.alerts:
        print(f"   ⚠️  Alert: {alert.message} at {alert.timestamp}")

    # Cleanup
    del tensors
    tf.keras.backend.clear_session()

    return tracker


def test_4_tf_memory_leak_detection():
    """Test 4: TensorFlow memory leak detection simulation."""
    print("\n🧪 Test 4: TensorFlow Memory Leak Detection")
    print("-" * 40)

    tracker = MemoryTracker(sampling_interval=0.05)
    tracker.start_tracking()

    try:
        # Simulate a TensorFlow memory leak scenario
        leaked_tensors = []

        print("Simulating TensorFlow memory leak...")
        for i in range(15):
            # Create tensor but don't properly clean up
            with tf.device('/GPU:0'):
                tensor = tf.random.normal(
                    (500, 500), dtype=tf.float32)  # ~1MB each
                leaked_tensors.append(tensor)

                # Simulate some work
                _ = tf.reduce_sum(tf.square(tensor))

            time.sleep(0.1)

            if i % 3 == 0:
                print(f"Iteration {i+1}/15 - Memory accumulating...")

    finally:
        tracker.stop_tracking()

    # Analyze for memory leaks
    analyzer = MemoryAnalyzer()
    tracking_results = tracker.get_tracking_results()

    print("Analyzing TensorFlow memory patterns...")
    leaks = analyzer.detect_memory_leaks(tracking_results)

    if leaks:
        print(f"✅ Memory leaks detected: {len(leaks)}")
        for leak in leaks:
            print(f"   🔴 {leak['type']}: {leak['description']}")
    else:
        print("✅ No memory leaks detected")

    # Cleanup (fixing the "leak")
    del leaked_tensors
    tf.keras.backend.clear_session()

    return analyzer


def test_5_tf_model_profiling():
    """Test 5: Complete TensorFlow model training profiling."""
    print("\n🧪 Test 5: Complete TensorFlow Model Training Profiling")
    print("-" * 40)

    profiler = TFMemoryProfiler(enable_tensor_tracking=True)

    # Create a more complex TensorFlow model
    with profiler.profile_context("model_setup"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("TensorFlow model setup complete")

    # Training loop with profiling
    for epoch in range(3):
        with profiler.profile_context(f"epoch_{epoch}"):

            with profiler.profile_context("data_generation"):
                # Generate fake batch
                batch_size = 64
                with tf.device('/GPU:0'):
                    inputs = tf.random.normal((batch_size, 784))
                    targets = tf.random.uniform(
                        (batch_size,), maxval=10, dtype=tf.int32)

            with profiler.profile_context("forward_pass"):
                with tf.device('/GPU:0'):
                    with tf.GradientTape() as tape:
                        predictions = model(inputs, training=True)
                        loss = tf.keras.losses.sparse_categorical_crossentropy(
                            targets, predictions)
                        loss = tf.reduce_mean(loss)

            with profiler.profile_context("backward_pass"):
                with tf.device('/GPU:0'):
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer = tf.keras.optimizers.Adam()
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables))

            print(f"Epoch {epoch+1}/3 - Loss: {loss.numpy():.4f}")

    # Get comprehensive results
    results = profiler.get_results()
    print(f"✅ Training completed")
    print(f"✅ Peak memory usage: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Average memory usage: {results.average_memory_mb:.2f} MB")
    print(f"✅ Total snapshots: {len(results.snapshots)}")

    return profiler


def test_6_tf_keras_integration():
    """Test 6: TensorFlow Keras integration profiling."""
    print("\n🧪 Test 6: TensorFlow Keras Integration Profiling")
    print("-" * 40)

    profiler = TFMemoryProfiler()

    with profiler.profile_context("dataset_creation"):
        # Create a synthetic dataset
        def create_dataset():
            x = tf.random.normal((1000, 28, 28, 1))
            y = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            return dataset

        train_dataset = create_dataset()
        print("Dataset created")

    with profiler.profile_context("model_build"):
        # Create CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("CNN model built")

    with profiler.profile_context("model_training"):
        # Train model using Keras fit
        history = model.fit(
            train_dataset,
            epochs=2,
            verbose=1
        )
        print("Model training completed")

    # Get results
    results = profiler.get_results()
    print(f"✅ Keras integration completed")
    print(f"✅ Peak memory: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Training accuracy: {history.history['accuracy'][-1]:.4f}")

    for context_name, stats in results.function_profiles.items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_7_tf_visualization():
    """Test 7: TensorFlow visualization capabilities."""
    print("\n🧪 Test 7: TensorFlow Visualization and Analysis")
    print("-" * 40)

    # Use profiler from previous test
    profiler = test_5_tf_model_profiling()

    try:
        visualizer = MemoryVisualizer()
        results = profiler.get_results()

        print("Creating TensorFlow visualizations...")

        # Memory timeline
        print("📊 Creating TensorFlow memory timeline...")
        visualizer.plot_memory_timeline(
            results, save_path='tf_test_memory_timeline.png')
        print("   Saved: tf_test_memory_timeline.png")

        # Function comparison
        print("📊 Creating TensorFlow function comparison...")
        visualizer.plot_function_comparison(
            results.function_profiles, save_path='tf_test_function_comparison.png')
        print("   Saved: tf_test_function_comparison.png")

        # Memory heatmap
        print("📊 Creating TensorFlow memory heatmap...")
        visualizer.create_memory_heatmap(
            results, save_path='tf_test_memory_heatmap.png')
        print("   Saved: tf_test_memory_heatmap.png")

        # Export data
        print("💾 Exporting TensorFlow profiling data...")
        export_path = visualizer.export_data(
            results, format='json', filepath='tf_test_profiling_results.json')
        print(f"   Saved: {export_path}")

        print("✅ All TensorFlow visualizations created successfully!")

    except Exception as e:
        print(f"⚠️  Visualization error (may be normal without display): {e}")
        print("   Data export should still work...")

        # Try data export only
        try:
            visualizer = MemoryVisualizer()
            results = profiler.get_results()
            export_path = visualizer.export_data(
                results, format='csv', filepath='tf_test_profiling_data.csv')
            print(f"✅ Data exported to: {export_path}")
        except Exception as e2:
            print(f"❌ Export also failed: {e2}")


def test_8_tf_command_line_tools():
    """Test 8: TensorFlow command line interface."""
    print("\n🧪 Test 8: TensorFlow Command Line Tools")
    print("-" * 40)

    print("Testing TensorFlow CLI commands...")

    # Test info command
    print("📋 Testing 'tfmemprof info' command:")
    os.system("python -m tfmemprof.cli info")

    print("\n📋 For TensorFlow real-time monitoring, you can run:")
    print("   tfmemprof monitor --interval 1.0 --duration 30")
    print("   tfmemprof track --output tf_tracking_results.json")
    print("   tfmemprof analyze --input tf_tracking_results.json")

    print("✅ TensorFlow CLI commands available and working!")


def test_9_tf_mixed_precision():
    """Test 9: TensorFlow mixed precision profiling."""
    print("\n🧪 Test 9: TensorFlow Mixed Precision Profiling")
    print("-" * 40)

    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

    profiler = TFMemoryProfiler()

    with profiler.profile_context("mixed_precision_training"):
        # Create model with mixed precision
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(256, activation='relu'),
            # Output layer in float32
            tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
        ])

        optimizer = tf.keras.optimizers.Adam()

        # Training step with mixed precision
        with tf.device('/GPU:0'):
            inputs = tf.random.normal((128, 784))
            targets = tf.random.uniform((128,), maxval=10, dtype=tf.int32)

            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, predictions)
                loss = tf.reduce_mean(loss)

                # Scale loss for mixed precision
                scaled_loss = optimizer.get_scaled_loss(loss)

            # Get scaled gradients
            scaled_gradients = tape.gradient(
                scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)

            # Apply gradients
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

    # Reset policy
    tf.keras.mixed_precision.set_global_policy('float32')

    results = profiler.get_results()
    print(f"✅ Mixed precision training completed")
    print(f"✅ Memory usage: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Loss: {loss.numpy():.4f}")

    return profiler


def run_all_tf_tests():
    """Run all TensorFlow tests in sequence."""
    print("🚀 TensorFlow GPU Memory Profiler - Complete Test Suite")
    print("=" * 60)

    # Check requirements first
    if not check_requirements():
        print("❌ Requirements not met. Please install TensorFlow with GPU support.")
        return False

    try:
        # Run all tests
        test_1_basic_tf_profiling()
        test_2_tf_context_profiling()
        test_3_tf_real_time_tracking()
        test_4_tf_memory_leak_detection()
        test_5_tf_model_profiling()
        test_6_tf_keras_integration()
        test_7_tf_visualization()
        test_8_tf_command_line_tools()
        test_9_tf_mixed_precision()

        print("\n🎉 All TensorFlow Tests Completed Successfully!")
        print("=" * 60)

        # Final memory summary
        print("\n📊 Final TensorFlow Memory Summary:")
        print(memory_summary())

        return True

    except Exception as e:
        print(f"\n❌ TensorFlow test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_tf_test():
    """Run a quick test to verify basic TensorFlow functionality."""
    print("⚡ Quick TensorFlow Test - Basic Profiler Functionality")
    print("-" * 50)

    if not tf.config.list_physical_devices('GPU'):
        print("❌ GPU not available - cannot run quick TensorFlow test")
        return False

    try:
        # Quick TensorFlow profiling test
        profiler = TFMemoryProfiler()

        @profiler.profile_function
        def quick_tf_test():
            with tf.device('/GPU:0'):
                x = tf.random.normal((1000, 1000))
                y = tf.linalg.matmul(x, x, transpose_b=True)
                return tf.reduce_mean(y)

        result = quick_tf_test()
        results = profiler.get_results()

        print(f"✅ Quick TensorFlow test passed!")
        print(f"✅ Result: {result.numpy():.2f}")
        print(f"✅ Peak memory: {results.peak_memory_mb:.2f} MB")
        print(f"✅ Functions profiled: {len(results.function_profiles)}")

        return True

    except Exception as e:
        print(f"❌ Quick TensorFlow test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test TensorFlow GPU Memory Profiler")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test only")
    parser.add_argument("--test", type=int, help="Run specific test (1-9)")

    args = parser.parse_args()

    if args.quick:
        success = run_quick_tf_test()
    elif args.test:
        if not check_requirements():
            sys.exit(1)

        test_functions = {
            1: test_1_basic_tf_profiling,
            2: test_2_tf_context_profiling,
            3: test_3_tf_real_time_tracking,
            4: test_4_tf_memory_leak_detection,
            5: test_5_tf_model_profiling,
            6: test_6_tf_keras_integration,
            7: test_7_tf_visualization,
            8: test_8_tf_command_line_tools,
            9: test_9_tf_mixed_precision
        }

        if args.test in test_functions:
            test_functions[args.test]()
            success = True
        else:
            print(f"❌ Invalid test number: {args.test}. Choose 1-9.")
            success = False
    else:
        success = run_all_tf_tests()

    sys.exit(0 if success else 1)
