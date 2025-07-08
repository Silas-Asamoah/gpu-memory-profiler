#!/usr/bin/env python3
"""
TensorFlow Basic Profiling Example

This example demonstrates the basic usage of the TensorFlow GPU Memory Profiler
including function profiling, context managers, model training, and visualization.
"""

from tfmemprof.context_profiler import profile_function, profile_context
from tfmemprof import TFMemoryProfiler, MemoryVisualizer, MemoryAnalyzer
import os
import sys
import time
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    # Enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found, using CPU")

except ImportError:
    print("TensorFlow not installed. Please install TensorFlow to run this example.")
    sys.exit(1)


def main():
    """Main example execution."""
    print("TensorFlow GPU Memory Profiling Example")
    print("=" * 50)

    # Initialize profiler
    profiler = TFMemoryProfiler(enable_tensor_tracking=True)
    visualizer = MemoryVisualizer()
    analyzer = MemoryAnalyzer()

    print("\n1. Basic Function Profiling")
    print("-" * 30)

    @profile_function
    def create_large_tensor():
        """Create a large tensor for demonstration."""
        print("  Creating large tensor...")
        x = tf.random.normal([2000, 2000], dtype=tf.float32)
        y = tf.random.normal([2000, 2000], dtype=tf.float32)
        z = tf.matmul(x, y)
        return z

    @profile_function
    def tensor_operations():
        """Perform various tensor operations."""
        print("  Performing tensor operations...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])

        # Multiple operations
        c = tf.add(a, b)
        d = tf.multiply(c, 2.0)
        e = tf.reduce_mean(d)

        return e

    # Execute profiled functions
    result1 = create_large_tensor()
    result2 = tensor_operations()

    print(f"  Large tensor shape: {result1.shape}")
    print(f"  Operation result: {result2.numpy():.4f}")

    print("\n2. Context Manager Profiling")
    print("-" * 32)

    with profile_context("data_loading"):
        print("  Loading and preprocessing data...")
        # Simulate data loading
        dataset_size = 1000
        x_data = tf.random.normal([dataset_size, 784])
        y_data = tf.random.uniform([dataset_size], maxval=10, dtype=tf.int32)
        y_data = tf.one_hot(y_data, depth=10)

    with profile_context("model_creation"):
        print("  Creating neural network model...")

        # Create a simple neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    print("\n3. Training Loop Profiling")
    print("-" * 30)

    @profile_function
    def training_step(model, x_batch, y_batch):
        """Single training step."""
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.categorical_crossentropy(
                y_batch, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        return loss

    # Create dataset
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.batch(batch_size).take(10)  # Only 10 batches for demo

    print("  Training model for 10 batches...")

    for i, (x_batch, y_batch) in enumerate(dataset):
        with profile_context(f"training_batch_{i}"):
            loss = training_step(model, x_batch, y_batch)
            print(f"    Batch {i+1}: Loss = {loss.numpy():.4f}")

    print("\n4. Memory-Intensive Operations")
    print("-" * 35)

    @profile_function
    def memory_intensive_operation():
        """Demonstrate memory-intensive operations."""
        print("  Creating multiple large tensors...")

        tensors = []
        for i in range(5):
            tensor = tf.random.normal([500, 500, 10])
            tensors.append(tensor)
            print(f"    Created tensor {i+1}: {tensor.shape}")

        # Perform operations on tensors
        combined = tf.concat(tensors, axis=2)
        result = tf.reduce_mean(combined)

        return result, tensors

    result, tensors = memory_intensive_operation()
    print(f"  Combined operation result: {result.numpy():.4f}")

    print("\n5. Profiling Results Analysis")
    print("-" * 32)

    # Get comprehensive results
    results = profiler.get_results()

    print(f"Peak Memory Usage: {results.peak_memory_mb:.2f} MB")
    print(f"Average Memory Usage: {results.average_memory_mb:.2f} MB")
    print(f"Total Profiling Duration: {results.duration:.2f} seconds")
    print(f"Total Memory Allocations: {results.total_allocations}")
    print(f"Total Memory Deallocations: {results.total_deallocations}")

    # Function profiling breakdown
    if results.function_profiles:
        print("\nFunction Profiling Breakdown:")
        print("-" * 30)

        for func_name, profile in results.function_profiles.items():
            avg_memory = profile['total_memory_used'] / \
                profile['calls'] if profile['calls'] > 0 else 0
            avg_duration = profile['total_duration'] / \
                profile['calls'] if profile['calls'] > 0 else 0

            print(f"  {func_name}:")
            print(f"    Calls: {profile['calls']}")
            print(f"    Peak Memory: {profile['peak_memory']:.2f} MB")
            print(f"    Avg Memory per Call: {avg_memory:.2f} MB")
            print(f"    Avg Duration per Call: {avg_duration:.4f} s")
            print()

    print("\n6. Memory Analysis")
    print("-" * 18)

    # Analyze efficiency
    efficiency_score = analyzer.analyze_efficiency(results)
    print(f"Memory Efficiency Score: {efficiency_score:.1f}/10")

    # Get optimization suggestions
    suggestions = analyzer.suggest_optimizations(results)
    if suggestions:
        print("\nOptimization Suggestions:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"  {i}. {suggestion}")

    print("\n7. Visualization")
    print("-" * 15)

    try:
        print("  Generating memory timeline plot...")
        visualizer.plot_memory_timeline(
            results, save_path="tf_memory_timeline.png")
        print("  ✓ Timeline saved as 'tf_memory_timeline.png'")

        if results.function_profiles:
            print("  Generating function comparison plot...")
            visualizer.plot_function_comparison(results.function_profiles,
                                                save_path="tf_function_comparison.png")
            print("  ✓ Function comparison saved as 'tf_function_comparison.png'")

        print("  Generating memory heatmap...")
        visualizer.create_memory_heatmap(
            results, save_path="tf_memory_heatmap.png")
        print("  ✓ Heatmap saved as 'tf_memory_heatmap.png'")

        # Export data
        print("  Exporting profiling data...")
        visualizer.export_data(results, "tf_profiling_data.csv", format="csv")
        visualizer.export_data(
            results, "tf_profiling_data.json", format="json")
        print("  ✓ Data exported to CSV and JSON files")

    except Exception as e:
        print(f"  ✗ Visualization error: {e}")
        print("  Note: Make sure matplotlib and plotly are installed for visualizations")

    print("\n8. Advanced Analysis")
    print("-" * 19)

    # Performance correlation analysis
    correlation = analyzer.correlate_with_performance(results)

    if correlation['function_efficiency']:
        print("Function Efficiency Analysis:")
        for func_name, efficiency in correlation['function_efficiency'].items():
            print(f"  {func_name}:")
            print(
                f"    Efficiency Score: {efficiency['efficiency_score']:.3f}")
            print(
                f"    Avg Memory/Call: {efficiency['avg_memory_per_call']:.2f} MB")
            print(
                f"    Avg Duration/Call: {efficiency['avg_duration_per_call']:.4f} s")

    # Optimization scoring
    optimization = analyzer.score_optimization(results)
    print(f"\nOptimization Scoring:")
    print(f"  Overall Score: {optimization['overall_score']:.1f}/10")
    print(f"  Categories:")
    for category, score in optimization['categories'].items():
        print(f"    {category}: {score:.1f}/10")

    print("\n9. Memory Cleanup")
    print("-" * 17)

    # Manual cleanup
    del tensors, result1, result2, x_data, y_data, model

    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    # Force garbage collection
    import gc
    gc.collect()

    print("  ✓ Manual cleanup completed")

    # Capture final memory state
    final_snapshot = profiler.capture_snapshot("final_cleanup")
    print(f"  Final memory usage: {final_snapshot.gpu_memory_mb:.2f} MB")

    print("\n" + "=" * 50)
    print("TensorFlow Memory Profiling Example Completed!")
    print("=" * 50)

    print(f"\nSummary:")
    print(f"  Peak Memory: {results.peak_memory_mb:.2f} MB")
    print(f"  Efficiency Score: {efficiency_score:.1f}/10")
    print(f"  Functions Profiled: {len(results.function_profiles)}")
    print(f"  Snapshots Captured: {len(results.snapshots)}")

    print(f"\nFiles Generated:")
    print(f"  - tf_memory_timeline.png (timeline plot)")
    print(f"  - tf_function_comparison.png (function comparison)")
    print(f"  - tf_memory_heatmap.png (memory heatmap)")
    print(f"  - tf_profiling_data.csv (profiling data)")
    print(f"  - tf_profiling_data.json (profiling data)")


if __name__ == "__main__":
    main()
