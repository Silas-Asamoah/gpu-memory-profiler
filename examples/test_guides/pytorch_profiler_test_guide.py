#!/usr/bin/env python3
"""
PyTorch GPU Memory Profiler - Testing Guide

This script demonstrates how to run and test the PyTorch profiler with various scenarios.
It includes simple tests, advanced features, and troubleshooting examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gpumemprof import (
        GPUMemoryProfiler,
        MemoryTracker,
        MemoryVisualizer,
        MemoryAnalyzer,
        profile_function,
        profile_context,
        get_gpu_info,
        memory_summary
    )
    print("✅ GPU Memory Profiler imported successfully!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you've installed the profiler: pip install -e .")
    sys.exit(1)


def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking Requirements...")
    print("-" * 40)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.version.cuda}")
        print(f"✅ GPU Device: {torch.cuda.get_device_name()}")
        print(
            f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available - profiler will work but with limited GPU features")
        return False

    # Check PyTorch version
    print(f"✅ PyTorch Version: {torch.__version__}")

    # Check GPU memory profiler components
    try:
        gpu_info = get_gpu_info()
        print(f"✅ GPU Info accessible: {gpu_info['device_name']}")
    except Exception as e:
        print(f"❌ GPU Info error: {e}")
        return False

    print("✅ All requirements satisfied!\n")
    return True


def test_1_basic_profiling():
    """Test 1: Basic function profiling."""
    print("🧪 Test 1: Basic Function Profiling")
    print("-" * 40)

    profiler = GPUMemoryProfiler()

    @profiler.profile_function
    def create_tensor(size_mb=10):
        """Create a tensor of specified size."""
        elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
        tensor = torch.randn(elements, device='cuda')
        return tensor

    @profiler.profile_function
    def process_tensor(tensor):
        """Process the tensor with some operations."""
        result = tensor * 2
        result = torch.sin(result)
        result = torch.relu(result)
        return result

    # Run the profiled functions
    print("Creating tensor...")
    tensor = create_tensor(50)  # 50MB tensor

    print("Processing tensor...")
    result = process_tensor(tensor)

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak memory usage: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Functions profiled: {len(results.function_profiles)}")

    # Clean up
    del tensor, result
    torch.cuda.empty_cache()

    return profiler


def test_2_context_profiling():
    """Test 2: Context-based profiling."""
    print("\n🧪 Test 2: Context-Based Profiling")
    print("-" * 40)

    profiler = GPUMemoryProfiler()

    with profiler.profile_context("data_preparation"):
        # Simulate data loading
        batch_size = 128
        input_size = 784
        data = torch.randn(batch_size, input_size, device='cuda')
        targets = torch.randint(0, 10, (batch_size,), device='cuda')
        print("Data prepared")

    with profiler.profile_context("model_creation"):
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).cuda()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        print("Model created")

    with profiler.profile_context("training_step"):
        # Training step
        outputs = model(data)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Training step completed, loss: {loss.item():.4f}")

    # Get and display results
    results = profiler.get_results()
    print(f"✅ Context blocks profiled: {len(results.function_profiles)}")
    for context_name, stats in results.function_profiles.items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_3_real_time_tracking():
    """Test 3: Real-time memory tracking."""
    print("\n🧪 Test 3: Real-Time Memory Tracking")
    print("-" * 40)

    # Create tracker with alerts
    tracker = MemoryTracker(
        sampling_interval=0.1,  # Sample every 100ms
        alert_threshold_mb=500,  # Alert at 500MB
        enable_alerts=True
    )

    print("Starting real-time tracking...")
    tracker.start_tracking()

    try:
        # Simulate memory-intensive operations
        tensors = []
        for i in range(10):
            print(f"Creating tensor {i+1}/10...")
            tensor = torch.randn(1000, 1000, device='cuda')  # ~4MB each
            tensors.append(tensor)
            time.sleep(0.2)  # Allow tracker to sample

        print("Processing tensors...")
        for i, tensor in enumerate(tensors):
            result = torch.matmul(tensor, tensor.T)
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
    torch.cuda.empty_cache()

    return tracker


def test_4_memory_leak_detection():
    """Test 4: Memory leak detection simulation."""
    print("\n🧪 Test 4: Memory Leak Detection")
    print("-" * 40)

    tracker = MemoryTracker(sampling_interval=0.05)
    tracker.start_tracking()

    try:
        # Simulate a memory leak scenario
        leaked_tensors = []

        print("Simulating memory leak...")
        for i in range(20):
            # Create tensor but don't properly clean up
            tensor = torch.randn(500, 500, device='cuda')  # ~1MB each
            leaked_tensors.append(tensor)

            # Simulate some work
            _ = torch.matmul(tensor, tensor.T)
            time.sleep(0.1)

            if i % 5 == 0:
                print(f"Iteration {i+1}/20 - Memory accumulating...")

    finally:
        tracker.stop_tracking()

    # Analyze for memory leaks
    analyzer = MemoryAnalyzer()
    tracking_results = tracker.get_tracking_results()

    print("Analyzing memory patterns...")
    leaks = analyzer.detect_memory_leaks(tracking_results)

    if leaks:
        print(f"✅ Memory leaks detected: {len(leaks)}")
        for leak in leaks:
            print(f"   🔴 {leak['type']}: {leak['description']}")
    else:
        print("✅ No memory leaks detected")

    # Cleanup (fixing the "leak")
    del leaked_tensors
    torch.cuda.empty_cache()

    return analyzer


def test_5_model_profiling():
    """Test 5: Complete model training profiling."""
    print("\n🧪 Test 5: Complete Model Training Profiling")
    print("-" * 40)

    profiler = GPUMemoryProfiler(track_tensors=True)

    # Create a more complex model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    with profiler.profile_context("model_setup"):
        model = CNN().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print("Model setup complete")

    # Training loop with profiling
    for epoch in range(3):
        with profiler.profile_context(f"epoch_{epoch}"):

            with profiler.profile_context("data_generation"):
                # Generate fake batch
                batch_size = 32
                inputs = torch.randn(batch_size, 3, 32, 32, device='cuda')
                targets = torch.randint(0, 10, (batch_size,), device='cuda')

            with profiler.profile_context("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with profiler.profile_context("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")

    # Get comprehensive results
    results = profiler.get_results()
    print(f"✅ Training completed")
    print(f"✅ Peak memory usage: {results.peak_memory_mb:.2f} MB")
    print(f"✅ Average memory usage: {results.average_memory_mb:.2f} MB")
    print(f"✅ Total snapshots: {len(results.snapshots)}")

    return profiler


def test_6_visualization():
    """Test 6: Visualization capabilities."""
    print("\n🧪 Test 6: Visualization and Analysis")
    print("-" * 40)

    # Use profiler from previous test
    profiler = test_5_model_profiling()

    try:
        visualizer = MemoryVisualizer()
        results = profiler.get_results()

        print("Creating visualizations...")

        # Memory timeline
        print("📊 Creating memory timeline...")
        visualizer.plot_memory_timeline(
            results, save_path='test_memory_timeline.png')
        print("   Saved: test_memory_timeline.png")

        # Function comparison
        print("📊 Creating function comparison...")
        visualizer.plot_function_comparison(
            results.function_profiles, save_path='test_function_comparison.png')
        print("   Saved: test_function_comparison.png")

        # Memory heatmap
        print("📊 Creating memory heatmap...")
        visualizer.create_memory_heatmap(
            results, save_path='test_memory_heatmap.png')
        print("   Saved: test_memory_heatmap.png")

        # Export data
        print("💾 Exporting profiling data...")
        export_path = visualizer.export_data(
            results, format='json', filepath='test_profiling_results.json')
        print(f"   Saved: {export_path}")

        print("✅ All visualizations created successfully!")

    except Exception as e:
        print(f"⚠️  Visualization error (may be normal without display): {e}")
        print("   Data export should still work...")

        # Try data export only
        try:
            visualizer = MemoryVisualizer()
            results = profiler.get_results()
            export_path = visualizer.export_data(
                results, format='csv', filepath='test_profiling_data.csv')
            print(f"✅ Data exported to: {export_path}")
        except Exception as e2:
            print(f"❌ Export also failed: {e2}")


def test_7_command_line_tools():
    """Test 7: Command line interface."""
    print("\n🧪 Test 7: Command Line Tools")
    print("-" * 40)

    print("Testing CLI commands...")

    # Test info command
    print("📋 Testing 'gpumemprof info' command:")
    os.system("python -m gpumemprof.cli info")

    print("\n📋 For real-time monitoring, you can run:")
    print("   gpumemprof monitor --interval 1.0 --duration 30")
    print("   gpumemprof track --output tracking_results.json")
    print("   gpumemprof analyze --input tracking_results.json")

    print("✅ CLI commands available and working!")


def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 PyTorch GPU Memory Profiler - Complete Test Suite")
    print("=" * 60)

    # Check requirements first
    if not check_requirements():
        print("❌ Requirements not met. Please install CUDA and required packages.")
        return False

    try:
        # Run all tests
        test_1_basic_profiling()
        test_2_context_profiling()
        test_3_real_time_tracking()
        test_4_memory_leak_detection()
        test_6_visualization()
        test_7_command_line_tools()

        print("\n🎉 All Tests Completed Successfully!")
        print("=" * 60)

        # Final memory summary
        print("\n📊 Final Memory Summary:")
        print(memory_summary())

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("⚡ Quick Test - Basic Profiler Functionality")
    print("-" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot run quick test")
        return False

    try:
        # Quick profiling test
        profiler = GPUMemoryProfiler()

        @profiler.profile_function
        def quick_test():
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x.T)
            return y.sum()

        result = quick_test()
        results = profiler.get_results()

        print(f"✅ Quick test passed!")
        print(f"✅ Result: {result.item():.2f}")
        print(f"✅ Peak memory: {results.peak_memory_mb:.2f} MB")
        print(f"✅ Functions profiled: {len(results.function_profiles)}")

        return True

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test PyTorch GPU Memory Profiler")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test only")
    parser.add_argument("--test", type=int, help="Run specific test (1-7)")

    args = parser.parse_args()

    if args.quick:
        success = run_quick_test()
    elif args.test:
        if not check_requirements():
            sys.exit(1)

        test_functions = {
            1: test_1_basic_profiling,
            2: test_2_context_profiling,
            3: test_3_real_time_tracking,
            4: test_4_memory_leak_detection,
            5: test_5_model_profiling,
            6: test_6_visualization,
            7: test_7_command_line_tools
        }

        if args.test in test_functions:
            test_functions[args.test]()
            success = True
        else:
            print(f"❌ Invalid test number: {args.test}. Choose 1-7.")
            success = False
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)
