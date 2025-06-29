#!/usr/bin/env python3
"""
CPU Memory Profiler - Testing Guide (No CUDA Required)

This script demonstrates how to run and test the memory profiler on CPU-only systems.
It focuses on CPU memory tracking and general profiling functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os
import psutil

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Checking System Configuration...")
print("-" * 50)
print(f"✅ PyTorch Version: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"✅ CPU Count: {torch.get_num_threads()}")
print(f"✅ System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# CPU-compatible profiler components


class CPUMemoryProfiler:
    """CPU Memory Profiler - works without CUDA."""

    def __init__(self, track_tensors=True):
        self.track_tensors = track_tensors
        self.results = []
        self.function_profiles = {}
        self.snapshots = []
        self.baseline_memory = self._get_cpu_memory()

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


class CPUMemoryTracker:
    """CPU Memory Tracker for real-time monitoring."""

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


def test_1_basic_cpu_profiling():
    """Test 1: Basic CPU function profiling."""
    print("\n🧪 Test 1: Basic CPU Function Profiling")
    print("-" * 40)

    profiler = CPUMemoryProfiler()

    @profiler.profile_function
    def create_large_array(size_mb=50):
        """Create a large numpy array."""
        elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
        array = np.random.randn(elements)
        return array

    @profiler.profile_function
    def process_array(array):
        """Process the array with some operations."""
        result = np.sin(array) + np.cos(array)
        result = np.maximum(result, 0)  # ReLU equivalent
        return np.mean(result)

    print("Creating large array...")
    array = create_large_array(30)  # 30MB array

    print("Processing array...")
    result = process_array(array)

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Functions profiled: {len(results['function_profiles'])}")
    print(f"✅ Processing result: {result:.6f}")

    # Clean up
    del array

    return profiler


def test_2_cpu_model_training():
    """Test 2: CPU model training profiling."""
    print("\n🧪 Test 2: CPU Model Training Profiling")
    print("-" * 40)

    profiler = CPUMemoryProfiler()

    # Create model (CPU only)
    with profiler.profile_context("model_creation"):
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print("CPU model created")

    # Training loop
    for epoch in range(3):
        with profiler.profile_context(f"epoch_{epoch}"):

            with profiler.profile_context("data_generation"):
                batch_size = 64
                inputs = torch.randn(batch_size, 784)  # CPU tensors
                targets = torch.randint(0, 10, (batch_size,))

            with profiler.profile_context("forward_pass"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with profiler.profile_context("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")

    # Get results
    results = profiler.get_results()
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Context blocks profiled: {len(results['function_profiles'])}")

    for context_name, stats in results['function_profiles'].items():
        print(
            f"   - {context_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_3_cpu_memory_tracking():
    """Test 3: CPU memory tracking."""
    print("\n🧪 Test 3: CPU Memory Tracking")
    print("-" * 40)

    tracker = CPUMemoryTracker(
        sampling_interval=0.1,
        alert_threshold_mb=200  # Lower threshold for CPU
    )

    print("Starting CPU memory tracking...")
    tracker.start_tracking()

    try:
        # Simulate memory-intensive operations
        arrays = []
        for i in range(8):
            print(f"Creating array {i+1}/8...")
            # Create 20MB numpy arrays
            array = np.random.randn(20 * 1024 * 1024 // 8)
            arrays.append(array)
            time.sleep(0.3)

        print("Processing arrays...")
        for i, array in enumerate(arrays):
            result = np.fft.fft(array[:1000])  # Expensive operation on subset
            arrays[i] = result
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
    del arrays

    return tracker


def test_4_cpu_memory_leak_simulation():
    """Test 4: CPU memory leak simulation."""
    print("\n🧪 Test 4: CPU Memory Leak Simulation")
    print("-" * 40)

    tracker = CPUMemoryTracker(sampling_interval=0.05)
    tracker.start_tracking()

    try:
        # Simulate memory leak
        leak_arrays = []

        print("Simulating CPU memory leak...")
        for i in range(15):
            # Create arrays and don't clean up properly
            array = np.random.randn(5 * 1024 * 1024 // 8)  # 5MB each
            leak_arrays.append(array)

            # Do some work
            _ = np.sum(array)
            time.sleep(0.1)

            if i % 3 == 0:
                print(f"Iteration {i+1}/15 - Memory accumulating...")

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
        if memory_growth > 50:  # 50MB growth indicates potential leak
            print("🔴 Potential memory leak detected!")
        else:
            print("✅ No significant memory leak detected")

    # Cleanup (fix the "leak")
    del leak_arrays

    return tracker


def test_5_pytorch_cpu_operations():
    """Test 5: PyTorch CPU operations profiling."""
    print("\n🧪 Test 5: PyTorch CPU Operations")
    print("-" * 40)

    profiler = CPUMemoryProfiler()

    @profiler.profile_function
    def matrix_operations():
        """Perform memory-intensive matrix operations."""
        # Large matrix multiplication
        a = torch.randn(2000, 2000)
        b = torch.randn(2000, 2000)
        c = torch.matmul(a, b)

        # More operations
        d = torch.sin(c) + torch.cos(c)
        e = torch.relu(d)

        return torch.mean(e)

    @profiler.profile_function
    def convolution_operations():
        """Perform convolution operations."""
        # Simulate image batch
        batch_size = 16
        channels = 3
        height = width = 224

        images = torch.randn(batch_size, channels, height, width)

        # Convolution layers
        conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        pool = nn.MaxPool2d(2, 2)

        x = pool(torch.relu(conv1(images)))
        x = pool(torch.relu(conv2(x)))

        return torch.mean(x)

    print("Running matrix operations...")
    result1 = matrix_operations()

    print("Running convolution operations...")
    result2 = convolution_operations()

    # Get results
    results = profiler.get_results()
    print(f"✅ Matrix result: {result1:.6f}")
    print(f"✅ Convolution result: {result2:.6f}")
    print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
    print(f"✅ Functions profiled: {len(results['function_profiles'])}")

    for func_name, stats in results['function_profiles'].items():
        print(
            f"   - {func_name}: {stats['total_duration']:.3f}s, {stats['total_memory_used']:.2f} MB")

    return profiler


def test_6_memory_analysis():
    """Test 6: Memory usage analysis."""
    print("\n🧪 Test 6: Memory Usage Analysis")
    print("-" * 40)

    # Run a comprehensive test
    profiler = CPUMemoryProfiler()

    # Multiple operations
    operations = [
        ("small_tensor", lambda: torch.randn(100, 100)),
        ("medium_tensor", lambda: torch.randn(500, 500)),
        ("large_tensor", lambda: torch.randn(1000, 1000)),
        ("matrix_multiply", lambda: torch.matmul(
            torch.randn(800, 800), torch.randn(800, 800))),
        ("neural_network", lambda: nn.Sequential(nn.Linear(1000, 500),
         nn.ReLU(), nn.Linear(500, 100))(torch.randn(64, 1000)))
    ]

    results = {}

    for name, operation in operations:
        @profiler.profile_function
        def wrapped_operation():
            return operation()

        wrapped_operation.__name__ = name
        print(f"Running {name}...")
        result = wrapped_operation()

    # Analyze results
    profiler_results = profiler.get_results()

    print("\n📊 Memory Usage Analysis:")
    print("-" * 30)

    for func_name, stats in profiler_results['function_profiles'].items():
        avg_memory = stats['total_memory_used'] / stats['calls']
        avg_time = stats['total_duration'] / stats['calls']

        print(f"  {func_name}:")
        print(f"    Memory: {avg_memory:.2f} MB")
        print(f"    Time: {avg_time:.4f}s")
        print(f"    Efficiency: {avg_memory/avg_time:.2f} MB/s")

    print(
        f"\n✅ Overall peak memory: {profiler_results['peak_memory_mb']:.2f} MB")
    print(f"✅ Average memory: {profiler_results['average_memory_mb']:.2f} MB")

    return profiler


def run_cpu_tests():
    """Run all CPU-compatible tests."""
    print("🚀 CPU Memory Profiler - Complete Test Suite")
    print("=" * 60)
    print("💡 Running CPU-only tests (no CUDA required)")
    print()

    try:
        # Run all CPU tests
        test_1_basic_cpu_profiling()
        test_2_cpu_model_training()
        test_3_cpu_memory_tracking()
        test_4_cpu_memory_leak_simulation()
        test_5_pytorch_cpu_operations()
        test_6_memory_analysis()

        print("\n🎉 All CPU Tests Completed Successfully!")
        print("=" * 60)

        # Final system memory summary
        print(f"\n📊 Final System Memory Summary:")
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
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_cpu_test():
    """Quick CPU test to verify basic functionality."""
    print("⚡ Quick CPU Test - Basic Profiler Functionality")
    print("-" * 50)

    try:
        profiler = CPUMemoryProfiler()

        @profiler.profile_function
        def quick_cpu_test():
            x = torch.randn(500, 500)
            y = torch.matmul(x, x.T)
            return torch.sum(y)

        result = quick_cpu_test()
        results = profiler.get_results()

        print(f"✅ Quick CPU test passed!")
        print(f"✅ Result: {result.item():.2f}")
        print(f"✅ Peak CPU memory: {results['peak_memory_mb']:.2f} MB")
        print(f"✅ Functions profiled: {len(results['function_profiles'])}")

        return True

    except Exception as e:
        print(f"❌ Quick CPU test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test CPU Memory Profiler (No CUDA required)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test only")
    parser.add_argument("--test", type=int, help="Run specific test (1-6)")

    args = parser.parse_args()

    if args.quick:
        success = run_quick_cpu_test()
    elif args.test:
        test_functions = {
            1: test_1_basic_cpu_profiling,
            2: test_2_cpu_model_training,
            3: test_3_cpu_memory_tracking,
            4: test_4_cpu_memory_leak_simulation,
            5: test_5_pytorch_cpu_operations,
            6: test_6_memory_analysis
        }

        if args.test in test_functions:
            test_functions[args.test]()
            success = True
        else:
            print(f"❌ Invalid test number: {args.test}. Choose 1-6.")
            success = False
    else:
        success = run_cpu_tests()

    sys.exit(0 if success else 1)
