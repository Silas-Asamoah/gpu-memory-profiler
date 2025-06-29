"""
Advanced Memory Tracking Example

This example demonstrates advanced features:
- Real-time memory tracking with alerts
- Automatic memory cleanup (watchdog)
- Memory leak detection
- Performance monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from typing import List

from gpumemprof import (
    MemoryTracker,
    MemoryWatchdog,
    GPUMemoryProfiler,
    MemoryAnalyzer,
    MemoryVisualizer,
    get_gpu_info
)


class MemoryLeakyModel(nn.Module):
    """A model that intentionally has memory management issues for demonstration."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 512)
        self.cached_tensors = []  # This will cause memory leaks

    def forward(self, x):
        # Create some intermediate tensors that we "forget" to clean up
        temp = torch.randn_like(x) * 0.1
        self.cached_tensors.append(temp)  # Memory leak!

        # Keep only last 5 to prevent unlimited growth
        if len(self.cached_tensors) > 5:
            self.cached_tensors.pop(0)

        x = self.linear(x + temp)
        return torch.relu(x)


def memory_intensive_operation():
    """Simulate a memory-intensive operation."""
    # Allocate large tensors
    tensors = []
    for i in range(10):
        size = np.random.randint(50, 200)  # 50-200 MB
        elements = int(size * 1024 * 1024 / 4)  # 4 bytes per float32
        tensor = torch.randn(elements, device='cuda')
        tensors.append(tensor)
        time.sleep(0.1)  # Simulate work

    # Process tensors
    for tensor in tensors:
        # Some computation
        result = torch.sin(tensor) + torch.cos(tensor)
        result = torch.relu(result)

    # Clean up most tensors (but not all - simulate partial cleanup)
    for tensor in tensors[:-2]:  # Keep last 2 tensors
        del tensor

    return tensors[-2:]  # Return some tensors (potential leak)


def background_workload(duration=30):
    """Run a background workload that continuously uses memory."""
    start_time = time.time()
    model = MemoryLeakyModel().cuda()

    while time.time() - start_time < duration:
        # Generate random input
        batch_size = np.random.randint(32, 128)
        input_data = torch.randn(batch_size, 1024, device='cuda')

        # Forward pass
        output = model(input_data)

        # Simulate some loss computation
        loss = output.mean()

        # Random sleep
        time.sleep(np.random.uniform(0.1, 0.5))

        # Occasionally trigger memory intensive operations
        if np.random.random() < 0.3:
            leaked_tensors = memory_intensive_operation()
            # Intentionally don't clean up all tensors


def alert_handler(event):
    """Handle memory alerts."""
    timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
    print(f"\n🚨 [{timestamp}] ALERT: {event.event_type.upper()}")
    print(f"   {event.context}")
    if event.metadata:
        for key, value in event.metadata.items():
            print(f"   {key}: {value}")
    print()


def main():
    """Main demonstration function."""
    print("GPU Memory Profiler - Advanced Tracking Example")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    # Show initial GPU state
    print("Initial GPU State:")
    gpu_info = get_gpu_info()
    print(f"Device: {gpu_info['device_name']}")
    print(f"Total Memory: {gpu_info['total_memory'] / (1024**3):.2f} GB")
    print(f"Allocated: {gpu_info['allocated_memory'] / (1024**3):.2f} GB")
    print()

    print("1. Setting up Real-time Memory Tracking")
    print("-" * 40)

    # Create memory tracker with custom settings
    tracker = MemoryTracker(
        sampling_interval=0.1,  # Sample every 100ms
        max_events=50000,       # Keep up to 50k events
        enable_alerts=True
    )

    # Configure alert thresholds
    tracker.set_threshold('memory_warning_percent', 70)    # Warn at 70%
    tracker.set_threshold('memory_critical_percent', 85)   # Critical at 85%
    tracker.set_threshold('memory_leak_threshold', 50 *
                          1024 * 1024)  # 50MB threshold

    # Add alert handler
    tracker.add_alert_callback(alert_handler)

    print("Configured thresholds:")
    print(f"  Warning: {tracker.thresholds['memory_warning_percent']}%")
    print(f"  Critical: {tracker.thresholds['memory_critical_percent']}%")
    print(
        f"  Leak threshold: {tracker.thresholds['memory_leak_threshold'] / (1024**3):.3f} GB")

    print("\n2. Setting up Memory Watchdog")
    print("-" * 40)

    # Create watchdog for automatic cleanup
    watchdog = MemoryWatchdog(
        tracker=tracker,
        auto_cleanup=True,
        cleanup_threshold=0.8,    # Trigger cleanup at 80% usage
        aggressive_cleanup_threshold=0.9  # Aggressive cleanup at 90%
    )

    print("Watchdog configured:")
    print(f"  Auto cleanup: {watchdog.auto_cleanup}")
    print(f"  Cleanup threshold: {watchdog.cleanup_threshold * 100}%")
    print(
        f"  Aggressive threshold: {watchdog.aggressive_cleanup_threshold * 100}%")

    print("\n3. Starting Real-time Tracking")
    print("-" * 40)

    # Start tracking
    tracker.start_tracking()
    print("✅ Memory tracking started")

    # Start background workload in separate thread
    print("🔄 Starting background workload...")
    workload_thread = threading.Thread(target=background_workload, args=(45,))
    workload_thread.daemon = True
    workload_thread.start()

    # Monitor for 45 seconds
    monitor_duration = 45
    start_time = time.time()

    print(f"📊 Monitoring for {monitor_duration} seconds...")
    print("   Real-time statistics (updated every 5 seconds):")
    print()

    try:
        last_report_time = 0
        while time.time() - start_time < monitor_duration:
            current_time = time.time() - start_time

            # Report statistics every 5 seconds
            if current_time - last_report_time >= 5:
                stats = tracker.get_statistics()
                current_mem = stats.get(
                    'current_memory_allocated', 0) / (1024**3)
                peak_mem = stats.get('peak_memory', 0) / (1024**3)
                utilization = stats.get('memory_utilization_percent', 0)
                total_events = stats.get('total_events', 0)
                alert_count = stats.get('alert_count', 0)

                print(f"   ⏱️  {current_time:5.1f}s | "
                      f"💾 {current_mem:5.2f} GB ({utilization:5.1f}%) | "
                      f"📈 Peak: {peak_mem:5.2f} GB | "
                      f"📋 Events: {total_events:5d} | "
                      f"⚠️  Alerts: {alert_count:3d}")

                last_report_time = current_time

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")

    # Wait for background thread to finish
    workload_thread.join(timeout=5)

    # Stop tracking
    tracker.stop_tracking()
    print("\n✅ Memory tracking stopped")

    print("\n4. Analysis Results")
    print("-" * 40)

    # Final statistics
    final_stats = tracker.get_statistics()
    cleanup_stats = watchdog.get_cleanup_stats()

    print("📊 Final Statistics:")
    print(f"   Total events: {final_stats.get('total_events', 0):,}")
    print(f"   Total allocations: {final_stats.get('total_allocations', 0):,}")
    print(
        f"   Total deallocations: {final_stats.get('total_deallocations', 0):,}")
    print(
        f"   Peak memory: {final_stats.get('peak_memory', 0) / (1024**3):.2f} GB")
    print(f"   Total alerts: {final_stats.get('alert_count', 0)}")
    print(f"   Automatic cleanups: {cleanup_stats.get('cleanup_count', 0)}")
    print(
        f"   Tracking duration: {final_stats.get('tracking_duration_seconds', 0):.1f} seconds")

    # Get recent alerts
    recent_alerts = tracker.get_alerts(last_n=10)
    if recent_alerts:
        print(f"\n⚠️  Recent Alerts (last {len(recent_alerts)}):")
        for i, alert in enumerate(recent_alerts[-5:], 1):  # Show last 5
            timestamp = time.strftime(
                '%H:%M:%S', time.localtime(alert.timestamp))
            print(f"   {i}. [{timestamp}] {alert.event_type}: {alert.context}")

    print("\n5. Memory Pattern Analysis")
    print("-" * 40)

    # Analyze events for patterns
    print("🔍 Analyzing memory patterns...")

    # Convert tracker events to a format suitable for analysis
    # This is a simplified conversion - in practice you'd want more sophisticated analysis
    if tracker.events:
        # Get memory timeline data
        timeline = tracker.get_memory_timeline(interval=1.0)

        print(f"📈 Memory Timeline Analysis:")
        if timeline['allocated']:
            min_mem = min(timeline['allocated']) / (1024**3)
            max_mem = max(timeline['allocated']) / (1024**3)
            avg_mem = sum(timeline['allocated']) / \
                len(timeline['allocated']) / (1024**3)

            print(f"   Min memory: {min_mem:.2f} GB")
            print(f"   Max memory: {max_mem:.2f} GB")
            print(f"   Avg memory: {avg_mem:.2f} GB")
            print(f"   Memory range: {max_mem - min_mem:.2f} GB")

    print("\n6. Exporting Results")
    print("-" * 40)

    # Export tracking data
    try:
        print("💾 Exporting tracking events...")
        tracker.export_events('tracking_events.csv', format='csv')
        print("   ✅ Saved: tracking_events.csv")

        tracker.export_events('tracking_events.json', format='json')
        print("   ✅ Saved: tracking_events.json")

    except Exception as e:
        print(f"   ❌ Export error: {e}")

    print("\n7. Visualization")
    print("-" * 40)

    # Create some visualizations if possible
    try:
        print("📊 Generating visualizations...")

        # Get timeline data for plotting
        timeline = tracker.get_memory_timeline(interval=0.5)

        if timeline['timestamps']:
            import matplotlib.pyplot as plt

            # Create timeline plot
            plt.figure(figsize=(12, 6))
            times = [(t - timeline['timestamps'][0])
                     for t in timeline['timestamps']]
            allocated_gb = [m / (1024**3) for m in timeline['allocated']]

            plt.plot(times, allocated_gb, 'b-',
                     linewidth=2, label='Allocated Memory')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory (GB)')
            plt.title('GPU Memory Usage Timeline')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('memory_tracking_timeline.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            print("   ✅ Saved: memory_tracking_timeline.png")

    except Exception as e:
        print(f"   ⚠️  Visualization warning: {e}")

    print("\n8. Cleanup and Final State")
    print("-" * 40)

    # Force cleanup
    print("🧹 Performing final cleanup...")
    watchdog.force_cleanup(aggressive=True)

    # Show final GPU state
    final_gpu_info = get_gpu_info()
    print(f"\n📊 Final GPU State:")
    print(
        f"   Allocated: {final_gpu_info['allocated_memory'] / (1024**3):.2f} GB")
    print(
        f"   Reserved: {final_gpu_info['reserved_memory'] / (1024**3):.2f} GB")

    memory_change = (
        final_gpu_info['allocated_memory'] - gpu_info['allocated_memory']) / (1024**3)
    print(f"   Memory change: {memory_change:+.2f} GB")

    print("\n" + "=" * 60)
    print("✅ Advanced tracking example completed!")
    print("\nGenerated files:")
    print("   - tracking_events.csv")
    print("   - tracking_events.json")
    print("   - memory_tracking_timeline.png")
    print("\nThis example demonstrated:")
    print("   ✓ Real-time memory tracking with alerts")
    print("   ✓ Automatic memory management (watchdog)")
    print("   ✓ Memory leak detection")
    print("   ✓ Performance monitoring")
    print("   ✓ Data export and visualization")


if __name__ == "__main__":
    main()
