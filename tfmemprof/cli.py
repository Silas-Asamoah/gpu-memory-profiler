"""TensorFlow Memory Profiler CLI"""

import argparse
import json
import time
import sys
import logging
from pathlib import Path

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from .utils import get_system_info, get_gpu_info, format_memory, generate_summary_report
from .tracker import MemoryTracker
from .analyzer import MemoryAnalyzer
from .visualizer import MemoryVisualizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_info(args):
    """Display system and GPU information."""
    print("TensorFlow Memory Profiler - System Information")
    print("=" * 50)

    system_info = get_system_info()

    print(f"Platform: {system_info['platform']}")
    print(f"Python Version: {system_info['python_version']}")
    print(f"TensorFlow Version: {system_info['tensorflow_version']}")
    print(f"CPU Count: {system_info['cpu_count']}")

    if 'total_memory_gb' in system_info:
        print(f"Total System Memory: {system_info['total_memory_gb']:.2f} GB")
        print(f"Available Memory: {system_info['available_memory_gb']:.2f} GB")

    print("\nGPU Information:")
    print("-" * 20)

    gpu_info = system_info.get('gpu', {})
    if gpu_info.get('available', False):
        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_info['count']}")
        print(
            f"Total GPU Memory: {format_memory(gpu_info['total_memory'] * 1024 * 1024)}")

        for i, device in enumerate(gpu_info.get('devices', [])):
            print(f"\nGPU {i}:")
            print(f"  Name: {device.get('name', 'Unknown')}")
            print(
                f"  Current Memory: {device.get('current_memory_mb', 0):.1f} MB")
            print(f"  Peak Memory: {device.get('peak_memory_mb', 0):.1f} MB")
    else:
        print("GPU Available: No")
        if 'error' in gpu_info:
            print(f"Error: {gpu_info['error']}")

    # TensorFlow specific information
    if TF_AVAILABLE:
        print(f"\nTensorFlow Build Information:")
        print("-" * 30)
        try:
            build_info = tf.sysconfig.get_build_info()
            print(f"CUDA Build: {build_info.get('is_cuda_build', 'Unknown')}")
            print(f"CUDA Version: {build_info.get('cuda_version', 'Unknown')}")
            print(
                f"cuDNN Version: {build_info.get('cudnn_version', 'Unknown')}")
        except Exception as e:
            print(f"Could not get build info: {e}")


def cmd_monitor(args):
    """Monitor GPU memory usage in real-time."""
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        return 1

    print(f"Starting TensorFlow memory monitoring...")
    print(f"Sampling interval: {args.interval} seconds")
    print(
        f"Duration: {args.duration} seconds" if args.duration else "Duration: Indefinite")
    if args.threshold:
        print(f"Alert threshold: {args.threshold} MB")
    print("Press Ctrl+C to stop\n")

    tracker = MemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device=args.device,
        enable_logging=True
    )

    try:
        tracker.start_tracking()

        start_time = time.time()
        while True:
            if args.duration and (time.time() - start_time) >= args.duration:
                break

            current_memory = tracker.get_current_memory()
            print(
                f"\rCurrent memory usage: {current_memory:.1f} MB", end="", flush=True)

            time.sleep(1.0)  # Update display every second

    except KeyboardInterrupt:
        print("\n\nStopping monitoring...")

    finally:
        results = tracker.stop_tracking()

        print("\nMonitoring Results:")
        print("-" * 20)
        print(f"Peak Memory: {results.peak_memory:.1f} MB")
        print(f"Average Memory: {results.average_memory:.1f} MB")
        print(f"Duration: {results.duration:.1f} seconds")
        print(f"Samples Collected: {len(results.memory_usage)}")

        if results.alerts_triggered:
            print(f"Alerts Triggered: {len(results.alerts_triggered)}")

        if args.output:
            # Save results
            output_data = {
                'peak_memory': results.peak_memory,
                'average_memory': results.average_memory,
                'duration': results.duration,
                'memory_usage': results.memory_usage,
                'timestamps': results.timestamps,
                'alerts': results.alerts_triggered
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {args.output}")


def cmd_track(args):
    """Start background memory tracking."""
    if not TF_AVAILABLE:
        print("Error: TensorFlow not available")
        return 1

    print("Starting background memory tracking...")

    tracker = MemoryTracker(
        sampling_interval=args.interval,
        alert_threshold_mb=args.threshold,
        device=args.device,
        enable_logging=True
    )

    # Add alert callback
    def alert_callback(alert):
        print(f"\n⚠️  MEMORY ALERT: {alert['message']}")

    tracker.add_alert_callback(alert_callback)

    try:
        tracker.start_tracking()
        print("Tracking started. Press Ctrl+C to stop and save results.")

        while True:
            time.sleep(5.0)  # Check every 5 seconds

            # Show periodic updates
            current_memory = tracker.get_current_memory()
            print(f"Current memory: {current_memory:.1f} MB")

    except KeyboardInterrupt:
        print("\nStopping tracking...")

    finally:
        results = tracker.stop_tracking()

        if args.output:
            output_data = {
                'peak_memory': results.peak_memory,
                'average_memory': results.average_memory,
                'duration': results.duration,
                'memory_usage': results.memory_usage,
                'timestamps': results.timestamps,
                'alerts': results.alerts_triggered,
                'events': results.events
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Results saved to {args.output}")

        print(
            f"\nTracking completed. Peak memory: {results.peak_memory:.1f} MB")


def cmd_analyze(args):
    """Analyze profiling results."""
    if not args.input:
        print("Error: Input file required for analysis")
        return 1

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    print(f"Analyzing results from {args.input}...")

    # Load results
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Create a simple result object for analysis
    class AnalysisResult:
        def __init__(self, data):
            self.peak_memory_mb = data.get('peak_memory', 0)
            self.average_memory_mb = data.get('average_memory', 0)
            self.min_memory_mb = min(data.get('memory_usage', [0]))
            self.total_allocations = len([m for i, m in enumerate(data.get('memory_usage', []))
                                          if i > 0 and m > data['memory_usage'][i-1]])
            self.total_deallocations = len([m for i, m in enumerate(data.get('memory_usage', []))
                                            if i > 0 and m < data['memory_usage'][i-1]])
            self.duration = data.get('duration', 0)

            # Create fake snapshots for analysis
            self.snapshots = []
            memory_usage = data.get('memory_usage', [])
            timestamps = data.get('timestamps', list(range(len(memory_usage))))

            for i, (mem, ts) in enumerate(zip(memory_usage, timestamps)):
                snapshot = type('Snapshot', (), {
                    'timestamp': ts,
                    'name': f'sample_{i}',
                    'gpu_memory_mb': mem,
                    'cpu_memory_mb': 0,
                    'gpu_memory_reserved_mb': mem * 1.1,  # Estimate
                    'gpu_utilization': min(100, mem / 1000 * 100),
                    'num_tensors': 0
                })()
                self.snapshots.append(snapshot)

    result = AnalysisResult(data)

    # Basic analysis
    print("\nBasic Analysis:")
    print("-" * 15)
    print(f"Peak Memory: {format_memory(result.peak_memory_mb * 1024 * 1024)}")
    print(
        f"Average Memory: {format_memory(result.average_memory_mb * 1024 * 1024)}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"Memory Allocations: {result.total_allocations}")
    print(f"Memory Deallocations: {result.total_deallocations}")

    if args.detect_leaks:
        print("\nMemory Leak Analysis:")
        print("-" * 22)

        analyzer = MemoryAnalyzer()

        # Create tracking result for leak detection
        class TrackingResult:
            def __init__(self, data):
                self.memory_usage = data.get('memory_usage', [])
                self.timestamps = data.get('timestamps', [])
                self.memory_growth_rate = 0
                if len(self.memory_usage) > 1 and result.duration > 0:
                    self.memory_growth_rate = (
                        self.memory_usage[-1] - self.memory_usage[0]) / result.duration

        tracking_result = TrackingResult(data)
        leaks = analyzer.detect_memory_leaks(tracking_result)

        if leaks:
            print("⚠️  Potential memory leaks detected:")
            for leak in leaks:
                print(
                    f"  - {leak['type']}: {leak['description']} (Severity: {leak['severity']})")
        else:
            print("✅ No memory leaks detected")

    if args.optimize:
        print("\nOptimization Analysis:")
        print("-" * 22)

        analyzer = MemoryAnalyzer()
        optimization = analyzer.score_optimization(result)

        print(f"Overall Score: {optimization['overall_score']:.1f}/10")
        print("\nCategory Scores:")
        for category, score in optimization['categories'].items():
            print(f"  {category}: {score:.1f}/10")

        if optimization['top_recommendations']:
            print("\nTop Recommendations:")
            for i, rec in enumerate(optimization['top_recommendations'], 1):
                print(f"  {i}. {rec}")

    if args.visualize:
        print("\nGenerating visualizations...")

        visualizer = MemoryVisualizer()

        try:
            visualizer.plot_memory_timeline(
                result, save_path="memory_timeline.png")
            print("✅ Timeline plot saved as memory_timeline.png")
        except Exception as e:
            print(f"❌ Could not generate timeline plot: {e}")

    if args.report:
        print(f"\nGenerating comprehensive report...")

        report = generate_summary_report(result)

        with open(args.report, 'w') as f:
            f.write(report)

        print(f"✅ Report saved to {args.report}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TensorFlow GPU Memory Profiler CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Info command
    info_parser = subparsers.add_parser(
        'info', help='Display system and GPU information')

    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor', help='Monitor GPU memory usage')
    monitor_parser.add_argument('--interval', type=float, default=1.0,
                                help='Sampling interval in seconds (default: 1.0)')
    monitor_parser.add_argument('--duration', type=float,
                                help='Monitoring duration in seconds (default: indefinite)')
    monitor_parser.add_argument('--threshold', type=float,
                                help='Memory alert threshold in MB')
    monitor_parser.add_argument('--device', default='/GPU:0',
                                help='TensorFlow device to monitor (default: /GPU:0)')
    monitor_parser.add_argument('--output', help='Output file for results')

    # Track command
    track_parser = subparsers.add_parser(
        'track', help='Background memory tracking')
    track_parser.add_argument('--interval', type=float, default=1.0,
                              help='Sampling interval in seconds (default: 1.0)')
    track_parser.add_argument('--threshold', type=float, default=4000,
                              help='Memory alert threshold in MB (default: 4000)')
    track_parser.add_argument('--device', default='/GPU:0',
                              help='TensorFlow device to monitor (default: /GPU:0)')
    track_parser.add_argument('--output', required=True,
                              help='Output file for tracking results')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze', help='Analyze profiling results')
    analyze_parser.add_argument('--input', required=True,
                                help='Input file with profiling results')
    analyze_parser.add_argument('--detect-leaks', action='store_true',
                                help='Detect memory leaks')
    analyze_parser.add_argument('--optimize', action='store_true',
                                help='Generate optimization recommendations')
    analyze_parser.add_argument('--visualize', action='store_true',
                                help='Generate visualization plots')
    analyze_parser.add_argument('--report',
                                help='Generate comprehensive report file')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == 'info':
        return cmd_info(args)
    elif args.command == 'monitor':
        return cmd_monitor(args)
    elif args.command == 'track':
        return cmd_track(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
