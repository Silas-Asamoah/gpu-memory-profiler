"""Command-line interface for GPU Memory Profiler."""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional

import psutil
import torch

from .profiler import GPUMemoryProfiler
from .tracker import MemoryTracker, MemoryWatchdog
from .visualizer import MemoryVisualizer
from .analyzer import MemoryAnalyzer
from .utils import memory_summary, get_gpu_info, get_system_info, format_bytes
from .cpu_profiler import CPUMemoryProfiler, CPUMemoryTracker


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Profiler - Monitor and analyze GPU memory usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpumemprof info                          # Show GPU information
  gpumemprof monitor --duration 60         # Monitor for 60 seconds
  gpumemprof track --output tracking.csv   # Track with CSV output
  gpumemprof analyze results.json          # Analyze profiling results
        """
    )

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Info command
    info_parser = subparsers.add_parser(
        'info', help='Show GPU and system information')
    info_parser.add_argument('--device', type=int, default=None,
                             help='GPU device ID (default: current device)')
    info_parser.add_argument('--detailed', action='store_true',
                             help='Show detailed information')

    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor', help='Monitor memory usage')
    monitor_parser.add_argument('--device', type=int, default=None,
                                help='GPU device ID (default: current device)')
    monitor_parser.add_argument('--duration', type=float, default=10.0,
                                help='Monitoring duration in seconds (default: 10)')
    monitor_parser.add_argument('--interval', type=float, default=0.1,
                                help='Sampling interval in seconds (default: 0.1)')
    monitor_parser.add_argument('--output', type=str, default=None,
                                help='Output file for monitoring data')
    monitor_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                                help='Output format (default: csv)')

    # Track command
    track_parser = subparsers.add_parser(
        'track', help='Real-time memory tracking with alerts')
    track_parser.add_argument('--device', type=int, default=None,
                              help='GPU device ID (default: current device)')
    track_parser.add_argument('--duration', type=float, default=None,
                              help='Tracking duration in seconds (default: indefinite)')
    track_parser.add_argument('--interval', type=float, default=0.1,
                              help='Sampling interval in seconds (default: 0.1)')
    track_parser.add_argument('--output', type=str, default=None,
                              help='Output file for tracking events')
    track_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                              help='Output format (default: csv)')
    track_parser.add_argument('--watchdog', action='store_true',
                              help='Enable automatic memory cleanup')
    track_parser.add_argument('--warning-threshold', type=float, default=80.0,
                              help='Memory warning threshold percentage (default: 80)')
    track_parser.add_argument('--critical-threshold', type=float, default=95.0,
                              help='Memory critical threshold percentage (default: 95)')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze', help='Analyze profiling results')
    analyze_parser.add_argument(
        'input_file', help='Input file with profiling results')
    analyze_parser.add_argument('--output', type=str, default=None,
                                help='Output file for analysis report')
    analyze_parser.add_argument('--format', choices=['json', 'txt'], default='json',
                                help='Output format (default: json)')
    analyze_parser.add_argument('--visualization', action='store_true',
                                help='Generate visualization plots')
    analyze_parser.add_argument('--plot-dir', type=str, default='plots',
                                help='Directory for visualization plots (default: plots)')

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == 'info':
            cmd_info(args)
        elif args.command == 'monitor':
            cmd_monitor(args)
        elif args.command == 'track':
            cmd_track(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_info(args):
    """Handle info command."""
    print("GPU Memory Profiler - System Information")
    print("=" * 50)

    # System info
    system_info = get_system_info()
    print(f"Platform: {system_info.get('platform', 'Unknown')}")
    print(f"Python Version: {system_info.get('python_version', 'Unknown')}")
    print(f"CUDA Available: {system_info.get('cuda_available', False)}")

    if not system_info.get('cuda_available', False):
        print("CUDA is not available. Falling back to CPU-only profiling.")
        process = psutil.Process()
        with process.oneshot():
            mem = process.memory_info()
        print(f"Process RSS: {format_bytes(mem.rss)}")
        print(f"Process VMS: {format_bytes(mem.vms)}")
        print(f"CPU Count: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")
        return

    print(f"CUDA Version: {system_info.get('cuda_version', 'Unknown')}")
    print(f"GPU Device Count: {system_info.get('cuda_device_count', 0)}")
    print(f"Current Device: {system_info.get('current_device', 0)}")
    print()

    # GPU info
    device_id = args.device if args.device is not None else torch.cuda.current_device()
    gpu_info = get_gpu_info(device_id)

    print(f"GPU {device_id} Information:")
    print(f"  Name: {gpu_info.get('device_name', 'Unknown')}")
    print(
        f"  Total Memory: {gpu_info.get('total_memory', 0) / (1024**3):.2f} GB")
    print(
        f"  Allocated: {gpu_info.get('allocated_memory', 0) / (1024**3):.2f} GB")
    print(
        f"  Reserved: {gpu_info.get('reserved_memory', 0) / (1024**3):.2f} GB")
    print(f"  Multiprocessors: {gpu_info.get('multiprocessor_count', 0)}")

    if args.detailed:
        print("\nDetailed Information:")
        print("-" * 30)

        # Memory summary
        summary = memory_summary(device_id)
        print(summary)

        # Additional stats if available
        if 'nvidia_smi_info' in gpu_info:
            smi_info = gpu_info['nvidia_smi_info']
            print("\nNVIDIA-SMI Information:")
            print(
                f"  GPU Utilization: {smi_info.get('gpu_utilization_percent', 0)}%")
            print(f"  Temperature: {smi_info.get('temperature_c', 0)}°C")
            print(f"  Power Draw: {smi_info.get('power_draw_w', 0):.1f} W")


def cmd_monitor(args):
    """Handle monitor command."""
    device = args.device
    duration = args.duration
    interval = args.interval

    cuda_available = torch.cuda.is_available()

    print(f"Starting memory monitoring for {duration} seconds...")
    print(f"Mode: {'GPU' if cuda_available else 'CPU'}")
    print(f"Sampling interval: {interval}s")
    print("Press Ctrl+C to stop early")
    print()

    if cuda_available:
        profiler = GPUMemoryProfiler(device=device)
        profiler.start_monitoring(interval)
    else:
        profiler = CPUMemoryProfiler()
        profiler.start_monitoring(interval)

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Print current status every 5 seconds
            if int((time.time() - start_time)) % 5 == 0:
                if cuda_available:
                    current_mem = torch.cuda.memory_allocated(
                        profiler.device) / (1024**3)
                else:
                    current_mem = profiler._take_snapshot().rss / (1024**3)
                elapsed = time.time() - start_time
                print(
                    f"Elapsed: {elapsed:.1f}s, Current Memory: {current_mem:.2f} GB")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    finally:
        profiler.stop_monitoring()

    # Show summary
    print("\nMonitoring Summary:")
    print("-" * 30)
    summary = profiler.get_summary()
    print(f"Snapshots collected: {summary.get('snapshots_collected', 0)}")
    peak = summary.get('peak_memory_usage', 0)
    change = summary.get('memory_change_from_baseline', 0)
    unit = "GB" if cuda_available else "MB"
    divisor = 1024**3 if cuda_available else 1024**2
    print(f"Peak memory usage: {peak / divisor:.2f} {unit}")
    print(f"Memory change from baseline: {change / divisor:.2f} {unit}")

    # Save data if requested
    if args.output:
        visualizer = MemoryVisualizer(profiler)
        output_path = visualizer.export_data(
            snapshots=profiler.snapshots,
            format=args.format,
            save_path=Path(args.output).stem
        )
        print(f"Data saved to: {output_path}")


def cmd_track(args):
    """Handle track command."""
    device = args.device
    duration = args.duration
    interval = args.interval

    print(f"Starting real-time memory tracking...")
    print(f"Device: {device if device is not None else 'current'}")
    print(f"Sampling interval: {interval}s")
    print(f"Duration: {duration}s" if duration else "Duration: indefinite")
    print("Press Ctrl+C to stop")
    print()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        tracker = MemoryTracker(
            device=device,
            sampling_interval=interval,
            enable_alerts=True
        )

        # Set thresholds
        tracker.set_threshold('memory_warning_percent', args.warning_threshold)
        tracker.set_threshold('memory_critical_percent', args.critical_threshold)

        # Add alert callback
        def alert_callback(event):
            timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
            print(f"[{timestamp}] {event.event_type.upper()}: {event.context}")

        tracker.add_alert_callback(alert_callback)

        # Create watchdog if requested
        watchdog = None
        if args.watchdog:
            watchdog = MemoryWatchdog(tracker)
            print("Memory watchdog enabled - automatic cleanup activated")
    else:
        tracker = CPUMemoryTracker(sampling_interval=interval)
        watchdog = None
        print("Running CPU memory tracker (CUDA unavailable).")

    # Start tracking
    tracker.start_tracking()

    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time

            # Check duration limit
            if duration and elapsed >= duration:
                break

            # Print status every 10 seconds
            if int(elapsed) % 10 == 0:
                stats = tracker.get_statistics()
                current_mem = stats.get(
                    'current_memory_allocated', 0) / (1024**3)
                peak_mem = stats.get('peak_memory', 0) / (1024**3)
                utilization = stats.get('memory_utilization_percent', 0)
                print(
                    f"Elapsed: {elapsed:.1f}s, Memory: {current_mem:.2f} GB ({utilization:.1f}%), Peak: {peak_mem:.2f} GB")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nTracking stopped by user")

    finally:
        tracker.stop_tracking()

    # Show final statistics
    print("\nTracking Summary:")
    print("-" * 30)
    stats = tracker.get_statistics()
    divisor = 1024**3 if cuda_available else 1024**2
    unit = "GB" if cuda_available else "MB"
    print(f"Total events: {stats.get('total_events', 0)}")
    print(f"Peak memory: {stats.get('peak_memory', 0) / divisor:.2f} {unit}")

    if watchdog:
        cleanup_stats = watchdog.get_cleanup_stats()
        print(f"Automatic cleanups: {cleanup_stats.get('cleanup_count', 0)}")

    # Save events if requested
    if args.output:
        tracker.export_events(args.output, args.format)
        print(f"Events saved to: {args.output}")


def cmd_analyze(args):
    """Handle analyze command."""
    input_file = args.input_file

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        return

    print(f"Analyzing profiling results from: {input_file}")

    # Load data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Create analyzer
    analyzer = MemoryAnalyzer()

    # For now, create dummy results for demonstration
    # In a real implementation, you'd parse the loaded data
    print("Analysis functionality is available through the Python API.")
    print("Please use the Python library for detailed analysis:")
    print()
    print("Example:")
    print("from gpumemprof import MemoryAnalyzer")
    print("analyzer = MemoryAnalyzer()")
    print("patterns = analyzer.analyze_memory_patterns(results)")
    print("insights = analyzer.generate_performance_insights(results)")
    print("report = analyzer.generate_optimization_report(results)")

    # Generate basic report
    print(f"\nBasic Analysis:")
    print(f"Input file: {input_file}")
    print(f"File size: {Path(input_file).stat().st_size} bytes")

    if 'results' in data:
        print(f"Number of results: {len(data['results'])}")
    if 'snapshots' in data:
        print(f"Number of snapshots: {len(data['snapshots'])}")


if __name__ == '__main__':
    main()
