#!/usr/bin/env python3
"""
CLI Demonstration Script

This script demonstrates the usage of both PyTorch and TensorFlow CLI tools
programmatically, showing how to integrate them into Python workflows.
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path


def run_cli_command(command, args=None, timeout=30):
    """Run a CLI command and return the result."""
    if args is None:
        args = []

    full_command = [command] + args

    try:
        print(f"Running: {' '.join(full_command)}")
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            print(f"✅ Command successful")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()[:200]}...")
            return result.stdout.strip()
        else:
            print(f"❌ Command failed: {result.stderr.strip()}")
            return None

    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        return None
    except FileNotFoundError:
        print(f"🚫 Command not found: {command}")
        return None
    except Exception as e:
        print(f"💥 Exception: {e}")
        return None


def demo_system_info():
    """Demonstrate system information commands."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION DEMO")
    print("="*60)

    print("\n1. PyTorch System Information:")
    print("-" * 40)
    run_cli_command("gpumemprof", ["info"])

    print("\n2. TensorFlow System Information:")
    print("-" * 40)
    run_cli_command("tfmemprof", ["info"])


def demo_monitoring():
    """Demonstrate monitoring commands."""
    print("\n" + "="*60)
    print("MONITORING DEMO")
    print("="*60)

    # Create output directory
    output_dir = Path("cli_demo_output")
    output_dir.mkdir(exist_ok=True)

    print("\n1. PyTorch Memory Monitoring (5 seconds):")
    print("-" * 45)
    pytorch_output = output_dir / "pytorch_monitoring.csv"
    run_cli_command("gpumemprof", [
        "monitor",
        "--duration", "5",
        "--interval", "0.5",
        "--output", str(pytorch_output)
    ])

    print("\n2. TensorFlow Memory Monitoring (5 seconds):")
    print("-" * 45)
    tf_output = output_dir / "tensorflow_monitoring.json"
    run_cli_command("tfmemprof", [
        "monitor",
        "--duration", "5",
        "--interval", "0.5",
        "--output", str(tf_output)
    ])


def demo_tracking():
    """Demonstrate tracking commands."""
    print("\n" + "="*60)
    print("TRACKING DEMO")
    print("="*60)

    output_dir = Path("cli_demo_output")

    print("\n1. PyTorch Memory Tracking (3 seconds):")
    print("-" * 40)
    pytorch_tracking = output_dir / "pytorch_tracking.json"

    # Start tracking in background
    try:
        process = subprocess.Popen([
            "gpumemprof", "track",
            "--duration", "3",
            "--interval", "0.2",
            "--output", str(pytorch_tracking),
            "--format", "json"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print("Tracking started...")
        time.sleep(3.5)  # Wait for tracking to complete

        if process.poll() is None:
            process.terminate()
            process.wait()

        if pytorch_tracking.exists():
            print(f"✅ Tracking results saved to {pytorch_tracking}")
        else:
            print("❌ Tracking results not found")

    except Exception as e:
        print(f"💥 Tracking failed: {e}")

    print("\n2. TensorFlow Memory Tracking (3 seconds):")
    print("-" * 40)
    tf_tracking = output_dir / "tensorflow_tracking.json"

    try:
        process = subprocess.Popen([
            "tfmemprof", "track",
            "--interval", "0.2",
            "--output", str(tf_tracking)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print("Tracking started...")
        time.sleep(3.5)  # Wait for tracking to complete

        if process.poll() is None:
            process.terminate()
            process.wait()

        if tf_tracking.exists():
            print(f"✅ Tracking results saved to {tf_tracking}")
        else:
            print("❌ Tracking results not found")

    except Exception as e:
        print(f"💥 Tracking failed: {e}")


def demo_analysis():
    """Demonstrate analysis commands."""
    print("\n" + "="*60)
    print("ANALYSIS DEMO")
    print("="*60)

    output_dir = Path("cli_demo_output")

    # Check if we have tracking results to analyze
    pytorch_tracking = output_dir / "pytorch_tracking.json"
    tf_tracking = output_dir / "tensorflow_tracking.json"

    print("\n1. PyTorch Results Analysis:")
    print("-" * 35)
    if pytorch_tracking.exists():
        run_cli_command("gpumemprof", [
            "analyze",
            str(pytorch_tracking),
            "--visualization"
        ])
    else:
        print("❌ No PyTorch tracking results found for analysis")

    print("\n2. TensorFlow Results Analysis:")
    print("-" * 35)
    if tf_tracking.exists():
        run_cli_command("tfmemprof", [
            "analyze",
            "--input", str(tf_tracking),
            "--detect-leaks",
            "--optimize"
        ])
    else:
        print("❌ No TensorFlow tracking results found for analysis")


def demo_help_commands():
    """Demonstrate help commands."""
    print("\n" + "="*60)
    print("HELP COMMANDS DEMO")
    print("="*60)

    print("\n1. PyTorch CLI Help:")
    print("-" * 25)
    run_cli_command("gpumemprof", ["--help"])

    print("\n2. TensorFlow CLI Help:")
    print("-" * 25)
    run_cli_command("tfmemprof", ["--help"])


def create_sample_data():
    """Create sample data for analysis if no real data exists."""
    output_dir = Path("cli_demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create sample PyTorch tracking data
    sample_pytorch_data = {
        "peak_memory": 2048.5,
        "average_memory": 1024.2,
        "duration": 10.0,
        "memory_usage": [512, 1024, 1536, 2048, 1536, 1024, 512],
        "timestamps": [0, 1, 2, 3, 4, 5, 6],
        "events": [
            {"timestamp": 1, "type": "allocation", "size": 512},
            {"timestamp": 3, "type": "allocation", "size": 512},
            {"timestamp": 5, "type": "deallocation", "size": 1024}
        ]
    }

    pytorch_file = output_dir / "sample_pytorch_tracking.json"
    with open(pytorch_file, 'w') as f:
        json.dump(sample_pytorch_data, f, indent=2)

    # Create sample TensorFlow tracking data
    sample_tf_data = {
        "peak_memory": 3072.0,
        "average_memory": 1536.0,
        "duration": 15.0,
        "memory_usage": [1024, 2048, 3072, 2048, 1024],
        "timestamps": [0, 3, 6, 9, 12],
        "alerts": [
            {"timestamp": 6, "type": "warning", "message": "High memory usage"}
        ]
    }

    tf_file = output_dir / "sample_tensorflow_tracking.json"
    with open(tf_file, 'w') as f:
        json.dump(sample_tf_data, f, indent=2)

    print(f"✅ Sample data created in {output_dir}")


def main():
    """Run the complete CLI demonstration."""
    print("GPU Memory Profiler CLI Demonstration")
    print("=" * 50)
    print("This script demonstrates the usage of both PyTorch and TensorFlow CLI tools.")
    print("It will run various commands and show their outputs.")

    # Check if CLI tools are available
    print("\n🔍 Checking CLI availability...")
    pytorch_available = run_cli_command("gpumemprof", ["--help"]) is not None
    tensorflow_available = run_cli_command("tfmemprof", ["--help"]) is not None

    if not pytorch_available and not tensorflow_available:
        print("❌ Neither CLI tool is available. Please install the package first:")
        print("   pip install -e .")
        return 1

    print(
        f"PyTorch CLI: {'✅ Available' if pytorch_available else '❌ Not available'}")
    print(
        f"TensorFlow CLI: {'✅ Available' if tensorflow_available else '❌ Not available'}")

    # Create sample data for analysis
    create_sample_data()

    # Run demonstrations
    demo_help_commands()
    demo_system_info()
    demo_monitoring()
    demo_tracking()
    demo_analysis()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\n📁 Output files have been saved to the 'cli_demo_output' directory.")
    print("\n📚 For detailed CLI usage, see CLI_USAGE_GUIDE.md")
    print("\n🔧 You can now use these commands in your own workflows:")
    print("   - gpumemprof monitor --duration 60 --output my_monitoring.csv")
    print("   - tfmemprof track --output my_tracking.json")
    print("   - gpumemprof analyze my_tracking.json --visualization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
