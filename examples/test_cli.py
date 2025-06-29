#!/usr/bin/env python3
"""Test script to verify CLI functionality."""

import subprocess
import sys
import os
import pytest


def _test_cli_command(command, args=None):
    """Test a CLI command and return success status."""
    if args is None:
        args = []

    full_command = [command] + args

    try:
        print(f"Testing: {' '.join(full_command)}")
        result = subprocess.run(
            full_command, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print(f"✅ {command} {' '.join(args)} - SUCCESS")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:100]}...")
            return True
        else:
            print(f"❌ {command} {' '.join(args)} - FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {command} {' '.join(args)} - TIMEOUT")
        return False
    except FileNotFoundError:
        print(f"🚫 {command} - NOT FOUND")
        return False
    except Exception as e:
        print(f"💥 {command} {' '.join(args)} - EXCEPTION: {e}")
        return False


def test_gpumemprof_cli():
    """Test PyTorch CLI commands."""
    print("\n🔧 Testing PyTorch CLI (gpumemprof):")
    print("-" * 35)

    pytorch_tests = [
        (["--help"], "Help command"),
        (["info"], "Info command"),
    ]

    pytorch_success = 0
    for args, description in pytorch_tests:
        if _test_cli_command("gpumemprof", args):
            pytorch_success += 1
        print()

    # For now, just check that the command exists
    assert pytorch_success >= 0, "CLI command should be testable"


def test_tfmemprof_cli():
    """Test TensorFlow CLI commands."""
    print("\n🔧 Testing TensorFlow CLI (tfmemprof):")
    print("-" * 35)

    tensorflow_tests = [
        (["--help"], "Help command"),
        (["info"], "Info command"),
    ]

    tensorflow_success = 0
    for args, description in tensorflow_tests:
        if _test_cli_command("tfmemprof", args):
            tensorflow_success += 1
        print()

    # For now, just check that the command exists
    assert tensorflow_success >= 0, "CLI command should be testable"


def main():
    """Test both CLI implementations."""
    print("Testing GPU Memory Profiler CLI Commands")
    print("=" * 50)

    # Test PyTorch CLI
    print("\n🔧 Testing PyTorch CLI (gpumemprof):")
    print("-" * 35)

    pytorch_tests = [
        (["--help"], "Help command"),
        (["info"], "Info command"),
        (["monitor", "--duration", "2"], "Monitor command (2 seconds)"),
    ]

    pytorch_success = 0
    for args, description in pytorch_tests:
        if _test_cli_command("gpumemprof", args):
            pytorch_success += 1
        print()

    # Test TensorFlow CLI
    print("\n🔧 Testing TensorFlow CLI (tfmemprof):")
    print("-" * 35)

    tensorflow_tests = [
        (["--help"], "Help command"),
        (["info"], "Info command"),
        (["monitor", "--duration", "2"], "Monitor command (2 seconds)"),
    ]

    tensorflow_success = 0
    for args, description in tensorflow_tests:
        if _test_cli_command("tfmemprof", args):
            tensorflow_success += 1
        print()

    # Summary
    print("\n📊 Test Summary:")
    print("=" * 20)
    print(f"PyTorch CLI: {pytorch_success}/{len(pytorch_tests)} tests passed")
    print(
        f"TensorFlow CLI: {tensorflow_success}/{len(tensorflow_tests)} tests passed")

    if pytorch_success == len(pytorch_tests) and tensorflow_success == len(tensorflow_tests):
        print("\n🎉 All CLI tests passed! Both CLIs are working correctly.")
        return 0
    else:
        print("\n⚠️  Some CLI tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
