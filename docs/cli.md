[← Back to main docs](index.md)

# Command Line Interface (CLI) Guide

This guide covers all CLI commands for both PyTorch (`gpumemprof`) and TensorFlow (`tfmemprof`).

- System info, monitoring, tracking, analysis
- Output formats, troubleshooting, integration

(Full CLI details follow...)

# GPU Memory Profiler CLI Usage Guide

This guide provides comprehensive documentation for the command-line interfaces of both the PyTorch (`gpumemprof`) and TensorFlow (`tfmemprof`) GPU memory profilers.

## Table of Contents

1. [Installation](#installation)
2. [PyTorch CLI (gpumemprof)](#pytorch-cli-gpumemprof)
3. [TensorFlow CLI (tfmemprof)](#tensorflow-cli-tfmemprof)
4. [Common Use Cases](#common-use-cases)
5. [Output Formats](#output-formats)
6. [Troubleshooting](#troubleshooting)

## Installation

The CLI tools are automatically installed when you install the GPU Memory Profiler package:

```bash
pip install -e .
```

This creates two command-line tools:

- `gpumemprof` - For PyTorch GPU memory profiling
- `tfmemprof` - For TensorFlow GPU memory profiling

## PyTorch CLI (gpumemprof)

### Overview

The `gpumemprof` command provides comprehensive GPU memory profiling for PyTorch applications.

### Available Commands

#### 1. `gpumemprof info` - System Information

Display GPU and system information.

**Options:**

- `--device DEVICE` - GPU device ID (default: current device)
- `--detailed` - Show detailed information

**Examples:**

```bash
# Basic GPU information
gpumemprof info

# Detailed information for specific device
gpumemprof info --device 0 --detailed
```

**Output:**

```
GPU Memory Profiler - System Information
==================================================
Platform: macOS-14.5.0-x86_64-i386-64bit
Python Version: 3.11.0
CUDA Available: True
CUDA Version: 12.1
GPU Device Count: 1
Current Device: 0

GPU 0 Information:
  Name: Apple M2 Pro
  Total Memory: 16.00 GB
  Allocated: 0.00 GB
  Reserved: 0.00 GB
  Multiprocessors: 10
```

#### 2. `gpumemprof monitor` - Real-time Monitoring

Monitor GPU memory usage in real-time.

**Options:**

- `--device DEVICE` - GPU device ID (default: current device)
- `--duration SECONDS` - Monitoring duration in seconds (default: 10)
- `--interval SECONDS` - Sampling interval in seconds (default: 0.1)
- `--output FILE` - Output file for monitoring data
- `--format FORMAT` - Output format: csv or json (default: csv)

**Examples:**

```bash
# Monitor for 60 seconds
gpumemprof monitor --duration 60

# Monitor with custom interval and save to file
gpumemprof monitor --duration 30 --interval 0.5 --output monitoring.csv

# Monitor indefinitely (until Ctrl+C)
gpumemprof monitor
```

**Output:**

```
Starting memory monitoring for 60.0 seconds...
Device: current
Sampling interval: 0.1s
Press Ctrl+C to stop early

Elapsed: 0.0s, Current Memory: 0.00 GB
Elapsed: 5.0s, Current Memory: 0.00 GB
...
```

#### 3. `gpumemprof track` - Advanced Tracking

Real-time memory tracking with alerts and automatic cleanup.

**Options:**

- `--device DEVICE` - GPU device ID (default: current device)
- `--duration SECONDS` - Tracking duration in seconds (default: indefinite)
- `--interval SECONDS` - Sampling interval in seconds (default: 0.1)
- `--output FILE` - Output file for tracking events
- `--format FORMAT` - Output format: csv or json (default: csv)
- `--watchdog` - Enable automatic memory cleanup
- `--warning-threshold PERCENT` - Memory warning threshold percentage (default: 80)
- `--critical-threshold PERCENT` - Memory critical threshold percentage (default: 95)
- `--oom-flight-recorder` - Enable automatic OOM artifact dump bundles
- `--oom-dump-dir DIR` - Dump output directory for OOM bundles (default: `oom_dumps`)
- `--oom-buffer-size N` - Ring-buffer event count for OOM pre-failure history
- `--oom-max-dumps N` - Retain at most N OOM dump bundles (default: 5)
- `--oom-max-total-mb MB` - Retain at most MB across all OOM dump bundles (default: 256)

**Examples:**

```bash
# Basic tracking
gpumemprof track --output tracking.csv

# Tracking with alerts and watchdog
gpumemprof track --watchdog --warning-threshold 70 --critical-threshold 90

# Limited duration tracking
gpumemprof track --duration 300 --output results.json --format json

# Enable OOM flight recorder with custom retention controls
gpumemprof track --oom-flight-recorder \
  --oom-dump-dir ./oom_dumps \
  --oom-buffer-size 5000 \
  --oom-max-dumps 10 \
  --oom-max-total-mb 1024
```

**Output:**

```
Starting real-time memory tracking...
Device: current
Sampling interval: 0.1s
Warning threshold: 80.0%
Critical threshold: 95.0%
Watchdog: Enabled

Tracking started. Press Ctrl+C to stop and save results.
Current memory: 0.00 GB
Current memory: 0.00 GB
...
```

#### 4. `gpumemprof analyze` - Results Analysis

Analyze profiling results and generate reports.

**Options:**

- `input_file` - Input file with profiling results
- `--output FILE` - Output file for analysis report
- `--format FORMAT` - Output format: json or txt (default: json)
- `--visualization` - Generate visualization plots
- `--plot-dir DIR` - Directory for visualization plots (default: plots)

**Examples:**

```bash
# Analyze results
gpumemprof analyze results.json

# Generate analysis with visualizations
gpumemprof analyze results.json --visualization --plot-dir analysis_plots

# Save analysis report
gpumemprof analyze results.json --output analysis_report.txt --format txt
```

**Output:**

```
Analyzing results from results.json...

Analysis Results:
================
Peak Memory Usage: 2.45 GB
Average Memory Usage: 1.23 GB
Memory Growth Rate: 0.15 GB/s
Memory Leaks Detected: 0
Optimization Score: 8.5/10

Top Recommendations:
1. Consider using gradient checkpointing for large models
2. Implement proper tensor cleanup in training loops
3. Use mixed precision training to reduce memory usage
```

## TensorFlow CLI (tfmemprof)

### Overview

The `tfmemprof` command provides comprehensive GPU memory profiling for TensorFlow applications.

### Available Commands

#### 1. `tfmemprof info` - System Information

Display TensorFlow-specific system and GPU information.

**Options:**

- `-v, --verbose` - Enable verbose logging

**Examples:**

```bash
# Basic information
tfmemprof info

# Verbose information
tfmemprof info --verbose
```

**Output:**

```
TensorFlow Memory Profiler - System Information
==================================================
Platform: macOS-14.5.0-x86_64-i386-64bit
Python Version: 3.11.0
TensorFlow Version: 2.13.0
CPU Count: 10

GPU Information:
--------------------
GPU Available: Yes
GPU Count: 1
Total GPU Memory: 16.00 GB

GPU 0:
  Name: Apple M2 Pro
  Current Memory: 0.0 MB
  Peak Memory: 0.0 MB

TensorFlow Build Information:
------------------------------
CUDA Build: False
CUDA Version: Unknown
cuDNN Version: Unknown
```

#### 2. `tfmemprof monitor` - Real-time Monitoring

Monitor TensorFlow GPU memory usage in real-time.

**Options:**

- `--interval SECONDS` - Sampling interval in seconds (default: 1.0)
- `--duration SECONDS` - Monitoring duration in seconds (default: indefinite)
- `--threshold MB` - Memory alert threshold in MB
- `--device DEVICE` - TensorFlow device to monitor (default: /GPU:0)
- `--output FILE` - Output file for results
- `-v, --verbose` - Enable verbose logging

**Examples:**

```bash
# Basic monitoring
tfmemprof monitor

# Monitor with alerts
tfmemprof monitor --threshold 4000 --duration 60

# Monitor specific device
tfmemprof monitor --device /GPU:1 --interval 0.5
```

**Output:**

```
Starting TensorFlow memory monitoring...
Sampling interval: 1.0 seconds
Duration: 60.0 seconds
Alert threshold: 4000 MB
Press Ctrl+C to stop

Current memory usage: 0.0 MB
Current memory usage: 0.0 MB
...

Monitoring Results:
--------------------
Peak Memory: 0.0 MB
Average Memory: 0.0 MB
Duration: 60.0 seconds
Samples Collected: 60
```

#### 3. `tfmemprof track` - Background Tracking

Start background memory tracking with alerts.

**Options:**

- `--interval SECONDS` - Sampling interval in seconds (default: 1.0)
- `--threshold MB` - Memory alert threshold in MB (default: 4000)
- `--device DEVICE` - TensorFlow device to monitor (default: /GPU:0)
- `--output FILE` - Output file for tracking results (required)
- `-v, --verbose` - Enable verbose logging

**Examples:**

```bash
# Basic tracking
tfmemprof track --output tracking_results.json

# Tracking with custom threshold
tfmemprof track --threshold 6000 --output results.json

# High-frequency tracking
tfmemprof track --interval 0.1 --output detailed_tracking.json
```

**Output:**

```
Starting background memory tracking...
Tracking started. Press Ctrl+C to stop and save results.
Current memory: 0.0 MB
Current memory: 0.0 MB
...

Stopping tracking...
Results saved to tracking_results.json
```

#### 4. `tfmemprof analyze` - Results Analysis

Analyze TensorFlow profiling results with advanced features.

**Options:**

- `--input FILE` - Input file with profiling results (required)
- `--detect-leaks` - Detect memory leaks
- `--optimize` - Generate optimization recommendations
- `--visualize` - Generate visualization plots
- `--report FILE` - Generate comprehensive report file
- `-v, --verbose` - Enable verbose logging

**Examples:**

```bash
# Basic analysis
tfmemprof analyze --input results.json

# Comprehensive analysis
tfmemprof analyze --input results.json --detect-leaks --optimize --visualize

# Generate report
tfmemprof analyze --input results.json --report analysis_report.txt
```

**Output:**

```
Analyzing results from results.json...

Basic Analysis:
---------------
Peak Memory: 0.0 MB
Average Memory: 0.0 MB
Duration: 60.0 seconds
Memory Allocations: 0
Memory Deallocations: 0

Memory Leak Analysis:
---------------------
✅ No memory leaks detected

Optimization Analysis:
----------------------
Overall Score: 10.0/10

Category Scores:
  Memory Efficiency: 10.0/10
  Allocation Patterns: 10.0/10
  Cleanup Practices: 10.0/10

Top Recommendations:
1. Current memory usage is optimal
2. No optimization needed at this time
```

## Common Use Cases

### 1. Quick System Check

```bash
# Check if GPU is available and get basic info
gpumemprof info
tfmemprof info
```

### 2. Training Session Monitoring

```bash
# Monitor during training (run in separate terminal)
gpumemprof monitor --duration 3600 --output training_monitor.csv

# Or for TensorFlow
tfmemprof monitor --duration 3600 --output tf_training_monitor.json
```

### 3. Memory Leak Detection

```bash
# Track with alerts and analyze for leaks
gpumemprof track --watchdog --output session_tracking.json
gpumemprof analyze session_tracking.json

# TensorFlow version
tfmemprof track --output tf_session_tracking.json
tfmemprof analyze --input tf_session_tracking.json --detect-leaks
```

### 4. Performance Optimization

```bash
# Get optimization recommendations
gpumemprof analyze results.json --visualization
tfmemprof analyze --input results.json --optimize --visualize
```

### 5. Continuous Monitoring

```bash
# Monitor indefinitely with alerts
gpumemprof track --watchdog --warning-threshold 70 --critical-threshold 90

# TensorFlow version
tfmemprof track --threshold 6000 --output continuous_monitoring.json
```

## Output Formats

### CSV Format

The CSV output includes columns for:

- Timestamp
- GPU Memory (MB/GB)
- Reserved Memory (MB/GB)
- GPU Utilization (%)
- Temperature (°C)
- Power Draw (W)

### JSON Format

The JSON output includes structured data with:

- Summary statistics
- Time series data
- Alert events
- System information
- Analysis results

### Visualization Output

When using `--visualization`, the profiler generates:

- Memory timeline plots
- Memory distribution histograms
- Memory growth rate analysis
- Alert event markers

## Troubleshooting

### Common Issues

1. **Command not found**

   ```bash
   # Reinstall the package
   pip install -e .
   ```

2. **CUDA not available**

   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Permission errors**

   ```bash
   # Check file permissions
   chmod +x $(which gpumemprof)
   chmod +x $(which tfmemprof)
   ```

4. **Memory monitoring not working**
   ```bash
   # Check if GPU is being used
   gpumemprof info
   tfmemprof info
   ```

### Getting Help

```bash
# Get help for any command
gpumemprof --help
gpumemprof info --help
gpumemprof monitor --help

tfmemprof --help
tfmemprof info --help
tfmemprof monitor --help
```

### Verbose Logging

Enable verbose logging for debugging:

```bash
gpumemprof monitor --duration 10 2>&1 | tee debug.log
tfmemprof monitor --verbose --duration 10
```

## Integration with Scripts

### Python Script Integration

```python
import subprocess
import json

# Run monitoring and capture results
result = subprocess.run([
    'gpumemprof', 'monitor',
    '--duration', '60',
    '--output', 'results.json',
    '--format', 'json'
], capture_output=True, text=True)

# Parse results
with open('results.json', 'r') as f:
    data = json.load(f)
    print(f"Peak memory: {data['peak_memory']} MB")
```

### Shell Script Integration

```bash
#!/bin/bash

# Monitor training session
gpumemprof monitor --duration 3600 --output training_$(date +%Y%m%d_%H%M%S).csv &

# Run your training script
python train.py

# Analyze results
gpumemprof analyze training_*.csv --visualization
```

This comprehensive CLI guide covers all available commands, options, and use cases for both PyTorch and TensorFlow GPU memory profilers.
