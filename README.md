# GPU Memory Profiler

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/nanaagyei/gpu-memory-profiler/actions)
[![PyPI Version](https://img.shields.io/pypi/v/gpu-memory-profiler.svg)](https://pypi.org/project/gpu-memory-profiler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4+-orange.svg)](https://tensorflow.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

A production-ready, open source tool for real-time GPU memory profiling, leak detection, and optimization in PyTorch and TensorFlow deep learning workflows.

## Why use GPU Memory Profiler?

-   **Prevent Out-of-Memory Crashes**: Catch memory leaks and inefficiencies before they crash your training.
-   **Optimize Model Performance**: Get actionable insights and recommendations for memory usage.
-   **Works with PyTorch & TensorFlow**: Unified interface for both major frameworks.
-   **Beautiful Visualizations**: Timeline plots, heatmaps, and interactive dashboards.
-   **CLI & API**: Use from Python or the command line.

## Features

-   Real-time GPU memory monitoring
-   Memory leak detection & alerts
-   Interactive and static visualizations
-   Context-aware profiling (decorators, context managers)
-   CLI tools for automation
-   Data export (CSV, JSON)
-   CPU compatibility mode

## Installation

### From PyPI (when released)

```bash
# Basic installation
pip install gpu-memory-profiler

# With optional dependencies
pip install gpu-memory-profiler[dev]    # Development tools
pip install gpu-memory-profiler[test]   # Testing dependencies
pip install gpu-memory-profiler[docs]   # Documentation tools
```

### From Source

```bash
git clone https://github.com/nanaagyei/gpu-memory-profiler.git
cd gpu-memory-profiler

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with testing dependencies
pip install -e .[test]
```

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/nanaagyei/gpu-memory-profiler.git
cd gpu-memory-profiler
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev,test]
pre-commit install
```

## Quick Start

### PyTorch Example

```python
from gpumemprof import GPUMemoryProfiler
profiler = GPUMemoryProfiler()
@profiler.profile_function
def train_step(model, data, target):
    output = model(data)
    loss = ...
    loss.backward()
    return loss
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

### TensorFlow Example

```python
from tfmemprof import TensorFlowProfiler
profiler = TensorFlowProfiler()
with profiler.profile_context("training"):
    model.fit(x_train, y_train, epochs=5)
results = profiler.get_results()
print(f"Peak memory: {results.peak_memory_mb:.2f} MB")
```

## Documentation

-   **[Full Documentation & Guides](docs/index.md)**
-   [CLI Usage](docs/cli.md)
-   [CPU Compatibility](docs/cpu_compatibility.md)
-   [Testing Guides](docs/pytorch_testing_guide.md), [TensorFlow](docs/tensorflow_testing_guide.md)
-   [In-depth Article](docs/article.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[MIT License](LICENSE)

---

**Version:** 0.1.0
