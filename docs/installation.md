[← Back to main docs](index.md)

# Installation Guide

This guide covers different installation methods for GPU Memory Profiler.

## Prerequisites

-   Python 3.10 or higher
-   pip (Python package installer)
-   Git (for source installation)

## Installation Methods

### 1. From PyPI

> **Note:** The package is not yet published on PyPI. Use the source
> installation below until the first release is available.

```bash
# Basic installation (includes PyTorch and TensorFlow)
pip install gpu-memory-profiler

# With visualization support (matplotlib, plotly, dash)
pip install gpu-memory-profiler[viz]

# With interactive TUI dashboard (textual)
pip install gpu-memory-profiler[tui]

# With optional dependencies
pip install gpu-memory-profiler[dev]    # Development tools
pip install gpu-memory-profiler[test]   # Testing dependencies
pip install gpu-memory-profiler[docs]   # Documentation tools (sphinx)
```

### 2. From Source

For development or if you need the latest features:

```bash
# Clone the repository
git clone https://github.com/nanaagyei/gpu-memory-profiler.git
cd gpu-memory-profiler

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with testing dependencies
pip install -e .[test]
```

### 3. Development Setup

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/nanaagyei/gpu-memory-profiler.git
cd gpu-memory-profiler

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all development dependencies
pip install -e .[dev,test]

# Install pre-commit hooks
pre-commit install
```

## Dependency Notes

The base installation pulls in **both** PyTorch (`torch>=1.8.0`) and
TensorFlow (`tensorflow>=2.4.0`) as required dependencies. There is
currently no way to install only one framework.

### CPU-Only Mode

For systems without GPU support, no extra steps are needed. The profiler
automatically detects the available hardware and falls back to CPU-based
memory tracking (RSS via psutil). See the
[CPU Compatibility Guide](cpu_compatibility.md) for details.

## Verification

After installation, verify that everything is working:

```bash
# Check version
python3 -c "from gpumemprof._version import __version__; print(__version__)"

# Test CLI tools
gpumemprof --help
tfmemprof --help

# Run basic tests
python3 -m pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**

    ```bash
    # Ensure you're using Python 3
    python3 --version

    # Reinstall in development mode
    pip install -e . --force-reinstall
    ```

2. **Missing Dependencies**

    ```bash
    # Install all dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

3. **Permission Issues**

    ```bash
    # Use user installation
    pip install --user gpu-memory-profiler
    ```

4. **Virtual Environment Issues**
    ```bash
    # Create fresh virtual environment
    python3 -m venv venv_new
    source venv_new/bin/activate
    pip install -e .
    ```

### Platform-Specific Notes

#### macOS

-   Use `python3` instead of `python`
-   Install Xcode command line tools if needed

#### Windows

-   Use `python` (if Python 3 is default) or `python3`
-   Install Visual C++ build tools if needed

#### Linux

-   Use `python3` instead of `python`
-   Install system dependencies: `sudo apt-get install python3-dev`

## Next Steps

After installation:

1. Open the [Documentation Home](index.md) for the full docs navigation.
2. Read the [Quick Start Guide](usage.md).
3. Explore the [CLI Usage](cli.md).
4. Review the [API Documentation](api.md).
5. Check out the [Examples](../examples/).

## Support

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/nanaagyei/gpu-memory-profiler/issues)
3. Create a new issue with detailed information

---

[← Back to main docs](index.md)
