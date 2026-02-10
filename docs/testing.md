[← Back to main docs](index.md)

# Testing & Validation Guide

This guide covers how to test and validate GPU Memory Profiler functionality.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python3 -m pytest

# Run with verbose output
python3 -m pytest -v

# Run with coverage
python3 -m pytest --cov=gpumemprof --cov=tfmemprof

# Run specific test file
python3 -m pytest tests/test_profiler.py

# Run specific test function
python3 -m pytest tests/test_profiler.py::test_basic_profiling
```

### Test Categories

```bash
# Run unit tests only
python3 -m pytest -m "unit"

# Run integration tests only
python3 -m pytest -m "integration"

# Run slow tests
python3 -m pytest -m "slow"

# Skip slow tests
python3 -m pytest -m "not slow"
```

### Framework-Specific Tests

```bash
# Run PyTorch tests only
python3 -m pytest -m "pytorch"

# Run TensorFlow tests only
python3 -m pytest -m "tensorflow"

# Run CPU-only tests
python3 -m pytest -m "cpu"

# Run GPU tests (if CUDA available)
python3 -m pytest -m "gpu"
```

### Parallel Testing

```bash
# Run tests in parallel
python3 -m pytest -n auto

# Run with specific number of workers
python3 -m pytest -n 4
```

### Textual Snapshot Tests

Use snapshot tests to detect unintended visual/layout regressions in the Textual
TUI tabs.

```bash
# Validate snapshots against the committed SVG baselines
conda run -n tensor-torch-profiler python -m pytest tests/tui/test_app_snapshots.py -m tui_snapshot

# Intentionally update snapshots after approved UI changes
conda run -n tensor-torch-profiler python -m pytest tests/tui/test_app_snapshots.py -m tui_snapshot --snapshot-update
```

Snapshot files are stored at:

`tests/tui/__snapshots__/test_app_snapshots/*.svg`

Only update snapshot baselines when UI changes are intentional and reviewed.

If you see `fixture 'snap_compare' not found` or plugin import errors, install
test dependencies and re-run:

```bash
conda run -n tensor-torch-profiler python -m pip install -r requirements-test.txt
conda run -n tensor-torch-profiler python -m pytest tests/tui/test_app_snapshots.py -m tui_snapshot
```

## Test Structure

```
tests/
├── conftest.py                    # Test configuration and fixtures
├── test_core_profiler.py         # Core profiling functionality
├── test_context_profiler.py      # Context managers and decorators
├── test_tracker.py               # Real-time tracking
├── test_visualizer.py            # Visualization components
├── test_analyzer.py              # Analysis algorithms
├── test_utils.py                 # Utility functions
├── test_profiler.py              # CLI/framework integration
├── test_integration.py           # Integration tests
└── test_performance.py           # Performance benchmarks
```

For manual smoke tests (CPU-only, PyTorch GPU, TensorFlow GPU, CLI) see
`docs/examples/test_guides/README.md`.

### CPU Smoke Test (No CUDA)

Need a fast signal on CPU-only machines? Run both steps below:

1. Force CPU execution and exercise the CLI walkthrough:

    ```bash
    # Windows (PowerShell/CMD)
    set CUDA_VISIBLE_DEVICES=
    python -m examples.cli.quickstart

    # macOS/Linux
    export CUDA_VISIBLE_DEVICES=
    python -m examples.cli.quickstart
    ```

    The quickstart script runs `gpumemprof --help`, `gpumemprof info`, and (if
    installed) the TensorFlow CLI without touching CUDA.

2. Validate the system-info fallbacks:

    ```bash
    pytest tests/test_utils.py
    ```

    This ensures `gpumemprof.utils.get_system_info()` still reports sensible
    metadata when GPUs are absent.

Add these steps to CPU-only CI jobs or use them locally before installing
CUDA.

### Enabling the CUDA Path

When you're ready to run GPU-accelerated tests or demos, install the CUDA
toolchain and the CUDA-enabled framework wheels, then re-run `pytest` without
forcing CPU-only mode. The full setup checklist (drivers, PyTorch CUDA builds,
TensorFlow GPU builds, verification commands) lives in
`docs/gpu_setup.md`.

## PyTorch Testing

### Basic PyTorch Test

```python
import pytest
import torch
from gpumemprof import GPUMemoryProfiler

def test_pytorch_basic_profiling():
    profiler = GPUMemoryProfiler()

    @profiler.profile_function
    def create_tensor():
        return torch.randn(1000, 1000).cuda()

    tensor = create_tensor()
    results = profiler.get_results()

    assert results.peak_memory_mb > 0
    assert len(results.function_calls) > 0
```

### Memory Leak Test

```python
def test_pytorch_memory_leak_detection():
    profiler = GPUMemoryProfiler()
    profiler.enable_leak_detection(threshold=50, window_size=5)

    # Simulate memory leak
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000).cuda()
        tensors.append(tensor)  # Don't delete tensors

    leaks = profiler.detect_leaks()
    assert len(leaks) > 0
```

## TensorFlow Testing

### Basic TensorFlow Test

```python
import pytest
import tensorflow as tf
from tfmemprof import TensorFlowProfiler

def test_tensorflow_basic_profiling():
    profiler = TensorFlowProfiler()

    with profiler.profile_context("test"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(100, activation='softmax')
        ])

        x = tf.random.normal((100, 1000))
        y = model(x)

    results = profiler.get_results()
    assert results.peak_memory_mb > 0
```

### Keras Model Test

```python
def test_tensorflow_keras_profiling():
    profiler = TensorFlowProfiler()

    with profiler.profile_context("model_training"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        x_train = tf.random.normal((1000, 500))
        y_train = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

        model.fit(x_train, y_train, epochs=2, batch_size=32)

    results = profiler.get_results()
    assert results.peak_memory_mb > 0
```

## CLI Testing

### PyTorch CLI Test

```python
import subprocess
import pytest

def test_gpumemprof_cli():
    # Test info command
    result = subprocess.run(['gpumemprof', 'info'],
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert 'GPU Memory Profiler' in result.stdout

def test_gpumemprof_monitor():
    # Test monitor command (short duration)
    result = subprocess.run(['gpumemprof', 'monitor', '--duration', '2'],
                          capture_output=True, text=True)
    assert result.returncode == 0
```

### TensorFlow CLI Test

```python
def test_tfmemprof_cli():
    # Test info command
    result = subprocess.run(['tfmemprof', 'info'],
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert 'TensorFlow' in result.stdout
```

## CPU Compatibility Testing

### CPU Mode Test

```python
def test_cpu_compatibility():
    # Test without CUDA
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from gpumemprof import GPUMemoryProfiler
    profiler = GPUMemoryProfiler()

    # Should work in CPU mode
    results = profiler.get_results()
    assert results is not None
```

## Performance Testing

### Benchmark Tests

```python
import pytest
import time

@pytest.mark.benchmark
def test_profiling_performance(benchmark):
    """Benchmark profiling performance."""
    def profiling_operation():
        # Profiling operation to benchmark
        pass

    result = benchmark(profiling_operation)
    assert result.stats.mean < 0.1  # Should complete in < 100ms
```

## Integration Testing

### End-to-End Test

```python
def test_end_to_end_pytorch():
    from gpumemprof import GPUMemoryProfiler

    profiler = GPUMemoryProfiler()

    # Simulate complete training workflow
    with profiler.profile_context("data_loading"):
        data = torch.randn(1000, 100).cuda()
        target = torch.randint(0, 10, (1000,)).cuda()

    with profiler.profile_context("model_creation"):
        model = torch.nn.Linear(100, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())

    with profiler.profile_context("training"):
        for epoch in range(3):
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    results = profiler.get_results()

    # Verify results
    assert results.peak_memory_mb > 0
    assert len(results.memory_snapshots) > 0
    assert len(results.function_calls) > 0
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --strict-config
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    pytorch: marks tests as PyTorch-specific
    tensorflow: marks tests as TensorFlow-specific
    gpu: marks tests as GPU-required
    cpu: marks tests as CPU-only
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["gpumemprof", "tfmemprof"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/.venv/*",
]
```

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
    test:
        runs-on: ubuntu-latest
        strategy:
            matrix:
                python-version: [3.10, 3.11, 3.12]
        steps:
            - uses: actions/checkout@v2
            - name: Set up Python ${{ matrix.python-version }}
              uses: actions/setup-python@v2
              with:
                  python-version: ${{ matrix.python-version }}
            - name: Install dependencies
              run: |
                  python3 -m pip install --upgrade pip
                  pip install -e .[test]
            - name: Run tests
              run: |
                  python3 -m pytest --cov=gpumemprof --cov=tfmemprof
```

### Local CI Simulation

```bash
# Run the same tests as CI
python3 -m pip install -e .[test]
python3 -m pytest --cov=gpumemprof --cov=tfmemprof --cov-report=xml
```

## Test Data

### Creating Test Data

```python
import numpy as np
import torch
import tensorflow as tf

def create_test_tensors():
    """Create test tensors for different frameworks."""
    # PyTorch tensor
    torch_tensor = torch.randn(100, 10)

    # TensorFlow tensor
    tf_tensor = tf.random.normal((100, 10))

    # NumPy array
    np_array = np.random.randn(100, 10)

    return torch_tensor, tf_tensor, np_array
```

### Mocking

```python
import pytest
from unittest.mock import Mock, patch

def test_with_mocked_gpu():
    """Test with mocked GPU functionality."""
    with patch('gpumemprof.utils.get_gpu_memory') as mock_gpu:
        mock_gpu.return_value = 1024  # Mock 1GB GPU memory
        # Test implementation
        pass
```

## Performance Testing

### Benchmark Tests

```python
import pytest
import time

@pytest.mark.benchmark
def test_profiling_performance(benchmark):
    """Benchmark profiling performance."""
    def profiling_operation():
        # Profiling operation to benchmark
        pass

    result = benchmark(profiling_operation)
    assert result.stats.mean < 0.1  # Should complete in < 100ms
```

### Memory Testing

```python
import pytest
import psutil
import os

def test_memory_usage():
    """Test memory usage during profiling."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Run profiling operation
    profiler = GPUMemoryProfiler()
    # ... profiling code ...

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

## Troubleshooting

### Common Test Issues

1. **Import Errors**

    ```bash
    # Ensure you're in the right environment
    python3 -c "import gpumemprof; print('Import successful')"
    ```

2. **Missing Dependencies**

    ```bash
    # Install test dependencies
    pip install -e .[test]
    ```

3. **GPU Tests Failing**

    ```bash
    # Run CPU-only tests
    python3 -m pytest -m "cpu"
    ```

4. **Slow Tests**
    ```bash
    # Skip slow tests
    python3 -m pytest -m "not slow"
    ```

### Debugging Tests

```bash
# Run with debug output
python3 -m pytest -v -s

# Run single test with debug
python3 -m pytest tests/test_profiler.py::test_specific_function -v -s

# Run with pdb on failure
python3 -m pytest --pdb
```

## Best Practices

1. **Write descriptive test names**
2. **Use appropriate markers**
3. **Keep tests independent**
4. **Use fixtures for common setup**
5. **Test both success and failure cases**
6. **Maintain good test coverage**
7. **Run tests before committing**

## Contributing to Tests

When adding new features:

1. Write tests for new functionality
2. Update existing tests if needed
3. Ensure all tests pass
4. Add appropriate markers
5. Update documentation

For more information, see the [Contributing Guide](../CONTRIBUTING.md).

---

[← Back to main docs](index.md)
