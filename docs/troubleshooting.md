[← Back to main docs](index.md)

# Troubleshooting Guide

This guide helps you resolve common issues with GPU Memory Profiler.

## Common Issues

### Import Errors

#### Problem: `ModuleNotFoundError: No module named 'gpumemprof'`

**Solution:**

```bash
# Install the package
pip install -e .

# Or install from PyPI (when available)
pip install gpu-memory-profiler
```

#### Problem: `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
# Install PyTorch
pip install torch

# Or install with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Problem: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**

```bash
# Install TensorFlow
pip install tensorflow

# Or install GPU version
pip install tensorflow-gpu
```

### CUDA Issues

#### Problem: `CUDA not available`

**Symptoms:**

- Error: `CUDA not available`
- Profiler falls back to CPU mode

**Solutions:**

1. **Check CUDA installation:**

```bash
nvidia-smi
nvcc --version
```

2. **Verify PyTorch CUDA:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

3. **Verify TensorFlow CUDA:**

```python
import tensorflow as tf
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
```

4. **Install CUDA-compatible versions:**

```bash
# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# TensorFlow with GPU support
pip install tensorflow-gpu
```

#### Problem: `CUDA out of memory`

**Symptoms:**

- Error: `CUDA out of memory`
- Training crashes

**Solutions:**

1. **Reduce batch size:**

```python
# Reduce batch size
dataloader = DataLoader(dataset, batch_size=16)  # Instead of 64
```

2. **Clear cache:**

```python
import torch
torch.cuda.empty_cache()
```

3. **Use gradient checkpointing:**

```python
model.use_checkpoint = True
```

4. **Monitor memory usage:**

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.start_monitoring(interval=0.5)

# Your training code here
profiler.stop_monitoring()
```

### Memory Leak Issues

#### Problem: Memory usage keeps increasing

**Symptoms:**

- Memory usage grows over time
- Profiler detects memory leaks

**Solutions:**

1. **Check for unreleased tensors:**

```python
# Ensure tensors are properly deleted
del tensor
torch.cuda.empty_cache()
```

2. **Use context managers:**

```python
with torch.no_grad():
    # Inference code here
    pass
```

3. **Monitor with profiler:**

```python
from gpumemprof import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
profiler.enable_leak_detection(threshold=100, window_size=10)

# Your code here
leaks = profiler.detect_leaks()
if leaks:
    print(f"Memory leaks detected: {leaks}")
```

### CLI Issues

#### Problem: `gpumemprof: command not found`

**Solution:**

```bash
# Reinstall the package
pip install -e .

# Check if entry points are installed
pip show gpu-memory-profiler
```

#### Problem: CLI commands fail

**Solutions:**

1. **Check Python path:**

```bash
which python
which gpumemprof
```

2. **Reinstall with entry points:**

```bash
pip uninstall gpu-memory-profiler
pip install -e .
```

3. **Use Python module directly:**

```bash
python -m gpumemprof.cli info
python -m tfmemprof.cli info
```

### Visualization Issues

#### Problem: Plots don't display

**Symptoms:**

- No plots appear
- Error: `No display name and no $DISPLAY environment variable`

**Solutions:**

1. **Use non-interactive backend:**

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

2. **Save plots to files:**

```python
profiler.plot_memory_timeline()
plt.savefig('timeline.png')
```

3. **Use Plotly for web-based plots:**

```python
profiler.create_dashboard()  # Opens in browser
```

#### Problem: Dash visualization fails

**Symptoms:**

- Error: `ImportError: No module named 'dash'`

**Solution:**

```bash
pip install dash
```

### Performance Issues

#### Problem: Profiler adds too much overhead

**Symptoms:**

- Training is significantly slower
- High CPU usage

**Solutions:**

1. **Increase sampling interval:**

```python
profiler = GPUMemoryProfiler(sampling_interval=2.0)  # Sample every 2 seconds
```

2. **Disable visualization during training:**

```python
profiler = GPUMemoryProfiler(enable_visualization=False)
```

3. **Use context profiling selectively:**

```python
# Only profile specific functions
@profiler.profile_function
def critical_function():
    pass
```

### Dependency Conflicts

#### Problem: `typing_extensions` version conflict

**Symptoms:**

- Error with TensorFlow CLI
- Version conflicts between packages

**Solutions:**

1. **Check versions:**

```bash
pip list | grep typing
```

2. **Install compatible version:**

```bash
pip install typing-extensions==4.5.0
```

3. **Use virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Platform-Specific Issues

#### macOS Issues

**Problem: CUDA not available on macOS**

**Solution:**

- Use CPU mode or MPS (Metal Performance Shaders)
- Install PyTorch with MPS support

**Problem: TensorFlow issues on Apple Silicon**

**Solution:**

```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos
```

#### Windows Issues

**Problem: Path issues**

**Solution:**

```bash
# Use forward slashes or raw strings
python -m gpumemprof.cli info
```

**Problem: Permission issues**

**Solution:**

```bash
# Run as administrator or use --user flag
pip install --user -e .
```

## Debug Mode

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from gpumemprof import GPUMemoryProfiler
profiler = GPUMemoryProfiler()
```

### Verbose CLI Output

```bash
# Use detailed/system output commands
gpumemprof info --detailed
gpumemprof monitor --duration 10
```

### Check System Information

```python
from gpumemprof import get_system_info
info = get_system_info()
print(info)
```

## Getting Help

### Before Asking for Help

1. **Check the documentation:**

   - [Installation guide](installation.md)
   - [Usage guide](usage.md)
   - [API reference](api.md)

2. **Run diagnostics:**

```python
from gpumemprof import get_system_info as get_torch_system_info
from tfmemprof import get_system_info as get_tf_system_info

# Check PyTorch setup
print(get_torch_system_info())

# Check TensorFlow setup
print(get_tf_system_info())
```

3. **Test with minimal example:**

```python
from gpumemprof import GPUMemoryProfiler
import torch

profiler = GPUMemoryProfiler()
@profiler.profile_function
def test():
    return torch.randn(100, 100).cuda()

result = test()
print(profiler.get_results())
```

### Reporting Issues

When reporting issues, include:

1. **System information:**

   - OS and version
   - Python version
   - PyTorch/TensorFlow versions
   - CUDA version (if applicable)

2. **Error messages:**

   - Full error traceback
   - Any warning messages

3. **Reproduction steps:**

   - Minimal code example
   - Expected vs actual behavior

4. **Environment:**
   - Virtual environment details
   - Package versions (`pip freeze`)

### Community Support

- **GitHub Issues**: [Create an issue](https://github.com/nanaagyei/gpu-memory-profiler/issues)
- **Documentation**: Check the [docs](index.md)
- **Examples**: See the [examples directory](../examples/)

---

[← Back to main docs](index.md)
