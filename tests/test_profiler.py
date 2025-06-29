import pytest
import tensorflow as tf
import torch

from gpumemprof import GPUMemoryProfiler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profiler():
    profiler = GPUMemoryProfiler()

    def dummy_func():
        return "Hello, World!"

    # Test basic profiling
    result = profiler.profile_function(dummy_func)()

    # Check that profiling was performed
    assert profiler.results is not None
    assert len(profiler.results) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_tensorflow():
    profiler = GPUMemoryProfiler()

    def dummy_tensorflow_func():
        x = tf.constant([1, 2, 3, 4, 5])
        return tf.reduce_sum(x)

    # Test TensorFlow profiling
    result = profiler.profile_function(dummy_tensorflow_func)()

    # Check that profiling was performed
    assert profiler.results is not None
    assert len(profiler.results) > 0
