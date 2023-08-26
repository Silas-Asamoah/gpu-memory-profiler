import pytest
import tensorflow as tf

from tensor_torch_profiler.profiler import TensorTorchProfiler


def test_profiler():
    profiler = TensorTorchProfiler()

    def dummy_func():
        return "Hello, World!"

    profiler.profile(dummy_func)

    assert "dummy_func" in profiler.stats
    assert profiler.stats["dummy_func"]["time"] >= 0
    assert profiler.stats["dummy_func"]["memory"] >= 0

    profiler.report()


def test_profile_tensorflow():
    profiler = TensorTorchProfiler()

    def dummy_tensorflow_func():
        x = tf.constant([1, 2, 3, 4, 5])
        return tf.reduce_sum(x)

    profiler.profile_tensorflow(dummy_tensorflow_func)

    assert "dummy_tensorflow_func" in profiler.stats
    assert profiler.stats["dummy_tensorflow_func"]["time"] >= 0

    profiler.report()
