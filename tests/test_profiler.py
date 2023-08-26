import pytest
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
