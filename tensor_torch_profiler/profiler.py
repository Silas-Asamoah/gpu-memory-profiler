import logging
import time
from collections import defaultdict

import tensorflow as tf
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorTorchProfiler:
    """
    A profiler for TensorFlow and PyTorch.
    """

    def __init__(self):
        self.stats = defaultdict(lambda: {"time": 0, "memory": 0})

    def profile(self, func, *args, **kwargs):
        """
        Profile a function.
        :param func: The function to profile.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function.
        :return: The result of the function.
        """
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()

        self.stats[func.__name__]["time"] += end_time - start_time
        self.stats[func.__name__]["memory"] += end_memory - start_memory

        return result

    def profile_tensorflow(self, func, *args, **kwargs):
        """
        Profile a tensorflow function.
        :param func: The tensorflow function to profile.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function.
        :return: The result of the function.
        """
        tf.profiler.experimental.start("logdir")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        tf.profiler.experimental.stop()

        self.stats[func.__name__]["time"] += end_time - start_time
        return result

    def report(self):
        """
        Report the profiling results.
        """
        for name, stat in self.stats.items():
            logger.info(f"Function: {name}")
            logger.info(f"Total execution time: {stat['time']} seconds")
            logger.info(f"Total memory usage: {stat['memory']} bytes")

    def reset(self):
        """
        Reset the profiling results.
        """
        self.stats = defaultdict(lambda: {"time": 0, "memory": 0})

    def profile_pytorch(self, func, *args, **kwargs):
        """
        Profile a torch function.
        :param func: The torch function to profile.
        :param args: The arguments to pass to the function.
        :param kwargs: The keyword arguments to pass to the function.
        :return: The result of the function.
        """
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        self.stats[func.__name__]["time"] += end_time - start_time
        self.stats[func.__name__]["memory"] += torch.cuda.max_memory_allocated()

        return result


def test_profile_pytorch():
    """
    Test the PyTorch profiler.
    """
    profiler = TensorTorchProfiler()

    def dummy_func(x):
        return x * 2

    profiler.profile_pytorch(dummy_func, torch.tensor([1.0]))

    assert "dummy_func" in profiler.stats
    assert profiler.stats["dummy_func"]["time"] > 0
    assert profiler.stats["dummy_func"]["memory"] > 0
