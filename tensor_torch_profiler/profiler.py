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
        tf.profiler.experimental.start('logdir')
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
