import logging
import time
from collections import defaultdict

import tensorflow as tf
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorTorchProfiler:
    def __init__(self):
        self.stats = defaultdict(lambda: {"time": 0, "memory": 0})

    def profile(self, func, *args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()

        self.stats[func.__name__]["time"] += end_time - start_time
        self.stats[func.__name__]["memory"] += end_memory - start_memory

        return result

    def report(self):
        for name, stat in self.stats.items():
            logger.info(f"Function: {name}")
            logger.info(f"Total execution time: {stat['time']} seconds")
            logger.info(f"Total memory usage: {stat['memory']} bytes")
