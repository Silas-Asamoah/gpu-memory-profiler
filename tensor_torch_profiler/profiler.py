import time
import torch
import tensorflow as tf
from collections import defaultdict

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
            print(f"Function: {name}")
            print(f"Total execution time: {stat['time']} seconds")
            print(f"Total memory usage: {stat['memory']} bytes")