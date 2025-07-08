"""
TensorFlow GPU Memory Profiler

A comprehensive GPU memory profiling tool for TensorFlow applications.
Provides real-time monitoring, leak detection, and optimization insights.
"""

__version__ = "0.1.0"
__author__ = "GPU Memory Profiler Team"
__email__ = "prince.agyei.tuffour@gmail.com"

from .profiler import TensorFlowProfiler
from .context_profiler import tensorflow_profiler, TensorFlowContextProfiler
from .tracker import TensorFlowMemoryTracker
from .visualizer import TensorFlowVisualizer
from .analyzer import TensorFlowAnalyzer
from .utils import get_system_info, validate_tensorflow_setup

__all__ = [
    "TensorFlowProfiler",
    "tensorflow_profiler",
    "TensorFlowContextProfiler",
    "TensorFlowMemoryTracker",
    "TensorFlowVisualizer",
    "TensorFlowAnalyzer",
    "get_system_info",
    "validate_tensorflow_setup",
]
