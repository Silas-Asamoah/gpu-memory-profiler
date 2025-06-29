"""TensorFlow Context Profiling"""

import functools
import threading
from contextlib import contextmanager
from typing import Optional, Any, Dict

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from .profiler import TFMemoryProfiler

# Global profiler instance
_global_profiler: Optional[TFMemoryProfiler] = None
_profiler_lock = threading.Lock()


def get_global_profiler() -> TFMemoryProfiler:
    """Get or create global profiler instance."""
    global _global_profiler

    with _profiler_lock:
        if _global_profiler is None:
            _global_profiler = TFMemoryProfiler()
        return _global_profiler


def set_global_profiler(profiler: TFMemoryProfiler):
    """Set global profiler instance."""
    global _global_profiler

    with _profiler_lock:
        _global_profiler = profiler


def profile_function(func=None, *, profiler=None, name=None):
    """
    Decorator to profile function memory usage.

    Args:
        func: Function to profile
        profiler: Profiler instance (uses global if None)
        name: Custom name for profiling
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            prof = profiler or get_global_profiler()
            func_name = name or f.__name__

            # Use the profiler's function profiling
            return prof.profile_function(f)(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def profile_context(name: str = "context", profiler: Optional[TFMemoryProfiler] = None):
    """
    Context manager for profiling code blocks.

    Args:
        name: Name for the profiling context
        profiler: Profiler instance (uses global if None)
    """
    prof = profiler or get_global_profiler()

    with prof.profile_context(name):
        yield


class ProfiledLayer:
    """Wrapper for TensorFlow layers with automatic profiling."""

    def __init__(self, layer, profiler: Optional[TFMemoryProfiler] = None, name: Optional[str] = None):
        """
        Initialize profiled layer.

        Args:
            layer: TensorFlow layer to profile
            profiler: Profiler instance
            name: Custom name for profiling
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        self.layer = layer
        self.profiler = profiler or get_global_profiler()
        self.name = name or getattr(layer, 'name', layer.__class__.__name__)

        # Wrap the call method
        self._original_call = layer.call
        layer.call = self._profiled_call

    def _profiled_call(self, *args, **kwargs):
        """Profiled version of layer call."""
        with self.profiler.profile_context(f"layer_{self.name}"):
            return self._original_call(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped layer."""
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        """Make the wrapper callable."""
        return self.layer(*args, **kwargs)


def profile_model(model, profiler: Optional[TFMemoryProfiler] = None):
    """
    Profile all layers in a TensorFlow model.

    Args:
        model: TensorFlow model
        profiler: Profiler instance

    Returns:
        Model with profiled layers
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    prof = profiler or get_global_profiler()

    # Profile each layer
    for i, layer in enumerate(model.layers):
        ProfiledLayer(layer, prof, f"{layer.name}_{i}")

    return model


class TensorFlowProfiler:
    """High-level TensorFlow profiling interface."""

    def __init__(self, device: Optional[str] = None):
        """Initialize TensorFlow profiler."""
        self.profiler = TFMemoryProfiler(device=device)
        set_global_profiler(self.profiler)

    def profile_training(self, model, dataset, epochs: int = 1, steps_per_epoch: Optional[int] = None):
        """
        Profile model training.

        Args:
            model: TensorFlow model
            dataset: Training dataset
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        # Profile the entire training process
        with self.profiler.profile_context("training"):
            for epoch in range(epochs):
                with self.profiler.profile_context(f"epoch_{epoch}"):
                    step_count = 0

                    for batch in dataset:
                        if steps_per_epoch and step_count >= steps_per_epoch:
                            break

                        with self.profiler.profile_context(f"step_{step_count}"):
                            # Assume the model has a train_step method or similar
                            if hasattr(model, 'train_step'):
                                model.train_step(batch)
                            else:
                                # Generic training step
                                with tf.GradientTape() as tape:
                                    if isinstance(batch, tuple):
                                        x, y = batch
                                        predictions = model(x, training=True)
                                        loss = model.compiled_loss(
                                            y, predictions)
                                    else:
                                        predictions = model(
                                            batch, training=True)
                                        loss = model.compiled_loss(
                                            batch, predictions)

                                gradients = tape.gradient(
                                    loss, model.trainable_variables)
                                model.optimizer.apply_gradients(
                                    zip(gradients, model.trainable_variables))

                        step_count += 1

    def profile_inference(self, model, data, batch_size: int = 32):
        """
        Profile model inference.

        Args:
            model: TensorFlow model
            data: Input data
            batch_size: Batch size for inference
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        with self.profiler.profile_context("inference"):
            # Batch the data if needed
            if hasattr(data, 'batch'):
                batched_data = data.batch(batch_size)
            else:
                # Assume data is a tensor or numpy array
                import numpy as np
                if isinstance(data, np.ndarray):
                    data = tf.constant(data)

                # Create batches manually
                num_samples = tf.shape(data)[0]
                num_batches = (num_samples + batch_size - 1) // batch_size

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    batch = data[start_idx:end_idx]

                    with self.profiler.profile_context(f"inference_batch_{i}"):
                        model(batch, training=False)

    def get_results(self):
        """Get profiling results."""
        return self.profiler.get_results()

    def reset(self):
        """Reset profiler state."""
        self.profiler.reset()


# Convenience functions for common use cases
def profile_keras_training(model, x_train, y_train, epochs: int = 1, batch_size: int = 32,
                           validation_data=None, profiler: Optional[TFMemoryProfiler] = None):
    """
    Profile Keras model training.

    Args:
        model: Keras model
        x_train: Training data
        y_train: Training labels
        epochs: Number of epochs
        batch_size: Batch size
        validation_data: Validation data tuple (x_val, y_val)
        profiler: Profiler instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    prof = profiler or get_global_profiler()

    with prof.profile_context("keras_training"):
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size)

        # Profile training
        for epoch in range(epochs):
            with prof.profile_context(f"epoch_{epoch}"):
                # Training
                with prof.profile_context("training_batches"):
                    for batch_x, batch_y in train_dataset:
                        with prof.profile_context("train_step"):
                            model.train_on_batch(batch_x, batch_y)

                # Validation
                if validation_data:
                    x_val, y_val = validation_data
                    with prof.profile_context("validation"):
                        model.evaluate(x_val, y_val, verbose=0)


def clear_global_profiler():
    """Clear global profiler state."""
    global _global_profiler

    with _profiler_lock:
        if _global_profiler:
            _global_profiler.reset()
            _global_profiler = None
