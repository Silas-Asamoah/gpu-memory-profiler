# gpu-memory-profiler

A simple tool to profile the memory usage of your GPU over time.

This profiler is designed to work with deep learning frameworks like Pytorch and Tensorflow. It provides real-time tracking and profiling of GPU memory usage during the execution of your deep learning models. This can be particularly useful for identifying memory leaks or inefficiencies in your model's memory usage. The tool provides a detailed breakdown of memory usage by layer, allowing you to pinpoint exactly where in your model the most memory is being used. This can help you optimize your model for better performance and efficiency.


## TensorFlow Profiler

The TensorFlow Profiler provides the following metrics:

- cpu_exec_micros: The execution time of the operation on the CPU.
- exec_micros: The total execution time of the operation.
- float_ops: The number of floating-point operations performed by the operation.
- total_accelerator_exec_micros: The total execution time of the operation on the accelerator (e.g., GPU).
- total_cpu_exec_micros: The total execution time of the operation on the CPU.
- parameters: The number of parameters in your model.
- tensor_value: The value of the tensor produced by the operation.
- input_shapes: The shapes of the input tensors to the operation.
- run_count: The number of times the operation was executed.

## PyTorch Profiler
The PyTorch Profiler provides the following metrics:
- CPU time: The execution time of the operation on the CPU.
- CUDA time: The execution time of the operation on the GPU.
- CPU memory usage: The amount of CPU memory used by the operation.
- CUDA memory usage: The amount of GPU memory used by the operation.
- Input shapes: The shapes of the input tensors to the operation.
- FLOPs: The number of floating-point operations performed by the operation (only for certain operations).
- Stack traces: The stack traces of the operation.
- Module hierarchy: The module hierarchy of the operation (only for TorchScript models).

## Differences between Tensorflow and Pytorch Profilers
1. Metrics Provided: Both profilers provide a range of metrics, including execution time, memory usage, and input shapes. However, there are some differences. For example, the TensorFlow profiler provides metrics like parameters (the number of parameters in your model) and tensor_value (the value of the tensor produced by the operation), which are not provided by the PyTorch profiler. On the other hand, the PyTorch profiler provides stack traces and module hierarchy, which are not provided by the TensorFlow profiler.

2. Ease of Use: Both profilers are relatively easy to use, but the PyTorch profiler is often praised for its simplicity and user-friendly API. The TensorFlow profiler, while powerful, can be a bit more complex to set up and use.

3. Visualization Tools: TensorFlow provides TensorBoard for visualizing profiling results, which is a powerful tool for understanding the performance of your models. PyTorch also provides a visualization tool, TensorBoardX, but it's not as fully featured as TensorBoard.