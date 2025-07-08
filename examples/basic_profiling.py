"""
Basic GPU Memory Profiling Example

This example demonstrates the core features of the GPU Memory Profiler:
- Basic function profiling
- Context-based profiling
- Memory tracking
- Visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Import the profiler
from gpumemprof import (
    GPUMemoryProfiler,
    profile_function,
    profile_context,
    MemoryVisualizer,
    MemoryAnalyzer,
    get_gpu_info,
    memory_summary
)


def create_large_tensor(size_mb=100):
    """Create a large tensor of specified size in MB."""
    elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    return torch.randn(elements, device='cuda')


def simple_computation(tensor):
    """Perform some simple computations on a tensor."""
    result = tensor * 2
    result = result + tensor
    result = torch.relu(result)
    return result


class SimpleModel(nn.Module):
    """A simple neural network for demonstration."""

    def __init__(self, input_size=1024, hidden_size=512, num_layers=3):
        super().__init__()
        layers = []

        # Create layers
        layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 10))  # 10 output classes

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


def main():
    """Main demonstration function."""
    print("GPU Memory Profiler - Basic Example")
    print("=" * 50)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    # Show GPU information
    print("GPU Information:")
    gpu_info = get_gpu_info()
    print(f"Device: {gpu_info['device_name']}")
    print(f"Total Memory: {gpu_info['total_memory'] / (1024**3):.2f} GB")
    print()

    # Initialize profiler
    profiler = GPUMemoryProfiler(track_tensors=True)

    print("1. Basic Function Profiling")
    print("-" * 30)

    # Profile tensor creation
    @profile_function
    def create_tensors():
        tensors = []
        for i in range(5):
            tensor = create_large_tensor(50)  # 50MB each
            tensors.append(tensor)
        return tensors

    # Profile computation
    @profile_function
    def process_tensors(tensors):
        results = []
        for tensor in tensors:
            result = simple_computation(tensor)
            results.append(result)
        return results

    # Run profiled functions
    tensors = create_tensors()
    results = process_tensors(tensors)

    # Clean up
    del tensors, results
    torch.cuda.empty_cache()

    print("\n2. Context-based Profiling")
    print("-" * 30)

    # Profile model training
    model = SimpleModel().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    with profile_context("model_training"):
        for epoch in range(3):
            with profile_context(f"epoch_{epoch}"):
                # Generate batch
                batch_size = 256
                inputs = torch.randn(batch_size, 1024, device='cuda')
                targets = torch.randint(0, 10, (batch_size,), device='cuda')

                # Forward pass
                with profile_context("forward_pass"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass
                with profile_context("backward_pass"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    print("\n3. Profiling Results Summary")
    print("-" * 30)

    # Get and display summary
    summary = profiler.get_summary()
    print(f"Total functions profiled: {summary['total_functions_profiled']}")
    print(f"Total function calls: {summary['total_function_calls']}")
    print(
        f"Total execution time: {summary['total_execution_time']:.3f} seconds")
    print(
        f"Peak memory usage: {summary['peak_memory_usage'] / (1024**3):.2f} GB")
    print(
        f"Memory change from baseline: {summary['memory_change_from_baseline'] / (1024**3):.2f} GB")

    # Show function-level statistics
    print("\nFunction-level Statistics:")
    for func_name, stats in summary['function_summaries'].items():
        print(f"  {func_name}:")
        print(f"    Calls: {stats['call_count']}")
        print(f"    Avg time: {stats['avg_time']:.3f}s")
        print(
            f"    Avg memory: {stats['avg_memory_allocated'] / (1024**3):.2f} GB")
        print(f"    Peak memory: {stats['peak_memory'] / (1024**3):.2f} GB")

    print("\n4. Memory Analysis")
    print("-" * 30)

    # Analyze memory patterns
    analyzer = MemoryAnalyzer(profiler)
    patterns = analyzer.analyze_memory_patterns()
    insights = analyzer.generate_performance_insights()

    print(f"Detected {len(patterns)} memory patterns")
    for pattern in patterns:
        print(
            f"  - {pattern.pattern_type}: {pattern.description} (severity: {pattern.severity})")

    print(f"\nGenerated {len(insights)} performance insights")
    for insight in insights:
        print(
            f"  - {insight.category}: {insight.title} (impact: {insight.impact})")

    # Generate optimization report
    report = analyzer.generate_optimization_report()
    print(
        f"\nOptimization Score: {report['optimization_score']['score']}/100 ({report['optimization_score']['grade']})")
    print(f"Description: {report['optimization_score']['description']}")

    print("\n5. Visualization")
    print("-" * 30)

    # Create visualizer
    visualizer = MemoryVisualizer(profiler)

    try:
        # Create memory timeline plot
        print("Generating memory timeline plot...")
        fig = visualizer.plot_memory_timeline(
            interactive=False, save_path='memory_timeline.png')
        print("Saved: memory_timeline.png")

        # Create function comparison plot
        print("Generating function comparison plot...")
        fig = visualizer.plot_function_comparison(
            save_path='function_comparison.png')
        print("Saved: function_comparison.png")

        # Create memory heatmap
        print("Generating memory heatmap...")
        fig = visualizer.plot_memory_heatmap(save_path='memory_heatmap.png')
        print("Saved: memory_heatmap.png")

        # Export data
        print("Exporting profiling data...")
        data_path = visualizer.export_data(
            format='json', save_path='profiling_results')
        print(f"Saved: {data_path}")

    except Exception as e:
        print(f"Visualization error (this might happen without display): {e}")

    print("\n6. Memory Summary")
    print("-" * 30)

    # Final memory summary
    summary_text = memory_summary()
    print(summary_text)

    print("\nExample completed successfully!")
    print("Check the generated files for detailed results.")


if __name__ == "__main__":
    main()
