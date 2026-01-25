"""Utility functions for GPU memory profiling."""

import os
import platform
import subprocess
import json
import sys
from typing import Dict, List, Optional, Union, Any
import torch
import psutil


def format_bytes(bytes_value: int, precision: int = 2) -> str:
    """
    Format bytes into human-readable format.

    Args:
        bytes_value: Number of bytes
        precision: Decimal precision

    Returns:
        Formatted string (e.g., "1.25 GB")
    """
    if bytes_value == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(bytes_value)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.{precision}f} {units[unit_index]}"


def convert_bytes(value: Union[int, float], from_unit: str, to_unit: str) -> float:
    """
    Convert between different byte units.

    Args:
        value: Value to convert
        from_unit: Source unit (B, KB, MB, GB, TB)
        to_unit: Target unit (B, KB, MB, GB, TB)

    Returns:
        Converted value
    """
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Invalid unit. Must be one of: {list(units.keys())}")

    bytes_value = value * units[from_unit]
    return bytes_value / units[to_unit]


def get_gpu_info(device: Optional[Union[str, int, torch.device]] = None) -> Dict[str, Any]:
    """
    Get comprehensive GPU information.

    Args:
        device: GPU device to query (None for current device)

    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available"}

    if device is None:
        device_id = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_id = device.index if device.index is not None else 0
    elif isinstance(device, str):
        device_id = int(device.split(":")[-1]) if ":" in device else 0
    else:
        device_id = int(device)

    # Basic PyTorch GPU info
    gpu_info = {
        "device_id": device_id,
        "device_name": torch.cuda.get_device_name(device_id),
        "device_capability": torch.cuda.get_device_capability(device_id),
        "total_memory": torch.cuda.get_device_properties(device_id).total_memory,
        "multiprocessor_count": torch.cuda.get_device_properties(device_id).multiprocessor_count,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }

    # Current memory usage
    try:
        gpu_info.update({
            "allocated_memory": torch.cuda.memory_allocated(device_id),
            "reserved_memory": torch.cuda.memory_reserved(device_id),
            "max_memory_allocated": torch.cuda.max_memory_allocated(device_id),
            "max_memory_reserved": torch.cuda.max_memory_reserved(device_id),
        })

        # Memory stats
        memory_stats = torch.cuda.memory_stats(device_id)
        gpu_info["memory_stats"] = {
            "active_bytes": memory_stats.get("active_bytes.all.current", 0),
            "inactive_bytes": memory_stats.get("inactive_split_bytes.all.current", 0),
            "reserved_bytes": memory_stats.get("reserved_bytes.all.current", 0),
            "num_alloc_retries": memory_stats.get("num_alloc_retries", 0),
            "num_ooms": memory_stats.get("num_ooms", 0),
        }
    except Exception as e:
        gpu_info["memory_error"] = str(e)

    # Try to get additional info via nvidia-ml-py or nvidia-smi
    try:
        gpu_info.update(_get_nvidia_smi_info(device_id))
    except Exception:
        pass  # nvidia-smi info is optional

    return gpu_info


def _get_nvidia_smi_info(device_id: int) -> Dict[str, Any]:
    """Get additional GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if device_id < len(lines):
                values = lines[device_id].split(',')
                if len(values) >= 8:
                    return {
                        "nvidia_smi_info": {
                            "memory_total_mb": int(values[2].strip()),
                            "memory_used_mb": int(values[3].strip()),
                            "memory_free_mb": int(values[4].strip()),
                            "gpu_utilization_percent": int(values[5].strip()),
                            "temperature_c": int(values[6].strip()),
                            "power_draw_w": float(values[7].strip()),
                        }
                    }
    except Exception:
        pass

    return {}


def _detect_platform_info() -> Dict[str, str]:
    """Safely detect platform and architecture across OSes."""
    if hasattr(os, "uname"):
        try:
            uname_result = os.uname()
            return {
                "platform": getattr(uname_result, "sysname", "Unknown"),
                "architecture": getattr(uname_result, "machine", "Unknown"),
            }
        except Exception:
            # Fall back to platform module if os.uname is unavailable or fails
            pass

    try:
        uname_result = platform.uname()
        system_name = getattr(uname_result, "system", platform.system())
        machine = getattr(uname_result, "machine", platform.machine())
    except Exception:
        system_name = platform.system()
        machine = platform.machine()

    return {
        "platform": system_name or "Unknown",
        "architecture": machine or "Unknown",
    }


def get_system_info() -> Dict[str, Any]:
    """Get system information relevant to GPU profiling."""
    platform_info = _detect_platform_info()
    system_info = {
        "platform": platform_info["platform"],
        "architecture": platform_info["architecture"],
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        system_info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "current_device": torch.cuda.current_device(),
        })

    # CPU and memory info
    try:
        system_info.update({
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
        })
    except Exception as e:
        system_info["system_info_error"] = str(e)

    return system_info


def check_memory_fragmentation(device: Optional[Union[str, int, torch.device]] = None) -> Dict[str, Any]:
    """
    Check GPU memory fragmentation.

    Args:
        device: GPU device to check

    Returns:
        Fragmentation analysis
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA is not available"}

    if device is None:
        device_id = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_id = device.index if device.index is not None else 0
    elif isinstance(device, str):
        device_id = int(device.split(":")[-1]) if ":" in device else 0
    else:
        device_id = int(device)

    memory_stats = torch.cuda.memory_stats(device_id)

    allocated = memory_stats.get("allocated_bytes.all.current", 0)
    reserved = memory_stats.get("reserved_bytes.all.current", 0)
    active = memory_stats.get("active_bytes.all.current", 0)
    inactive = memory_stats.get("inactive_split_bytes.all.current", 0)

    total_gpu_memory = torch.cuda.get_device_properties(device_id).total_memory

    fragmentation_info = {
        "device_id": device_id,
        "total_memory": total_gpu_memory,
        "allocated_memory": allocated,
        "reserved_memory": reserved,
        "active_memory": active,
        "inactive_memory": inactive,
        "free_memory": total_gpu_memory - reserved,
        "fragmentation_ratio": inactive / reserved if reserved > 0 else 0,
        "utilization_ratio": allocated / total_gpu_memory,
        "reservation_ratio": reserved / total_gpu_memory,
        "waste_ratio": (reserved - allocated) / total_gpu_memory if reserved > allocated else 0,
    }

    # Add formatted versions
    for key, value in fragmentation_info.items():
        if key.endswith("_memory") or key == "total_memory":
            fragmentation_info[key + "_formatted"] = format_bytes(value)

    return fragmentation_info


def suggest_memory_optimization(fragmentation_info: Dict[str, Any]) -> List[str]:
    """
    Suggest memory optimization strategies.

    Args:
        fragmentation_info: Output from check_memory_fragmentation

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    fragmentation_ratio = fragmentation_info.get("fragmentation_ratio", 0)
    utilization_ratio = fragmentation_info.get("utilization_ratio", 0)
    waste_ratio = fragmentation_info.get("waste_ratio", 0)

    if fragmentation_ratio > 0.3:
        suggestions.append(
            "High memory fragmentation detected. Consider calling torch.cuda.empty_cache() "
            "periodically or restructuring your code to reduce fragmentation."
        )

    if utilization_ratio > 0.9:
        suggestions.append(
            "Very high GPU memory utilization. Consider reducing batch size, "
            "using gradient checkpointing, or model parallelism."
        )

    if waste_ratio > 0.2:
        suggestions.append(
            "Significant memory waste detected. Review memory allocation patterns "
            "and consider using more efficient data structures."
        )

    if utilization_ratio < 0.3:
        suggestions.append(
            "Low GPU memory utilization. Consider increasing batch size or "
            "using a larger model to better utilize available memory."
        )

    # General suggestions
    suggestions.extend([
        "Use torch.no_grad() context for inference to reduce memory usage.",
        "Consider using mixed precision training (torch.cuda.amp) to reduce memory footprint.",
        "Profile memory usage at different points in your code to identify bottlenecks.",
        "Use del statement to explicitly delete large tensors when no longer needed.",
    ])

    return suggestions


def memory_summary(device: Optional[Union[str, int, torch.device]] = None) -> str:
    """
    Generate a comprehensive memory summary.

    Args:
        device: GPU device to summarize

    Returns:
        Formatted memory summary string
    """
    gpu_info = get_gpu_info(device)
    fragmentation_info = check_memory_fragmentation(device)
    suggestions = suggest_memory_optimization(fragmentation_info)

    summary = []
    summary.append("=" * 60)
    summary.append("GPU MEMORY SUMMARY")
    summary.append("=" * 60)

    # Device info
    summary.append(
        f"Device: {gpu_info.get('device_name', 'Unknown')} (cuda:{gpu_info.get('device_id', 0)})")
    summary.append(
        f"Total Memory: {format_bytes(gpu_info.get('total_memory', 0))}")
    summary.append("")

    # Current usage
    summary.append("Current Memory Usage:")
    summary.append(
        f"  Allocated: {format_bytes(fragmentation_info.get('allocated_memory', 0))}")
    summary.append(
        f"  Reserved:  {format_bytes(fragmentation_info.get('reserved_memory', 0))}")
    summary.append(
        f"  Free:      {format_bytes(fragmentation_info.get('free_memory', 0))}")
    summary.append("")

    # Ratios
    summary.append("Memory Ratios:")
    summary.append(
        f"  Utilization: {fragmentation_info.get('utilization_ratio', 0):.1%}")
    summary.append(
        f"  Reservation: {fragmentation_info.get('reservation_ratio', 0):.1%}")
    summary.append(
        f"  Fragmentation: {fragmentation_info.get('fragmentation_ratio', 0):.1%}")
    summary.append(f"  Waste: {fragmentation_info.get('waste_ratio', 0):.1%}")
    summary.append("")

    # Suggestions
    if suggestions:
        summary.append("Optimization Suggestions:")
        # Show top 5 suggestions
        for i, suggestion in enumerate(suggestions[:5], 1):
            summary.append(f"  {i}. {suggestion}")
        summary.append("")

    summary.append("=" * 60)

    return "\n".join(summary)


class MemoryContext:
    """Context manager for tracking memory usage in a block of code."""

    def __init__(self, name: str = "memory_context", device: Optional[Union[str, int, torch.device]] = None):
        self.name = name
        self.device = device
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_memory = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize(self.device)
        self.end_memory = torch.cuda.memory_allocated(self.device)
        self.peak_memory = torch.cuda.max_memory_allocated(self.device)

    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary for this context."""
        if self.start_memory is None or self.end_memory is None:
            return {"error": "Context not properly initialized"}

        return {
            "name": self.name,
            "start_memory": self.start_memory,
            "end_memory": self.end_memory,
            "peak_memory": self.peak_memory,
            "memory_diff": self.end_memory - self.start_memory,
            "peak_memory_usage": self.peak_memory - self.start_memory,
            "start_memory_formatted": format_bytes(self.start_memory),
            "end_memory_formatted": format_bytes(self.end_memory),
            "peak_memory_formatted": format_bytes(self.peak_memory),
            "memory_diff_formatted": format_bytes(abs(self.end_memory - self.start_memory)),
            "peak_memory_usage_formatted": format_bytes(self.peak_memory - self.start_memory),
        }
