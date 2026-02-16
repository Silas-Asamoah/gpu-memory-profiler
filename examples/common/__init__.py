"""Shared helper utilities for example scripts.

Framework-specific helpers are loaded lazily so examples can be imported in
single-framework environments (for example, torch-only CI jobs).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .cli import ensure_cli_available, print_cli_result, run_cli_command
from .formatting import print_header, print_kv, print_section
from .summary import print_profiler_summary

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "seed_everything": (".device", "seed_everything"),
    "get_torch_device": (".device", "get_torch_device"),
    "get_tf_device": (".device", "get_tf_device"),
    "describe_torch_environment": (".device", "describe_torch_environment"),
    "describe_tf_environment": (".device", "describe_tf_environment"),
    "build_simple_torch_model": (".torch_workflow", "build_simple_torch_model"),
    "generate_torch_batch": (".torch_workflow", "generate_torch_batch"),
    "run_torch_train_step": (".torch_workflow", "run_torch_train_step"),
    "build_simple_tf_model": (".tf_workflow", "build_simple_tf_model"),
    "generate_tf_batch": (".tf_workflow", "generate_tf_batch"),
    "run_tf_train_step": (".tf_workflow", "run_tf_train_step"),
}

__all__ = [
    "print_header",
    "print_section",
    "print_kv",
    "run_cli_command",
    "print_cli_result",
    "ensure_cli_available",
    "print_profiler_summary",
    "seed_everything",
    "get_torch_device",
    "get_tf_device",
    "describe_torch_environment",
    "describe_tf_environment",
    "build_simple_torch_model",
    "generate_torch_batch",
    "run_torch_train_step",
    "build_simple_tf_model",
    "generate_tf_batch",
    "run_tf_train_step",
]


def __getattr__(name: str) -> Any:
    lazy_target = _LAZY_EXPORTS.get(name)
    if lazy_target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = lazy_target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
