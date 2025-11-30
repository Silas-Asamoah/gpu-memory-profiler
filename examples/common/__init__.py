"""Shared helper utilities for example scripts."""

from .device import (
    seed_everything,
    get_torch_device,
    get_tf_device,
    describe_torch_environment,
    describe_tf_environment,
)
from .formatting import print_header, print_section, print_kv
from .cli import run_cli_command, print_cli_result, ensure_cli_available
from .summary import print_profiler_summary
from .torch_workflow import (
    build_simple_torch_model,
    generate_torch_batch,
    run_torch_train_step,
)
from .tf_workflow import (
    build_simple_tf_model,
    generate_tf_batch,
    run_tf_train_step,
)

__all__ = [
    "seed_everything",
    "get_torch_device",
    "get_tf_device",
    "describe_torch_environment",
    "describe_tf_environment",
    "print_header",
    "print_section",
    "print_kv",
    "run_cli_command",
    "print_cli_result",
    "ensure_cli_available",
    "print_profiler_summary",
    "build_simple_torch_model",
    "generate_torch_batch",
    "run_torch_train_step",
    "build_simple_tf_model",
    "generate_tf_batch",
    "run_tf_train_step",
]
