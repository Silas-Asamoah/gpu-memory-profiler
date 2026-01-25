"""Console formatting helpers for examples."""

from __future__ import annotations

from typing import Any


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def print_section(title: str) -> None:
    line = "-" * len(title)
    print(f"\n{title}\n{line}")


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")
