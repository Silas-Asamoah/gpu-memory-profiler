"""Shared helpers for hidden-memory gap analysis tests."""

from gpumemprof.telemetry import SCHEMA_VERSION_V2, TelemetryEventV2

BASE_NS = 1_700_000_000_000_000_000
INTERVAL_NS = 100_000_000  # 100 ms between samples
DEVICE_TOTAL_BYTES = 16 * 1024**3  # 16 GiB device


def build_gap_event(
    *,
    index: int,
    allocator_allocated: int,
    allocator_reserved: int,
    device_used: int,
    collector: str,
    device_total: int = DEVICE_TOTAL_BYTES,
) -> TelemetryEventV2:
    """Build a minimal valid TelemetryEventV2 for gap analysis tests."""
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=BASE_NS + index * INTERVAL_NS,
        event_type="sample",
        collector=collector,
        sampling_interval_ms=100,
        pid=1,
        host="test",
        device_id=0,
        allocator_allocated_bytes=allocator_allocated,
        allocator_reserved_bytes=allocator_reserved,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=0,
        device_used_bytes=device_used,
        device_free_bytes=device_total - device_used,
        device_total_bytes=device_total,
        context=None,
    )
