"""Tests for TelemetryEvent v2 schema and legacy conversions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate as jsonschema_validate

from gpumemprof.telemetry import (
    SCHEMA_VERSION_V2,
    UNKNOWN_HOST,
    UNKNOWN_PID,
    TelemetryEventV2,
    load_telemetry_events,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)


def _schema() -> dict:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "schemas"
        / "telemetry_event_v2.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _make_valid_event() -> TelemetryEventV2:
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=1_700_000_000_000_000_000,
        event_type="sample",
        collector="gpumemprof.cuda_tracker",
        sampling_interval_ms=100,
        pid=1234,
        host="host-a",
        device_id=0,
        allocator_allocated_bytes=1024,
        allocator_reserved_bytes=2048,
        allocator_active_bytes=512,
        allocator_inactive_bytes=1536,
        allocator_change_bytes=256,
        device_used_bytes=2048,
        device_free_bytes=4096,
        device_total_bytes=6144,
        context="unit",
        metadata={"origin": "test"},
    )


def test_telemetry_event_v2_serialization_validates_against_schema() -> None:
    event = _make_valid_event()
    record = telemetry_event_to_dict(event)

    validate_telemetry_record(record)
    jsonschema_validate(instance=record, schema=_schema())


def test_validate_telemetry_record_rejects_missing_fields() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record.pop("collector")

    with pytest.raises(ValueError, match="Missing required telemetry fields"):
        validate_telemetry_record(record)


def test_validate_telemetry_record_rejects_negative_allocator_counter() -> None:
    record = telemetry_event_to_dict(_make_valid_event())
    record["allocator_allocated_bytes"] = -1

    with pytest.raises(ValueError, match="allocator_allocated_bytes must be >= 0"):
        validate_telemetry_record(record)


def test_legacy_gpumemprof_record_converts_to_v2() -> None:
    legacy = {
        "timestamp": 1700000000.25,
        "event_type": "allocation",
        "memory_allocated": 10_000,
        "memory_reserved": 15_000,
        "memory_change": 512,
        "device_id": 0,
        "context": "alloc",
        "metadata_usage_percent": 75.5,
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="gpumemprof.cuda_tracker",
        default_sampling_interval_ms=100,
    )
    record = telemetry_event_to_dict(event)

    assert record["schema_version"] == 2
    assert record["collector"] == "gpumemprof.cuda_tracker"
    assert record["allocator_allocated_bytes"] == 10_000
    assert record["allocator_reserved_bytes"] == 15_000
    assert record["allocator_change_bytes"] == 512
    assert record["metadata"]["usage_percent"] == 75.5
    jsonschema_validate(instance=record, schema=_schema())


def test_legacy_cpu_record_converts_with_defaults() -> None:
    legacy = {
        "timestamp": 1700000001.0,
        "event_type": "allocation",
        "memory_allocated": 8_192,
        "memory_change": 1_024,
        "context": "cpu",
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="gpumemprof.cpu_tracker",
        default_sampling_interval_ms=200,
    )
    record = telemetry_event_to_dict(event)

    assert record["collector"] == "gpumemprof.cpu_tracker"
    assert record["device_id"] == -1
    assert record["allocator_reserved_bytes"] == record["allocator_allocated_bytes"]
    assert record["device_total_bytes"] is None
    assert record["pid"] == UNKNOWN_PID
    assert record["host"] == UNKNOWN_HOST
    jsonschema_validate(instance=record, schema=_schema())


def test_legacy_tf_record_converts_with_defaults() -> None:
    legacy = {
        "timestamp": 1700000002.5,
        "type": "sample",
        "memory_mb": 2.0,
        "device": "/GPU:0",
    }

    event = telemetry_event_from_record(
        legacy,
        default_collector="legacy.unknown",
        default_sampling_interval_ms=500,
    )
    record = telemetry_event_to_dict(event)

    assert record["collector"] == "tfmemprof.memory_tracker"
    assert record["device_id"] == 0
    assert record["allocator_allocated_bytes"] == 2 * 1024 * 1024
    assert record["device_used_bytes"] == 2 * 1024 * 1024
    jsonschema_validate(instance=record, schema=_schema())


def test_load_telemetry_events_reads_dict_events_payload(tmp_path: Path) -> None:
    payload = {
        "peak_memory": 123.4,
        "events": [
            {
                "timestamp": 1700000003.0,
                "type": "sample",
                "memory_mb": 1.0,
                "device": "/GPU:0",
            }
        ],
    }
    path = tmp_path / "tf_track.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    events = load_telemetry_events(path, events_key="events")

    assert len(events) == 1
    assert events[0].schema_version == SCHEMA_VERSION_V2


def test_legacy_conversion_can_be_disabled() -> None:
    with pytest.raises(ValueError, match="Legacy record conversion is disabled"):
        telemetry_event_from_record(
            {"timestamp": 1.0, "memory_allocated": 1},
            permissive_legacy=False,
        )
