[← Back to main docs](index.md)

# TelemetryEvent v2 Schema

`TelemetryEvent v2` is the canonical event format for tracker exports.

Schema file:

`docs/schemas/telemetry_event_v2.schema.json`

## Required fields

- `schema_version` (`2`)
- `timestamp_ns`
- `event_type`
- `collector`
- `sampling_interval_ms`
- `pid`
- `host`
- `device_id`
- `allocator_allocated_bytes`
- `allocator_reserved_bytes`
- `allocator_active_bytes`
- `allocator_inactive_bytes`
- `allocator_change_bytes`
- `device_used_bytes`
- `device_free_bytes`
- `device_total_bytes`
- `context`
- `metadata`

## Collector values

- `gpumemprof.cuda_tracker`
- `gpumemprof.cpu_tracker`
- `tfmemprof.memory_tracker`

## Legacy v1 to v2 conversion defaults

Conversion is permissive by default in `gpumemprof.telemetry.telemetry_event_from_record`.

- Missing `pid` -> `-1`
- Missing `host` -> `"unknown"`
- Missing `device_id` -> inferred from `device` if possible, otherwise `-1`
- Missing `allocator_reserved_bytes` -> `allocator_allocated_bytes`
- Missing `allocator_change_bytes` -> `0`
- Missing `device_used_bytes` -> `allocator_allocated_bytes`
- Missing `device_total_bytes` and `device_free_bytes` -> `null`
- Missing `event_type` -> `type` field if present, else `"sample"`
- Legacy `metadata_*` fields are folded into the v2 `metadata` object

If a legacy record is missing a valid timestamp, conversion fails.

## Python API

Use the public conversion/validation helpers in `gpumemprof.telemetry`:

```python
from gpumemprof.telemetry import (
    load_telemetry_events,
    telemetry_event_from_record,
    telemetry_event_to_dict,
    validate_telemetry_record,
)
```

- `load_telemetry_events(path, permissive_legacy=True, events_key=None)`
- `telemetry_event_from_record(record, permissive_legacy=True, ...)`
- `validate_telemetry_record(record)`

These APIs normalize legacy records to `schema_version: 2` and enforce required fields.
