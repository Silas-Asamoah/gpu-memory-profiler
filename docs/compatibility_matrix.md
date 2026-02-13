[← Back to main docs](index.md)

# Compatibility Matrix

This matrix reflects current backend behavior in this repository.

## PyTorch (`gpumemprof`)

| Runtime backend | Typical platform | `gpumemprof info` | `gpumemprof monitor` | `gpumemprof track` | Telemetry collector | `device_total/free` support |
| --- | --- | --- | --- | --- | --- | --- |
| `cuda` | NVIDIA + CUDA | ✅ | ✅ | ✅ | `gpumemprof.cuda_tracker` | ✅ |
| `rocm` | AMD + ROCm (Linux) | ✅ | ✅ | ✅ | `gpumemprof.rocm_tracker` | ✅ |
| `mps` | Apple Silicon (macOS) | ✅ | ✅ | ✅ | `gpumemprof.mps_tracker` | Partial (backend-dependent) |
| `cpu` | Any host | ✅ | ✅ | ✅ | `gpumemprof.cpu_tracker` | N/A |

## TensorFlow (`tfmemprof`)

| Runtime backend | Typical platform | `tfmemprof info` diagnostics | `tfmemprof monitor/track` | Telemetry collector |
| --- | --- | --- | --- | --- |
| `cuda` | NVIDIA + CUDA | ✅ | ✅ | `tfmemprof.memory_tracker` |
| `rocm` | AMD + ROCm (Linux) | ✅ (build/runtime diagnostics) | ✅ | `tfmemprof.memory_tracker` |
| `metal` | Apple Silicon / Apple GPU path | ✅ (backend diagnostics) | ✅ (runtime-dependent counters) | `tfmemprof.memory_tracker` |
| `cpu` | Any host | ✅ | ✅ | `tfmemprof.memory_tracker` |

## Notes

- Backend capability metadata is emitted in tracker exports under `metadata`:
  - `backend`
  - `supports_device_total`
  - `supports_device_free`
  - `sampling_source`
- Some platform backends may expose zero/limited runtime device counters despite successful backend detection (for example, certain TensorFlow Metal configurations).
