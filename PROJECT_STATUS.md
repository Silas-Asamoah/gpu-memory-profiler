# GPU Memory Profiler - Project Status

## Current Status: v0.2 Launch Candidate

**Version Target**: `0.2.0`  
**Branch**: `release/v0.2-readiness`  
**State**: Launch QA hardening in progress

## Completed in This Launch Hardening Pass

### 1. Scenario-Coverage Expansion

- Added launch scenario modules under `examples/scenarios/`:
  - `cpu_telemetry_scenario.py`
  - `mps_telemetry_scenario.py`
  - `oom_flight_recorder_scenario.py`
  - `tf_end_to_end_scenario.py`
- Added launch orchestrator:
  - `examples/cli/capability_matrix.py`
- Added shared matrix utilities:
  - `examples/common/capability_matrix_utils.py`

### 2. Example Reliability Improvements

- Fixed TensorFlow basic demo API drift in `examples/basic/tensorflow_demo.py`.
- Stabilized benchmark harness defaults for less noisy local/CI budget checks.

### 3. TUI Launch Workflow Alignment

- Updated TUI playbook and CLI action content to surface launch-critical workflows:
  - diagnose command
  - OOM scenario command
  - capability matrix smoke command
- Updated TUI pilot/snapshot coverage and snapshot baselines for these intentional UI changes.

### 4. Documentation and Release Process Refresh

- README now documents launch scenario workflows and capability matrix usage.
- Example docs now include scenario matrix guidance.
- Release checklist rewritten for v0.2 criteria and command-level QA gates.
- CI includes capability matrix smoke execution in `cli-test`.

## QA Evidence Model for v0.2

Release confidence is based on:

1. Command-level QA gate in `tensor-torch-profiler` env.
2. Capability matrix smoke/full reports.
3. CI checks (unit/integration, TUI gates, build/twine).

## Known Local Risk (Tracked)

- In this local macOS environment, `pytest` startup can segfault in `tensor-torch-profiler`.
- This is tracked as an environment-specific risk and does not replace CI gating.
- v0.2 sign-off is based on CI + command-level QA evidence until local env issue is resolved.

## Remaining Steps to Publish

1. Run the full `RELEASE_CHECKLIST.md` and archive artifacts.
2. Confirm CI green on final release PR.
3. Tag and publish `v0.2.0`.
4. Post-release smoke install + quickstart verification.

