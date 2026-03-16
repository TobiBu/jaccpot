# Octree FMM Status

This note summarizes the current octree-native FMM status in `jaccpot` and the
remaining work items that still deserve attention.

## Current State

The octree backend now exists as a real execution path beside the radix
backend.

Implemented and validated:

- explicit `execution_backend` support with `auto|radix|octree`
- octree-native prepared-state metadata attached to solver state
- octree-native upward/downward scaffolding in explicit octree node space
- native octree far-pair consumption from `yggdrax`
- native octree near-neighbor interop for prepared-state and target-subset
  evaluation
- prepared-state evaluation for:
  - full outputs
  - target subsets
  - potentials
  - JIT/eager traversal
  - prepared-state cache reuse
- non-default runtime coverage for:
  - baseline nearfield mode
  - class-major farfield mode with grouped interactions

Current practical scope:

- the explicit octree backend is validated for `basis="solidfmm"`
- radix remains the default execution path unless octree execution is requested
- topology reuse remains radix-only by design

## Remaining Work

### 1. Broaden config coverage

- add more octree coverage for non-default MAC and runtime combinations
- add explicit guardrail tests for unsupported basis/backend combinations
- add more coverage for large-memory and minimum-memory runtime settings

Status: in progress

### 2. Benchmarking and reporting

- compare radix vs octree prepare/evaluate throughput on representative CPU/GPU
  workloads
- document preferred validation hardware and known device-sensitive behavior
- add a stable octree benchmark/example workflow

Status: pending

### 3. User-facing polish

- keep README/examples aligned with the supported octree workflow
- expand example coverage beyond the prepare/evaluate comparison script
- document recommended solver settings for octree validation runs

Status: in progress
