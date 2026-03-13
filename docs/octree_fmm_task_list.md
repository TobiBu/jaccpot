# Octree FMM Task List

This plan tracks the work needed to add an octree-native FMM path beside the
existing radix-based execution path.

## Principles

- Keep the current radix FMM path intact.
- Add an octree FMM backend next to radix, not as a rewrite.
- Share only topology-agnostic code.
- Fail fast when users request an octree execution backend that has not yet
  been implemented.

## Milestones

### 1. Backend Split

- Add explicit `execution_backend` config with `auto|radix|octree`.
- Preserve current behavior under `auto`.
- Make explicit `octree` backend requests fail fast until the backend exists.
- Store the resolved execution backend in prepared state.

Status: in progress

### 2. Octree-Native Prepared-State Scaffolding

- Add octree-native execution containers for upward/downward/interactions.
- Define octree node-space indexing rules.
- Keep radix<->octree mapping arrays as bridge data only.

Status: pending

### 3. Octree Upward Sweep

- Implement octree-native P2M using `oct_leaf_nodes`.
- Implement octree-native M2M using `oct_children` and `oct_level_offsets`.
- Add parity tests against the current radix execution path.

Status: pending

### 4. Octree Interaction Scheduling

- Build or remap far-field interaction lists into octree node space.
- Add octree-native scheduling buffers for levelwise M2L.

Status: pending

### 5. Octree Downward Sweep

- Implement octree-native M2L.
- Implement octree-native L2L.
- Evaluate locals on octree leaf nodes.

Status: pending

### 6. Benchmark and Compare

- Compare radix vs octree for:
  - prepare_state time
  - upward sweep
  - interaction scheduling
  - downward sweep
  - total runtime
- Compare correctness across accelerations and potentials.

Status: pending
