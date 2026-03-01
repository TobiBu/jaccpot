# Adaptive Traversal Design

## Problem

`jaccpot` previously relied on solver-specific adaptive traversal concepts being
implemented directly in `yggdrax`, including:

- `mac_type="dehnen_error"`
- traversal-time `node_features`
- structured far-pair buckets by expansion gear

That mixes responsibilities across the two repositories. Multipole-derived
error estimates and adaptive expansion-order selection are solver concerns and
should remain in `jaccpot`.

## Ownership Boundary

`yggdrax` owns:

- tree construction
- per-node geometry
- generic dual-tree traversal
- generic JAX-traceable `pair_policy`
- generic integer `interaction_tags` on accepted far pairs

`jaccpot` owns:

- multipole data and derived moment summaries
- target force scales
- adaptive accept / near / refine logic
- required expansion-order selection
- bucketing far pairs by selected order

## Target Data Flow

1. Build tree and geometry with `yggdrax`
2. Build upward multipoles in `jaccpot`
3. Derive solver-owned adaptive policy state in `jaccpot`
4. Call `yggdrax.build_interactions_and_neighbors(...)` with:
   - `pair_policy`
   - `policy_state`
   - `return_result=True`
5. Read:
   - `interaction_sources`
   - `interaction_targets`
   - `interaction_tags`
6. Bucket accepted far pairs by `interaction_tags` inside `jaccpot`
7. Dispatch far-field kernels by selected order bucket

## Migration Steps

1. Introduce a dedicated `jaccpot.runtime._adaptive_policy` module
2. Move multipole-derived adaptive proxy construction there
3. Remove all traversal-time `node_features` / `dehnen_error` dependencies
4. Switch to generic `pair_policy` + `interaction_tags`
5. Replace old structured far-pair gear buckets with tag-based bucketing
6. Update tests, examples, and docs

## Legacy Path To Remove

The following legacy integration path should be removed during this branch:

- `build_dehnen_error_node_features(...)`
- `_tail_power_by_gear_from_multipoles(...)` naming and API
- `p_gears` / `eps` / `node_features` passed into `yggdrax` traversal
- structured `far_pairs_by_gear` returned by traversal
- fallback logic that assigns all far pairs to the highest gear

## Current Acceptance Model

The current jaccpot runtime uses a solver-owned adaptive acceptance rule built
on yggdrax's generic traversal hook:

- `yggdrax` provides generic pair traversal plus integer `interaction_tags`
- `jaccpot` computes multipole-derived residual proxies and target force scales
- acceptance is decided by the highest available order together with a relaxed
  geometric guard
- order selection uses the smallest passing order among `p_gears`

This keeps tree traversal generic while making both the MAC decision and the
required expansion order solver-owned.
