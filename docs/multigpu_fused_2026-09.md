# The fused fast lane on many GPUs (2026-09, in progress)

Plan: `~/.claude/plans/ok-we-have-implemented-nested-clarke.md`. Target: jz-fmm's multi-GPU front, as its
single-GPU front was the target for `docs/sub10ms_2026-09.md`.

## Why

The sub-10 ms work took the single-GPU force to **11.45 ms at N = 2x10^5** (cells64, theta 0.8, p6), at parity
with jz-fmm. None of it reaches `jaccpot/distributed/fmm.py`, which is a separately assembled per-device pipeline
-- bucket radix tree rebuilt in-trace, traced dual-tree wavefront walk, by-level JAX cascades, pure-JAX
rotate-scale M2L, rectangle near-field kernel -- running **0.2-1.3 M particles/s against the fused lane's 17.5**.

The gap is not communication: the LET halo exchange is 0.4-0.7 % of a step, while ~43 % of a 10^7/5-card force is
fixed overhead (in-trace tree build, launch count). Both are what the fused lane fixed on one card.

## Decisions

One-sided cross-domain evaluation on the receiving device (jz-fmm's scheme: no reverse halo, no cross-domain
canonicalisation); multi-node from the start, so nothing host-global may remain; forward force first, gradients
later.

## Built so far

### Padded shards (yggdrax)

`num_valid` through `adaptive_cell_leaf_partition` -> `build_static_cells_tree` ->
`rebuild_static_radix_tree_from_template`, refused on the bucket path.

Sorting the dead rows last under the reserved sentinel is necessary and **not sufficient**: they all carry the
same code, so no Morton level splits them and they collapse into one leaf -- measured **4384 rows at leaf_size
64, 68x the contract width**. Under trace that truncates into a live, massless, zero-radius leaf at a real
position, which fails the MAC against everything near it. `num_valid` cuts the run boundaries (a dead row is a
boundary at every level, without which the last live cell's occupancy counts the padding), the leaf starts, the
live-leaf count and the last leaf's end.

A padded shard then partitions **exactly** like the same particles unpadded, and its tree has the same live node
count, covers exactly the live particles and conserves mass.

### Static shapes without an eager visit (jaccpot)

`jaccpot/runtime/capacity_plan.py`. The level cascades take their per-level batch width and level count from a
process-level registry that an eager `prepare_state` fills before the traced refresh reads it. Inside `shard_map`
no eager prepare ever runs, and the failure is silent: the width falls back to `num_internal`, and
`registered_num_levels` returns `None`, which drops the `pallas_levels` key and **deselects the per-level Pallas
L2L cascade outright**. At N = 2x10^5 the plan gives `level_batch_width` 6788 against `num_internal` 16383 -- the
fallback is 2.4x wider per level, and the cascade it would have run is the one that cost 44 ms per step.

A second failure appears only with one process per GPU: the registry is process-global and filled from each
process's own shard, so the key matches everywhere while the values differ and two processes compile different
static shapes for one SPMD program. Three same-sized shards measured widths 250 / 233 / 262 -- `merge_plans` takes
the field-wise maximum for exactly this.

`upward_num_levels` is a separate field on purpose: the upward sweep bounds tree DEPTH plus headroom while the
registry counts the level table's live levels plus headroom, and substituting one for the other truncates a sweep
silently.

### The per-device force (jaccpot)

`jaccpot/distributed/fused.py`: `global_mesh_bounds`, `reduce_flag_across_mesh`, `fused_force_step`.

`fused_force_step` is the seam `strict_run_v2` calls per endpoint, lifted out of its velocity-Verlet scan. The two
neighbouring entry points are both wrong for a force evaluator: `strict_run_v2` bundles the integrator, and
`strict_fused_prepared_eval_fn` omits the refresh by design.

`bounds` is **required, not defaulted**. `_resolve_prepare_state_bounds` already honours an explicit box, so a
global Morton frame needed no library change -- but a silently re-inferred per-shard box is exactly the failure
that would de-align the mesh while every test still passed.

In `global_mesh_bounds`, dead rows are folded onto a live position rather than masked with infinities: masking
would make a device whose shard is entirely padding reduce to `inf` and poison the box for the whole mesh.

## Gate G1 (ndev = 1): measured

One A100, N = 2x10^5, cells64, theta 0.8, p4, against an fp64 direct sum on 2048 targets.
Probe: `bench/multigpu_fused_shardmap_probe.py`. Host load ~35, so **no timing is quoted**.

| arm | aggL2 |
|---|---|
| outside `shard_map` | 2.2625e-03 |
| inside `shard_map`, same box | 2.2625e-03 |
| inside `shard_map`, box from the all-reduce | 2.2392e-03 |

| comparison | max abs | rel-L2 |
|---|---|---|
| same box, inside vs outside | 6.985e-10 | **1.760e-11** |
| capacity-padded vs unpadded, live rows | 4.657e-10 | **1.534e-11** |

**The fused pipeline traces and runs unchanged under a manual axis** -- Pallas kernels, the walk's `while_loop`
and the nested jits included. Both differences are fp32 accumulation noise.

The control is the point. The production arm sits 2.6e-3 from the reference, the size of the FMM error itself,
purely because the all-reduced box carries slack that `infer_bounds` does not and so builds a different (equally
valid) tree. Comparing only that arm would leave "the trace changed the answer" and "the box changed the tree"
indistinguishable.

### Per-device states (jaccpot)

`stack_prepared_states` + `make_fused_force_evaluator`. Beyond one device the prepared state cannot ride as a
closure constant -- each device needs its own tree, multipoles and lists -- so it becomes a `shard_map` input,
which requires every leaf to share a shape across devices.

Measured at N = 2x10^5 over two Morton domains: **the states already stack** -- identical treedef, 55 leaves each,
no shape disagreement -- because the interaction-list capacities are fixed by configuration rather than measured
per shard. What does differ per shard is the level-shape plan: **widths 3205 and 3460, depths 44 and 50** on those
two domains. That is `merge_plans` earning its place at production scale, not on a toy tree.

`stack_prepared_states` names the leaf that disagrees when one does; "cannot stack" over 55 anonymous leaves is
not an actionable message.

## Gate G1 (ndev = 2): measured

Two A100s (cards 1 and 7, loadavg 23), N = 2x10^5 split into two Morton domains of 100,000, cap 114,999,
leaf_capacity 8192, cells64, theta 0.8, p4. Probe: `bench/multigpu_fused_ndev2_probe.py`. No cross-domain field
yet -- each device computes the force of its OWN shard on itself, and the reference is that same computation run
singly, so any difference is the mesh plumbing alone.

| device | mesh vs single-device, max abs | rel-L2 | local-only aggL2 vs fp64 direct over its shard |
|---|---|---|---|
| 0 | 9.313e-10 | **2.361e-11** | 2.7920e-03 |
| 1 | 2.328e-10 | **2.080e-11** | 3.2663e-03 |

No overflow. **The mesh plumbing is transparent to fp32 noise** -- stacking, the per-device slice, the specs, the
box all-reduce and the plan installation together contribute 2e-11, the same level as ndev = 1's 1.76e-11.

The per-device plans differed as expected (width 3205 / depth 44 on device 0, 3460 / 50 on device 1) and merged
to 3460 / 50, which is the width both devices then compiled against.

## Traps found

* The fused lane's profile gate keys on the **exact array length**. A shard's length is the padded capacity, not
  the live count, so `JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET` must admit `cap`.
* The **per-leaf capacity fits are real**: the default fast-lane environment is the leaf-256 entry, and leaf 64
  has ~10x the far pairs and does not fit without its own preset plus traversal overrides. Both are whole-problem
  host state that no process can compute under one process per GPU -- which is what the capacity plan must absorb.
* **The solver carries host-side caches that only an eager prepare fills**, and a cold one fails *inside* the
  traced body: `_resolve_dual_downward_planner_hint` calls a jitted planner and takes `bool()` of its result,
  which is concrete only when `_refresh_dual_planner_cache` is warm. Building the evaluator around a fresh solver
  dies on a `TracerBoolConversionError`. This is the **third** site of the same "an eager prepare always precedes
  a traced refresh" assumption, after the level-shape registry and the upward depth stash -- the first two degrade
  silently, this one at least fails loudly. The driver must run its eager per-shard prepares on the SAME solver
  instance it then builds the `shard_map` around.
* **`shard_map` does not remove the mapped axis.** With `in_specs=P("gpus")` a `(ndev, cap, 3)` input arrives
  inside the body as `(1, cap, 3)`, so a `min(axis=0)` reduces the DEVICE axis instead of the particles and every
  collective after it compares whole shards elementwise -- silently, with plausible shapes. It surfaced here as a
  global box that came back equal to one shard's raw positions. Particle arrays are therefore passed FLAT as
  `(ndev * cap, ...)`, matching `jaccpot/distributed/fmm.py`; per-device pytrees are sliced `[0]` on entry.
  jz-fmm ships a wrapper (`expanding_shard_map`) whose whole purpose is stripping this axis.
* yggdrax exposes the tree builders **twice**, an implementation and a public wrapper, and jaccpot imports the
  wrapper. Threading a parameter through the implementation alone leaves it unreachable, surfacing as a jaxtyping
  signature-bind `TypeError` three frames from the cause; tests run from inside the yggdrax worktree call the
  implementation directly and miss it.

## Next

Phase 1 is done: the fused lane runs per device under one `shard_map`, at parity with the single-device lane to
fp32 noise, at one and at two devices. What it does NOT yet have is any cross-domain field -- each device sees
only its own shard.

That is plan Phase 3, and it begins with the volume probe that decides its shape
(`bench/multigpu_cross_volume_probe.py`): the cross-domain near volume as a fraction of the local one, the
exported node set per ordered domain pair, and the opened particles per pair. The first of those can invalidate
the approach -- with bucket leaves the cross half is already 54-80 % of near leaf pairs at 4-6 devices, and cell
leaves' effect on that share has never been measured.
