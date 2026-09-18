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

## Phase 3.1 -- the cross-domain volume probe (partial, and one number is unexplained)

`bench/multigpu_cross_volume_probe.py`, one A100, Plummer, cells64, theta 0.8, Morton domains snapped to a
level-`align_level` cell edge (zero straddling leaves, which is what alignment buys).

**Solid, and favourable.** The cross-domain share of the near field is small and the tree is what makes it so:

| N | ndev | cross share of near VOLUME | cross share of near PAIRS |
|---|---|---|---|
| 2x10^5 | 2 / 4 / 8 | 0.099 / 0.099 / 0.112 | 0.097 / 0.097 / 0.112 |
| 10^6 | 2 / 4 / 8 | 0.014 / 0.014 / 0.043 | 0.039 / 0.039 / 0.078 |

At 10^6 only 1.4-4.3 % of the near volume crosses a domain boundary. One-sided evaluation doubles the cross half,
so it costs a few per cent of the near field, not a factor. That is the number that could have killed the design
and it did not.

**RESOLVED (see the section below): the import is fine, and ONE leaf was hiding that.** The text that follows
records the state before the resolution.

**Measured but UNEXPLAINED at the time -- superseded.** The import volume per device comes out at exactly its own
particle count at every device count:

| ndev | imported leaves | share of EACH foreign domain | imported particles | halo / local |
|---|---|---|---|---|
| 2 | 25,944 | 1.000 | 499,972 | 1.00 |
| 4 | 12,972 | 0.333 | 249,986 | 1.00 |
| 8 | 6,486 | 0.143 | 124,996 | 1.00 |

The figures are internally consistent (leaves x occupancy = particles, occupancy 19.3 matching the tree's own
mean), but the per-source share is exactly `1/(ndev-1)` in all three rows and the particle count matches `N/ndev`
to 0.006 %. A boundary region has no reason to scale that way. Either a Morton bisection's boundary really is
fractal enough to reach every leaf -- which is plausible, a z-order split interlocks at every scale -- or the
union logic has a structural flaw. **Resolve this before the export design depends on it**; jz-fmm's comparable
figure is 10-60 % of local data, and 100 % would change the design.

**The RCB control could not be run.** RCB cuts through cells (3,133-9,382 straddling leaves at 10^6), so the
leaf-to-domain assignment by first particle is wrong for those and they are miscounted as cross-domain, inflating
RCB's cross fraction from 0.014 to 0.168. That is a measurement of the assignment breaking, not of RCB. Comparing
the two partitioners needs per-domain trees, or a leaf assignment that handles straddling.

Two mistakes in the probe worth remembering: means over ordered domain pairs summarise nothing when most pairs are
distant and export zero (use per-device unions), and `occ` is indexed over ALL nodes, so counting `occ > 0` across
it includes every internal node and doubles the leaf count -- which halves the apparent per-domain leaf share and
inverts the reading of the import volumes.

## The import volume, resolved: one leaf, not the decomposition

The earlier probe measured the wrong object -- ONE global tree with leaves merely LABELLED by domain -- so it was
rebuilt as `bench/multigpu_import_volume_probe.py`: each domain gets its own cell tree in the shared Morton frame,
and the cross relation is the real `dual_tree_walk_cross_impl` the distributed lane uses. Per-domain trees also
dissolve the RCB problem, because leaves cannot straddle a cut they were never built across.

**The decomposition was never the problem.** Cross-neighbours per target leaf, plane split, N = 2x10^5, ndev = 2,
cells64, theta 0.8:

| statistic | foreign leaves needed |
|---|---|
| median | **0** |
| p90 | 8 |
| p99 | 23 |
| max | **5,523** (of 5,534 -- the entire source domain) |

Half of all leaves import NOTHING. The import is one leaf:

| excluded target leaves | imported leaves | imported particles |
|---|---|---|
| none | 99.8 % | 100.0 % |
| **the single worst** | **28.6 %** | **30.5 %** |
| top 10 | 27.1 % | 29.1 % |
| top 50 | 24.6 % | 26.7 % |

The offender has radius **119.88 against a median leaf radius of 0.128** -- about 900x. Its two runners-up are
33.83 and 33.28. **A Morton cell bounds OCCUPANCY, not EXTENT**: in a sparse halo the coarsest cell holding 64
particles is geometrically vast, and such a cell fails the MAC against everything. Cell leaves fixed exactly this
for the LOCAL near field (direct share 0.1225 N -> 0.0052 N); across a domain boundary it returns, and a single
instance on the receiving side forces a whole-domain import.

Excluding it, the halo is 30.5 % of a neighbour domain -- inside jz-fmm's stated 10-60 % of local data. So the
design is viable and the fix is narrow: treat the few oversized cells as FAR (ship a multipole) rather than
importing their neighbours' particles, or split them. It is a handful of leaves, not a policy change.

**The partitioner is not the lever.** Plane, RCB and Morton give essentially the same import (0.663 / 0.663 /
0.692 at ndev = 2), and RCB reproduces the plane exactly at two devices, as it must -- its first cut IS a plane on
the longest axis. The cost is set by a few oversized cells, not by boundary shape, so the aligned-Morton partition
of Phase 2 stands.

**A gap in yggdrax this exposed:** `dual_tree_walk_cross_impl` takes **no `node_active` mask**, unlike
`dual_tree_walk_mutual`. The production lane must pad for static shapes, and unmasked padding leaves (radius 0)
fail the MAC against everything -- the documented 30M-spurious-edge failure. Phase 3's cross walk needs that mask
added. The probe sidesteps it by building unpadded trees, which a probe may do and the lane may not.

Two more measurement traps, both of which produced plausible wrong readings:
* **A union over target leaves is not a summary.** The mean (3.2 foreign leaves per target) and the union (99.8 %)
  describe the same data and disagree completely, because one saturated row swallows the union.
* A padding leaf carries `start == end == n`, which a naive `end - start + 1` reads as ONE particle rather than as
  empty.

## Phase 3.2 -- `node_active` on the cross walk (yggdrax)

`dual_tree_walk_cross_impl` now takes `target_node_active` and `source_node_active`: **two** masks, not one,
because unlike `dual_tree_walk_mutual` this walk spans two index spaces. A pair whose target OR source node is
inactive is dead -- never accepted, never near, never refined. `None` on either side is bit-identical to the walk
without the argument, and the masks are shape-checked against their own tree (unlike `policy_state`, whose shapes
are not static and which therefore reads the wrong node rather than raising).

The mask must be **ancestor-closed**, and nothing can check that for you: an inactive node is never refined, so
marking an internal node inactive prunes every live descendant with it. The rule that satisfies it is the one the
padding produces naturally -- inactive exactly when a node's own particle range is empty.

The failure it prevents is worse across two trees than within one. In the self walk the padding leaves share a
centre and only fail the MAC against each other; here each is a distinct point that is near to part of the other
tree and far from the rest, so it emits spurious pairs of BOTH kinds against the whole live tree. Measured on two
padded cell trees (300 points each, leaf 8, 87 and 97 live leaves in a capacity of 256): the per-leaf near demand
goes 28 -> 156 and the total cross-near count 140 -> 22,535, a factor of 161. In practice the first thing the
padding blows is the pair QUEUE, so at the walk's default queue the unmasked run never reaches a near decision at
all.

`tests/distributed/test_cross_walk_node_active.py`, 7 tests. The control is the admissibility partition of
`test_cross_walk.py` -- far sources from a target leaf and its ancestors plus its near sources must tile every
source particle exactly once -- and it was **mutation-checked rather than assumed**: killing one live source leaf
breaks 86 of 87 target leaves, one live internal source node 84, and a target mask that is not ancestor-closed 2.
It is also blind to the flood (padding nodes carry no particles, so an unmasked walk partitions just as well),
which is why the flood test is separate. One test goes through the jitted `dual_tree_walk_cross` wrapper rather
than the impl, because that is the name jaccpot imports and a parameter has been left unreachable there before.

## Phase 3.3 -- the oversized-cell policy, measured on both ICs

Probe: `bench/multigpu_oversized_cell_probe.py`. Per-domain cell trees in the shared Morton frame, the real
`dual_tree_walk_cross_impl` for the cross relation, and the reclassification applied POST HOC to the walk's own
neighbour lists -- exact for "this pair is served by an expansion instead of by particles", and not a model of
SPLITTING the cell, which would change the walk. Capacities are grown until the walk reports no overflow; nothing
is ever accepted as a truncation.

### The statistic changed, and so did one number in the section above

The import is reported as **`worst1src`: the largest single ordered (receiver, sender) pair**, not a mean. The two
are not interchangeable and the difference is threefold. At the plane split, N = 2x10^5, ndev = 2, device 0 imports
**100 %** of device 1 and device 1 imports **33 %** of device 0 (their max/median leaf radii are 1689 and 653), so
a mean over devices reads 66 %. The "100.0 %" in the section above is the FIRST of those two ordered pairs; it is
correct, and it is one pair.

The probe reproduces that configuration exactly as a control -- `worst1src` 1.000 with nothing cut, 0.326 after the
single worst importer -- against the record's 30.5 %. The remaining 2 pp is that
`multigpu_import_locality_probe.py` asserts on `near_overflow` and `queue_overflow` but **not on `far_overflow`**,
which is also one of the walk's termination conditions.

### Radius is a weak discriminant; leaf DEPTH is the right one

Spearman(leaf radius, import size) is **+0.05** over all leaves -- radius only works because the top few importers
happen to be the top few radii. Spearman(-depth, radius) is **+0.73**. And a per-device radius quantile is the
wrong SHAPE regardless: at ndev = 4 on a Plummer sphere one device has max/median leaf radius 1838 and another
**4.4**, so a per-device quantile makes the healthy device cut good leaves while under-cutting the sick one.

Depth needs no order statistic and no reduction. The Morton frame is global, so depth `d` is exactly cell size
`box / 2^d` -- the same absolute test on every device. It is also the direct statement of the cause: a
cell-partition leaf's bounding radius is at most `sqrt(3)/2` of its own cell, so a leaf is vast only when its CELL
is shallow.

**Plummer, Morton partition, N/device = 10^5**, worst single ordered import:

| ndev | no policy | radius quantile | leaf depth |
|---|---|---|---|
| 2 | 1.000 | 0.018 % of cells -> 0.384 | depth <= 1: 0.055 % -> 0.384 |
| 4 | 1.000 | 0.21 % -> 0.269 | depth <= 3: **0.135 %** -> 0.273 |
| 8 | 1.000 | 0.21 % -> 0.384 | depth <= 5: **0.080 %** -> 0.387 |

Converted from percentages to counts, the knee sits at **3-7 leaves per device at every ndev** -- so the rule is
"reclassify the shallowest handful of leaves", a bincount over an integer array with ~20 distinct values. The sweep
is flat past the knee (cutting 11 % of cells at ndev = 4 buys 0.273 -> 0.273), so the threshold is not knife-edge.

### The disc+bulge IC has NO pathology, and that is the finding

| ndev | disc+bulge, no policy | after the best cut measured |
|---|---|---|
| 2 | **0.411** | 0.375 at 6.6 % of cells |
| 4 | **0.395** | 0.387 at 1.0 % |
| 8 | **0.280** | 0.275 at 1.4 % |

The application IC is already at the gate before any policy, and the policy barely moves it -- its import is simply
not concentrated in a few cells. Its max/median leaf radius is 145-320 against Plummer's 835-1916.

**Why**: the Plummer sphere's bounding box is **745 for a system of scale radius 1**, because one particle in its
unbounded tail sets it. The Morton hierarchy therefore burns about nine levels before reaching the body of the
system, and its shallow cells are enormous -- the worst leaf spans half the box. The disc+bulge IC is clipped at
`rmax_code = 20` (box ~40) and has no such cells. **This is the same trap as "a Hernquist tail needs clipping --
one particle sets the tree's bounding box"** from the rollout work (memory `disc-bulge-rollout-and-cap-cliff`),
resurfacing as the cross-domain import.

The subsampling runs the safe way: the disc numbers are a 1:105 uniform subsample of the 21M IC, which makes each
halo cell about 4.7x larger in linear size than at full N, so the production disc is FURTHER from the pathology
than what is measured here, not closer.

**So the oversized-cell policy is a defence against unclipped ICs, not a prerequisite for the application.** It
should still be built -- it is a bincount and a mask, and Plummer is a standard test case -- but Phase 3.5's
exchange does not wait on it.

### A trap that was nearly shipped as a scaling trick

To reach ndev = 8 the probe first capped `max_neighbors_per_leaf` at 256 and counted the truncated rows as
importing their whole source domain, on the reasoning that an upper bound errs against the design. It does not
work, because **`near_overflow` is one of the walk's `cond_fun` termination conditions**: an overflowing row does
not truncate itself, it HALTS the walk and every other row loses its remaining rounds. Measured against the
full-capacity arm on the same configuration, it read one device's import as **0.010 of its neighbour against a true
0.304** -- 30x too small -- while the other device's number and the entire radius sweep were unaffected and looked
perfectly sane. Both capacities are now GROWN until the walk reports no overflow, which reproduces the
full-capacity control row for row and is still cheap, because almost every ordered pair needs a small fraction of
the worst pair's buffers.

### Where the floor is

Cutting by import size directly -- the best any target-side rule could do -- reaches 0.180 (Plummer, ndev = 2) and
0.287 (ndev = 8) only after cutting 3.6-3.9 % of cells. So roughly **30 % of a neighbour domain is structural**,
not removable by reclassifying cells, and that is the number Phase 3.5 should size the exchange for. It is inside
jz-fmm's stated 10-60 % of local data.

## Phase 3.4 -- the cross FAR import is the BIGGER half

Probe: `bench/multigpu_far_import_probe.py`, same per-domain cell trees and the same real
`dual_tree_walk_cross_impl`, capacities grown until no overflow (`far_overflow` is a `cond_fun` termination
condition too, so it may not be accepted any more than `near_overflow` may).

Pairs are not the payload: the exchange ships each distinct source node ONCE however many target nodes name it,
and that compression is **8-13 far pairs per node shipped**. What is left after it is still large.

Worst ordered (receiver, sender) pair, N/device = 10^5, Morton, cells64, theta 0.8:

| IC | ndev | far nodes | as a fraction of the sender's tree | near particles | p=4 far/near bytes | p=6 |
|---|---|---|---|---|---|---|
| Plummer | 2 | 5,755 | 0.53 | 38,409 | 0.94 | 1.84 |
| Plummer | 4 | 4,169 | 0.38 | 25,549 | 1.02 | 2.00 |
| Plummer | 8 | 5,767 | 0.53 | 36,630 | 0.98 | 1.93 |
| disc+bulge | 2 | 6,024 | 0.59 | 41,074 | 0.92 | 1.80 |
| disc+bulge | 4 | 5,583 | 0.49 | 32,006 | 1.09 | 2.14 |
| disc+bulge | 8 | 5,513 | 0.47 | 27,496 | 1.25 | 2.46 |

(A real multipole of order p is (p+1)^2 coefficients at 4 bytes; a particle is 4 floats.)

**Three things follow.**

1. **The far import is not a coarse summary.** It is a multipole for **38-59 % of the sender's whole tree**, at
   median depth 10-27 of a tree 15-35 deep -- deep nodes near the leaves, not a handful near the root. The reason
   is structural rather than pathological: different target nodes are served at different levels, so the UNION
   over one receiver's targets spans most of the sender's levels even though each individual cut is thin. The
   per-target row is small (median 1-11) and the union is not one saturated row.
2. **At p >= 5 the far half is the DOMINANT half**, 1.2-2.5x the near payload. At p = 4 the two are within 25 %
   of each other. Expansion order is therefore a communication knob in the distributed lane in a way it is not on
   one card, and the p = 6 the single-GPU record uses is the expensive end.
3. **The disc+bulge IC is NOT better here, and at ndev = 8 it is worse** (1.25 against 0.98 at p = 4). That is the
   opposite of the near half, where the disc has no pathology at all. So the far import is a property of the
   method, not of the IC, and unlike the oversized-cell policy there is nothing IC-specific to exploit.

**What this means for 3.5.** A single frontier exchange that ships half of every sender's tree to every receiver
is ~0.5-1.2 MB per ordered pair at 10^5/device, so ~4-8 MB received per device per step at ndev = 8, on top of the
near half. That is affordable at this size and is NOT affordable at 10^6/device, where it scales to tens of MB
against a budget of tens of ms. This is exactly the case for jz-fmm's progressive per-plane request -- ask only
for the nodes each level's interaction list names, when that level runs -- rather than one frontier round, and it
is a second, independent reason to drop `build_coarse_frontier`'s every-leaf export.

These are counted BYTES, not timings; the box has been loaded throughout and nothing here was timed.

## Phase 4 (kernel side) -- no new Pallas kernel, now verified rather than read

The plan's claim was that the one-sided cross field needs argument changes to kernels that already exist. That
was read off the signatures; it is now measured. Three commits, each with the same shape: an optional static
argument, `None` keeping the single-tree behaviour exactly, and a test that the rows the old call produced beyond
the new bound were *exactly zero* -- pure waste, not an approximation.

**4.1 -- the walk can be seeded with many pairs** (yggdrax `4b07908`). `dual_tree_walk_mutual` took only
`(root, root)`. It now takes `seed_a` / `seed_b` / `seed_count`. Concatenate an imported source set at
`[n_local, n_local + n_remote)`, seed `(local_root, imported_i)`, and the walk's existing `(min, max)`
canonicalisation orders every emitted pair as (local target, imported source) for free, because every local index
is strictly below every imported one. Seeds are canonicalised here too, so a seed and its own descendants cannot
be emitted in opposite orientations.

**That ordering is asserted directly, not inferred.** The test builds the combined index space and checks every
emitted pair satisfies `a < n_local <= b`, and that every imported node stayed whole -- the walk refined only on
the local side, as the design requires. Putting the imports first instead makes the M2L expand the wrong way
round with a plausible-looking result, and nothing else would catch it.

**4.2 -- `n_targets` on the lanes M2L** (jaccpot `f4d0ede`). The grid, the `out_shape` and `csr_by_target`'s
`total_nodes` all came from `multipoles.shape[0]`, so a concatenated source array emitted `n_local + n_remote`
rows: twice the launches and twice the buffer for discarded rows, then a shape mismatch against the local-only
accumulator. Hence "mandatory", not an optimisation.

**4.3 -- `num_target_leaves` on the leafpair near field** (jaccpot `62650b4`). Here the claim really did hold:
`leaf_positions` is targets and source gather table alike, and the grid comes from the chunk table built over the
target rows, so `L_source = L_local + L_halo` already works. Only the tail `segment_sum` spanned the whole pool.

Both 4.2 and 4.3 pin the part that could have been quietly false: **zeroing the imported half must change the
local rows**. Without that check the kernel could be ignoring precisely the half of the source array the import
provides, and every shape assertion would still pass.

**The reverse of both cvjps REFUSES a non-trivial value** with `NotImplementedError` rather than returning a
quietly wrong gradient. Each would have to rebase its own second pass and accept a cotangent shorter than the
pool; gradients through the cross-domain import are a later phase by decision, so this is left unplumbed and loud
instead of half-plumbed and silent.

**Still open in 4.3**, because they need the driver that does not exist yet: slicing `_combined_neighbors`' counts
and offsets to the local rows (`build_leafpair_chunk_table` gives every empty row one chunk with `is_first = 1`,
so an unsliced table runs a discarded self-pass on every halo leaf), asserting that `import_near_halo` never
silently drops a remote leaf wider than the contract width, and rebasing the local half's node ids through the
scatter table rather than by subtracting `leaf_nodes[0]`.

## Next

Phase 1 is done: the fused lane runs per device under one `shard_map`, at parity with the single-device lane to
fp32 noise, at one and at two devices. What it does NOT yet have is any cross-domain field -- each device sees
only its own shard.

That is plan Phase 3, and its volume question is now settled: the cross-domain near field is a few per cent of
the local one, and the halo is ~30 % of a neighbour domain once the oversized outer cells are handled. Both are
inside the regime the design assumed.

Phase 3.2 and 3.3 are now done, and 3.3 reshaped what follows.

`node_active` is on the cross walk. The oversized-cell policy is measured on both ICs, and the answer is that it is
a defence rather than a prerequisite: reclassify by Morton leaf DEPTH (not radius, and not a per-device quantile),
about the shallowest 3-7 leaves per device, which takes an unclipped Plummer sphere from importing a whole
neighbour domain to ~0.3 of one; the disc+bulge IC never needed it, being already at 0.28-0.41.

3.4 is now done too, and it moved the cost: the FAR half is the bigger one. A device needs multipoles for 38-59 %
of each sender's tree, which at p >= 5 outweighs the near particles and is no better on the disc than on Plummer.

Phase 4's kernel side is done as well, so the only thing between here and a force is the exchange and the driver.

Next is **3.5, the exchange itself**, and every measurement points the same way: build it as jz-fmm's progressive
per-level request rather than one frontier round. Size the near half for the ~30 % of a neighbour domain that is
structural, and treat expansion order as a communication knob, not just an accuracy one. **Phase 2** (the
device-resident partition) is still not built and the driver needs it; the probes so far have used host-side
domains, which a probe may do and the lane may not.
