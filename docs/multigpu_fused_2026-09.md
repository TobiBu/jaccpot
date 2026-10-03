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

## Phase 3.5 -- a flat imported POOL double-counts, and the plan's 4.1 data model has to change

Probe: `bench/multigpu_export_walk_probe.py`, plus three diagnostics under it.

The exchange has to answer "the receiver does not have the sender's tree". The scheme tried here is a **sender-side
export walk**: every device publishes a small level-k Morton summary of itself, and each SENDER walks its own full
tree against every receiver's summary cells and decides unilaterally what to send -- one ragged round, no request
round. It is correct by construction, because a summary cell BOUNDS every real target inside it, so a node far
from the cell is far from each target in it. The export walk is `dual_tree_walk_mutual` seeded with
`(cell_i, sender_root)` over a combined `[cells ; sender_tree]` space: Phase 4.1's multi-pair seed used in the
opposite direction from the evaluation.

**The export itself is sound.** Fed the receiver's own leaves as its "summary", the export names every source the
exact cross walk names -- the 23 it appears to miss are every one of them covered by a shipped ANCESTOR, so
nothing is lost, and it adds 177 extras. A set difference was simply the wrong way to compare two cuts.

**But the shipped set is not a cut: 2,033 of 2,034 exported nodes also have a shipped ancestor in the set.** That
is not waste -- different receiver cells genuinely need different granularities, which is the same fact Phase 3.4
measured as "the union spans most of the sender's levels". It is fatal to the plan's 4.1 data model, which has the
receiver seed `(local_root, imported_i)` against a FLAT pool and refine locally. Measured on that exact
construction at N = 2x10^4, ndev = 2:

| | |
|---|---|
| exported nodes | 1,284 |
| pairs emitted, orientation `a < n_local <= b` | 144,368, **all correct** |
| target leaves whose source coverage DOUBLE-COUNTS | **583 of 583** |
| target leaves with a coverage hole | 0 |

Every target leaf accepts an ancestor for one pair and a descendant of it for another, and counts that mass twice.
**A conservation check cannot see this**: double counting leaves momentum exact, which is the failure mode this
project has already been bitten by (memory `cross-m2l-theta-lift-design`). It would surface only as a percent-level
force error in a global comparison.

So Phase 4.1's seed mechanism is right and its *payload* is wrong. The imported object cannot be a flat node pool;
it must carry PROVENANCE -- which receiver cell each node was exported for. The corrected design:

* ship each distinct node payload ONCE, plus a small per-cell CSR naming which nodes each receiver cell needs;
* the receiver seeds `(cell_root_local, imported_node)` only for the pairs in that cell's list, and refines
  locally. Each cell's list IS a cut by construction -- it is one export walk's output from a single target -- so
  each local target below that cell gets a valid partition.

**This ties Phase 2 to Phase 3.5 harder than the plan says.** "The local root of cell C" only exists if the cell is
a subtree of the receiver's tree, which is exactly what Morton `align_level` guarantees. Alignment was justified in
Phase 2 as a tree-structure nicety; it is in fact what makes the corrected exchange addressable at all.

Not yet measured: the size of the per-cell CSR against the node payload, which is what decides whether this stays
one round. That is the next thing to probe, before any exchange code is written.

## Phase 3.5 -- sizing the corrected exchange: the CSR is cheap, the SUMMARY is the design

`bench/multigpu_export_walk_probe.py` now reports the per-cell interaction list itself, not just the node set
behind it. The node set is the PAYLOAD; the pair list is the CSR that makes it addressable; they scale
differently, and a node needed by many cells costs one payload and many CSR entries.

**The question is answered: provenance is affordable.** A CSR entry is 4 bytes against ~120 (p = 4) or ~164
(p = 5) for a node payload, and the duplication factor is 3-25 pairs per node, so the CSR runs **5-60 % of the
payload**. The corrected exchange stays one round. Every row also verifies the property the fix rests on --
`cutOK`: no single cell's list contains both a node and one of its ancestors, so a target below that cell gets a
partition. Checked on every configuration, because the flat pool over all cells demonstrably is not a cut.

**Bytes are not the only currency and they move OPPOSITE to flops.** A coarse summary ships fewer, deeper nodes
(the union is over fewer cuts) but every target inside a cell is evaluated against that cell's whole list. A probe
reporting only bytes would have picked the coarsest level available; `evalPairs` is reported beside it.

**Fixed Morton levels are not merely suboptimal, they are unusable, and it gets worse with N.** Against the
leaf-granularity floor:

| summary | ndev 2, N/dev 10^4 | ndev 2, N/dev 10^5 | disc, N/dev 10^5 | ndev 4, N/dev 10^5 |
|---|---|---|---|---|
| occ4 | 1.26x | **1.28x** | **1.28x** | **1.13x** |
| occ16 | 2.41x | 2.21x | 1.99x | 1.48x |
| Morton level 6 | 6.09x | **61x** | 12x | **61x** |
| Morton level 3 | 7.01x | **64x** | 48x | **65x** |

At N/dev = 10^4 a fixed level cost 7x; at 10^5 it costs 64x, while the occupancy cut is flat at ~1.3x. The cost is
`sum over cells of |list| x leaves_in_cell`, which a few big cells with many leaves AND long lists dominate -- at
N/dev = 10^4, occ16 and Morton level 6 had the SAME cell count (88 against 90) and differed 2.5x. It is the
population imbalance of a fixed level, not the summary's size, and imbalance grows with N.

**So the summary is an occupancy-balanced cut of the receiver's own tree** (`occupancy_cut`), descending until a
subtree holds at most a few leaves. Operating band **occ4 to occ16**: 1.1-2.2x the ideal evaluation work with the
CSR at 0.30-0.64 of the payload, so total bytes ~1.3-1.6x the payload. Leaf granularity is the flop floor but
costs 2.8-4.7x the payload in CSR alone, and is 5,000 cells to publish.

**A correction to what this document said above.** The Phase 3.5 entry before this one claimed the corrected
exchange is addressable only because Morton `align_level` makes a cell a subtree. That is not right: an occupancy
cut is made of REAL TREE NODES, so its cells are subtrees by construction whatever the partition does, and each
has a local root for free. `align_level` keeps its Phase 2 justification -- no top node straddles a device, so
each device's node set is disjoint -- but the exchange does not depend on it for addressability.

Not measured here: the accuracy of the conservative export, which over-ships by up to 1.55x in nodes
(`nodes/need`) and is therefore never less accurate, only more expensive. Nothing here was timed.

## Phase 3.5 BUILT -- the exchange, five pieces, all on CPU

yggdrax `4b07908`, `99c1560`, `a880135`, `0138c7a`, `4b769cb`. 51 tests, none needing a GPU.

| piece | what it is |
|---|---|
| `dual_tree_walk_mutual(seed_a=, seed_b=)` | many seed pairs instead of one root pair |
| `distributed/summary.py::occupancy_cut` | the cut of a tree the exchange is addressed by |
| `distributed/export.py::export_walk` | what this device owes every other, in ONE walk |
| `distributed/export.py::build_send_buffers` | grouped by destination, payload deduplicated |
| `distributed/import_cells.py` | two ragged rounds, then receiver-side assembly |

**The multi-pair seed is used three times, in three directions**: the sender seeds
`(cell, sender_root)` to decide what to export; the receiver seeds
`(cell_root_local, imported_entry)` to expand what arrived down to its own targets; and the
evaluation seeds `(local_root, imported_i)`. Each time the walk's `(min, max)` canonicalisation
supplies the pair ordering for free, because one index space is placed wholly below the other.

**The whole thing rests on one property, and it is tested end to end.** Summary -> export -> send
buffers -> assembly, then: for every local target leaf, the sender's particles reached through far
sources accepted for it or any ancestor, plus its near sources, are covered **exactly once**. Green
at `max_leaves` 1, 4 and 16. The pooled construction fails the same check on 583 of 583 leaves, and
neither failure mode is visible to a conservation check -- a gap loses force, an overlap
double-counts it, and double counting leaves momentum exact.

**Two traps made into named, separately tested functions** rather than inline steps, because both
produce plausible values, correct shapes and wrong forces:

* `rebase_csr`. Each sender numbers `csr_row` from zero inside its own block; the blocks land
  end-to-end in one receive buffer, so entries from every sender after the first must be shifted.
  Tested on hand-made sizes where the arithmetic is visible, with a single-sender identity control.
* A device never exports to ITSELF. Its particles are already in its own local field, and counting
  them twice leaves momentum exact. The test hands the sender's own cells in ACTIVE so the function
  is the only thing masking them.

**What the tests deliberately do not claim.** A padded shard's cut is NOT the unpadded one's -- the
two trees have different balanced structures, so it lands on different nodes; what holds is that it
tiles exactly the LIVE particles. And `num_valid` is a measured NO-OP on this ranges convention
(padding carries `start > end`, which the `sub > 0` guard already excludes); a synthetic tree built
to exercise the other convention passed for an accidental reason and was removed rather than kept
as false assurance.

Output is `(local node, imported payload row)`, which is exactly what Phase 4's arguments take: the
far list feeds an M2L over `[local ; imported]` with `n_targets = n_local`, the near list a leafpair
kernel with `num_target_leaves = L_local`.

**Not done**: gathering the real payloads (multipoles, particles) into the send buffers, and wiring
the lists into the fused lane's kernels -- that is the driver, and it needs Phase 2's device-resident
partition. Nothing here has been timed.

## Phase 2 -- `align_level` is NOT worth having, and the plan's value for it is unusable

Probe: `bench/multigpu_align_level_probe.py`. The plan gave two reasons to snap domain boundaries to level-k
Morton cell edges. Both are now answered, and the answer is to drop the idea.

**Reason one is void.** "It makes each device's node set disjoint" belonged to the LET design, where every device
built a coarse tree over other devices' leaves in a shared numbering. The Phase 3 lane gives each device its own
tree over its own particles in its own index space, and the exchange addresses cells by an occupancy cut of real
tree nodes. Disjointness holds whatever the partition does.

**Reason two is real, testable, and too small to matter.** A pivot cutting inside a cell splits it across two
devices, and the halves occupy the same region on either side of the boundary. Alignment does remove that -- but
removing it changes nothing measurable:

| align | occupied cells | imbalance % | straddling cells @11 | imported nodes | near import / own |
|---|---|---|---|---|---|
| none | -- | 0.000 | 1 | 6056 | **0.411** |
| 9 | 20,023 | 0.306 | **0** | 6055 | 0.410 |
| 11 | 117,169 | 0.004 | **0** | 6056 | 0.411 |

(disc+bulge, N/device = 10^5, ndev = 2. The 0.411 independently reproduces Phase 3.3's figure for the same point
from a probe written separately.) There was exactly ONE straddling cell out of 117,169 occupied, so the mechanism
is real and far too rare to price.

**And the plan's `align_level = min(ceil(log8(512 x ndev)), leaf_depth_min - 1)` = 3 is unusable.** It assumed
particles spread over the level-3 grid. They do not: the bounding box is set by the tail, so a Plummer sphere
occupies **21 cells of 512** at level 3. Aligning there leaves a device with 9 particles at ndev = 2 and **zero**
at ndev = 4; levels 5 and 7 also empty a device at 4 devices. Only levels 9 and finer are affordable, and by then
alignment buys nothing.

**Decision: no `align_level` selector.** Use unaligned equal-count Morton pivots, which balance exactly (0.000 %).

**Two probe errors, both caught by the numbers being impossible or immovable**, and both recorded because the
second is the more dangerous kind:
1. Summing each imported node's particle range double-counts, because the import holds nodes together with their
   own descendants -- it read an import of 19.8x a domain. Fixed with a coverage mask.
2. **A saturated statistic cannot answer anything.** Coverage over far AND near is pinned at 1.000 by
   construction, because the far list reaches nodes near the root whose ranges cover the sender outright. Three
   runs were spent before that was spotted. The near list alone is the one with headroom, and the node count --
   which is the payload the exchange actually pays for -- was unsaturated the whole time and flat.

Nothing here was timed.

## Forced CPU devices do NOT cover the GPU tracing path

Everything in Phases 2 and 3 was developed against `--xla_force_host_platform_device_count=4`. On two real A100s
(cards 3 and 4, the only idle ones) the whole stack passes -- **62 passed, 5 skipped**, the skips being the
ndev = 4 cases -- with one exception that matters more than the pass.

`maybe_repartition` failed on GPU and passed on CPU, with **every shape and dtype agreeing**:

```
cond branches must have equal output types but they differ.
  the output of true_fun has type float32[96,3] but the corresponding output of
  false_fun has type float32[96,3]{V:gpus}, so the manual axis types do not match
```

`resolve_ragged_method` picks the `buf` all-gather fallback on CPU and the native `ragged_all_to_all` on GPU.
Under native, `sfc_partition`'s result is inferred **axis-invariant**; under buf it is varying. The identity
branch is varying either way, so the `lax.cond` is well-typed on one backend and rejected on the other. The two
backends are not two ways of running the same program.

Fixed by pushing both branches to varying, and **the form matters**: `pcast(x, (), to='varying')`, which is what
the error message itself suggests, is a NO-OP in both directions and changed nothing.
`pcast(x, axis_name, to='varying')` does convert invariant to varying, RAISES on an input that is already
varying, and the variance is not exposed as an attribute to test -- so the cast is attempted and the raise is
read as "already varying", at trace time.

The native ragged path itself is sound on jax 0.10.2: `auto` resolves to it and the exchange round-trips every
reference to the right payload, which is the thing the 0.9.0 corruption would have broken silently.

## Phase C -- the cross field interleaved between the sweeps (C0-C4 done, far half only)

Plan `~/.claude/plans/phase-c-interleaved-cross-field.md`. Decided after Phase 3.4 measured the FAR half as the
bigger one: adding the cross field after the force would need a SECOND L2L cascade, so it goes where it belongs,
between the upward and downward sweeps. Gate agreed with the user: **G4 clause two only** -- match an fp64 direct
sum to the accuracy the single-GPU lane achieves at the same `(p, theta, leaf)`, with a monotone per-order sweep.
Clause one (1e-6 against the single-GPU lane) is unreachable in principle: Phase 1 measured that a device building
its own tree from an all-reduced box already sits 2.6e-3 away, "purely because it builds a different but equally
valid tree".

**It is a concatenation, not a second pass.** `_solidfmm_downward_accumulate_from_multipoles` already takes the far
pair list and the multipoles as arguments and does `locals_updated = initial_locals_coeffs + m2l_inc` before the
cascade. So: `[local ; imported]` multipoles, local ++ cross pairs, `n_targets = n_local`, one cascade.

| step | what it establishes | commit |
|---|---|---|
| C0 | `LargeNPreparedState.upward` is **None** -- multipoles are an unretained intermediate, so the hook cannot live outside the refresh | `27818b9` |
| C1 | the hook fires, sees `packed (4095, 25)`, force **bit-identical** | `27818b9` |
| C2 | 8 unreferenced GARBAGE rows + `n_targets`, force **bit-identical** | `25427c6`, `c866d5b` |
| C3 | the REAL pipeline under `shard_map` with live collectives, every count 0, force **bit-identical** | `83e7e30`, `dddca04` |
| C4 | **the cross FAR field reaches the force at ndev = 2**: 5.825e-01 -> 2.409e-01 against an fp64 direct sum over ALL sources | `0145f06` |

**C4 does NOT meet the gate and cannot yet**: the hook returns far pairs and discards the near list, so the
residual is the missing cross NEAR field. That is C5.

**What the bit-identity chain caught**, none of which would have surfaced as a wrong force:

* The hook placed in the PREPARE path (which runs once) and then inside one of TWO downward branches (the run took
  the other). Caught only because the probe asserts the hook FIRED -- without that, "bit-identical" is vacuous and
  both wrong placements report PASS.
* C3's first version checked that diagnostic KEYS existed, not their VALUES. A non-empty-but-dropped import would
  have passed; it now requires every count to be zero at ndev = 1.
* Those diagnostics cannot be read host-side -- the record dict holds TRACERS under jit.
* The C4 control was written as `local > 3x cross` with no basis for the 3, and rejected a real 2.42x improvement.
  It is a liveness check, not an accuracy assertion.

**Two environment facts worth keeping.** The fused lane runs under `shard_map` with **`check_vma=False`**, because
its Pallas kernels build `out_shape` without a `manual_axis_type` -- which sits beside the `maybe_repartition` fix
that needed `pcast` precisely because THAT path runs with the check on; different call sites, not a contradiction.
And the single-GPU comparison arm needs its own process: `apply_fast_lane_env` is process-wide and tuned to the
shard length, while the fused profile gate keys on the exact array length.

Nothing in this phase was timed.

## Phase C5 -- the cross NEAR half is ADDED, not interleaved, and it is verified without a GPU

The far half goes BETWEEN the sweeps because a far contribution arrives as a local expansion and has to ride the
L2L cascade down. That argument does not extend to the near half: a near contribution is a direct sum with no
cascade behind it, so it can be computed once and added to the finished acceleration. Gravity is linear in the
sources, so this is exact, not an approximation, and it costs one extra kernel instead of a second plumbing path
through the prepared state. Phase 3.1 measured the cross near field at 1.4-4.3 % of the local near list, so this
is the cheap half by a wide margin.

**What was built.** `make_cross_hook` takes a `near_sink` dict. Given one it does the near half on the SAME export
walk as the far half -- one walk, one set of exported cells -- building `W`-wide leaf particle tiles from
`node_ranges` + `positions_sorted`, exchanging them, and running `receiver_interaction_lists` on the near CSR.
`cross_near_acceleration` then concatenates the imported leaves onto the local pool, builds a CSR over LOCAL target
leaf rows, and calls the leafpair kernel with `num_target_leaves = L_local` and `include_self=False`, so it emits
local rows only and never re-counts the local self term that is already in the force.

**The near overflow flag is folded into the one the evaluator returns.** It is a tracer the caller cannot read
host-side, and the cross buffers saturate independently of the local lane's, so without this fold a saturated
cross near list would be invisible.

**Two traps in the pair list, both real.**

* A target arrives as a leaf NODE id and the pool is indexed by leaf ROW. They differ by the internal-node count,
  which is passed in explicitly: deriving it from the pool's own length happens to work for a balanced structure,
  and that is exactly why it should not be done here.
* The first version CLIPPED an out-of-range target into `[0, L-1]`. That silently ADDS a foreign leaf's sources to
  a real local row -- a wrong force. The bound is now part of the liveness mask, so such a pair is dropped instead.
  Neither outcome is acceptable, but only the second is detectable, and the test asserts nothing is dropped.

**Verification, and it needed no GPU at all.** `tests/unit/distributed/test_cross_near_acceleration.py` builds a
pool by hand -- ragged occupancy, `num_internal` deliberately NOT `L-1` so a row/node mixup cannot hide -- and
requires the kernel's answer to equal a numpy Plummer sum written out in the test, over exactly the paired
sources, to rtol 1e-10 under Pallas `interpret`. Around it: untouched leaves stay zero, poisoned padding beyond
`count` changes nothing, an out-of-range target is dropped rather than folded, and doubling the imported masses
doubles the term.

Three mutants were run against that suite and all three were killed: `row -> row + 1` (the node/row off-by-one),
`G -> 2G`, and `include_self False -> True`. This is the first piece of the cross field that is checked exactly
rather than by a ratio.

**Not yet run on GPUs.** The C4 probe now has a third arm (local-only / +FAR / +FAR and NEAR) sharing one sink,
but it needs two idle cards and the box has had one, at loadavg 36-66. The arm is wired, not measured.

## Two geometry bugs in the cross exchange, found by reading rather than by a failing number

Both were found while checking the near half against the plan's `_combined_neighbors` traps, and neither would
have announced itself: no crash, no overflow, no momentum signature.

**The imported RADIUS was never shipped.** The receiver built `combined_rad` as the local radii followed by
`zeros(recv_node_cap)`. A zero source radius passes `radius / distance < theta` at ANY distance, so every imported
node was unconditionally admissible: multipoles used at close range, and the near list starved of exactly the
pairs that carry the largest forces. The radius now travels with the multipole, and `imported_zero_radius` is a
diagnostic so the run can confirm they arrive.

**The near walk scored the FAR import's geometry.** The two halves export DIFFERENT node sets -- far sends
internal nodes, near sends leaves -- so payload row k means a different node in each. The near call reused
`combined_cen` / `combined_rad`, pairing each imported leaf's particles with an unrelated node's centre and
radius. The near payload now carries its own centre and radius.

**The prediction below was FALSIFIED, and the section after it records what was actually wrong.** It is left
here unedited because the way it failed is the useful part.

**A prediction, recorded before the run so it cannot be fitted afterwards.** At order 4, ndev = 2, N = 2e5,
leaf 64, theta 0.8, the single-GPU reference is **2.9401e-03** and the mesh arms before these fixes were
local-only 5.825e-01 and far-only 2.409e-01 -- the cross field was reaching the force but leaving it ~82x worse
than the same lane on one card. If the zero-radius defect is the cause, the far-only error must fall
substantially from 2.409e-01 on its own, and `imported_zero_radius` must be 0. If it does not fall, the diagnosis
is wrong and the gap is elsewhere -- most likely the near half, whose few per cent of PAIRS are the closest
sources and so can carry most of the residual regardless.

**The single-GPU reference, measured (card 4, pinned).** Order 4 `2.9401e-03`, order 5 `1.5599e-03`, order 6
`8.2045e-04`: monotone, about 1.9x per order, so the reference arm itself behaves. This is the second clause of gate G4 -- the first clause, rel-L2 below 1e-6
against the single-GPU lane, is unreachable because one-sided evaluation is not exact here.

## The prediction failed, and the real fault was three silent truncations

Shipping the correct radii made the far-only error WORSE, not better: 2.409e-01 -> 5.1256e-01. That is consistent
rather than contradictory. A zero radius made every imported node unconditionally far-admissible, so close cross
pairs were evaluated by multipole -- a bad approximation, but nonzero. With true radii those pairs are correctly
rejected as far and nothing picked them up, so they became simply ABSENT. A bad approximation beats a missing
term, which is why the better-looking number came from the buggier code. The radius fixes are still right; they
exposed a hole they had been papering over.

**The per-order sweep is what identified it.** At p = 4, 5, 6 the error was 5.1256e-01, 5.1261e-01, 5.1263e-01 --
FLAT. An expansion missing sources cannot be improved by more terms. A single-point C4 ratio would have read as a
mild regression rather than a structural gap; the sweep read it correctly. This is the third time in this project
that a coverage defect was invisible to everything except an order sweep.

**The diagnostics, which had to leave the `shard_map` as outputs (the record holds tracers).** At `max_cells=1024`:
`export_far` 69717/71281 but `recv_nodes` only 499/653, against Phase 3.4's measured need of 38-59 % of a
16383-node tree. A 12-19x shortfall, reported with no overflow anywhere.

**Three truncations, one cause: the flags existed and were wired to nothing.**

* `summary.overflow` was never read. At `max_cells=1024` with ~5500 live leaves and `max_leaves_per_cell=4` the cut
  needs ~1500 cells and got 1024. `TreeSummary`'s own docstring, written in Phase 3.5, says verbatim: "**Must be
  read**: a truncated summary silently drops part of the receiver from the exchange, which loses force rather than
  accuracy." It was then left out of the OR. Losing FORCE rather than accuracy is exactly what makes the error flat
  in p.
* `export_far` was 69717 against an `export_far_cap` of 65536. `ex.far_overflow` was set, but only reachable
  through `record["overflow"]`, which nothing consumed.
* The near sink's flag omitted `ex.far_overflow` entirely.

All caps are now over-allocated and every flag is OR-ed into what the caller reads. The standing rule in this
record -- over-allocate and read the flags -- only ever worked on its first half.

**A sizing constraint that is now a hard error rather than a silent one.** The receiver walk seeds one pair per
received CSR entry, so `walk_queue` MUST exceed `recv_csr_cap`. Over-allocating both in the wrong ratio raises
`ValueError: the wavefront cannot hold its own seed`, which is the behaviour every one of these caps should have.

**After the fixes, at order 4, ndev = 2, N = 2e5, leaf 64, theta 0.8, all flags clean:**

| arm | rel-L2 vs fp64 direct |
| --- | --- |
| local-only (no cross) | 5.7593e-01 |
| + cross FAR | 2.0061e-01 |
| + cross FAR and NEAR | 1.9515e-01 |
| single-GPU reference | 2.9401e-03 |

`recv_nodes` is now 6010/6734 -- 37-41 % of the tree, which is Phase 3.4's measured 38-59 %, so the import volume
is finally the predicted one. C4's liveness control passes at 2.95x. The gate does NOT pass: 1.95e-01 against a
2.94e-03 reference is still ~66x short.

**The open question, stated precisely so it is not guessed at.** The far walk produces `near_pairs` of
166505/85124 -- pairs where the receiver's MAC failed and the source is an imported INTERNAL node whose particles
were never shipped, so nothing can consume them. Against that, the usable near import is only 34618/34637 pairs.
Two hypotheses were checked and BOTH are wrong: the imported radii do arrive (`imported_zero_radius` = 0), and the
sender and receiver use the same criterion (both call `dual_tree_walk_mutual` with the same `mac_type`, and the
receiver seeds from `(cell_root, imported node)` -- exactly the pair the sender judged). So an admissible seed
should pass straight through to the far list and `near_pairs` should be near zero. It is not.

`recv_csr` was then measured and it settles half of it. `recv_csr` = 180409/128919 is the exact mirror of
`export_far` = 128919/180409, so the exchange itself loses nothing -- every pair the sender emitted arrives. Of
device 0's 180409 seeds, 149223 pass straight through to the far list and about 31186 (17 %) do NOT: those descend
and produce the 166505 near pairs, ~5.3 each. So the fault is neither the exchange nor a capacity. **17 % of the
pairs the sender judged far-admissible are rejected by the receiver's re-test of the SAME pair.**

Three further explanations were checked in the source and all three are wrong: `build_send_buffers` rebases
`csr_cell` by `s_cell - s_dev * mc` and `csr_row` by `g_row - node_offsets[s_dev]`, which is one rebase each and
correct; the sender cannot descend the cell side (cells are childless in its combined tree) so its far list really
is judged against the whole cell; and the canonicalisation puts the cell in slot `a` and the source node in slot
`b` on BOTH sides, so the two tests are not mirror images of each other.

What has NOT been checked, and is the next measurement rather than the next guess: recompute the MAC for the seed
pairs on the receiver and count the failures directly, instead of inferring them from the walk's output. 

**The MAC recomputation, built and queued.** Two independent recomputations of the REAL predicate -- imported as
`_compute_mac_ok` from `yggdrax._interactions_impl` rather than retyped, so the test cannot diverge from what the
walk does. `export_mac_fail` runs it on the sender over the pairs `export_walk` emitted as far, with the sender's
own geometry; `seed_mac_fail` runs it on the receiver over the seeds it starts from, with the geometry that
arrived. Same pairs, same predicate, two ends of the wire. Both near zero means the seeds are admissible and the
walk expands them anyway; sender clean and receiver failing means the geometry changes in transit; both failing
means `export_walk` emits inadmissible pairs.

It carries its own control: `export_near_mac_fail` runs the same check over the sender's NEAR list, whose pairs
bottomed out precisely because they fail the MAC. If the far list comes back clean and the near list comes back
rejected, the instrument discriminates; if the two look alike, the instrument is broken and none of the other
numbers mean anything.

The GPU form of it never got a slot -- two waiters ran 6 h and 10 h and all eight cards stayed busy (16-37 GiB
each, 22-86 % utilisation, other users). So the question was answered on CPU instead, which turned out to be
possible for every link in the chain: the exchange logic is yggdrax and needs no Pallas kernel.

### The sender's list is clean, and the instrument proves it can tell

`tests/distributed/test_export_walk_admissibility.py` builds two Morton-split domains, runs `export_walk`, and
recomputes the imported `_compute_mac_ok` over what it emitted:

| list | pairs | MAC failures |
| --- | --- | --- |
| FAR | 19997 | **0** |
| NEAR | 14029 | **14029** |

0 % against 100 %. The near list is the control -- those pairs bottomed out precisely because they fail the MAC --
so the recomputation discriminates perfectly, and the sender's far list is fully admissible by its own geometry.
The defect is therefore introduced AFTER emission.

### The send buffers preserve every pairing

`tests/distributed/test_send_buffer_pairing.py` checks the exact invariant at ndev = 3 with duplicate nodes and
duplicate cells: every live `(global_cell, node)` must reappear as a CSR entry whose rebased cell is
`global_cell % max_cells`, in destination `global_cell // max_cells`, whose `csr_row` points at a `node_rows` slot
holding that same node. Five cases pass, including a non-vacuity control. Dedup, grouping and the double rebase
are not the defect either, and `test_import_cells` already covers the exchange round-trip.

### The defect: the MAC was re-tested at a point the sender never used

`export_walk` is handed `geom.center` -- geometric, bounding-box centres. The far payload shipped
`mp.centers` -- EXPANSION centres, a different point. The receiver then built
`combined_cen = concat([geom.center, imp_cen])`, so the one array the walk reads was geometric for local nodes and
expansion for imported ones. Every imported node was re-tested at coordinates the sender's decision was never
based on, which is precisely how a pair the sender accepted comes back inadmissible.

Both centres now travel, because the two consumers need different ones: `mp.centers` is the expansion centre and
belongs to the M2L, `geom.center` is what the MAC was computed with and belongs to the walk. This is the same
class as memory `mac-geometry-inconsistent-with-com-centres`, and jax here is 0.10.2, so the
`ragged_all_to_all` corruption is not in play.

The diagnostics channel was switched from int64 to float64 for this -- an int cast would have truncated a
DISTANCE to zero and made a real effect look like nothing.

### Confirmed on GPU, and the prediction was exact

| diagnostic | before | after |
| --- | --- | --- |
| `center_mismatch` (of 16383 nodes) | -- | 11116 / 10628 |
| `center_max_delta` (box spans ~600) | -- | 180.9 / 114.1 |
| `seed_mac_fail` | ~31186 (inferred) | **0 / 0** |
| far walk's `near_pairs` | 166505 / 85124 | **0 / 0** |
| `far_pairs` vs `recv_csr` | 149223 of 180409 | **180409 of 180409** |
| `export_mac_fail` / `export_near_mac_fail` | -- | 0 / 49623 (control discriminates) |

The two centre arrays differ on two thirds of the nodes by up to 180 units, so the fix was not a no-op. Every
seed now survives the receiver's re-test, the far walk emits no unusable near pairs at all, and `far_pairs`
equals `recv_csr` exactly -- which is precisely what the theory said an admissible seed should do. The far-only
error fell 2.0061e-01 -> **7.1542e-02** and C4's control went 2.95x -> 6.59x.

### Coverage is still not restored, and the near half makes it worse

far-only across p = 4, 5, 6 is 7.1542e-02, 7.1427e-02, 7.1296e-02 -- 0.3 % over two orders, while the single-GPU
reference improves 3.58x across the same range. Still FLAT. Part of that is expected, since the far-only arm is
missing the cross near sources by construction, so this arm alone cannot distinguish "the near half is missing"
from "the far half still has a hole". It becomes a real discriminator only once the near half is correct.

### The near half is WRONG, and it is NOT double counting



Adding it makes the answer worse, not better: 7.1542e-02 -> 8.7360e-02, C5 = 0.82x.

**The first explanation written here was double counting, and it is WRONG.** The argument was that the sender's
two lists are complementary only at cell granularity, so after the receiver expands them a target could get both
node X's multipole and the particles of a leaf beneath X. That is testable: if the walk stops at X when X is
far-admissible for cell C, it never descends into X, so no near source for C can be a descendant of a far source
for C. Measured on CPU over a real export walk: of 14029 near pairs, all of whose cells also have far sources,
**0** have a near source descended from a far source. The two imports are disjoint subtrees and double counting
cannot be the explanation.

What fitted the evidence instead was a MISDIRECTED contribution: a term of roughly the right magnitude landing on
the wrong particles adds noise rather than signal, and a rising error is what that looks like.

The first suspect -- that pool row `r` might not be leaf node `num_internal + r` -- was WRONG:
`leaf_pool_mismatch` came back 0/0 over 5558/5314 occupied rows. But that check is what identified the real
fault, because it passes by comparing against `node_ranges`, **which is defined in Morton-sorted order**. So
`leaf_particle_indices` indexes the SORTED array, while the lane's acceleration is in the caller's original
particle order. The near term was being scattered by a sorted index into an array indexed by original position.

The scatter now maps sorted slot -> original index through `tree.particle_indices`. `perm_nonidentity` guards it
the way `center_mismatch` guarded the centre fix: it reads 99989/99980 of ~100000, so essentially every particle
was permuted and the remap is emphatically not a no-op.

| arm | before the remap | after |
| --- | --- | --- |
| + cross FAR and NEAR | 8.7360e-02 | **3.5859e-02** |
| C5 (the near half) | 0.82x, harmful | **2.00x** |
| C4 control | 6.59x | **16.06x** |

The unit test could not have caught this: it built its own pool and its own index array, so sorted and original
order coincided by construction. This is also the second time in this phase that a PASSING diagnostic located
the defect, and it matches the note in memory `mesh-galaxy-rollout` that the mesh lane's output rows come back
permuted even with zero padding.

A second, separate loss is already visible and is NOT the cause of the regression: `near_walk_far_pairs` is
89568/93005, pairs the receiver reclassified as far. They are dropped because the near import ships particles and
not multipoles, and they are not in the far import either, since they came from the sender's near list. That is
missing force, but dropping a term cannot make the answer worse than omitting the whole half.

### The dropped pairs WERE the remaining hole

`near_walk_far_pairs` = 89568/93005 -- 72 % of everything the receiver's near walk produces -- were pairs the
sender exported as NEAR (so it shipped particles) that the receiver reclassifies as FAR. With no multipole behind
them they were discarded, and they are absent from the far import too because the two export lists are disjoint
subtrees.

Sized with a one-line experiment before building anything: `near_theta = 0` makes every pair in that walk bottom
out as near, so nothing is dropped. It costs direct sums and changes no physics -- direct summation is exact --
so it measures the hole without fixing it efficiently.

| | dropped (near_theta = theta) | nothing dropped (near_theta = 0) |
| --- | --- | --- |
| `near_walk_far_pairs` | 89568 / 93005 | **0 / 0** |
| `near_list_pairs` | 34618 / 34637 | 150608 / 165105 |
| rel-L2 at p = 4 | 3.5859e-02 | **3.6959e-03** |

One discarded list was the entire 12x gap.

### Gate G4, second clause: the coverage check PASSES, the accuracy match does NOT

| p | distributed (far+near) | single-GPU reference | ratio |
| --- | --- | --- | --- |
| 4 | 3.6959e-03 | 2.9401e-03 | 1.26x |
| 5 | 2.3956e-03 | 1.5599e-03 | 1.54x |
| 6 | 1.9403e-03 | 8.2045e-04 | 2.36x |

**Monotone at last** -- 3.70e-03 -> 2.40e-03 -> 1.94e-03, after three rounds where this sweep was dead flat. The
coverage hole is closed: the field now responds to expansion order, which an expansion missing sources cannot do.

But the RATIO widens, 1.26x -> 1.54x -> 2.36x. The distributed error improves 1.90x over p = 4 -> 6 while the
reference improves 3.58x, so the distributed lane is approaching a floor near 1.9e-03 that the reference passes
straight through. Something in it does not improve with order and becomes dominant at high p. The gate asks for a
match to the lane's own accuracy, and 2.36x and widening is not a match.

**The leading hypothesis is fp32, and it is not a guess.** The lane runs `working_dtype=jnp.float32`, and memory
`fp32-roundoff-floor-at-1e7` records exactly this signature: fp32 pins the distributed force near 1e-3 and makes
expansion order inert. A floor at 1.9e-03 sits right there. The reference arm is fp32 too, but its near field
goes through the two-level fp64 accumulator of `nearfield-accumulator-fix`, whereas the cross near term is a
SEPARATE accumulation added to the lane's acceleration afterwards -- and at `near_theta = 0` it now carries
150k-165k direct pairs per device. Testing it needs a `working_dtype` knob on the probe; NOT YET RUN.

**Efficiency, deliberately not addressed.** `near_theta = 0` is correct but not the efficient answer: it converts
~90k pairs per device from an M2L into direct sums. The efficient fix is to ship multipole coefficients for the
near-exported leaves as well -- they are leaves, they already have multipoles, and the near payload already
carries their centre and radius -- so the receiver can serve those pairs by M2L. Its cost cannot be quantified on
this box, where nothing may be timed.

**A width defect found while reading the walk, fixed and guarded.** `receiver_interaction_lists` passed
`seed_a`/`seed_b` but not `seed_count`, so `init_size` fell back to the CSR CAPACITY -- 524288 slots per round
against 180409 live ones. Dead slots carry `-1` and are filtered by the walk's own liveness mask, so this is width
and not correctness, and it is NOT the cause of the rejections. `tests/distributed/test_receiver_seed_count.py`
asserts the lists are identical across three capacities with every dead slot poisoned, plus a control that
declaring those slots live DOES change the answer -- without which the invariance test could pass because the
fixture's padding is harmless rather than because the walk ignores it.

## Task 1 -- the floor: what the CPU could settle before a card came free (2026-09-22)

The handoff's hypothesis was fp32, on the strength of one asymmetry: the reference arm's near field goes
through the two-level fp64 accumulator, the cross near term does not. **Half of that premise is false.**
The accumulator is selected by `JACCPOT_NEARFIELD_ACCUM`, read in `_fast_lane.py` with default `"input"`,
and the benchmark's `apply_fast_lane_env` never sets it -- so BOTH arms of the gate probe ran with a
single fp32 accumulator. The other half was true: `cross_near_acceleration` passed no `accum` at all, so
it was pinned to `"input"` whatever the lane did. It now takes `accum` and, when `None`, reads the same
env choice the lane reads (jaccpot `f8e0fea`). Its final scatter is a permutation, one leaf slot per
live particle, so the kernel is the only place the width matters.

The sweep that decides the fp32 question needs two arms at the same width, so the probe has
`PROBE_DTYPE`; the runs are queued behind a saturated box (all eight cards foreign, loadavg 18-45). While
waiting, two other explanations for a p-independent residual were tested where they can be tested --
on CPU, with yggdrax alone, `bench/multigpu_far_import_dropped_pairs_probe.py`, at N = 2e4 and at the
gate's N = 2e5, both orderings:

1. **Pairs the far import cannot serve.** A far payload row is a childless multipole, so the receiver
   walk treats it as a leaf; a local leaf that fails the MAC against it yields a NEAR pair with no
   particles behind it, which the hook records (`near_pairs`) and drops. If that happened it would be a
   coverage hole and exactly p-independent. **It does not happen**: far pairs = CSR entries
   (115232 and 145507 at 2e5), receiver NEAR pairs against the far import **0**, seeds failing the MAC
   at the receiver **0**. The receiver walk is an exact pass-through of the sender's decisions, as the
   `import_cells` docstring claims and the centre fix made true.
2. **A looser MAC for cross pairs than for local ones.** The lane's walk uses `_build_mac_extents`
   (radii PROPAGATED up the tree plus a depth pad on zero-radius leaves); the export and receiver walks
   use the raw `geom.radius`. A pair accepted near theta converges slowly in p, so a systematically
   looser cross MAC would widen the ratio with order. **They are the same number**: propagation only
   fills ZERO extents, eff/raw = 1.000 (median and max) on every exported and every target node, and
   **0 of the accepted cross pairs would fail the lane's own test**. The accepted cross pairs sit at
   (r_t + r_s)/d median 0.68, p90 0.78, max 0.800.

**A confound in the probe itself, fixed (jaccpot `cb0ac32`).** One `default_rng(12345)` was advanced once
per arm, so the reference arm scored the FIRST 512-per-device draw and the far+near arm the THIRD. The
record's own `rel_l2` finding says a different target sample alone swings a ratio by tens of percent. The
draw is now made once per device and shared by every arm and by the SOLO process. The recorded fp32
figures (1.26x / 1.54x / 2.36x) were taken under the old sampling and are re-run in the sweep below.

**The bisection arm.** `make_cross_hook(export_theta=)`: the sender's export walk at theta 0 sends every
pair as leaf particles, the far list is empty, and with `near_theta = 0` the ENTIRE cross field is a direct
sum in the working dtype. If that arm matches the single-GPU lane, the residual lives in the multipole path
(import, M2L, cascade); if it does not, it lives in the local lane on a shard, in fp32, or in the probe.
It needs the near caps raised (~1560 leaves per device pair with ~1560 imported leaves): `PROBE_*_BITS`.

Queued, in this order, all N = 2e5 ndev = 2 leaf 64 theta 0.8 `near_theta = 0`, every flag read:
fp64 both arms p4/p6; fp32 all-direct + fp32 reference + fp32 multipole path p4/p6 (the last two
re-take the recorded numbers under the shared draw); fp32 with `accum = wide` on both arms p4/p6;
then p5 everywhere. Results below when the cards free up. **Nothing here is a timing.**

### Task 1, the sweep (2026-09-22, cards 2+7 after a 15 h wait): fp32 is NOT the floor, and fp64 cannot be run

**fp64 is unavailable on this lane by construction.** `resolve_large_n_execution_config` raises
`radix_fast_lane requires working_dtype=float32`; both fp64 arms fail in under a minute at every order. The
instrument the handoff asked for does not exist here, so the fp32 question was answered the other way round.

**The two-level accumulator changes nothing.** N = 2e5, ndev = 2, leaf 64, theta 0.8, `near_theta = 0`, ONE
shared target draw for every arm, every flag clean:

| p | mesh fp32 `accum=input` | mesh fp32 `accum=wide` | reference `input` | reference `wide` | ratio |
| --- | --- | --- | --- | --- | --- |
| 4 | 3.6120e-03 | 3.6120e-03 | 2.9401e-03 | 2.9401e-03 | 1.23x |
| 5 | 2.2580e-03 | -- | 1.5599e-03 | -- | 1.45x |
| 6 | 1.7045e-03 | 1.7045e-03 | 8.2045e-04 | 8.2046e-04 | 2.08x |

Identical to five digits on the mesh arm and to the last digit on the reference (which proves the wide path
ran). With 10-30 source leaves per target at this N the fp32 accumulation error is orders below the residual;
the 439x of `nearfield-accumulator-fix` was a 10^7-particle, many-leaves-per-target effect. **Dead.**

**The reference reproduces the record exactly** under the shared draw (2.9401e-03 / 1.5599e-03 / 8.2045e-04),
because the reference had always used the FIRST draw; only the mesh arm moved (3.6959 -> 3.6120e-03 at p4,
1.9403 -> 1.7045e-03 at p6). The confound was worth 2-12 %, not the widening.

**What is left, in quadrature** (mesh^2 - reference^2)^1/2: **2.10e-03 -> 1.63e-03 -> 1.49e-03** over p = 4/5/6,
a 1.41x improvement where the reference improves 3.58x. Not flat -- it converges, slowly.

**A third CPU negative.** The lane's own mutual pairs on the full tree sit at (r_t + r_s)/d median **0.718**,
p90 0.785, and the per-device local trees are identical to three digits; the cross pairs sit LOWER (median
0.66-0.67, p90 0.78). The MAC sum does not distinguish the populations, so "cross pairs are accepted closer
to theta" is dead too.

**The live hypothesis: lopsided pairs.** The export walk refines ONLY the source against a fixed receiver cell
(<= 4 leaves), so a cross pair can be a small cell against a node of radius up to 0.8 d. The MAC bounds the
SUM; the M2L's multipole truncation is bounded by r_s/(d - r_t) alone, which for a symmetric pair at the MAC
limit is 0.667 and for a lopsided one 0.8 -- per order. Over p = 4 -> 6 that predicts 2.25x vs 1.56x, and the
measured 3.58x (reference) vs 1.41x (excess) sit on either side. The mutual walk splits the LARGER node, so
its pairs are near-symmetric by construction. Being measured now (per-side truncation factors of the two
populations); the all-direct bisection arm is queued behind the Task 2 check and decides whether the excess
is in the multipole path at all.
## Task 2 -- the near-exported leaves ship their multipoles; `near_theta` returns to theta (2026-09-22)

Branch `perf/multigpu-near-multipoles` (worktree `jaccpot-mgpu-task2-wt`, off the Task 1 head). The near
payload is now `[3W positions | W masses | 3 geometric centre | 1 radius | n_coeff multipole | 3 expansion
centre]` per leaf tile, and the receiver's near walk runs at theta again: its near pairs are summed directly as
before, its far pairs -- **89568 / 93005 per device at N = 2e5, 72 % of the near walk, the ones that were
unservable and were the final 12x** -- become a SECOND imported multipole block behind the far import.
`merge_imported_blocks` stacks the two and shifts the near block's source rows by the far payload's
**capacity**, not its live count (the far payload's dead rows are still rows of the concatenated array); a
control test reads the multipole each pair points at, so a live-count shift would fail it. `_l2l.py` needs no
change: the imported block is simply longer and `n_targets` stays `n_local`.

**Verified on cards 2+7, same probe, same shared draw, every flag clean:**

| p | `near_theta = theta`, multipoles shipped | `near_theta = 0` (all direct) | difference |
| --- | --- | --- | --- |
| 4 | 3.6200e-03 | 3.6120e-03 | +0.2 % |
| 5 | 2.2580e-03 | 2.2580e-03 | 0 |
| 6 | 1.7073e-03 | 1.7045e-03 | +0.2 % |

The two agree to expansion accuracy at every order, which is the check the handoff demanded: pairs are neither
double-served nor dropped. The same worktree at `near_theta = 0` reproduces 3.6120e-03 exactly, so the code
change is inert when the knob removes the far pairs, and the 0.2 % is the M2L truncation on ~90k pairs that
were exact direct sums before. The cost side cannot be quoted on this box (nothing here is a timing); what
moved is ~90k direct-sum leaf pairs per device back into the M2L, at the price of `n_coeff + 3` floats per
near-exported leaf (+20 % of the near payload at p6, W = 64).

Double counting is excluded twice over: the sender's far and near source sets are disjoint subtrees (measured
0 of 14029 earlier), and at the receiver each (local leaf, imported leaf) pair ends in exactly one of the two
lists the walk emits.

### Task 2 re-verified under the COM geometry

With Task 1's geometry fix merged in (`66494a8`), the near walk at theta classifies MORE of the near import as
far -- 125472 / 173775 pairs per device (was 89568 / 93005 under the box geometry) -- and every one goes
through the M2L from the shipped leaf multipole:

| p | `near_theta = theta`, multipoles shipped | `near_theta = 0` (all direct, COM geometry) | difference | vs reference |
| --- | --- | --- | --- | --- |
| 4 | 2.3806e-03 | 2.3623e-03 | +0.8 % | 0.81x |
| 5 | 1.1951e-03 | 1.1804e-03 | +1.2 % | 0.77x |
| 6 | 6.9536e-04 | 6.8569e-04 | +1.4 % | 0.85x |

Flat in p against the single-GPU lane and within the M2L's own truncation of the direct-sum control. This is
the configuration to carry forward: `near_theta` defaults to `theta`, no knob set, every flag clean.

### Task 1 RESOLVED: the cross MAC was tested about the box, the expansions are about the COM

**Bisection first.** With `PROBE_EXPORT_THETA=0` the entire cross field travels as particles and is summed
directly (8.4M export entries, 29.5M near leaf pairs per device, caps 2^24/2^25, every flag clean):

| p | all-direct cross | multipole-path cross (AABB geometry) | reference | multipole-path component (quadrature) |
| --- | --- | --- | --- | --- |
| 4 | 1.9192e-03 | 3.6120e-03 | 2.9401e-03 | 3.06e-03 |
| 5 | 1.0598e-03 | 2.2580e-03 | 1.5599e-03 | 1.99e-03 |
| 6 | 6.0038e-04 | 1.7045e-03 | 8.2045e-04 | 1.60e-03 |

The all-direct arm is BETTER than the single-GPU lane at every order and converges 3.2x over p = 4 -> 6, so the
local lane on a shard, the near path and the probe are all sound; the residual is entirely in the cross
MULTIPOLE path, and its component decelerates (1.54x, then 1.25x): a floor of ~1.4e-03 plus a part that
converges like the local lane. The per-particle dump (`PROBE_DUMP`) puts 67 % of the distributed error^2 at
p6 in the shell r in [0.5, 1) and half of the excess in 1 % of the targets, inner particles with ~1 % force
error: a few pairs, not a diffuse approximation.

**The cause is memory `mac-geometry-inconsistent-with-com-centres`, one level up.** The lane's real-basis
sweep expands about the COM (`center_mode='com'` only) and since sub-10ms Phase 1.2 its own walk tests the
MAC about those centres with exact particle radii about them (`resolve_walk_geometry`, default `"com"` in
the strict fused lane). The cross exchange used `upward.geometry` -- AABB centres and box half-diagonals --
for the summary cells, the export walk and both receiver walks, and shipped `mp.centers` beside it for the
M2L. `center_mismatch` 11116/16383, `center_max_delta` 180 was that gap, read the wrong way: the fix that
made "both centres travel" kept the walk consistent with ITSELF, not with the expansions. A pair admissible
about the box centre can be divergent about the COM the receiver expands from, and a divergent pair does not
improve with order. The CPU probe's exact COM radii show such pairs in both populations, but the lane's own
walk never emits one and the export walk did.

**Fix (jaccpot `6d3097c`):** the hook builds the walk geometry exactly as the lane does
(`resolve_walk_geometry(tree, positions_sorted, upward.geometry, mp.centers, leaf_cap, default_mode="com")`)
and uses it everywhere the MAC is tested. `JACCPOT_CROSS_MAC_GEOMETRY=aabb` keeps the old geometry as a
control. `center_mismatch` must now read 0, and does.

**Gate G4, second clause, PASSES.** Same probe, same shared draw, `near_theta = 0`, every flag clean:

| p | distributed, COM geometry | reference | ratio | (AABB geometry, for the record) |
| --- | --- | --- | --- | --- |
| 4 | 2.3623e-03 | 2.9401e-03 | 0.80x | 3.6120e-03 (1.23x) |
| 5 | 1.1804e-03 | 1.5599e-03 | 0.76x | 2.2580e-03 (1.45x) |
| 6 | 6.8569e-04 | 8.2045e-04 | 0.84x | 1.7045e-03 (2.08x) |

Flat in p (0.80 / 0.76 / 0.84) and BELOW the single-GPU lane -- below because the cross half is direct-summed
at `near_theta = 0` here; Task 2 puts those pairs back through the M2L and is re-verified under this geometry
below. The distributed error improves 3.44x over p = 4 -> 6 against the reference's 3.58x. Cost: the exact COM
MAC is less conservative about the centre and more honest about the radius; far pairs 180k -> 293k and near
list pairs 150k -> 249k / 165k -> 329k per device. **Nothing here is a timing.**

**What the four dead hypotheses bought.** fp32 (wide accumulator inert to five digits; fp64 refused by the
lane), dropped far-import pairs (0), a looser cross MAC (identical extents), and cross pairs sitting closer to
theta (they sit LOWER) were each killed by a measurement before the bisection arm pointed at the multipole path
and the per-particle dump at a handful of pairs. The method note is the one the record already carries: the
diagnostic that "passed" (`center_mismatch` -> both centres travel) was answering a narrower question than the
one that mattered.

### Why the one-sided export was hit and the mutual walk was not (CPU, N = 2e5)

`bench/multigpu_far_import_dropped_pairs_probe.py` with `PROBE_LOCAL_PAIRS=1`, both walks run on the BOX
geometry, then scored about the COM with the exact particle radius about it: the mutual walk's pairs are
near-symmetric (r_big/r_small median 1.2) and carry **0.3 %** of their mass/distance weight in pairs whose
expansion cannot converge about the COM (rho >= 1); the export walk's pairs are lopsided (r_big/r_small
median 10-13, p90 ~150: a fixed <= 4-leaf cell against whatever node first passes the MAC) and carry
**17.5 % and 29.3 %** of their weight in non-converging pairs. The capped proxy sum m_s min(rho,1)^(p+1) / d^2
predicts a p4 -> p6 improvement of 1.12-1.19x for the cross pairs against 2.25-2.49x for the mutual pairs --
the measured 1.41x vs 3.58x, in order and in magnitude. So both surviving hypotheses were right TOGETHER: the
box/COM inconsistency exists for every pair, and the lopsided pairs the one-sided export produces are the ones
without slack to absorb it. Testing the MAC about the COM with exact radii removes the inconsistency; the
lopsidedness stays, and is why the cross field still costs more pairs per unit of accuracy than the local one.

## First timings (2026-10-01): correct and accurate, but slower than one card

The first timed forces of this lane, from a quiet-enough host (loadavg 8-18, cards 5/6/7 idle, cards 6+7 on one
PCIe switch; `common.gpu_guard` contention monitor, no foreign process on any timed card). Every number is a FULL
force -- tree rebuild, walk, exchange, evaluation -- the same scope as the single-GPU record's `scan_full` step,
with the record's command-buffer flags. Probe `bench/multigpu_c4_cross_force_probe.py` with `PROBE_TIME_REPS=20`;
min of 20 after 3 warm-up calls, Plummer, leaf 64, theta 0.8, fp32, COM cross geometry, `near_theta = theta`.

| N total | cards | p | local only (ms) | full force (ms) | error | particles/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1e5 | 1 | 6 | 8.62 | -- | 7.30e-04 | 11.6 M |
| 2e5 | 1 | 6 | 11.87 | -- | 6.77e-04 | 16.8 M |
| 2e5 | 1 | 4 | 11.52 | -- | 2.17e-03 | 17.4 M |
| 4e5 | 1 | 6 | 22.58 | -- | 5.51e-04 | 17.7 M |
| 2e5 | 2 | 6 | 9.65 | **25.45** | 6.95e-04 | 7.9 M |
| 2e5 | 2 | 4 | 9.15 | **24.55** | 2.38e-03 | 8.1 M |
| 4e5 | 2 | 6 | 13.85 | **39.99** | 6.10e-04 | 10.0 M |

* **The one-card arm reproduces the single-GPU record** (11.87 vs 11.45 ms at 2e5 p6, 11.52 vs 11.35 at p4), so
  the mesh evaluator costs nothing at ndev = 1 and these rows are on the record's scale.
* **Two cards are slower than one at every N measured**: 2.1x at 2e5, 1.8x at 4e5. At 2e5 per device the force
  is 3.4x the single-GPU lane's per-device time; the plan's final gate is 1.5x.
* **Against the old distributed lane** (199.5 ms at 1.3e5 per device on 2 cards, 1.3 M particles/s) the new lane
  is ~7.7x faster per particle. That gap was the plan's starting point and it is mostly closed.
* **The cross field costs 16 ms at 1e5 per device and 26 ms at 2e5 per device, nearly the same at p4 and p6** --
  it is not expansion work. At ndev = 1, with nothing to import, the hook alone costs 4.1-4.4 ms.

**Where the 16 ms goes** (per-device perfetto trace, 2 cards, N = 2e5, p6; cross arm minus local arm, device 0):

| stage | extra per force |
| --- | --- |
| kernel launches | +1,864 (1,576 -> 3,440) |
| ragged all-to-all (4 rounds: far + near, payload + CSR) | 4.5 ms |
| summary all-gather | 1.2 ms |
| generic fusions, sorts, gathers/scatters (the export and receiver walks run as TRACED JAX, not the Pallas walk) | ~7 ms |
| M2L + extra near-field kernel | < 1 ms |
| idle gaps (launches + collective synchronisation) | +6.6 ms |

**The levers, in order of size.** (1) Run the export walk and both receiver walks through the Pallas mutual walk
the local lane already uses -- the same cure the sub-10 ms work applied to the local walk. (2) Merge the far and
near exchanges into one, halving the ragged rounds, and ship live particles instead of 64-slot padded tiles;
each round also memsets a capacity-sized receive buffer. (3) Cut the launch count, which drives the idle gaps.
A one-card baseline at N/dev = 1e6 and the jz-fmm multi-GPU front (Phase 0.2) are still unmeasured.

## The capacity flags could not fire (2026-10-01), on either lane

Found while planning the large-N sweep, then measured. Under `jit` a saturated walk
(far / near / queue) or cell-leaf partition cannot raise: it surfaces in ONE place,
`compact_far_pairs.far_pair_count`, saturated to the far-pair buffer's length. Two
defects hid that signal.

1. **The mesh lane read three attributes that do not exist.** `_local_overflow` OR-ed
   `leaf_capacity_overflow`, `walk_overflow` and `capacity_overflow`;
   `LargeNPreparedState` has none of them, so the mesh lane's local flag was the
   constant `False`. The cross FAR half's flags reached the caller only through the
   diagnostics `record`, which the production and timing paths do not pass.
2. **The refresh replaced the list whose count carries the signal.** In the
   fresh-rebuild mode (the default) `_refresh_large_n_same_topology` swaps the far-pair
   list it just built for the cached placeholder before returning, so the carried
   state keeps its shapes. A guard reading the returned state saw the PREPARE's count.
   That disabled the far-pair arm of the single-GPU `strict_run_v2` guard too -- its
   own error message describes exactly this saturation and could not be reached.

**Measured (1 card, N = 2e5, the mesh evaluator's local arm):** evaluating at theta 0.3
against capacities sized for 0.8 gives a force that is 99 % wrong (rel-L2 9.91e-01)
with `overflow=False` before the fix and `overflow=True` after; the theta-0.8 control
stays `False` with an unchanged error (6.7740e-04). The single-GPU record
configuration through `strict_run_v2` still runs without raising (11.68 ms/step,
aggL2 7.4121e-04, the record's value).

**Fixes (jaccpot `ff4f4ef`, `d173d19`):** one shared guard
(`jaccpot.runtime.capacity_guard.fused_state_capacity_ok`) with a structural check
against the buffer's own length; the refresh evaluates it on the lists it built and
leaves the verdict on the engine for the same trace (`last_refresh_capacity_ok`); the
cross hook carries an always-on `flag_sink` (far half, the near walk's far list, and
both exchanges' received counts against their receive capacities). Earlier ACCURACY
results stand -- they were checked against fp64 direct sums -- but every
`overflow=False` before this was vacuous for the local half and, in timing mode, for
the far half.

**Also fixed while there (`13bb9fa`):** each eager per-shard prepare overwrote the
engine's walk-capacity record, so every device's traced walk was sized for the LAST
shard; the records are now merged (field-wise max) and installed. And the traced
cell-tree depth check now reads the same bound the upward sweep loops to (the
installed plan first).

## Gate G2: the rollout repartitions on device, and nothing but ownership changes (2026-10-01)

`jaccpot/distributed/rollout.py` + yggdrax `sfc_repartition`; gate `bench/multigpu_rollout_gate.py`, rows in
`bench/results/multigpu_rollout_gate/`. Plummer N = 2e5 with isotropic DF velocities (virial), leaf 64,
theta 0.8, p6, dt 0.005, 100 steps, 2 cards (1 and 4; a correctness gate, not a timing), every flag live.

**CPU first (4 forced devices, partition-independent reference force):** repartitioning every step and never
repartitioning give BITWISE-identical positions, velocities and accelerations by id over 30 steps at ndev 2
and 4, with different ownership; a mutated arm that leaves velocities in the old row order is caught.

**GPU, the fused lane:**

| clause | result |
| --- | --- |
| flags | clean on all 100 steps |
| ids exactly once | at every repartition and probe step |
| balance | counts 99542 / 100458 at worst (bound 101,563 = ceil(N/2) x (1 + 4/256)) |
| repartitions | 6 (steps 16 .. 96), each moving ~3,100 particles |
| ownership vs the never-repartition arm at step 100 | 17,731 of 200,000 particles on a different device |
| force vs the single-GPU lane on the same positions | ratio 0.87 - 1.07 over the 7 probe steps (gate 1.2) |
| energy dE/E at step 100 | 2.48e-04 (2 cards) vs 2.50e-04 (1 card) vs 2.47e-04 (2 cards, never repartitioned) |

The fp64 probe errors stay between 5.8e-04 and 8.6e-04 over the run, the single-GPU lane's class. The energy
drift is the time step's and the force error's, not the repartition's: the three arms agree to three digits,
including a rise between steps 50 and 75 that all of them show.

**~3,100 moved per repartition is mostly pivot jitter, not dynamics:** at 256 samples per device the sample-sort
boundary moves by ~2 ndev / num_samples = 1.6 % of a shard between repartitions, about that many particles.
Harmless for correctness (the moves are exact), but it is exchange volume a cheaper criterion could skip.

**Not covered:** the cross-volume comparison against the control arm (the timing-mode hook carries no
diagnostics), a disc IC, ndev > 2, and a run long enough for the repartition to matter for the cross cost.

## Cutting the cross cost (2026-10-02): two cards now beat one from 1e6 up

**Setup.** Cards 1+2 (one CPU socket, different PCIe switches), Plummer seed 0, leaf 64, theta 0.8, p6, fp32,
full force per call (local and cross arms), min of 10-20 after 3 warm-ups, host load 8-14. Every change below
was kept only after an interleaved A/B on the same cards, with rel-L2 vs fp64 unchanged in every printed digit.
One A/B was discarded because another user's job landed on its cards mid-run (that trace ran 10x slow).

**What each change bought** (2-card cross arm at N = 2e6, i.e. 1e6 per card):

| change | ms |
| --- | --- |
| start of the day | 189 |
| NCCL ragged exchange: `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` | 138 |
| near-field padding dropped instead of piled on one row (both lanes); send counts without atomics | 122 |
| one walk geometry shared by the cross hook and the local walk | 121 |
| far receiver walk skipped (a pass-through), queue 2^24 -> 2^22, export and near walks on the Pallas walk | 97 |
| export walk's own queue; receive caps from live counts; XLA latency-hiding scheduler | 88 |
| `cell_min_level = 8` (no leaf coarser than 1/256 of the box) | 75 |
| `summary_cell_level = 8` (no summary cell coarser either) | 72 (71.7 / 72.5) |

**Where it stands** (both levels 8; the probe's leaf capacity is now 1.15x the live leaves, see below):

| N | 1 card (ms) | 2 cards (ms) | speed-up | error 1 / 2 cards |
| --- | --- | --- | --- | --- |
| 4e5 | 17.9 | 18.6 | 0.96x | 5.51e-04 / 6.08e-04 |
| 1e6 | 42.0 | 35.2 | 1.19x | 7.13e-04 / 7.02e-04 |
| 2e6 | 88.9 | 71.7 | 1.24x | 8.71e-04 / 8.81e-04 |
| 8e6 | 330.6 | 231.6 | 1.43x | 4.56e-04 / 5.36e-04 |

The final gate (per-device time within 1.5x of one card) is NOT met: at 1e6 per card, two cards at 2e6 would
have to take <= 1.5 x 42.0 = 63 ms. They take 71.7.

**Findings, in the order they were made**

1. **The exchange kernel, not the bytes.** `bench/multigpu_exchange_bench.py` times the cross hook's own helper
   at its row widths. XLA's default one-shot `ragged_all_to_all` kernel stores straight into peer memory and
   moves ~2 GB/s over PCIe; the NCCL path moves 11-13 GB/s, the same on a pair under one switch and a pair across
   switches. `xla_gpu_ragged_all_to_all_mode` takes `peer|private|symmetric` (not the enum names); `symmetric`
   hangs and `private` is 0.75 GB/s. XLA reads its flags once, so the library cannot set this: the mesh evaluator
   warns when it is missing (`fused.RAGGED_EXCHANGE_XLA_FLAG`) and the probe and rollout gate set it.
2. **Padding piled on one address.** The near-field segment sum sent every padding chunk to one extra row (520k
   of 655k chunks at 1e6 per card, 10 ms of a 74 ms local force in a trace without command buffers), and the
   cross near term sent every dead leaf slot to one discard row. Both now drop padding (`FILL_OR_DROP`). The
   per-destination send counts were a `segment_sum` of 2^23 rows into two counters (2.6 ms a call); they are now
   two lookups at the device boundaries of the sorted order.
3. **The far receiver walk is a pass-through by construction.** It pairs the receiver's cell root with a node the
   sender accepted against that very cell, on the same centre and radius, so it can only re-accept: far pairs ==
   received CSR entries (5,083,648 / 4,640,119), zero near pairs, and the seed MAC re-check fails 0 of them. It is
   now read off the CSR; that walk's seed was also why the queue had to be 2 x the receive CSR capacity.
4. **The Pallas walk needs a right-sized queue.** Seeded (`seed_a/seed_b/seed_count`), it emits exactly the
   traced walk's pair sets, but at a 2^24 queue it bought nothing: its grid is one program per 64 queue slots on
   EVERY round and its loop carries are copied each iteration. At the measured peaks (export 0.9M, near 2.7M ->
   2^22) it wins 9 ms at 2e6 and 4.6 ms at 4e5 over the traced walk.
5. **Payload compaction is not a speed lever.** Shipping the near import's live particles flat (plus a count per
   leaf) instead of 64-slot tiles is bit-identical (CPU test) and cuts that payload from ~65 to ~16 MB per
   direction, but times +1.3 ms at 2e6 and -0.5 at 4e5: the exchange is bound by its synchronisation, not its
   bytes. It stays (less receive memory; `JACCPOT_CROSS_NEAR_TILES=1` is the control).
6. **It exposed the real waste: the near export shipped the WHOLE remote shard** (999,999 of 1e6 particles, 51k
   of 51.4k leaves) every force. Leaves are the coarsest Morton cells holding <= leaf_size particles, so a
   sparse outskirt cell with a few far-apart outliers stays one leaf (Morton depth 1-5) whose bounding sphere
   spans much of the box; it fails the MAC against everything. One summary cell per device held every remote
   leaf in its near CSR. The same leaves have near rows of hundreds of thousands on ONE card (the long serial
   rows of the near-field kernel). `TreeConfig.cell_min_level = 8` splits them: +0.7 % leaves, one card 4-18 %
   faster from 2e5 to 8e6, two cards 88 -> 75 ms at 2e6. `summary_cell_level = 8` bounds the cells the same way
   (monotone Morton-cell edge, so the occupancy cut stays a cut): 75 -> 72 ms at 2e6, 254 -> 232 ms at 8e6.
   Near pairs at 2e6 fell from 1.0M to 150k per device; the particles shipped did not (one small cell next to
   the dense core, radius 16 at r = 17, is still near most of the remote core at cell granularity).
7. **A measurement trap: the power-of-two leaf capacity.** The probe sized it as the next power of two above
   1.25x the leaves. Time is linear in that padding (4e6 on one card: 299 / 374 / 541 ms at 1.2 / 2.4 / 4.8x with
   the same pairs to the digit), so 800 extra leaves that crossed 65,536 made `cell_min_level` look 23 % SLOWER at
   1e6. It is now 1.15x in steps of 1024 (`PROBE_LEAF_CAP_RULE=pow2` restores the old rule); that alone took one
   card at 4e5 from 22.5 to 18.6 ms.
8. **The cubic Morton box is a trade, not a fix.** A per-axis box gives every cell its aspect; a 4e6 draw with a
   3.2:1 box walked 58.1M far pairs (2.7e-3) against 14.2M (4.9e-4) in a cube -- but at 2e6 and 8e6 the cube is
   13 % and 28 % slower for a smaller error. Opt-in: `JACCPOT_CUBIC_BOUNDS=1`.

**What remains** (stage-split trace at 2e6, `bench/analyse_trace_by_stage.py`; command buffers off, so the
proportions and not the totals count): the cross arm adds ~33 ms of kernel time per device. ~20 ms of it is the
cell-level FAR pairs -- ~5.5M per device, treecode-style (~360 sender nodes per summary cell, because only the
sender side refines): the cross M2L +6.6 ms, its gathers +2.7, the wider CSR sort +1.4, the export walk 3.3, the
far CSR on the wire. The rest is the exchange (NCCL 7 ms), device copies (+2.3), the near term (2.8) and send
buffers. The lever is a **two-sided export walk** over the receiver's summary tree, so large sender nodes pair with
large receiver nodes as the local mutual walk does; plan `two-sided-export-walk.md` (2026-10-02).

## The two-sided export walk (2026-10-03): the 1e6-per-card gate row passes

Plan `two-sided-export-walk.md`. yggdrax `summary_tree` + `export_walk_two_sided`; jaccpot `cross.py`
(`JACCPOT_CROSS_TWO_SIDED`, default on; `=0` is the one-sided control).

**What changed.** The one-sided export walked the sender's tree against the receiver's summary CELLS, which are
childless, so only the sender refined and every cell collected a treecode-style list of sender nodes: 4.6-5.1M
far pairs per device at 1e6 per card. Each device now publishes the top of its tree -- the occupancy cut plus every
ancestor of it, with child links, root at index 0, in ONE packed all_gather (centre, radius, children, active;
indices exact as floats) -- and the export walk splits whichever side is larger, as the local mutual walk does.
Far pairs land on the receiver's internal nodes; its L2L cascade carries them down; the far receiver lists stay a
pass-through (the CSR index names the summary node the sender tested). Near pairs name cut cells only.

Two details that a test pinned rather than an argument:
* The walks read `left < 0` as a leaf. Where the padding subtree joins the live tree, an ancestor of the cut has
  one EMPTY child (one such node in every padded shard). It is kept as an inactive summary leaf; dropped, the
  walk would stop at that ancestor.
* Ancestors are "live, not in the cut, no strict ancestor in the cut"; the last is one pointer-doubling pass over
  `parent` (ceil(log2 nodes) rounds of two gathers).

**Coverage, tested first.** Momentum cannot see a coverage error, so `tests/unit/distributed/test_cross_coverage.py`
counts: every pair the pipeline emits (publish, export, send buffers, the receiver's slice, the far pass-through,
the receiver near walk -- the hook's own `_summary_rows` / `_export_from_rows`) is expanded to its particle block,
and the (receiver particle, sender particle) matrix must be exactly 1. It passes one-sided, two-sided over cells
and two-sided over leaves, both directions, theta 0.8 and 0.5, with a size-bounded cut; a dropped summary entry
and a duplicated far pair are caught. yggdrax tests the export lists the same way at ndev 2 and 3, and the
Pallas walk's two-sided pair sets equal the traced walk's.

**Volume** (two A100s, Plummer seed 0, leaf 64, theta 0.8, p6, per device):

| | 1e5 / card one-sided | two-sided | 1e6 / card one-sided | two-sided cells | two-sided leaves |
| --- | --- | --- | --- | --- | --- |
| export far pairs | 283-367k | 54k | 4.6-5.1M | 437-458k | 606k |
| export walk peak | 36-47k | 6.2k | 0.72-0.90M | 61-63k | 71k |
| near CSR | 12-14k | 12-14k | -- | 265-288k | 149k |
| near-walk far pairs (M2L) | 21-25k | 20-25k | -- | 479-501k | 0 |
| near particles shipped | | | | 0.98M / 0.80M | 0.23M / 0.52M |

Two-sided far pairs are 8.7 per leaf at both N: O(N), as an FMM's should be.

**Accuracy** (N = 2e5, rel-L2 vs fp64 over all sources): p4 / p5 / p6 = 2.38 / 1.19 / 0.695e-3 one-sided,
2.24 / 1.20 / 0.663e-3 two-sided (one-card lane 2.94 / 1.56 / 0.820e-3). Flat in p, in class. At 2e6 8.81e-4 ->
8.80e-4 (cells) / 8.85e-4 (leaves); at 8e6 5.36e-4 -> 4.95e-4.

**Timing** (cards 6+7, a PIX pair, interleaved arms from frozen worktrees, min of 15, cross arm):

| N | one-sided | two-sided cells | one-sided + caps | two-sided cells + caps | two-sided leaves + caps |
| --- | --- | --- | --- | --- | --- |
| 4e5 | 17.35 / 17.43 | 17.05 / 17.06 | 16.73 / 16.55 | 16.46-16.59 | 16.65 / 16.43 |
| 2e6 | 71.12 / 71.22 | 64.15 / 64.01 | 67.94 / 67.93 | 61.11-61.47 | 57.35 / 57.31 |
| 8e6 | 229.4 / 229.6 | 202.4 / 202.3 | 219.5 / 219.2 | 188.9-189.1 | 188.3 / 188.3 |

**Findings, in order.**
1. **The far pairs were not what the M2L paid for.** Two-sided cut the export far pairs 12x but the 2e6 force by
   only 7 ms, not the ~20 the stage trace had attributed to them. The M2L kernel took 5.4 ms with ~5.6M cross
   pairs and 5.2 ms with ~0.95M -- against 1.4 ms in the local arm. Its grid is ONE program per target walking its CSR
   row in 32-lane tiles, so the cost is the longest row, not the pair count. The long rows came from the NEAR
   import: one outskirt cell (4 leaves, radius 15.8 at r = 17.4) was near 48k sender leaves, and the receiver's
   near walk gave its leaves those leaves as M2L sources.
2. **Publishing receiver LEAVES fixes that.** With the summary a tree, cells of one leaf
   (`max_leaves_per_cell = 1`, now the two-sided default) cost only a deeper walk: the sender decides "needs
   particles" per receiver leaf, the receiver near walk becomes a pass-through (0 far pairs out of it; tested),
   the near CSR halves and the near particles shipped fall 2-4x. 2e6: 61.3 -> 57.3 ms. At 8e6 and 4e5 it is
   neutral (-0.7 / 0 ms): the 2e6 seed-0 draw is the one with the pathological outskirt cell.
3. **The receiver caps were 8-17x oversized, in both arms.** The probe's per-leaf factors predate
   `summary_cell_level = 8`, which cut the near import. At 1e6 per card: near-walk queue 4.2M for a 0.52M peak,
   near-walk far list 8.4M for 0.50M, near list 2.1M for 0.15M, near CSR 4.2M for 0.29M. Each is padded work: the
   far list joins the merged M2L CSR sort (27M wide at 2e6) and two more 8M sorts, the queue sets the Pallas grid.
   Re-derived from the live counts: -3.3 ms one-sided and -2.9 two-sided at 2e6, -10 / -13 ms at 8e6, forces
   identical in every digit.

**A pre-existing silent truncation, found by the three-seed gate** (fixed in this branch). Two of nine two-card
gate rows were wrong with every capacity flag False -- seed 2 at 4e5 rel-L2 4.2e-2, seed 1 at 1e6 3.6e-2 -- and
identically so in the one-sided and both two-sided modes (4.2287 / 4.2288 / 4.2298e-2), against 7.8e-4 on one
card. A per-particle dump (`PROBE_DUMP`) put it on ONE particle that kept its near field and lost its far field
(0.024 against 0.37). Cause: `measure_shard_plan` ran the eager prepare in each shard's OWN box, while the traced
force rebuilds in the GLOBAL mesh box; different boxes cut different cells, and shard 0's widest level was 4564
nodes against a planned 4525 (= 1.25 x 3620, its own-box width), so the M2M/L2L level loops dropped 39 nodes and
everything under them. A larger leaf capacity (which widens the plan through the padding levels) or the plan
widened by 10 % (`PROBE_PLAN_WIDEN=1.1`) restored 8.8e-4 exactly. The refresh's own width guard folds into the
walk's far-pair saturation and never reached the mesh flag: with the plan's width HALVED on purpose the force was
49 % wrong and every flag read False. Fixed: the eager prepare builds in the mesh box
(`strict_fused_prepared_eval_fn(bounds=)`), the registry is lifted to the tree rebuilt exactly as the refresh will,
and `capacity_plan.plan_level_overflow` is ORed into the mesh flag (the halved plan now raises it). The eager
own-box tree also over-counted leaves (33,388 against 26,302 on a 1e6 shard), which is why the probe's leaf
capacity raised at setup on seed 1. Every gate row before this one was seed 0, which does not trip it.

**The gate** (one card at N on card 6 vs two cards at 2N on 6+7, back to back, min of 15, after the fix):

| per card | seed | 1 card at N | 2 cards at 2N, local / cross | ratio | gate 1.5 |
| --- | --- | --- | --- | --- | --- |
| 2e5 | 0 | 10.73 | 12.18 / 16.59 | 1.55 | fail by 0.5 ms |
| 2e5 | 1 | 11.24 | 11.73 / 17.46 | 1.55 | fail by 0.6 ms |
| 2e5 | 2 | 11.59 | 11.56 / 17.02 | 1.47 | pass |
| 5e5 | 0 | 21.75 | 21.29 / 28.70 | 1.32 | pass |
| 5e5 | 1 | 19.36 | 21.65 / 29.03 | 1.50 | pass (on the line) |
| 5e5 | 2 | 22.84 | 23.57 / 30.78 | 1.35 | pass |
| 1e6 | 0 | 41.98 | 46.28 / 56.24 | 1.34 | pass |
| 1e6 | 1 | 42.54 | 44.75 / 54.66 | 1.28 | pass |
| 1e6 | 2 | 43.92 | 42.39 / 52.05 | 1.19 | pass |

The 1e6-per-card row -- the one this plan was for (needed <= 63 ms at 2e6, was 71.0) -- passes on every draw.
At 2e5 per card the cross hook's fixed cost (~5 ms over the local arm on a launch-bound one-card time of
10.7-11.6 ms) is what fails; the volume this plan cut is not. Same-N accuracy, two cards against one: seed 1 at
4e5 1.36e-3 / 1.22e-3, seed 2 at 4e5 8.81e-4 / 7.82e-4, seed 2 at 1e6 1.24e-3 / 1.17e-3 (the seed-1 4e5 and
seed-2 1e6 draws are harder for both lanes).

**Gate G2.2 on the new defaults** (`bench/multigpu_rollout_gate.py`, arm D, N = 2e5, 100 steps, repartition every
16; rows in `bench/results/multigpu_two_sided_export/rollout_gate/`): flags clean and ids exactly once on every
step, 6 repartitions moving 18,725 particles, fp64 probe errors 6.10-8.01e-4 (the 2026-10-01 run: 5.8-8.6e-4);
the one-card lane on the same positions 6.10-8.04e-4, a ratio of 0.96-1.01 (gate 1.2); dE/E at step 100
2.44e-4 on two cards against 2.50e-4 on one.

**Not covered:** ndev > 2 on GPU (the CPU coverage and yggdrax tests run ndev 3); the summary all_gather moves its
CAPACITY (2 x the leaf capacity x 7 floats: 3.4 MB per device at 1e6 per card), which grows with ndev; one outskirt
leaf (radius 9.4) still has 17.5k near sender leaves at 2e6, a long row in the cross near-field kernel.

## Beyond two cards (2026-10-03): the symmetric exchange

Four free A100s (cards 4-7: two PIX pairs, one CPU socket). Probe as before, arms from frozen worktrees.

**Four cards worked out of the box and are in class.** At N = 8e5, seed 0: one card 1.13e-3, two cards 1.30e-3,
four cards 1.16e-3 (rel-L2 vs fp64; this draw is harder for every lane). No flag fired.

**But weak scaling was poor**, at 2e5 per card: one card 10.7 ms, two cards 16.3, four cards 25.5. The four-card
trace (command buffers off) put it in the exchanges, not in compute:
* NCCL `SendRecv` 6.45 ms per call on four cards against 0.88 on two, over five ragged exchanges (far payload, far
  CSR, near rows, near CSR, near particles), plus a summary all_gather and two size all_gathers;
* XLA's NCCL path for `ragged_all_to_all` reads the send/receive sizes back to the HOST before posting the sends, so
  every ragged exchange is a host sync: each `ragged-all-to-all-start` host thunk held its device thread for
  3.7-13.7 ms per call (summed over the four device threads);
* shrinking every exchange capacity to ~1.5x its live count bought only 1 ms, so it is the synchronisation, not
  the bytes.

**What each change bought** (cross arm, min of 15, interleaved; forces identical in every printed digit unless
noted):

| change | 2 cards, 4e5 | 4 cards, 8e5 | 2 cards, 2e6 | 4 cards, 4e6 |
| --- | --- | --- | --- | --- |
| #356 (two-sided over leaves) | 16.59 | 26.55 | 56.15 | 117.96 |
| near receiver walk skipped (a pass-through over leaves) | 16.34 | 25.47 | 55.90 | 99.02 |
| probe capacity floors 2^21 -> 2^18 | 15.72 | 23.00 | 55.82 | 99.34 |
| symmetric exchange, sort-based rows | 16.01 | 23.65 | 56.70 | 96.33 |
| symmetric exchange, presence-map rows | 15.06 | 22.21 | 54.63 | 91.97 |

1. **The near receiver walk is a pass-through over leaves.** Every near pair the sender emits is (my one-leaf
   cell, its leaf) on the same geometry, so re-walking it can only re-find it (0 far pairs, near == CSR, measured).
   Read off the CSR, and the near rows carry no geometry and no multipole. Worth 19 ms at 4e6 on four cards.
2. **The symmetric exchange.** Over leaves every device publishes its whole live tree, so the summary all_gather
   already hands every device every tree. Each device walks its tree against each peer's over the gathered blocks,
   seeded (lower device, higher device) -- the identical walk, in the identical orientation, that the peer runs
   (the walk splits side `a` on an exact radius tie, so "peer vs mine" and "mine vs peer" could decompose
   differently) -- and reads BOTH lists off the same pairs. Only multipoles and particles travel: two ragged
   exchanges instead of five, no CSR. Both ends order rows by the sender's summary index; the received sizes are
   checked against the locally predicted ones, so a divergence raises the flag. Tested on CPU: both devices derive
   the same pair set, the sender's rows equal the receiver's expected rows, the receiver's lists cover every cross
   pair exactly once, and the Pallas walk's sets equal the traced walk's.
3. **No sort for the row layout.** The first version deduplicated (peer, node) with an argsort over the pair-list
   capacity, four per force, and LOST at two cards. (peer, node) lives in the fixed `[ndev x S]` summary space, so
   a presence map and one prefix sum give every row: -0.6 to -7 ms against the floors arm everywhere.

4. **Unrolled `searchsorted` on the hot paths.** The default `jnp.searchsorted` is a while loop with one small
   kernel per bisection step -- ~23 for the near-field CSR offsets (13k queries into 8.4M sorted keys at 2e5 per
   card) and as many for the M2L CSR by target -- launch-bound inside the fused step. `method="scan_unrolled"`
   is the same search as straight-line code: one card at 2e5 10.80 -> 10.12 ms, four cards at 8e5 22.1 -> 21.0
   ms, forces identical (measured with the probe's `PROBE_SEARCHSORTED` switch, then applied at 11 call sites).
5. **A finer `cell_min_level` fixes the near-import imbalance but loses.** On four cards device 0 imported 554k
   near particles against 120-200k on the others: a big sparse leaf of device 0 next to a neighbour's dense core
   pairs with thousands of small leaves, and the symmetric near pairs carry very asymmetric particle counts. Level
   10 (11) evens that out (124k / 152k / 197k / 126k) but adds 22 % (one card) to 49 % (four-card shards) leaves:
   one card 10.72 -> 12.24 ms, four cards 21.70 -> 21.96. Level 8 stays.
6. **What remains at four cards is compute and waiting, evenly spread.** Per-device compute 18.4-19.8 ms per call
   (no skew worth a cost-weighted partition), NCCL 5.6-9.5 ms of which most is waiting at collectives; and the
   cross VOLUME per device roughly doubles from two to four cards at equal particles per card (export far pairs
   ~150k -> 283-403k at 2e5 per card): more domain boundary per device.

**Weak scaling on the final state** (`bench/results/multigpu_ndev_scaling/weak/`; one card at N, two at 2N, four at
4N, min of 15, seeds 0-2; host load 8-20 from other users' jobs, so the launch-bound 2e5-per-card single-card time
moves 10.1-12.7 ms; every run flag-clean):

| per card | seed | 1 card | 2 cards | 4 cards | 2 / 1 | 4 / 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2e5 | 0 | 12.36 | 15.43 | 21.95 | 1.25 | 1.78 |
| 2e5 | 1 | 12.70 | 16.02 | 24.24 | 1.26 | 1.91 |
| 2e5 | 2 | 11.06 | 15.38 | 20.63 | 1.39 | 1.87 |
| 5e5 | 0 | 21.29 | 26.21 | 36.31 | 1.23 | 1.71 |
| 5e5 | 1 | 18.67 | 26.54 | 37.01 | 1.42 | 1.98 |
| 5e5 | 2 | 22.34 | 29.47 | 34.80 | 1.32 | 1.56 |
| 1e6 | 0 | 41.20 | 53.55 | 87.72 | 1.30 | 2.13 (*) |
| 1e6 | 1 | 41.85 | 51.15 | 61.93 | 1.22 | 1.48 |
| 1e6 | 2 | 42.73 | 49.29 | 59.01 | 1.15 | 1.38 |

(*) the 4e6 seed-0 draw is the known per-axis-box anomaly (rel-L2 2.4e-3 on one card as on four).

**Two cards pass the 1.5x gate on all nine rows; four cards pass at 1e6 per card on both ordinary draws and fail at
2e5-5e5 per card** (1.56-1.98). rel-L2 4.0e-4 to 1.4e-3 throughout, draw-dependent; at the same N (8e5, seed 0)
one card 1.13e-3, two 1.30e-3, four 1.16e-3.

**Domain shape is the next lever, partly.** From two to four cards at equal particles per card the cross volume per
device grows (more boundary per domain). A diagnostic recursive-coordinate-bisection split in the probe
(`PROBE_PARTITION=rcb`: equal-count cuts along the longest axis; the lane takes any partition) cuts the export far
pairs at 2e6 on four cards from 560-691k to 360-515k per device and the near pairs from 143-190k to 70-176k, but
concentrates the near import further (one device 1.36M particles against 0.93M under Morton). Interleaved, two
rounds, cross arm:

| point | Morton | RCB |
| --- | --- | --- |
| 2 cards, 1e6 | 26.15 / 26.36 | 24.61 / 24.89 |
| 4 cards, 8e5 | 21.29 / 21.60 | 20.11 / 19.46 |
| 4 cards, 2e6 | 36.16 / 36.48 | 35.15 / 35.43 |

-1 to -2 ms. (Its rel-L2 reads 25-35 % lower, but the probe samples different targets per partition, so that is not
an accuracy claim.) Production partitions in Morton order (`sfc_partition`), so taking this needs yggdrax work.

**Not done, and why:** the near-import asymmetry (a big sparse leaf next to a dense core pulls in thousands of small
leaves' particles) wants an asymmetric treatment -- the far side's multipoles evaluated at the few target particles
(M2P), or the light side shipped and forces returned -- worth at most ~1.5 ms at 8e5 on four cards by the level-10
bound; and dropping the per-exchange size gather (fixed per-sender receive slots) would remove two small collectives
whose cost is mostly waiting at them.

## Next

**2026-10-03, later:** four cards run (section "Beyond two cards"): the symmetric exchange, the near pass-through,
smaller probe floors and an unrolled searchsorted; two cards pass the 1.5x gate on every row of three draws, four
cards at 1e6 per card. What follows: the four-card rows at 2e5-5e5 per card (domain shape -- RCB is -1 to -2 ms --
and the near-import asymmetry); eight cards; the 25M disc+bulge rollout on the tuned code.

**2026-10-03:** the two-sided export walk is built and the default (section above); the 1e6-per-card gate row
passes on three draws, and a silent level-loop truncation in the mesh lane's capacity plan is fixed.

**2026-10-02:** the cross field is built, correct, and two cards now beat one from N = 1e6 up (section above).
The paragraphs below are the Phase 1-3 history.


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

Phase 3 is now complete: measured (3.1, 3.3, 3.4), gated (3.2) and built (3.5). The exchange turned out to need
ONE round, not jz-fmm's progressive per-level request, because the per-cell CSR that makes the import addressable
costs only 5-60 % of the payload it addresses.

Phase 2 is under way. `sfc_partition` now routes a particle's identity WITH the particle (yggdrax `99125ff`), and
`align_level` has been measured and dropped -- unaligned equal-count Morton pivots balance exactly and the import
does not care. What remains of Phase 2 is the repartition CADENCE (every ~16 steps, with the predicate coming from
an all-reduce, because a device-divergent predicate deadlocks) and Gate G2.

Then the driver: gather the real payloads into the send buffers, wire the imported lists into the fused kernels
through `n_targets` and `num_target_leaves`, and take Gate G4 (the one-sided force against the single-GPU lane on
the identical particle set). Size the near half for the ~30 % of a neighbour domain that is structural, and
remember that expansion order is a communication knob here, not only an accuracy one.
