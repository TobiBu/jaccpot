# The fused lane's memory, sized by the pairs (2026-10-03)

Plan: `fused-memory-reduction` (single-card memory without losing step time). Branch `perf/fused-memory`, stacked
on #357. Rows: `bench/results/fused_memory/` (its README names each arm's commit). Bench:
`bench/fused_memory_budget.py`, one configuration per process (the allocator's peak is never reset).

**Where it started.** One A100 held 8e6 particles of the sub-10 ms lane (leaf 64 cell leaves, theta 0.8, p5) at
22.2 GiB; 1.6e7 ran out of memory in the eager prepare. That ceiling measured a cap SETTING: the Odisseo harness
named both flat-walk list caps as pow2(200k fit x ceil(N / 200k)) while its N-max script said they were unnamed,
and N does not decide the pair counts (the 4e6 draw has 3x the far pairs of 8e6).

## Step 0: where the bytes were

Card 3, Plummer seed 0, cell_min_level 8, leaf capacity 1.15x the live cells in steps of 1024; three interleaved
processes per arm (min). The card held an idle foreign process and the host load was ~20 throughout, so every row
carries the contention flag; arms are only compared against each other.

| N | caps | far used / cap | near used / cap | per force | per step | prepare peak | peak after the step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2e6 | harness | 11.0M / 33.6M | 5.1M / 67.1M | 23.0 ms | 95.3 ms | 5.56 GiB | 5.70 GiB |
| 2e6 | 1.5x counts | 11.0M / 16.6M | 5.1M / 7.7M | 21.8 | 76.0 | 1.23 | 1.87 |
| 8e6 | harness | 20.9M / 134M | 19.2M / 268M | 78.9 | 349.6 | 21.97 | 22.47 |
| 8e6 | 1.5x counts | 20.9M / 31.3M | 19.2M / 28.8M | 72.0 | 262.7 | 3.40 | 7.08 |

(This code was #357's head; "1.5x counts" names the caps by hand.)

* **The eager peak was the post-walk list build**, run op by op so every capacity-width local stayed alive: at 8e6
  the walk left 5.3 GiB of peak, the list build 22.0 (a 16.6 GiB transient), the downward pass nothing new.
* **Every per-step cost tracked the cap, not the pairs**: compiled-step temporaries 11.1 -> 3.2 GiB, the prepared
  state 3.06 -> 1.02 GiB (its far list alone 1.50 -> 0.35), and 25 % of the step time.
* **With tight caps the STEP holds the peak** (8e6: arguments 1.44 + outputs 1.44 + temporaries 3.18 GiB). Its
  temporaries are near-field scratch and the far field's leaf-major evaluation, not the lists.
* **Count drift along a rollout is small.** Plummer at 2e6 with equilibrium velocities (dt 1e-2, 200 steps): under
  1 % on far pairs, near edges and peak wavefront. The 25M disc+bulge IC subsampled to 2e6 (prefix of the shuffled
  file, masses rescaled, NFW halo as external force, dt 5e-4, 500 steps ~ a quarter orbit): far +2.3 %, near
  +2.6 %, live leaves +3.5 %. The default headroom stays 1.5 (see "Headroom" below).
* **Sorts.** At 8e6 sizes a keys-only int64 composite sort orders (target, source) in HALF the time of two stable
  int32 argsorts (6.4 against 13.3 ms for 28.8M near entries; identical output); int32 instead of int64 argsort
  indices alone is -13 %; a two-key `lax.sort` is 5x slower (XLA hands only single-key sorts to CUB).

## What changed

* **Unnamed list caps are sized from the counts the eager walk measured**: headroom x count
  (`JACCPOT_FLAT_WALK_CAP_HEADROOM`, default 1.5), rounded up to 2^20 (a smaller list to the next power of two),
  never below the width validated before, so a re-prepare keeps the compiled step's shapes. That width is what the
  traced refresh builds. Named caps stay exact (#333's rule).
* **The Pallas walk counts past a full list.** A far or near overflow no longer stops it; it reports what the list
  needed (`far_needed` / `near_needed`, exact unless the queue overflowed), so the eager ladder sizes a list in ONE
  retry instead of doubling from 2^17 (eight walks at 8e6). A walk stopped by its queue counted only part of the
  pairs, so only the queue grows then. The flat walk's queue ceiling is 2^28 (the dual walk's 2^25 stopped 3.2e7).
* **Eager memory.** The walk runs in one jit (its buffers are created inside and updated in place, not held twice
  as loop arguments), the previous attempt is dropped before a retry allocates, and the post-walk list build is
  one jit.
* **Sorts.** The lists' (target, source) order is one keys-only composite sort; the M2L CSR and the
  non-deterministic near sort are one key-value sort, with no permutation, no gathers and no int64 iota.
* **Segment retry.** The traced walk's needs leave the scan as a running maximum
  (`capacity_guard.last_refresh_walk_needs`). When the capacity flag fires, `strict_run_v2` re-plans the unnamed
  caps from them, re-prepares from the segment's start, recomputes the starting force, recompiles and runs the
  segment once more (fallback reason `capacity_segment_retry`; `JACCPOT_STRICT_SEGMENT_RETRY=0` raises as
  before). The plan's version -- re-prepare from the start state and let it measure the new counts -- cannot
  work: the start state's counts are the ones that already fit.
* **Masses bug.** The compiled runner closed over `masses` while its cache key held only their shape, so a
  second call with different masses of the same shape refreshed with the first call's (9.1e-3 at N = 2e4 after two
  steps, A-vs-A 0). The masses are a runner argument now (own commit and test).
* **Mesh.** Shards prepared one after another get non-decreasing widths (each starts from the previous floor) and
  a stacked state needs one width, so `setup_fused_force` and the probe re-prepare the narrower shards against the
  merged record (`capacity_plan.list_widths`). The probe's local caps are unnamed unless `PROBE_NAMED_CAPS=1`.
* **The out-of-memory report** lists the flat-walk lane's own buffers (lists, near partials, sort scratch, queue)
  instead of the dual walk's per-node and per-leaf ones, which this lane never builds.
* **Harness** (Odisseo `bench/fused-memory-caps`): above 2e5 the flat walk's caps are no longer named; the N-max
  script runs at memory fraction 0.9, its comment says what the harness does, its failure grep names the first
  cause; leaf capacity 1.15x in steps of 1024 and cell_min_level 8 in `smallleaf_baseline.py`.

**Three latent traps the width change exposed** (each a hard error or a wrong verdict, never a silent one):
* the fused pipeline pinned the neighbour-edge width to the FIRST prepare's ("static runtime sizing neighbor-edge
  cap exceeded" on any wider list); an unnamed cap now follows the list;
* the scan checks its initial state before its own refresh records the traced caps, so after a re-plan it read the
  previous, narrower trace's record and failed a state that fit; the record is cleared when a new runner traces;
* the mesh assembly refuses mixed widths (above).

## Base against new (interleaved A/B)

Base = #357's head with the harness's caps; new = this branch with the caps unnamed. Same card and protocol as
Step 0, three processes per arm, min (median):

| N | arm | per force | per step | prepare peak | peak after the step |
| --- | --- | --- | --- | --- | --- |
| 2e5 | base | 2.65 (2.97) ms | 14.6 (17.0) ms | 0.34 GiB | 0.42 GiB |
| 2e5 | new, the same named caps | 2.98 (2.99) | 10.3 (11.5) | 0.14 | 0.43 |
| 2e5 | new | 2.94 (2.96) | 11.4 (11.6) | 0.14 | 0.26 |
| 2e6 | base | 23.2 (23.2) | 94.7 (95.4) | 5.56 | 5.67 |
| 2e6 | new | 21.8 (22.7) | 72.6 (73.1) | 0.94 | 1.82 |
| 8e6 | base | 78.9 (79.1) | 349.7 (349.8) | 21.97 | 22.50 |
| 8e6 | new | 71.6 (71.8) | 251.8 (269.2) | 3.06 | 7.14 |
| 8e6 | new, headroom 1.25 | 71.7 (72.0) | 249.9 (250.1) | 3.06 | 7.21 |

* **Lists**: identical to the base's live prefix, as sets and in row order, at 2e5 and 2e6.
* **Forces**: bitwise equal at 2e5. At 2e6 0.03 % of the rows differ, by at most 1.3e-7 relative (rel-L2
  4.9e-10), against an A-vs-A control of exactly 0: the far width changed and with it one kernel's fp32 summation
  order. Not chased.
* **2e5**: no regression on this host; the record configuration (9.64 ms per step) needs a quiet card to re-check.
* **Headroom**: 1.25 saves no peak memory at 8e6 (the step's peak is not the lists) and at most a few per cent of
  the step, inside this host's noise. 1.5 stays the default: the multi-GPU rollout has no segment retry yet, and
  per-shard counts move more than global ones.

## The new ceiling on one card

New code, unnamed caps, memory fraction 0.9 (35.5 GiB limit; the card held a 4.2 GiB foreign process), one fresh
process per rung, Plummer seed 0 (per-force and per-step min of 3 and 2):

| N | far / near (directed) | per force | per step | prepare peak | peak after the step | fp64 rel-L2 (512) |
| --- | --- | --- | --- | --- | --- | --- |
| 8e6 | 20.9M / 19.2M | 72.1 ms | 265 ms | 3.07 GiB | 7.08 GiB | 7.3e-4 (median 4.6e-4) |
| 1.2e7 | 45.2M / 45.5M | 130 | 469 | 4.78 | 11.6 | -- |
| 1.6e7 | 69.9M / 130.0M | 334 | 1030 | 8.91 | 16.9 | 1.1e-3 (median 6.2e-4) |
| 2.4e7 | 66.0M / 211.1M | 520 | 1432 | 12.96 | 24.2 | -- |
| 3.2e7 | 56.3M / 355.5M | 845 (force fits) | out of memory | 19.81 | -- | -- |

* **The one-card ceiling moved from 8e6 to 2.4e7 Plummer particles**, and the 2.4e7 step peaks at 24.2 GiB.
* **Above 1.2e7 these draws are pathological**, not the code: the unclipped sampler's outliers stretch the
  per-axis box (extent 5.2e3 at 8e6, 2.7e4 at 1.6e7, 2.6e4 at 2.4e7) and the near list grows from 2.4 edges per
  particle at 8e6 to 8.1, 8.8 and 11.1. Hence the per-force jump from 1.2e7 to 1.6e7.
* **3.2e7 fails in the compiled step, not in the prepare**: one 16.9 GiB temporary for the traced near-list build
  (an int64 composite sort over 534M slots, its inputs and outputs). Doubling the near-field chunk
  (`JACCPOT_NEARFIELD_LEAFPAIR_CSR_CHUNK=128`, time-neutral at 8e6) shrinks the force's temporaries 14.2 -> 12.4
  GiB but not the step's.
* **The force's own temporary peak at 8e6 is the far field's leaf-major evaluation**: ~20 arrays of
  [leaf capacity, 64] slots, while cell leaves hold ~20 particles on average -- 3.7x the particle count. That, and
  donating the step's carry (the plan's Step 3: the arguments and outputs are each ~1.4 GiB at 8e6), are the levers
  for going further.

## The production IC on one card -- it fits, and the lane is not accurate on it

The 25M disc+bulge IC (`disk_bulge_25m.npz`, 25,165,824 particles, the quarter orbit's softening 0.0076) through
the same configuration fits ONE card: prepare 14.0 GiB, step 28.2 GiB (arguments 5.8 + outputs 5.7 + temporaries
14.2), 516 ms per force, 1533 ms per step, far 60.1M / near 231.6M directed pairs.

**But the force is wrong near the bulge centre**: rel-L2 0.30 against fp64 on 512 targets (median 2.9e-3, max
1.32). This is not the branch: on the same IC subsampled to 2e6 the base code gives 0.45, the new code 0.45 in
every printed digit. The worst particles are bulge particles at r ~ 0.01-0.03 with forces ~3x too large, and with
the softening at 1e-4 the same 2e6 particles score 1.06e-3 (median 7.2e-4): the fused lane's far field is the
UNSOFTENED expansion, and its geometric MAC accepts cells closer than the softening length, where that is the
wrong force law (up to 2^1.5 at r = eps). Leaf 256, theta 0.6, p6 only halves it (0.27). The quarter orbit ran
the mesh lane with the mass-dependent `dehnen_error` criterion, whose error model takes the softening
(`runtime/_adaptive_policy.py`); the flat walk cannot carry it (no pair-policy seam). **The 25M production rollout cannot move to the fused lane until its MAC knows the
softening.**

## Two cards at the new ceiling

The c4 probe, p6, cards 2+1 (a NODE pair; both carried other users' jobs at up to 98 % utilisation, so these are
indicative), local caps unnamed, min of 5:

| N | one card | two cards (cross) | fp64 rel-L2 one / two |
| --- | --- | --- | --- |
| 8e6 | 523 ms | -- | 4.6e-4 / -- |
| 1.6e7 | 1718 ms | 609 ms (local arm 713) | 6.3e-4 / 5.9e-4 |
| 2.4e7 | out of memory (a 20 GiB block in the mesh body; card 2 had ~32 GB free) | local arm 1081 ms; the cross arm ran out of memory on card 1 (~24 GB free, a 15.5 GiB block) | -- |

* At 1.6e7 two cards are 2.8x faster than one, and two cards at 1.6e7 cost 1.17x one card at 8e6. Padding no
  longer decides the comparison. Treat the factor as indicative: the probe's one-card 1.6e7 arm (p6, mesh
  evaluator, card 2 under load) took 5x what `strict_run_v2`'s force took on card 3 at p5 (334 ms).
* The one-card arm at 1.2e7 (709 ms) overlapped a 64-worker CPU test run (load 184) and is not in the table.
* The probe's one-card arm runs the mesh evaluator; at 2.4e7 it needs more than `strict_run_v2` (which fit 2.4e7
  on card 3), and these two cards had a quarter to a half of their memory taken by other jobs.

## Next

* A softening-aware acceptance on the flat walk (accept only beyond a multiple of the softening), then the 25M
  disc+bulge production rollout on this code.
* The leaf-major evaluation's 3.7x slot padding; the step's carry (plan Step 3) when a real IC needs it.
* A segment retry for the multi-GPU `FusedRollout` (it raises `RolloutFlagError` today).
* The record configuration at 2e5 on a quiet card.
