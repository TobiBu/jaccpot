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
| 3.2e7 | 56.3M / 355.5M | 845 (force fits) | out of memory (see below) | 19.81 | -- | -- |

* **The one-card ceiling moved from 8e6 to 2.4e7 Plummer particles** with the benches' on-demand allocator (the
  2.4e7 step peaks at 24.2 GiB), and to 3.2e7 with a preallocated arena (next bullet, and Step 3 below).
* **Above 1.2e7 these draws are pathological**, not the code: the unclipped sampler's outliers stretch the
  per-axis box (extent 5.2e3 at 8e6, 2.7e4 at 1.6e7, 2.6e4 at 2.4e7) and the near list grows from 2.4 edges per
  particle at 8e6 to 8.1, 8.8 and 11.1. Hence the per-force jump from 1.2e7 to 1.6e7.
* **3.2e7 failed on ALLOCATOR FRAGMENTATION, not on the card** (found with Step 3, below). The step needs one
  16.9 GiB temporary for the traced near-list build (an int64 composite sort over 534M slots, its inputs and
  outputs). The benches run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the allocator grows regions on demand
  and keeps them. The eager prepare's 20 GiB peak had left regions that no contiguous 16.9 GiB fit beside. With a
  preallocated arena (jax's default; fraction 0.88 here, because the card held another process) the same rung
  fits: 845 ms per force, 2274 ms per step, step peak 33.5 GiB of 34.8. Doubling the near-field chunk
  (`JACCPOT_NEARFIELD_LEAFPAIR_CSR_CHUNK=128`, time-neutral at 8e6) shrinks the force's temporaries 14.2 -> 12.4
  GiB but not the step's.
* **The force's own temporary peak at 8e6 is the far field's leaf-major evaluation**: ~20 arrays of
  [leaf capacity, 64] slots, while cell leaves hold ~20 particles on average -- 3.7x the particle count. That is
  the lever after Step 3.

## Step 3: the step's carry, donated and slimmer (2026-10-04)

Branch `perf/fused-carry`, stacked on #358. The prepared state is the compiled scan's carry, so the runner held it
twice: as its argument and as its output.

* **Donation.** `strict_run_v2(..., donate_prepared_state=True)` hands the state's buffers to the scan, which
  writes the returned state into them. It is opt-in for a caller's state: the Odisseo coupling passes ONE
  prepared state to every warm-up and timed run, and donating it by default would delete it under them. A state
  `strict_run_v2` prepares itself (`prepared_state=None`) is always donated, since nothing else holds it.
  `strict_fused_prepared_eval_fn(..., donate_prepared=True)` gives a one-shot eval closure.
* **The far list rides outside the scan.** On the default fresh far-pair rebuild the carried far list is a
  placeholder: each refresh builds its own and returns the input's unchanged. It is detached before the scan and
  re-attached to the returned state, which the gradient path reads (3 x P int32, 1.0 GiB at 2.5e7 on the disc).
* **Two things donation exposed:**
  * the flat-walk neighbour list holds one array under two fields, and XLA refuses to donate a buffer twice, so
    repeats are copied first (`_unaliased`);
  * the engine's topology-reuse entry kept the prepare's tree, which a donated scan deletes, so the donating call
    drops it and the one-slot prepared-state cache (they only save a rebuild).
  * A test walks the engine for deleted arrays after donating calls: none.

Same card, unnamed caps, interleaved where marked:

| row | before (#358) | Step 3 | per step before / after |
| --- | --- | --- | --- |
| 8e6 (2 x 2 interleaved; "before" = Step 3 code without donation) | 6.76-6.86 GiB | 5.74-5.81 GiB | 264 / 264 ms |
| 25M disc+bulge | 28.2 GiB (args 5.8 + out 5.7 + temp 14.2) | 22.6 GiB (args 4.8, 3.8 aliased; temp 14.3) | 1533 / 1515 ms |
| 3.2e7 Plummer, preallocated arena (0.88) | 33.5 GiB | 26.4 GiB | 2274 / 2244 ms |

Results are bitwise equal with and without donation (A-vs-A controlled), and the far list comes back as the same
object. With the asynchronous allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`) 3.2e7 also fits on demand
(26.2 GiB), but the force ran 1319 against 845 ms.

**One card now holds 3.2e7 Plummer particles at 26.4 GiB, and the 25M production IC at 22.6 GiB.** What is left
in the step is its temporaries: at 3.2e7, the near-list build over 534M slots.

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

## Round 2: particle order, lane cascades, the particle carry (2026-10-04)

Plan: `yes-we-now-have-greedy-thimble` (measure the one-card ceiling, then make the lane fit more and run
faster). Branch `perf/fused-particle-major` on #359; yggdrax #83. Rows: `bench/results/fused_memory/round2/`.
Bench additions: `--ic plummer_clipped --rmax 20` (the Plummer inverse CDF truncated at 20 scale radii,
deterministic, no box-stretching outliers), `--prealloc FRAC`, `--save-forces`, `--trace-dir`,
`--no-command-buffers`; `jax.named_scope("fmm_*")` on the fused step's stages and
`bench/analyse_trace_by_stage.py` (kernel time per scope from a trace, buffer attribution from a dump).

**First finding: the centres of mass were wrong at large N (yggdrax #83).** The node moments were differences
of float32 prefix sums over all N particles, so at large N a deep node's centre of mass carried a rounding error
of order N eps |x| and landed outside the node. The COM MAC radii, and with them the near lists, grew with N
(near edges per particle 2.4 at 8e6 -> 11 at 3.2e7 on seed-0 Plummer). Accumulated in float64:

| run | near pairs | force | prepare peak |
| --- | --- | --- | --- |
| 3.2e7 clipped, before / after | 515.9M / 35.8M | 774 / 200 ms | 28.2 / 16.7 GiB |
| 25M disc+bulge, before / after | -- | 516 / 149 ms | -- |

So part of "the disc near field is physical" was this bug (memory `disc-nearfield-is-physical` now carries the
caveat).

**The changes**, measured at 8e6 clipped Plummer (leaf 64 cell leaves, theta 0.8, p5; card 3, which carried an
idle foreign process; same frozen worktree per comparison, interleaved, PREALLOCATE off):

| step | force | per step | step peak | forces |
| --- | --- | --- | --- | --- |
| #359 + COM fix (table near field, leaf-major L2P) | 53.2 ms | 342 ms | 9.34 GiB (1254 B/p) | reference |
| particle-major L2P + direct near field (now default) | 34.2 | 266 | 5.66 | bitwise (force and scan state) |
| COM radii by level passes | 30.3 | 224 | 4.60 (prepare 5.59 -> 3.69) | bitwise |
| lane cascades (now default) | 29.5 | 145 | 4.52 | round-off: rel-L2 7.8150e-4 -> 7.8154e-4 |
| `carry="particles"` | -- | -- | 3.55 (contended card; prepare 3.25) | as above |

* **Far-field L2P in particle order.** The leaf-major evaluation vmapped a `value_and_grad` over `[leaves, 64]`
  slots: ~20 residuals at 3.7-6 slots per particle on cell leaves, the force's temporary peak. Now every particle
  is evaluated in its own leaf's expansion, chunk by chunk (`JACCPOT_L2P_LAYOUT=particle`, the default).
* **Near field on the sorted particles.** The CSR kernel read `(L, 64)` tables gathered every force, emitted a
  `(64, 4)` partial for every chunk (every leaf, padding leaves included, owns one) and reduced them with a
  segment sum. `JACCPOT_NEARFIELD_LAYOUT=direct` (default) reads `pm[start + j]`, runs each leaf's first chunk
  and its self term in one program that stores straight into particle order, and scatter-adds only the rows'
  remaining chunks; the potential lane runs only on request.
* **COM MAC radii.** The (leaves x 64 levels) ancestor and distance tables, a `(L, 64, 4, 3)` broadcast and a
  scatter-max whose non-final lanes hit one sentinel row became one pass per ancestor level, bounded by the
  upward sweep's depth, reducing squared distances (one square root per leaf), jitted as one program (the eager
  prepare had run it op by op and materialised the `(L, 64, 3)` gather). Four levels per pass was tried and
  reverted: XLA did not fuse the `(L, 64, 4)` norm into the reduction (35 against 26 ms per step, +0.2 GiB).
* **Lane cascades.** The level kernels ran one program per node, loaded ~40 KB of constant tables each, and
  launched the widest level's grid at every level. One node per lane (the M2L lane kernel's design) on the
  packed table: M2M 54 -> 1.6 ms, L2L 44 -> 2.7 ms per step (`JACCPOT_CASCADE_KERNEL=lanes`, default).
* **The particle carry.** Jax's dead-code elimination of the fused scan step, with only the particle outputs
  marked used, keeps none of the carried state's 51 leaves on the default fresh far-pair rebuild.
  `strict_run_v2(carry="particles")` (or `JACCPOT_STRICT_CARRY`) carries positions, velocities and forces;
  each step materialises the state from a shape template (NaN broadcasts XLA removes) and the call returns a
  `StrictParticleCarry` handle that also spares the next call its eager initial force evaluation. Default
  stays `"state"`.
* **Smaller items:**
  * the walk's pair queue is double-buffered (a 33 MB copy per round), and its round kernel strides over a
    capped grid;
  * the deterministic near CSR comes from one sort of the canonical pairs;
  * yggdrax: one key-value Morton sort, and the cell partition without sentinel atomics and without a scan
    pair per level (bitwise; numpy reference kept).

**The one-card ceiling.** Preallocated arena 0.88 of a 40 GB A100 (35.2 GiB). Base = #359 + the COM fix,
state carry, card 3. Round 2 = head (43bef3d) with `JACCPOT_STRICT_CARRY=particles`, card 5 (empty). The cards
differ, so the per-step times compare only roughly; the 8e6 A/B above is the clean timing.

| run | base: step peak, per step | round 2: step peak (B/p), force, per step | rel-L2 |
| --- | --- | --- | --- |
| clip8 | 9.40 GiB, 306 ms | 3.51 GiB (471), 29.8 ms, 111 ms | 7.82e-4 |
| clip32 | 29.91 GiB, 1077 ms | 12.13 GiB (407), 121 ms, 400 ms | 7.10e-4 |
| clip48 | out of memory (eval) | 17.63 GiB (394), 180 ms, 601 ms | -- |
| clip64 | out of memory (eval) | 23.44 GiB (393), 237 ms, 815 ms | 7.03e-4 |
| clip80 | -- | 29.06 GiB (390), 292 ms, 1020 ms | -- |
| clip88 | -- | prepare 28.89 GiB and force (320 ms) fit; the scan's 4.0 GiB block did not, after the bench's second eager prepare | -- |
| p32 (seed 0) | 22.43 GiB, 923 ms | 8.89 GiB (298), 144 ms, 366 ms | -- |
| p48 | out of memory (step) | 13.70 GiB (306), 334 ms, 713 ms | -- |
| p64 | -- | 20.44 GiB (343), 316 ms, 845 ms | -- |
| p80 | -- | prepare 29.29 GiB (393 B/p: the draw's outliers stretch the box, 591M far pairs against 219M at p64); the force ran out of memory | -- |
| p96 | -- | prepare 28.05 GiB, force 541 ms; the scan ran out of memory as at clip88 | -- |
| p112 | -- | prepare 33.14 GiB; the force ran out of memory | -- |
| 25M disc+bulge | 20.68 GiB, 779 ms | 8.91 GiB (380), 96 ms, 303 ms | 0.34 (softening, see above) |

* **A production rollout's sequence** (`--skip-eval`: one prepare, then the scan; the ladder above prepares twice,
  once for the force-only timing and once inside `strict_run_v2`):

  | run | allocator | peak (B/p) | per step |
  | --- | --- | --- | --- |
  | clip88 | preallocated arena 0.88 | 31.76 GiB (388) | 1126 ms |
  | clip96 | preallocated arena 0.88 | the scan's 4.4 GiB block does not fit after the prepare | -- |
  | clip96 | `cuda_async` (on demand) | 34.20 GiB (383) | 1580 ms |
  | clip104 | `cuda_async` | **36.87 GiB (381)** | 1730 ms |
  | p80 (seed 0) | preallocated arena 0.88 | prepare 31.97 GiB; the scan's 3.3 GiB block does not fit | -- |

  **One 40 GB A100 now holds 1.04e8 particles** (the base: 3.2e7). The arena's limit is fragmentation, not
  size: right before the scan only the particles are live (checked at 2e4: 1.1 MB; at 8.8e7: 6.75 GiB after the
  call), yet a 4.4 GiB block will not fit after a 30 GiB prepare. The asynchronous allocator fits it and costs
  ~25 % of the step (1580 ms at 9.6e7 against 1126 at 8.8e7 in the arena, per particle), as in Step 3.
* **What binds now.** With the particle carry, the scan itself is no longer the peak: after it only the
  particles stay resident (0.8 GiB at 8e6). The binding peak is the eager prepare: the downward pass's
  transients at 8e6, ~360 B per particle at large N.
* **Clipped vs seed-0 Plummer.** The clipped draw has twice the leaves per particle (cell_min_level 8 is relative
  to its smaller box: 10.5 against 19.7 particles per leaf). On it, cell_min_level 6 is 17 % faster and 18 %
  lighter at the same rel-L2 (91 against 110 ms, 396 against 481 B/p at 8e6). On seed 0 it changes nothing. The
  default stays 8: it is what keeps sparse outskirts from forming huge leaves (outlier draws, the multi-GPU
  cross export).
* **Against jz-fmm on the same IC and card class.** jz-fmm, leaf 32, p4, theta 0.6, aggL2 3-8e-4, arena 0.88,
  card 6: 241-251 B per particle flat from 8e6 to 1e8 (23.4 GiB at 1e8); 104 ms at 8e6, 398 at 3.2e7, 1278 at
  1e8, tree build included. Per particle, our full step (refresh, force, kick) now costs what its force
  evaluation costs, at ~1.6x its memory.

**Where a step goes now** (8e6 clipped, stage trace without command buffers, lane cascades): walk 27 ms,
COM radii 26, near field 23.5, tree 19, M2L 15, P2M 10, the walk queue's copies 10 (since removed), each call's
eager initial force ~11 per step at 2 steps per call (the particle carry's handle removes it). The cascades,
40 % of the step before, are 3 %.

**jz-fmm on the same IC and card** (`codes/jzfmm_force_eval.py --memory`, Odisseo; card 2 under other users'
load, so its times are indicative): 8e6 clipped at leaf 32 peaks at 213-244 B per particle (p3-p5); p5 theta
0.8 gives aggL2 4.4e-4.

## Round 3: the step's liveness, lists without sorts (2026-10-04)

Branch `perf/fused-round3` on #360. Rows: `bench/results/fused_memory/round3/`. Configuration as in round 2, with
`JACCPOT_STRICT_CARRY=particles` throughout. Card 3 of the A100 node carried its idle 4.2 GiB foreign process.

**New tool: `bench/analyse_step_liveness.py`.** A buffer assignment gives offsets, not live ranges. This tool reads
the scheduled HLO next to it and reports three things:
- what is live at the compiled step's peak, per `fmm_*` scope;
- the live maximum while each stage runs (the next windows);
- what is live at any given instant.

Its peak matched the assignment's temporary block to 1 % (1.560 against 1.570 GiB at 8e6). Every window below was
found with it.

**Where the step's peak was** (8e6, the runner at 09dff2c, 1.57 GiB temporary block):
- **Far CSR build: 1.56 GiB.** The placement held both sorted copies of the canonical pairs, the materialised slot
  arrays and the two 2W outputs (sources and targets): 40 B per canonical slot.
- **M2L window.** The presorted CSR read made a `where(valid, src, 0)` copy of the list, kept through the L2L.
- **Upward window.** The leaf P2M's `(L, 64)` row array sat next to the `(nodes, 36)` table it was concatenated into.

**Changes.** 8e6 clipped Plummer; each pair is the same frozen worktrees, interleaved. Forces and the scan state
are bitwise equal against the previous arm.

| commit | change | 8e6 | large N (production sequence, arena 0.88) |
| --- | --- | --- | --- |
| 7381938 | COM radii by a Pallas kernel: each leaf's particles read once per 8 ancestors | step 113.7 / 111.6 -> 106.5 / 108.0 ms | -- |
| c2298a0 | far list emitted in CSR order; the M2L reads it without its sort | step peak 3.70 -> 2.91 GiB, prepare 3.38 -> 2.76 | 6.4e7: 23.71 -> 19.47 GiB (398 -> 327 B/p) |
| 26a9de8 | box geometry deferred on the COM lane (prepare and refresh) | neutral | 6.4e7: tree+upward 16.5 -> 10.9 GiB, 19.08 GiB (320 B/p); 9.6e7 fits the arena: 27.92 GiB (312 B/p), 1326 ms |
| 09dff2c | far tags zero-length (were 2W int32 of -1) | -- | 1.12e8: 32.94 GiB (316 B/p), 1527 ms; 1.2e8: 31.47 GiB (275), 1617 ms; 1.28e8: the scan OOMs |
| 2911274 | far list carries row offsets; the presorted M2L reads it without a copy | step 2.82 -> 2.59 GiB; runner block 1.570 -> 1.375 GiB | -- |
| 8da2071 | leaf P2M writes its rows of the table in place | step 2.59 -> 2.53 GiB | 1.28e8 fits: 30.41 GiB (255 B/p), 1728 ms |
| d35526a | directed CSR lists without a sort (Pallas placement + in-row ranks) | step 2.59 -> 2.31 GiB (348 -> 310 B/p), prepare 2.17 -> 1.88; 103.9 / 104.4 ms; runner block 1.375 -> 1.229 GiB | 1.28e8: 29.81 GiB (250 B/p), 1763 ms |
| 6690a15 | the particle-carry runner donates the initial acceleration it built and returns no total acceleration | step 2.31 -> 2.26 GiB (303 B/p), 104.1 ms | 1.36e8 still fails in the arena (below) |

Commit details:
- **Row offsets (2911274).**
  - `TargetSortedFarPairs.targets` holds the CSR row offsets; `far_pair_targets()` expands them for every reader
    that wants one target per entry.
  - Every M2L route other than lanes expands them first. Checked: the `pair` route on the new code is bitwise
    equal to the old code's `pair` route.
  - The lanes backward pass expands them before its by-source sort.
- **CSR without a sort (d35526a).** Three passes:
  1. row counts by an integer scatter-add, and their prefix sum;
  2. placement from per-row atomic cursors;
  3. in-row ranks, tile against tile.

  A row's entries are distinct, so the ranks undo whatever order the atomics left: bitwise equal to the sorted build
  (a GPU test at 3e5 pairs with a 1000-entry row). Live set: 16 B per canonical slot against the sorts' 24 at their
  placement. Time is neutral: the 8e6 far list builds in 10.3-10.7 ms against 10.6-11.3. The cursors are float32:
  on jax 0.10.2 the int32 vector form of `plgpu.atomic_add` neither updates memory nor returns old values (probed);
  float32 does both.

**Negative.** 16-lane near-field sub-tiles: 111 -> 124-151 ms per step.

**The step's windows now** (8e6, d35526a): upward 1.00 GiB (the P2M's padded position and mass copies next to
the table and the far list), walk 0.95, near 0.94, lists 0.94. The step is balanced; each further cut is ~5 %.

**What binds the ceiling now.** At 1.28e8 (d35526a) the peak is the arrays in use before the scan (~9 GiB: the
particles, masses, accelerations) plus the scan's 15 GiB temporary block plus 5.7 GiB of outputs: the measured
29.81 GiB exactly. The outputs (state, acceleration, self-gravity: 48 B/p) were fresh buffers beside the
arguments. 6690a15 drops the unused total acceleration and donates the initial one. That is -24 B/p at the scan,
-0.05 GiB at 8e6.

From 1.36e8 the first scan call cannot place a 6.5-7.2 GiB allocation: the eager prepare left the arena with two
free regions, neither large enough. (XLA's rematerialisation log is no guide here: its estimate at 1.36e8 did not
move with 6690a15.)

**The ceiling now** (production sequence, `JACCPOT_STRICT_CARRY=particles`, 6690a15):

| run | allocator | peak (B/p) | per step |
| --- | --- | --- | --- |
| 1.28e8 clipped | preallocated arena 0.88 | 29.81 GiB (250) | 1763 ms |
| 1.36e8 - 1.52e8 | preallocated arena 0.88 | the first scan call cannot place a 6.5-7.2 GiB allocation (two free regions, neither large enough) | -- |
| 1.36e8 | `cuda_async` | **31.59 GiB (249)** | 2199 ms |
| 1.52e8 / 1.68e8 | `cuda_async` | the scan's temporary block itself (22.6 / 24.7 GiB) does not fit beside the particles | -- |

**One 40 GB A100 now holds 1.28e8 particles in the arena, 1.36e8 with the asynchronous allocator** (round 2:
8.8e7 / 1.04e8). The asynchronous allocator costs ~17 % per particle (2199 ms at 1.36e8 against 1763 at 1.28e8).
Beyond that the step's own temporary block (~150 B per particle at large N) binds.

## Round 4: optional state donation, the start force, whole near rows (2026-10-04)

Branch `perf/fused-round4` on #362. Rows: `bench/results/fused_memory/round4/`. Configuration as in rounds 2-3
(`JACCPOT_STRICT_CARRY=particles`); card 3 with its idle foreign process.

**What the step held** (liveness of the 1.28e8 runner at the end of round 3, 15.0 GiB block, 13.3 GiB live peak):

| window | what | size at 1.28e8 |
| --- | --- | --- |
| every step-long | the OLD positions, kept because the final velocity-Verlet update re-drifted them | 1.43 GiB |
| upward | the leaf P2M's padded copies of positions and masses | 1.95 GiB |
| near | the extra-chunk partials `f32[E/64, 64, 3]` sized for the list's capacity | 2.58 GiB |
| list build | the CSR placement's shifted and padded copies of the walk's pairs | ~1.4 GiB |
| outside the block | the runner's output state, beside the input state | 24 B/p |

**Changes**

| commit | change | effect |
| --- | --- | --- |
| bec501f + 59c03b3 | `strict_run_v2(carry="particles", donate_state=False)`: OPTIONAL donation of the input state (off by default, the input is kept); with it the call keeps a host copy of the start state for a capacity retry | -24 B/p at the scan when on; 8e6 in-use after the scan 0.655 -> 0.494 GiB; bitwise |
| 30dcb69 | P2M gathers instead of padded copies; the CSR placement reads the walk's arrays directly | bitwise |
| b5eab21 | the step kicks the drifted state (`_velocity_verlet_kick_drifted`) | -12 B/p under every window; bitwise |
| f91ee92 | the direct near field runs WHOLE ROWS (`JACCPOT_NEARFIELD_DIRECT_ROWS`, default `whole`): no extra-chunk partials, no scatter | 8e6: rel-L2 vs fp64 7.70271221e-4 vs 7.70271223e-4 (chunked); 0.18 % of particles differ by <= 3e-7; step 101.6-101.9 vs 102.7-103.1 ms |
| 836b4b1 + ec1f50b | the first particle-carry call evaluates its start force as the steps' own refresh in a small program of its own, after the prepared state is freed (no eager force beside the prepare) | bitwise (2e5, 2e6, 8e6); 8e6 program blocks: start 1.12, one-step runner 1.21, two-step 1.19 GiB |
| d73a970 | each of the walk's carried buffers gets its own negative fill (identical fills were one broadcast, copied into every loop-carry slot) | bitwise; 8e6 step peak 2.26 -> 2.19 GiB |

**Tried and reverted**
- Freeze-and-resume retry (where(ok, new, old) on the carry): kept the step's old state and self-gravity alive
  through every step, +36 B/p; the runner's block went 15.0 -> 19.8 GiB at 1.28e8. The 8e6 process peak hid it
  (the eager prepare sets it there): measure the runner's block, not the process peak, at small N.
- Relocating the self-gravity through the host after the prepare: did not fix 1.36e8 (the eager force's block was
  the failing allocation, not the self-gravity's place).
- The start force inside the scan's program: a one-step scan is inlined and XLA overlapped the start refresh with
  the step, doubling the block (8e6: 1.19 -> 2.52 GiB).
- `--xla_disable_hlo_passes=while-loop-invariant-code-motion`: no change (the overlap was the inlining).

**At 1.28e8** (`--donate-state`, d73a970): process peak 26.19 GiB (220 B/p; round 3: 29.81, 250 B/p). It is now the
EAGER PREPARE's: the runner's temporary block is 14.9 GiB (round 3: 15.0; 19.8 with the reverted freeze), its live
peak 12.2 GiB (round 3: 13.3), and at 1.52e8 the prepare peaks in its list build (21.3 GiB in use after the walk +
9.3 GiB transient).

**The ceiling now** (production sequence, arena 0.88, `--donate-state`, d73a970; card 3):

| run | peak (B/p) | per step |
| --- | --- | --- |
| 1.36e8 | 27.72 GiB (219) | 2156 ms |
| 1.44e8 | 29.11 GiB (217) | 2321 ms |
| 1.52e8 | 30.62 GiB (216) | 2497 ms |
| 1.60e8 | 32.20 GiB (216) | 2629 ms |
| 1.68e8 | 33.45 GiB (214), one recovered allocator retry in the prepare | 2704 ms |
| 1.76e8 | the eager prepare runs out (a 5.57 GiB block in its tree stage) | -- |
| 1.36e8 without `--donate-state` | 31.40 GiB (248), the scan's peak | 1906 ms |

**One 40 GB A100 now holds 1.68e8 particles in the preallocated arena** (round 3: 1.28e8; round 2: 8.8e7), at 214
B per particle. The per-step times with `--donate-state` include the host copy of the start state that every call
makes (24 B/p, ~0.25 s per 2-step bench call at 1.36e8: 2156 against 1906 ms); a production call of many steps
pays it once.

**What binds now: the eager prepare** (~215 B/p, its list build at 1.52e8). With the start force in its own
program, the particle carry needs only the walk's caps, the shape template and the initial verdict from the eager
prepare, so a prepare that stops after the walk (deriving the template's shapes rather than materialising the
lists and the downward pass) is the next lever.

## Round 5: jax 0.11.2, order 6 by default, three kernels (2026-10-05)

Branch `perf/fused-round5`, on the jax 0.11.2 PR (#365) on #363. Rows: `bench/results/fused_memory/round5/`
(its directories below). One A100 40 GB (card 6, held by a guard process between runs), clipped Plummer unless
named, `JACCPOT_STRICT_CARRY=particles`, unnamed caps, preallocated arena 0.88.

**Where it started: jz-fmm, apples to apples** (`compare_jzfmm/`). Same IC draw, the same 4096 targets and the
same fp64 direct-sum reference (cached under the harness's `artifacts/reference/`), interleaved on one card.
jz-fmm's time is one `force()` including its tree build; ours is one full `strict_run_v2` step (rebuild + force +
kick). Our force on an already prepared state is only ~30 % of a step, so it is never the comparison.

| N | jz-fmm best (p5 theta 0.8) | ours best then (p6 theta 0.8 cell_min_level 6, round-4 kernels) | ratio |
| --- | --- | --- | --- |
| 8e6 | 72.3 ms at 3.8e-4 | 88.4 ms at 4.3e-4 | 1.22x |
| 3.2e7 | 284.3 ms at 4.0e-4 | 353.2 ms at 3.6e-4 | 1.24x |
| 1e8 | 921.5 ms at 4.8e-4 | 1224.2 ms at 3.4e-4 | 1.33x |

Memory was level (1e8: 243 against 272 B/p). The ~1000 ms at 1e8 often quoted for jz-fmm is its p5/theta 0.8
point (921.5 ms here). Theta 0.7 bought accuracy at +30 % of the step; order 6 at +5 %.

**Where a step went** (1e8, p5 theta 0.8 cell_min_level 8, trace without command buffers, kernel ms per step):
near 287, walk 292, lists 146 (of which the far CSR placement 85, superlinear: 3.5 ms at 8e6 for 10x fewer
pairs), COM radii 112, upward 96 (P2M 78), M2L 92, tree 92. COM radii and P2M ran one program per leaf over the
64-lane capacity for ~10-20 particles.

**Defaults** (`defaults/`, jax 0.11.2, the frozen round-4 tree). Step ms / fp64 rel-L2 / longest near row:

| case | p5 cml8 (old) | p6 cml8 | p5 cml6 | p6 cml6 |
| --- | --- | --- | --- | --- |
| clipped 2e5 | 14.2 / 1.33e-3 / 85 | 9.2 / 7.8e-4 / 85 | 10.0 / 1.30e-3 / 85 | 13.3 / 7.5e-4 / 85 |
| clipped 2e6 | 32.4 / 9.2e-4 / 101 | 34.0 / 5.0e-4 / 101 | 29.2 / 9.2e-4 / 101 | 28.9 / 5.0e-4 / 101 |
| unclipped 2e6 | 44.0 / 1.21e-3 / 634 | 40.1 / 8.3e-4 / 634 | **16627** / - / **101870** | **17797** / - / 101870 |
| unclipped 8e6 | 96.5 / 8.3e-4 / 262 | 98.1 / 4.9e-4 / 262 | ended / - / 1342 | ended / - / 1342 |
| disc 2e6 | 29.7 / 0.58 / 144 | 28.5 / 0.58 / 144 | 28.6 / 0.58 / 145 | 34.2 / 0.58 / 145 |
| disc 8e6 | 91.3 / 0.36 / 107 | 98.8 / 0.36 / 107 | 93.6 / 0.36 / 107 | 91.6 / 0.36 / 107 |

* **Order 6 is the new default** (bench `--order 6`): ~1.7x lower error for +2-8 % of the step at 8e6 (the disc's
  error is the softening's, the known follow-up). Memory +4-7 % at 8e6-1e8.
* **cell_min_level 6 broke the near field on outliers**: an outskirt cell of the unclipped draw stays one leaf near
  101,870 leaves, and round 4's whole-row near field ran that row in ONE program -- 16.6 s per step. It helps
  the clipped draw only (8e6 101 -> 83 ms, 1e8 1252 -> 1155 ms), so level 8 stays the default.
* **Two long-row fixes.** The force at level 6 on the unclipped draw went 240.8 -> 34.2 ms with the first; the
  step needed the second (a trace put 19.87 s of its 19.93 s in one kernel):
  * near field (`851e1fa`): a row's own program runs it up to 1024 entries, the rest goes in pieces of 1024
    through the chunked path's partials, under a `lax.cond` that skips them when no row is that long;
  * CSR rank (`6d97fe0`, `bdf0674`): the rank pass ordered each row tile against tile, `(n / 32)^2` pairs, in ONE
    program; rows past 2048 entries now spread their tiles over 256 programs, 64 rows a launch, under a
    `lax.cond`. Ranks are exact, so the lists are the same to the bit.
  Every row of the standard configurations is far below both limits (85-634), so those keep their bits.

**cell_min_level 8 against 6 with all of round 5** (`cml/`, p6; step ms, the same rel-L2 at both levels):

| case | level 8 | level 6 |
| --- | --- | --- |
| clipped 2e5 | 10.8 | 9.7 (-10 %), 458 vs 681 B/p |
| clipped 2e6 | 31.0 | 25.2 (-19 %), 289 vs 395 B/p |
| clipped 8e6 | 95.8 | 80.6 (-16 %), 261 vs 318 B/p |
| clipped 1e8 | 1169.0 | 1094.0 (-6 %), 255 vs 270 B/p |
| disc 2e6 / 8e6 | 31.1 / 82.7 | 27.4 / 80.5 |
| unclipped 2e6 | 38.6 | **135.6** (was 17,445 before the two fixes) |
| unclipped 8e6 | 88.4 | 92.1 (+4 %) |

Level 6 pays on bounded distributions and costs 3.5x on the unclipped draw at 2e6, where one outskirt leaf is
still near ~all leaves (36 % more near pairs, a 10^10-compare rank, now in parallel): level 8 stays the bench
default, level 6 is the option for bounded ICs. The LIBRARY default (`TreeConfig.cell_min_level=None`) is
unconstrained, which is worse than either on outliers.

**Kernels** (`tunes/` alone on the 8e6 and 1e8 cell trees; `kernel_ab/` in the full step):

| change | alone, 8e6 | alone, 1e8 | result |
| --- | --- | --- | --- |
| COM radii: ancestors gathered first (pointer jumping in XLA), particles 8 lanes at a time, 16 leaves a program (`540b0d6`) | 11.4 -> 4.8 ms | 78 -> 36.5 ms | bitwise |
| P2M: 16 leaves a program as rows of a (16, 8) lane tile, chunks up to the block's largest leaf (`2e618db`) | p5 10.4 -> 1.6, p6 9.0 -> 2.4 ms | p5 66 -> 17, p6 56 -> 27 ms | fp32 order (1.1e-7) |
| CSR placement in passes over row ranges, one per 4M rows (`8978dcd`, `4d15ff8`) | 1 pass (unchanged) | 1180 -> 1142 ms step (4 passes) | bitwise |

In the step (p6, cell_min_level 8, jax 0.11.2, interleaved, frozen trees; old = `JACCPOT_P2M_BLOCK=0
JACCPOT_COM_RADII_VARIANT=chain`):

| N | old kernels | new kernels | + 4 placement passes | rel-L2 old / new | peak |
| --- | --- | --- | --- | --- | --- |
| 8e6 | 109.8, 108.1 ms | 94.5, 95.2 ms (-13 %) | (one pass) | 4.333e-4 / 4.333e-4 | 2.37 / 2.38 GiB |
| 1e8 | 1291.7, 1295.1 ms | 1181.1, 1179.9 ms (-9 %) | 1141.7 ms (-12 %) | 3.364e-4 / 3.364e-4 | 25.19 / 25.19 GiB |

At 1e8 that is 1142 ms at 3.4e-4 against jz-fmm's 921.5 ms at 4.8e-4 (1.24x, more accurate).

**Where a step goes now** (p6, cell_min_level 8, new kernels, one placement pass; `kernel_ab/trace_*`), kernel ms
per step at 8e6 / 1e8: near 22.6 / 286, walk 18.2 / 287, M2L 14.4 / 137 (p6: 92 at p5), lists 8.5 / 143
(placement 3.5 / 82), tree 6.6 / 94, unscoped 3.9 / 57 (one gather fusion, 40 ms at 1e8), COM radii 5.4 / 51
(kernel 2.5 / 24), upward 4.3 / 37 (P2M 1.1 / 11.6, M2M 2.3 / 17.6), L2L 2.9 / 23, L2P 2.5 / 29. The walk and the
near field are half the step.

**jax 0.11.2** (#365; yggdrax#86): the 0.11.0 CPU regression is gone (characterization suite 218 s against 246-250
s on 0.10.2); the fused step is 0-3 % faster on the same code (8e6 105.8 -> 101-104 ms; 1e8 p6 cml6 1224-1230 ->
1197-1199 ms) but its peak is +5 % at 1e8 (22.61 -> 23.75 GiB, the same code), so the one-card ceiling has to be
re-measured. Two XLA:CPU stalls had to be fixed on the way, both LLVM's loop vectorizer in a deep recursion
(`llvm::vputils::onlyFirstLaneUsed`): the unrolled `searchsorted` (kept off the CPU now) and every Pallas kernel in
interpret mode (the test conftests turn the vectorizer off). And 0.11.2 deprecates the Pallas Triton backend.

**jax 0.11.2's allocator once crashed a preallocated run** (1 of ~120 on 0.11.2, `rank_scan_ab/f_s1`): `Check
failed: central_gap_ == kInvalidChunkHandle ... spatial partitioning expects one central gap`, in
`BFCAllocator::InsertFreeChunk` while the eager prepare freed a buffer. The BFC allocator's spatial partitioning is
on by default with preallocation; `--xla_gpu_enable_allocator_spatial_partitioning=false` costs nothing measured
(1e8 step 1147.9 / 1146.0 ms on, 1147.1 / 1148.3 off, peak 25.185 GiB both; `allocator_ab/`), so the bench turns it
off when it preallocates, and production runs with a preallocated arena should set it too.

**The one-card ceiling on jax 0.11.2** (`ceiling/`, `--donate-state`, arena 0.88): round 4's configuration (p5,
round-4 code) still fits 1.68e8 (33.68 GiB, 215 B/p, 2960 ms/step, one recovered allocator retry; 33.45 GiB on
0.10.2). The new default, p6 with round 5, fits 1.60e8 (33.34 GiB, 224 B/p, 2667 ms/step, one recovered retry;
1.44e8 30.23 GiB at 2561 ms, 1.52e8 31.76 GiB at 2551 ms) and runs out at 1.68e8, in the eager prepare, where the
p6 coefficient tables (2 x nodes x 49) are 8.7 GiB: order 6 costs ~5 % of the ceiling. Runs at the limit can pass
order 5.

## Round 6: the walk's atomics, a tiled near field, a tree-order carry (2026-10-06)

Branch `perf/fused-round6`, on main `a7da8a6` (#366). Rows: `bench/results/fused_memory/round6/` (README there).
One A100 40 GB (card 0, held by the guard between runs), jax 0.11.2, clipped Plummer, p6, theta 0.8,
cell_min_level 8, `JACCPOT_STRICT_CARRY=particles`, unnamed caps, arena 0.88, frozen worktrees per arm.

**Step 0: the gap at EQUAL accuracy** (`compare_jzfmm/`, interleaved with jz-fmm on the same card, targets and fp64
reference; jz-fmm's time is a force including its tree, ours a full step). Round 5's 1.24x at 1e8 compared our
3.4e-4 with jz-fmm's 4.8e-4. Measured at matching error:

| N | jz-fmm (ms at rel-L2) | ours, main (p6) | ours, round 6 |
| --- | --- | --- | --- |
| 8e6 | p5 theta 0.8: 72.8 at 3.81e-4; p6 theta 0.8: 102.5 at 1.78e-4 | 93.5 at 4.33e-4 | 81.1 / 80.6 at 4.33e-4 (79.2 / 79.1 with the tree-order carry) |
| 1e8 | p5 theta 0.8: 921.5 at 4.76e-4; p5 theta 0.7: 1183.3 at 3.65e-4; p6 theta 0.8: 1258.6 at 2.27e-4; p6 theta 0.7: 1621.1 at 1.37e-4 | 1155.7 at 3.36e-4 | 909.3 at 3.36e-4 (868.0 with the tree-order carry) |

At 1e8 main was already level with jz-fmm's measured points at our accuracy (1156 against 1183 ms at 3.65e-4, our
p90 4.9e-4 against its 7.1e-4); only jz-fmm's front interpolated between p5 and p6 (~1080 ms at 3.4e-4) was ahead.
The real gap was at moderate N: 1.3x at 8e6 (jz-fmm ~71 ms at our 4.3e-4). jz-fmm's error grows with N at a fixed
setting (p5 theta 0.8: 3.8e-4 at 8e6, 4.8e-4 at 1e8); ours shrinks (4.3e-4, 3.4e-4).

**The walk: atomics, not memory.** The round kernel claimed every block's slots with nine atomics on one six-word
counter array (far, near, four child pairs, three overflow flags), and read each node from five arrays. Two
options, the same pair sets (counts, rounds, peak and order-free checksums of both lists equal; the unit tests
compare the sets with the flat walk):

* `fused_emit`: one claim for the four child pairs, the overflow flags only when the block overflowed: three
  atomics per block instead of nine;
* `node_layout="record"`: one 32-byte record per node `(cx, cy, cz, r, left, right, active, 0)`, built once per
  walk (12 B/node more than the padded centres it replaces).

Alone on the captured 1e8 tree (`tunes/w1e8.txt`, `w2_1e8.txt`, three interleaved rounds): 286.5 / 288.7 ms ->
fused emit 166.5, record 247.8, both **96.0-96.4 ms (-66 %)**; at 8e6 the old walk is bimodal (21.5-33.9 ms), both
options 14.6-15.3. In the step: 1155.1 / 1155.5 -> 954.2 / 953.0 ms at 1e8 and 93.5 / 93.7 -> 83.8 / 83.5 at 8e6,
the force bitwise unchanged and the peak slightly lower (25.19 -> 24.96 GiB at 1e8). **Both are the default now**
(`JACCPOT_WALK_NODE_LAYOUT=soa`, `JACCPOT_WALK_FUSED_EMIT=0` restore the old walk).

**The near field is FP32-bound on its pair count, not latency-bound.** The plan's estimate (2.6e10 pairs at 1e8,
30-50 ms of arithmetic against 286 measured) undercounted: the captured lists hold **6.25e10** particle pairs at 1e8
(4.8e9 at 8e6; cell leaves of 14.9 / 10.5 particles, median 12 / 7, rows of 17.7 / 12.3 leaves). The scalar kernel
runs 32-lane target tiles 43 % / 31 % full, and its 1.45e11 lane-evaluations in 0.29 s are ~5e11 per second -- the
A100's FP32 issue rate at ~19 instructions per pair. Variants behind `source_tile` / `source_flags` /
`target_classes` (`tunes/t*.txt`, alone on the captured 8e6 and 1e8 lists):

| variant | 8e6 | 1e8 |
| --- | --- | --- |
| scalar loop, 32-lane targets (old) | 23.0 ms | 289-295 ms |
| (16, 16) source tiles, sum per tile | 38.5 | 493 |
| + 2D accumulation, one sum at the end (`a`) | 26.7 | 324 |
| (16, 8) tiles, `a` | 21.0 | 261 |
| + lean pair body (`l`: mask and -G folded into the masses, one select per pair) | 20.3 | 253 |
| + runs of touching leaves merged (`r`) -- **the new default** | **19.7** | **245.6** |
| one launch per occupancy class (target tiles 4-64 wide) | 20.7-27 | 253-341 |
| operands loaded through 2D indices (`g`, no layout conversion) | 20.5 | 251 |

So -14 % / -16 % alone: the gate (-40 %) is missed. Smaller target tiles fill more lanes but pay the same per-pair
instructions plus per-tile overhead, and a 2D load changed nothing (Triton had already avoided the conversion). The
sums are tree sums now: the fp64 error of the near sums themselves 3.4e-7 -> 1.5e-7 (8e6) and 1.1e-7 -> 7.6e-8
(1e8); the force's rel-L2 is unchanged at four digits on the clipped draw (4.333e-4, 3.364e-4) and the disc
(0.3574), 8.286e-4 -> 8.285e-4 on the unclipped 2e6 draw. In the step: -45 ms at 1e8 (913 -> 868 with the tree-order
carry), -3.7 ms at 8e6. **The default now** (`JACCPOT_NEARFIELD_SOURCE_TILE=0` restores the scalar loop). The lever
left is the pair count itself (leaf size and the MAC), or symmetric pairs, which need a scatter of the reactions.

**The tree-order carry** (`JACCPOT_STRICT_CARRY_ORDER=tree`, particles carry only, opt-in). The scan carries the rows in
the previous step's Morton order with their masses and input indices; the force is never gathered back (the
evaluation's new `sorted_output`), the inverse permutation is never built, the tree's gathers read an almost sorted
array, and the state goes back to input order once per call. **Bitwise** against the input-order carry over 9 steps
at 2e6 and 8e6 (0 rows differing; A-vs-A control bitwise too; `carry_bitwise/`); ties of the 63-bit Morton code keep
the carry's order instead of the input's, which would differ (a few pairs per step in the core at 1e8, none at these
sizes). 1e8: 954 -> 913 ms (-41 ms) on the walk defaults, 909 -> 868 ms on all of round 6; 8e6: -1.3 ms. **But +30 B
per particle at 1e8** (24.96 -> 27.81 GiB): the kick can no longer update the state in place (24 B/p) and the carry
holds the masses and indices. Neither an optimization barrier after the force (28.55 GiB, +6 ms) nor reusing the
tree's own sorted positions and masses (no change) recovered it, so it stays opt-in; it also needs an external field
that acts row by row and is not used with `return_history` or a step callback.

**The step, interleaved A/B** (`step_ab/`, frozen `k3`, env toggles; B = main's paths):

| N | B | + walk | + tree-order carry | + near field |
| --- | --- | --- | --- | --- |
| 8e6 | 93.5 / 93.7 ms | 83.8 / 83.5 | 82.5 / 82.5 | **78.8 / 78.8** |
| 1e8 | 1155.1 / 1155.5 ms | 954.2 / 953.0 | 913.0 / 914.0 | **867.7 / 868.2** |
| peak at 1e8 | 25.19 GiB | 24.96 | 27.81 | 27.81 |

The defaults as merged (frozen `k6`, `final/`; main = `base`): 8e6 81.1 / 80.6 ms (79.2 / 79.1 with the
tree-order carry), 1e8 909.3 ms at 24.96 GiB (868.0 at 27.81), 2e6 30.0 -> 26.5 ms (23.9), 2e5 10.0 -> 10.6 ms
(10.7; medians 11.9 / 10.6 / 13.0: launch-bound and noisy at this size); the force alone at 1e8 382.6 -> 338.3 ms,
rel-L2 3.364e-4.

**Other distributions** (`ics/`, main against the new defaults, step ms): unclipped Plummer 2e6 at cell_min_level 6
(the long-row case) 136.7 -> 126.6 (round 5: 135.6; the gate held), at level 8 39.1 -> 32.7, at 8e6 88.1 -> 73.6;
the disc at 8e6 81.8 -> 70.0 (force 35.6 -> 26.8 ms, rel-L2 unchanged).

**Where a step goes now** (round-6 defaults + tree-order carry, trace without command buffers, kernel ms per step,
8e6 / 1e8; `ics/stages_*.txt`): near 19.5 / 244, M2L 14.4 / 139, tree 7.9 / 105, lists 8.2 / 101, walk 8.5 / 92,
COM radii 5.3 / 51, upward 4.3 / 38, L2P 2.7 / 31, L2L 2.9 / 23, the step's copies and integrator ~6 / 68. The near
field and the M2L (~40 % at 1e8) now issue FP32 at the card's rate; what is left above a few percent is data
movement: the tree build (cub sort 18 ms, scatters ~20 at 1e8), the list placement (four passes) and rank (~70 of the 101 ms), the COM radii.

## Round 7: the force's way back to input order, a Cartesian L2P, the tree's level tables (2026-10-06)

Branch `perf/fused-round7`, on main `6cca378` (#367); yggdrax `perf/tree-depth-levels` on main `bad5445`. Rows:
`bench/results/fused_memory/round7/` (README there). The round-6 setup: one A100 40 GB (card 0, held by the guard
between runs), jax 0.11.2, clipped Plummer, p6, theta 0.8, cell_min_level 8, `JACCPOT_STRICT_CARRY=particles`,
unnamed caps, arena 0.88, frozen worktrees per arm, interleaved A/B.

**Step 0: where the data movement goes -- and two misreadings.** Main profiled at 8e6 and 1e8 with the input-order
carry as merged (round 6's table had the tree-order carry), every kernel named by the HLO instruction that launched it
and its source line (`step0/kernels_*.txt`). Two things that table had wrong:

* `bench/analyse_trace_by_stage.py` mapped instruction names across ALL dumped modules, first module wins, so
  `_compiled_runner_start`'s names shadowed `_compiled_runner`'s and some kernels landed in the wrong stage. It now
  looks each kernel up in its own module (the trace's `hlo_module`; a module dumped twice keeps its last compile).
* the "kick fusion" (17.6 ms at 1e8) is the **L2P**: XLA drops the metadata of that multi-output fusion, so it read
  as unmapped; its body is the jvp-transpose of `evaluate_local_real`. A fusion without metadata now takes the
  op_name most of its body carries. The L2P is 4.4 ms at 8e6 and ~52 ms at 1e8, not 2.7 / 31.

The rows that were far from their floors (kernel ms per step, 8e6 / 1e8; floors at ~1.3 TB/s):

| what | 8e6 | 1e8 | why |
| --- | --- | --- | --- |
| the force back to input order, `acc[inverse_permutation]` | 2.44 | 39.7 | XLA fused near + far INTO the gather: four random sector reads per particle (floor 0.3 / 3.4) |
| the inverse permutation (a scatter of N int32) | 0.40 | 16.3 | it only fed that gather |
| L2P (spherical angles + reverse-mode autodiff, XLA, 2^21-particle chunks) | 4.4 | 52 | ~500 flops per particle |
| far-list placement / rank | 3.15 / 2.19 | 45.6 / 20.3 | one pass at 8e6, four at 1e8 |
| COM radii: kernel / per-chunk segment scan + scatter | 2.45 / 2.9 | 24.7 / 27 | |
| tree: node-level histogram (scatter-add into 64 bins) | 1.03 | 10.3 | every node on a few counters |
| tree: depths by pointer doubling (22-25 `fori_loop` rounds, two carry copies each) | 0.74 | 9.6 | |
| tree: cub sort of the 64-bit Morton keys | 1.5 | 16.9 | at its floor |

**What changed** (each arm interleaved against main at 8e6, two rounds, `step_ab/`; "bitwise" = the state after 4
steps at 2e6 equal to main's, with a main-vs-main control, `bitwise/`):

* **The force goes back by a scatter** through the sort permutation (`out[perm[i]] = acc_sorted[i]`, unique
  indices): the sum fuses into the scatter's contiguous reads and the inverse permutation becomes dead code. 8e6
  81.0 -> 79.3 ms; bitwise; no extra peak. An optimization barrier before the gather (the sum materialised) got 1.2 of
  the 1.7 ms for +10 B/p at 2e6. `JACCPOT_FASTLANE_UNPERMUTE=gather` restores the gather.
* **A Pallas L2P** (`jaccpot/pallas/l2p_real.py`, the default where Pallas lowers; `JACCPOT_L2P_KERNEL=xla` restores
  XLA): one particle per lane, the gradient from the Cartesian recurrence of the complex inner solid harmonics
  (no Condon-Shortley phase, `1/(n+m)!`, the basis of `evaluate_local_real`) and the identities
  `d_z Y_n^m = Y_{n-1}^m`, `(d_x - i d_y) Y_n^m = Y_{n-1}^{m-1}` (`-conj(Y_{n-1}^1)` at m = 0),
  `(d_x + i d_y) Y_n^m = -Y_{n-1}^{m+1}` -- harmonics to degree p-1, no square root, no division by the radius.
  The same gradient to 1.9e-15 (float64) / 1.2e-6 (float32) relative. 8e6 step 80.8 -> 75.3 ms with the scatter
  (-3.8 for the L2P); force rel-L2 4.333e-4 unchanged (p90 6.242 -> 6.243e-4), the force alone 30.1 -> 28.4 ms. A
  leaf-major form (a program per 16 leaves, their particles 8 lanes at a time) ran its tiles ~1/3 full on cell leaves:
  -3.7 in the step but a slower force (30.9), dropped. The particle's leaf came from a per-particle array
  (`jnp.repeat` of the leaf counts and a gather: 0.38 ms at 8e6, 9.0 at 1e8, 8 B/p at the 1e8 peak). The kernel now
  bisects its block's window of leaf ends instead (a block of 128 particles lies in at most 128 consecutive leaves,
  from the leaf of its first particle, one `searchsorted` per block): bitwise; 8e6 73.2 / 73.2 -> 72.6 / 72.5 ms,
  1e8 805.7 -> 803.1 ms and 24.78 -> 23.97 GiB. Counting the window's ends per lane (a 128 x 128 compare) cost more
  than the array: 74.5-74.9 ms at 8e6, 825.4 at 1e8.
* **yggdrax: the tree's level tables from the level sort.** The stable sort of the node levels (which
  `nodes_by_level` needed anyway) gives the level offsets by `searchsorted` -- no histogram -- and the depth doubling
  runs 8 unrolled rounds (depth up to 256; the tables hold 64 levels) instead of a 22-25-round loop. Bitwise (trees
  equal on CPU; the step state equal on the GPU). 8e6 81.1 -> 79.3 ms (round 1; round 2 ran under a host load of 82
  on 64 cores -- a CPU test suite of ours -- and is not counted); tree 6.4 -> 4.3 ms at 8e6 and 90 -> 52 at 1e8
  (the inverse permutation's scatter left with the gather above).
* **COM radii:** blocks whose ancestors are all past the root read no particles, and 32 leaves a program (alone on
  the 8e6 tree: 5.43 -> 5.07 ms); bitwise. In the step 5.35 -> 4.90 ms (8e6), 51.7 -> 47.5 (1e8).
* **Lists:** the CSR route reads the walk's pairs without a masked copy, and the placement's scratch list no longer
  shares its fill with the output (XLA merged the two fills, then copied the one into both aliased operands); both
  bitwise. 8.1 -> 7.8 ms (8e6), 99.9 -> 95.8 (1e8).

**What did not work** (`tunes/`, `logs/`):

* **Relaxed atomics.** Pallas lowers every atomic as acq_rel at GPU scope; a relaxed primitive (a copy of the
  lowering with `MemSemantic.RELAXED`) raised the placement's atomic rate only 17.1 -> 18.6 G/s, and the far-list build
  not at all at one pass. Dropped. The probe found the cause of the "int32 vector atomics do nothing" trap: the lowering
  picks the integer opcode by `isinstance(val.type, IntegerType)`, which a vector (a ranked tensor) never is, so a
  vector int32 `atomic_add` is emitted as a FLOAT add on the int bits. Testing the element type fixes it.
* **More placement passes at 8e6:** alone, 2-4 passes take the far list 6.8 -> 6.2 ms; in the step 2 / 4 passes give
  -0.3 / -0.2 ms. Left at one pass below 4M rows.
* **A row-parallel rank** (several rows side by side, a `(rows, K, K)` compare): alone, the far-list build 10.6 ->
  12.0-13.9 ms and the near 2.4 -> 3.0-4.4 (single rounds). Dropped.
* **COM radii folded by atomic max** (the run reduction in the kernel, one atomic max per run per block): exact in any
  order, but every block that spans a top node raises it: 6.0 / 9.3 / 28.9 ms alone at 16 / 32 / 64 leaves a block,
  against 5.4 for the scan and scatter (5.1 at 32 leaves). Dropped.

**The tree-order carry no longer pays.** With the scatter the input-order carry lost what made the tree-order carry
fast: 8e6 72.9 / 73.2 against 73.0 / 73.2 ms (+20 B/p), 1e8 801.6 against 805.7 ms (+3.5 GiB, 24.78 -> 28.27). It
stays opt-in; Step 4 of the plan (its memory) is moot.

**The defaults as of this round** (`final/`; main = `base`, the round = frozen `a8` + yggdrax `y1`, then `a10` =
`a8` + the bisecting L2P, interleaved with `a8` in chain 10):

| N | main | round 7 (`a8`) | round 7 (`a10`, as merged) | jz-fmm at its nearest error (force incl. tree) |
| --- | --- | --- | --- | --- |
| 8e6 | 81.1 / 80.7 ms at 4.333e-4 | 73.0 / 73.2 at 4.333e-4 (force 24.9 ms) | **72.6 / 72.5** | p5 theta 0.8: 74.2 / 73.6 ms at 3.81e-4 |
| 1e8 | 911.2 ms, 24.96 GiB | 805.7 ms, 24.78 GiB, rel-L2 3.364e-4 (force 278.5 ms) | **803.1 ms, 23.97 GiB** | p5 theta 0.7: 1183.8 ms at 3.65e-4 |

At 8e6 the full step is now level with jz-fmm's force (round 6: 1.12x behind), -10 % against main; at 1e8 1.47x
ahead, -12 % against main and 1 GiB less. Everything but the L2P is bitwise against main (`bitwise/`, `B1` vs `X`);
the L2P forms are bitwise against each other (`C` vs `C2`, `C3`). At 2e6 the round's peak is main's (393 against
392 B/p; 400 with the per-particle leaf array).

**Where a step goes now** (`final/stages_*.txt`, the `a8` profile, kernel ms, 8e6 / 1e8): near 19.6 / 246, M2L 14.4 /
140, lists 7.8 / 96, walk 8.4 / 90, tree 4.3 / 52, COM radii 4.9 / 48, upward 3.8 / 34, L2L 2.9 / 23, L2P 1.2 / 19.6
(of it the per-particle leaf array `a10` removed: 0.38 / 9.0), the scatter back to input order 1.3 / 18.6, the
integrator's state updates ~1.5 / ~17. Kernel time 80.3 -> 72.7 ms (8e6), 915 -> 808 (1e8).

## Next

Two follow-ups stand between the fused lane and the 25M disc+bulge production rollout (both recorded
2026-10-03, neither started):

* **Consider the softening properly.** The far field is the unsoftened expansion and the geometric MAC ignores
  epsilon, so in the bulge centre it accepts cells closer than the softening length. Options, to be measured:
  * a distance floor `d - r_a - r_b >= c eps` in the walk's acceptance (cheap; with Plummer softening c must be
    ~20-30 for 1e-3, since the Plummer force never turns Newtonian);
  * compact-support (spline) softening in the near-field kernels plus a floor at its support, exact beyond it;
  * softened far-field corrections (heaviest; the real-harmonic M2L assumes a harmonic kernel).
* **`dehnen_error` on the fused lane.** The flat walk has no pair-policy seam, and the per-node fold
  (`dehnen_theta`) is refuted. The Pallas walk (and yggdrax's flat walk, to keep their pair sets identical) has
  to evaluate the criterion per pair from the policy's per-node arrays, rebuilt from the refreshed tree inside the
  scan. Its force-scale estimate already takes the softening, and the mesh lane was in class with it on this IC,
  so measure whether it alone fixes the bulge centre before building the distance floor.

Then (updated after round 7):

* **Speed, by the 1e8 profile** (round-7 defaults, ~808 ms of kernels): the near field (246 ms) and the M2L (140)
  issue FP32 at the card's rate, so they shrink only with less work -- fewer near pairs (leaf size, the MAC; 6.25e10
  at 1e8) or symmetric pairs (each computed once; needs a deterministic scatter of the reactions). The rest: the lists
  (96: four placement passes and the rank; a walk that emits pairs already grouped by target would remove both), the
  walk (90), the tree (52: cub sort 17, the positions' and masses' gathers ~19), COM radii (48), the scatter back to
  input order (18.6, random 12-byte rows), the integrator's column updates of the `[N, 2, 3]` state (~17, ~1 ms of
  it above its floor at 8e6). At 8e6 the step is level with jz-fmm's force; the next lever there is the same list.
* **The tree-order carry** no longer pays (round 7: -4 ms at 1e8 for +3.5 GiB); it can go unless a row-wise
  external field wants it for another reason.
* **cell_min_level per IC.** Level 6 saves 6-19 % (and 6-33 % memory) on bounded distributions and costs 3.5x on
  the unclipped draw at 2e6. An adaptive cut (split a cell leaf whose particles are spread wide relative to its
  neighbours) would take the gain without the outlier leaf; the library default (`None`, unconstrained) should
  become 8.
* **Memory: the eager prepare binds** (~215 B/p at 1.5-1.7e8, its list build). The particle carry needs only the
  walk's caps, the shape template and the initial verdict from it: a prepare that stops after the walk and
  derives the template's shapes would drop the eager list build and downward pass. Re-measure the one-card ceiling
  on jax 0.11.2 first (+5 % peak at 1e8 on the same code).
* **The Pallas Triton backend is deprecated** in jax 0.11.2: every kernel will need its Mosaic GPU form before
  JAX drops Triton.
* **Production.** Odisseo can opt into `JACCPOT_STRICT_CARRY=particles` and `donate_state` (and, with a row-wise
  external field, `JACCPOT_STRICT_CARRY_ORDER=tree`); the int32 index default is in (#361, yggdrax #84).
* Still open: a segment retry for the multi-GPU `FusedRollout`, and the GPU-only unit test failures that predate
  0.11.2 (10 in the Pallas unit files on an A100, two of them on ARCHITECTURE section 9's list).
