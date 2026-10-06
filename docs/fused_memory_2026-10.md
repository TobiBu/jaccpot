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

Then (updated after round 4):

* **Memory: the eager prepare binds** (~215 B/p at 1.5-1.7e8, its list build). The particle carry needs only the
  walk's caps, the shape template and the initial verdict from it (the start force is its own program now): a
  prepare that stops after the walk and derives the template's shapes would drop the eager list build and downward
  pass.
* **Then the step** (live peak 12.2 GiB at 1.28e8): the list build, the walk, the near field's padded table
  (`(N + W, 4)`, 16 B/p) and the step-long per-particle arrays are within ~1 GiB of each other.
* **Speed** (8e6: ~101 ms per step): the list build ~11 ms; the walk ~27 ms.
* **Production.**
  * Odisseo can opt into `JACCPOT_STRICT_CARRY=particles` without code changes; `donate_state` needs one keyword
    in its `strict_run_v2` call (it is off by default and keeps the input state).
  * The int32 index default is its own PR pair (jaccpot #361, yggdrax #84).
* Still open: a segment retry for the multi-GPU `FusedRollout` (it raises `RolloutFlagError` today), and the
  record configuration at 2e5 on a quiet card.
