# Production readiness for large galaxy runs, 2026-10

**[open]** This is the record of the multi-session plan that makes jaccpot production-ready for large galaxy
simulations. There is one section per session. The plan itself (sessions S0-S13, R1) lives outside the repo; this
file records what each session measured and decided.

**The user's decisions behind it (2026-10-13):**

| | decision |
| --- | --- |
| D1 | Block (individual) timesteps that conserve momentum exactly and evaluate only the active interactions. |
| D2 | `dehnen_error` on multi-GPU too, with its runtime cost measured. |
| D3 | fp32 and fp64 both selectable. |
| D4 | Production settings live in jaccpot. Integration moves towards nornax solvers; Odisseo becomes a client. |
| D5 | Multi-node is future work. |
| D6 | Gradients at all N, with respect to numerical and potential parameters. Rollouts run until chaos sets in; long runs use multiple shooting (mimirax + nornax). |

## S0 — baseline, guards, block-step probes (2026-10-13)

**Setup.**
- Card: one A100-PCIE-40GB (GPU 7, booked through autocvd; no foreign process during any quoted row). Host load
  was 27-41 on 64 cores, the same as the round-7 1e8 row (39).
- Software: jax 0.11.2 (`/export/scratch/tbuck/jax0112-venv`).
- Code:
  - **main** = jaccpot `5bc4959` + yggdrax `2758fa9`;
  - **round 7** = the frozen `jaccpot-r7-a10` + `yggdrax-r7-y1`, the round-7 code as merged.
- Harness: `bench/fused_memory_budget.py`, one configuration per process, with:
  - clipped Plummer, cells of 64, cell_min_level 8, p6, theta 0.8;
  - `JACCPOT_STRICT_CARRY=particles`, unnamed caps, arena 0.88.
- "Step" is the minimum of 3 calls of 4 steps each.
- Rows: `bench/results/production_baseline_2026-10-13/` (its README names every file).

### The single-card lane has not regressed; the ferrers3 default costs 8-9 %

| N | round 7, plummer | main, plummer | main, ferrers3 (the default since #375) | peak (all three) | rel-L2 (4096 targets) |
| --- | --- | --- | --- | --- | --- |
| 8e6 | 72.4 / 72.4 ms | 72.8 / 72.8 ms | 78.2 / 78.7 ms | 2.36 GiB, 317 B/p | 4.333e-4 in every arm |
| 1e8 | 793.8 ms | 798.5 ms | **872.6 ms** | 23.97 GiB, 257 B/p | — |

- **Round 7 reproduces its record** (72.5 ms at 8e6, 803.1 ms at 1e8), so the card and the host are comparable to
  the round's.
- **The cleanup and the x64 requirement cost nothing.** At the same kernel main is within 0.6 %, with identical
  far-pair counts (39,093,974 at 8e6 and 402,173,018 at 1e8) and identical peaks.
- **The ferrers3 default costs +74 ms at 1e8.** 1e8 is still under the 1 s budget, but the margin fell from about
  200 ms to about 127 ms.
- **The cost is the near-field kernel.** The eval-only rows differ by only 0.7 ms at 8e6 (27.4 vs 28.1 ms), which
  first suggested the refresh. The step trace below corrects that: the whole +5.75 ms is the step's near-field
  kernel. Why the eval-only rows hide it is not explained yet.

### The 25M disc+bulge runs on one card, and `dehnen_error` costs 2.9x per step

The disc+bulge IC (25,000,000 of `disk_bulge_25m.npz`, shuffled, so the prefix is a random subsample) runs in its
NFW halo with ferrers3 at eps 1.5e-3, the production softening (`docs/softening_2026-10.md`).

| MAC | step | peak (step) | force eval | rel-L2 | p90 | far pairs |
| --- | --- | --- | --- | --- | --- | --- |
| geometric, theta 0.8 | **238.9 ms** | 6.17 GiB, 265 B/p | 100.6 ms | 4.18e-4 | 7.48e-4 | 94.8M |
| `dehnen_error` 1e-4 | **695.3 ms** | 7.05 GiB, 303 B/p | 99.2 ms | **1.49e-4** | 2.46e-4 | 93.2M |
| `dehnen_error` 1e-5 | 1399.5 ms | 8.41 GiB, 361 B/p | — | — | — | 215.1M |

- **One card does the 25M production run at 0.24 s/step.** The quarter orbit ran on the mesh lane at 82 s/step on
  8 cards (Odisseo `781c412`).
- **`dehnen_error` 1e-4 is 2.8x more accurate at equal force cost, with equal list sizes, but its step is 2.9x
  slower.** The extra 456 ms is all in the refresh (walk, f_b, tables), which the traces below split. It has to be
  fixed (plan S3b) before `dehnen_error` can be a production default or go multi-GPU.

### Odisseo's `integrate()` is 40-1600x slower per step than the lane it drives

The script is `odisseo_time.py`, at 8e6 on the same IC. It makes three calls: A (4 steps), B (4 steps again) and
C (12 steps). The per-step time is (C - B) / 8.

| Odisseo config | per call (4 steps) | per step | peak |
| --- | --- | --- | --- |
| Odisseo defaults (leaf 16, p4, theta 0.6) | A 746 s, B 697 s | **124 s** | 19.0 GiB, 2550 B/p |
| leaf 64, p6, theta 0.8, fp64 state | A 253 s, B 192 s | **3.9 s** | 5.1 GiB, 682 B/p |
| the same, fp32 state | A 240 s, B 200 s | **3.0 s** | 4.1 GiB, 554 B/p |
| the same, fp64, `return_snapshots=True` (the history path) | A 215 s, B 173 s | **35.7 s** | 10.7 GiB with the 12 stored snapshots (4.7 GiB for 4) |
| (the harness, same knobs) | — | 78.5 ms | 2.36 GiB, 317 B/p |

- **Each call pays about 190 s of fixed cost, and pays it again on an identical call.** That is the prepare plus a
  recompile; the cache key holds `id()` of the external closure and `float(dt)`.
- **Even with the harness's knobs, Odisseo's step is 38-50x slower.** Odisseo cannot select cell leaves or
  cell_min_level, so it builds bucket leaves. It also passes `advanced=`, which replaces the preset's bundle; it
  carries the state rather than particles; and it evaluates an eager start force. Plan S4 traces the split.
- **The history path is the first-order integrator that the review found** (`jaccpot_coupling.py:312-343`). It is
  also 9x slower than the strict lane (35.7 vs 3.9 s/step), because it leaves the device scan for a host loop
  that prepares or refreshes the state at every step. A run that wants snapshots therefore pays an order of
  magnitude and loses second-order accuracy.

### Gradients: the jitted gradient is 7-8x the force eval, and its forward has drifted from production

The bench is `bench/grad_step_profile.py` at its operating point (cells 64, theta 0.8, p5, fp32), with the
device peak taken over the whole process.

| N | production eval | jitted grad (pos + mass) | upward / far / near grad | process peak | aggL2 production / differentiable forward |
| --- | --- | --- | --- | --- | --- |
| 2e5 | 4.63 ms | **31.9 ms** (Sept: 22.1) | 5.0 / 20.8 / 11.7 ms | 3.22 GiB | 1.27e-3 / **1.60e-3** |
| 1e6 | 13.0 ms | **101.6 ms** | 14.5 / 56.8 / 47.1 ms | 15.8 GiB | 1.22e-3 / **2.41e-3** |
| 8e6 | 74.5 ms | **out of memory** (a 27.37 GiB request in `jit_loss`) | 70.5 / **OOM** (27.35 GiB, `jit_far_loss`) / 330.5 ms | 19.5 GiB | 8.31e-4 / **1.24e-3** |

The first 8e6 run failed earlier, in the bench's own direct-sum reference: a fixed 1024-target block is 61 GiB at
8e6. The block is sized to the budget on this branch, and the row above is the re-run.

- **A single-force gradient at 8e6 does not fit on one 40 GB card through today's gradient seam.**
  - The full and the far-field gradients each ask for about 27.4 GiB on top of the prepared state; the eager arms
    fail too.
  - The near-field gradient alone takes 330 ms (4.4x the whole production eval).
  - The far field's reverse goes through the leaf-major padded L2P and the leaf-table layouts that the forward
    left in round 2. Those are what plan S8 replaces (a Pallas L2P reverse reducing into per-leaf cotangents, a
    direct-layout near reverse).
- **A correctness regression.** In September the differentiable forward equalled the production forward to
  1.8e-11, at the same aggL2 of 1.27e-3 (`docs/fast_gradients_2026-09.md`). Now:
  - it is 1.3x less accurate at 2e5, 2.0x at 1e6 and 1.5x at 8e6;
  - at 8e6 the bench prints it directly: the differentiable forward against the production eval, rel-L2 1.40e-3.

  Pin S4b compares the gradient lane only with itself, so it could not see this. S1 bisects it and adds a pin
  asserting that the differentiable forward equals the production forward on the fused lane.

  **S1:** not a regression of main but a MAC-geometry split since `bc62978`: the bench's fresh solver walked the
  box MAC, the production eval the COM one. Fixed in S1 (see there); pin G1 guards it.
- **The 2e5 grad went from 22.1 to 31.9 ms.** Both rows were taken on a loaded host; S8 profiles it.
- **The gradient's memory grows faster than the forward's** (15.8 GiB at 1e6, out of memory at 8e6). The process
  peak includes the bench's eager stage arms, so S8 measures the jitted gradient alone.

### The block-step lane

The lane is Odisseo `integrate_blockstep_jaccpot` on the jitted device lane: fp64, k_max 3, theta 0.6, leaf 64,
clipped Plummer.

| N | per base step | peak | rungs used | momentum drift |
| --- | --- | --- | --- | --- |
| 1e5 | 2.19 s | 0.34 GiB (3.7 kB/p, fixed buffers dominate) | 0-1 of 0-3 | 8.6e-19 |
| 1e6 | **15.4 s** | 0.96 GiB, 1.03 kB/p | 0-1 of 0-3 | 2.2e-18 |

A base step at 1e6 takes 10 full force evaluations: a rebuild, a rung force and 9 boundary kicks. That is about
1.5 s per fp64 force, against the fused lane's 13 ms fp32 force at 1e6.

### Pins added (`bench/dce_pins.py`)
- **L1-L3 are timing pins:**
  - L1: 8e6 step;
  - L2: 1e8 step;
  - L3: the 25M disc step, plus its force rel-L2.

  They run without the deterministic-ops flag. `compare --envelope` passes a pin on clean rows within max(3 %,
  twice the A-vs-A spread) in time and max(2 %, twice the spread) in peak, with rel-L2 equal to 2 digits.
- **DET1/DET2 are S1 and S4b on the production flags.** Two records of the same code must be bitwise equal.
- **Baseline tags** `main-5bc4959` and control `main-5bc4959-b` (`/export/scratch/tbuck/dce_pins/`; summaries in the
  results directory). Every row is clean.

| pin | baseline | control | change |
| --- | --- | --- | --- |
| L1, 8e6 step | 78.3 ms, 317 B/p | 78.5 ms | +0.3 % |
| L2, 1e8 step | 871.8 ms, 257 B/p | 873.0 ms | +0.1 % |
| L3, 25M disc step | 240.0 ms, 265 B/p | 239.3 ms | -0.3 % |
| L3 force | 100.1 ms, rel-L2 4.1791e-4 | 100.0 ms, 4.1791e-4 | -0.1 % (peak -1.8 %) |
| DET1 (fused step, 2e5) | | | force and state **bitwise** |
| DET2 (large-N grad, 2e4) | | | d/dpos and d/dmass **bitwise** |

**The production path is bitwise reproducible without `--xla_gpu_deterministic_ops`, for the forward and for the
gradient.** The plan's later bitwise gates (S6) and any reversible-adjoint work can rely on it. The A-vs-A spread of
the timing pins is 0.1-0.3 %, so the 3 % floor is the binding envelope.

### D1: which momentum-exact level map

Both maps give each interaction one level and kick both partners, so both conserve momentum exactly; they differ
in what has to be evaluated at a fine boundary. The three probes:
- **Activity** (`bench/blockstep_activity_probe.py`): the 25M disc, the lane's own tree and lists, rungs from
  `sqrt(eps/|a|)` (halo included) with the fastest particle on the finest rung.
- **Frozen lists** (`bench/blockstep_frozen_list_probe.py`): the mutual lane at 1e6, p6, theta 0.8, ferrers3 eps
  1.5e-3, against an fp64 direct sum on 4096 targets. The drift is `v0 * dt_max` with `dt_min = 5e-4`, the disc's
  production step.
- **Accuracy A/B** (`bench/blockstep_hold_vs_max_ab.py`): a direct sum on a 2e4 prefix of the same disc, in its
  halo, with Plummer eps `0.02 R_d`. Each scheme runs 256-512 fine steps and is scored against a shared-step
  reference at `dt_min / 4`.

**1. The timestep distribution is narrow, which caps any block scheme at 1.93x for this IC.** The rung histogram
for k_max 6 is

    [0, 0, 12583, 87715, 1098213, 22320242, 1481247]

- 5.9 % of particles need the finest step.
- 89 % need 2-4x the finest step.
- 0.4 % need more than 8x.

The particle-step bound `N 2**k / sum_i 2**rung_i` is 1.93x at every k_max from 3 to 6; more levels only add empty
rungs. The criterion's `|a|` spans only a factor of about 16-64 here, because the softening caps the centre.

**2. Under the max map every particle is kicked at every boundary, and the frozen lists that would avoid the
per-boundary rebuild are wrong.**

At a finest-only boundary (half of all boundaries) the max map still evaluates:
- 13.8 % of the near tiles;
- 7.3 % of the far pairs;
- and kicks **100 %** of the particles. Every cell far-pairs with an active one, so every particle needs L2L/L2P,
  and every particle drifts, so P2M/M2M too.

At every other boundary it evaluates 92-100 % of the work.

Keeping the lists for the whole base step would save the topology stage (about 35 % of a step). But the
frozen-list force error grows with the base step:

| k_max | dt_max | displacement p99 | frozen rel-L2 / max | fresh rel-L2 / max |
| --- | --- | --- | --- | --- |
| 3 | 4e-3 | 15 eps | 1.30e-2 / 0.13 | 7.6e-4 / 0.14 |
| 4 | 8e-3 | 30 eps | 1.59e-1 / 1.55 | 6.1e-4 / 0.12 |
| 5 | 1.6e-2 | 60 eps | 6.5 / 131 | 7.6e-4 / 0.13 |
| 6 | 3.2e-2 | 121 eps | 1.8e3 / 5.8e4 | 7.1e-4 / 0.09 |

The median stays small (3e-4 to 2e-2). The tail explodes: particles drift inside cells whose far expansion they
still use. Neither the overflow flag nor momentum sees this. **This is also what today's block-step lane does**
(`BlockStepFMM`, lists frozen per base step). Its forces on a galaxy at these steps are wrong in the tail from
k_max 3, which is a correctness item for S1.

**3. Predicted gain per base step against a shared dt_min**, from the round-7 stage costs:

| k_max | max map, frozen lists (invalid) | max map, rebuilt at every boundary | HOLD | bound |
| --- | --- | --- | --- | --- |
| 3 | 2.19x | **1.29x** | **1.93x** | 1.93x |
| 4 | 2.30x | 1.29x | 1.93x | 1.93x |
| 5 | 2.37x | 1.29x | 1.93x | 1.93x |
| 6 | 2.40x | 1.29x | 1.93x | 1.93x |

A multipole per rung (option B1) changes nothing: 99 % of the leaves are single-rung, and B1 inflates the far work by
only 3.7-3.9 %.

**4. Accuracy (direct sum, 2e4).** The energy error is `|E - E0| / |W0|` over the run; dx is the position error
against the reference, in eps.

| k_max | shared dt_min: dE, dx median / p99 | max: dE, dx med / p99, pairs | HOLD: dE, dx med / p99, pairs |
| --- | --- | --- | --- |
| 3 | 9.7e-7, 2.6e-4 / 9.1e-3 | 3.3e-6, 1.3e-3 / 3.0e-2, 0.58 | 7.8e-7, 1.3e-3 / 3.1e-2, 0.49 |
| 4 | 9.8e-7, 7.5e-4 / 6.0e-2 | 2.2e-6, 5.0e-3 / 9.5e-2, 0.59 | 4.0e-6, 5.3e-3 / 1.1e-1, 0.49 |
| 5 | 9.8e-7, 7.5e-4 / 6.0e-2 | 2.3e-8, 4.9e-3 / 9.6e-2, 0.59 | 7.2e-6, 5.3e-3 / 1.1e-1, 0.49 |

- Both maps' pair impulses balance to about 1e-18; momentum is exact.
- HOLD's position error is within 5-15 % of the max map's.
- Its energy error is 0.2-300x the max map's, and every value is at the 1e-6 level or below.
- HOLD evaluates 49 % of the shared run's pairs, the max map 58 %.

**Recommendation: the HOLD map (level = min rung), built as the fused force on padded rung subsets.**
- It reaches the bound.
- It needs no frozen lists: a fresh subset tree per evaluation.
- It reuses the whole fused lane, with yggdrax `num_valid` for the padded subset.
- Its accuracy is in class.

**But the bound for this IC is 1.93x.** The block-step track (S5-S7) pays off for ICs with a wider timestep range
(cusps, mergers, smaller softening), so the rung distribution of each planned production IC should decide its
priority. On this disc, S2-S4 (Odisseo's 40-1600x), S3b (`dehnen_error`'s 2.9x) and the ferrers3 near kernel (9 %)
are worth more.

**5. The cap comes from the timestep criterion, not from the disc** (`bench/blockstep_timestep_spread.py`: a 3e4
direct-sum prefix, NFW halo included, eps 1.5e-3).

| R / R_d | fraction | median \|a\| | dt / dt_min, `sqrt(eps/\|a\|)` | dt / dt_min, `sqrt(r/\|a\|)` |
| --- | --- | --- | --- | --- |
| < 0.25 | 3.3 % | 71 | 1.9 | 13 |
| 1-2 | 28.9 % | 25 | 3.2 | 64 |
| 4-6 | 10.7 % | 13 | 4.5 | 156 |
| 6-10 | 3.2 % | 8 | 5.6 | 241 |

- **Why the acceleration criterion (nornax's, GADGET's) spreads so little.** |a| varies only about 10x across the
  galaxy (max/median 10.9): the rotation curve is nearly flat (|a| = v_c^2 / R) and the softening caps the centre.
  The criterion then takes the square root, so dt grows only as R^0.26.
  - The median particle wants 3.3x dt_min, and 99 % are within 6.8x.
  - The shared step set by the centre is therefore almost right for every particle, and the bound is about 2x.
- **An orbit-time criterion spreads much more.** `sqrt(r / |a|)`, which scales with the orbital time, spreads about
  35x from the 1st to the 99th percentile. The bound at k_max 6 is then about 40x.
- At equal eta the acceleration criterion's step is `sqrt(eps / r)` of the orbit-time step, about 1/25 at 4 R_d.
  Most of the outer disc's steps resolve the softening scale, not its orbit.

**6. HOLD keeps its accuracy when the rungs are far apart.** The A/B was rerun with `--criterion dyn` (eta 0.02,
N = 5000, eps 1.5e-3, scored against the shared-step reference):

| k_max | rung histogram | shared: dE | max: dE, dx median / p99, pair work | HOLD: dE, dx median / p99, pair work |
| --- | --- | --- | --- | --- |
| 4 | [4794, 145, 41, 12, 8] | 7.1e-7 | 1.4e-8, 5.3e-5 / 2.7e-3, 7.4 % | 4.6e-7, 5.5e-5 / 2.9e-3, 6.3 % |
| 6 | [2040, 2340, 414, 141, 45, 12, 8] | 7.4e-7 | 9.6e-5, 9.2e-4 / 5.2e-2, 4.8 % | 8.1e-5, 1.2e-3 / 1.2e-1, 2.2 % |

The same run with the acceleration criterion (k_max 4, eta 0.1) gave:
- max: dE 1.5e-6, dx median 1.5e-4, pair work 42 %;
- HOLD: dE 7.7e-6, dx median 2.3e-4, pair work 29 %;
- shared: dE 1.0e-6.

At k_max 6 both maps lose accuracy together. That is the coarsest rung's step being too long for eta 0.02, which
is the criterion's calibration, not the map. The comparison that decides is accuracy at equal cost, which these
fixed-rung runs do not give.

**Decision (2026-10-13):**
- **HOLD is the map to build.**
- **Production keeps the acceleration criterion for now**, and the block-step track is deferred behind the
  gradient work.
- **Before it resumes, a criterion calibration run.** Shared vs HOLD, the acceleration vs the orbit-time criterion
  (possibly with a softening floor), at equal cost over at least one orbit. Scored on energy, angular momentum,
  Sigma(R), sigma_z(R) (disc heating) and A2(t).
- That run decides whether galaxy discs use the orbit-time criterion, which would make block steps worth about
  10-40x on this IC.

### Where the ferrers3 and `dehnen_error` time goes

Traces were taken without command buffers, per kernel (`bench/analyse_trace_by_stage.py`; kernel ms per call).

**ferrers3 vs plummer, 8e6:**

| stage | plummer | ferrers3 |
| --- | --- | --- |
| near (`nearfield_leafpair_csr_direct_first`) | 19.17 | **24.92** |
| M2L | 14.45 | 14.45 |
| walk | 7.73 | 7.73 |
| lists | 7.83 | 7.77 |
| COM radii / tree / upward / L2L / L2P | 4.93 / 4.67 / 3.86 / 2.97 / 0.73 | identical |
| **kernel total** | 72.52 | 78.26 |

The whole difference is the near kernel's pair arithmetic: the branch-free `G(c) / max(r, h)^3` is evaluated for
every pair. The walk's softening floor costs nothing. The lever: skip the polynomial for a tile whose pairs all lie
beyond the support, which is nearly every tile at production softening.

**Geometric vs `dehnen_error` 1e-4, 25M disc+bulge** (kernel ms per call; total 245.4 vs 699.7):

| stage | geometric | `dehnen_error` 1e-4 |
| --- | --- | --- |
| near | 93.8 (`..._ferrer`) | 116.1 (`..._fb_fer`: the f_b lane +22) |
| lists | 35.8 | 35.5 |
| M2L | 30.9 | 30.5 |
| walk | 21.9 | **42.2** (eq 16a per pair) |
| tree / COM radii / upward / L2L / L2P | 14.7 / 13.7 / 8.6 / 5.9 / 2.2 | 12.6 / 13.8 / 8.9 / 5.8 / 2.1 |
| unscoped + unmapped | 16.7 | **429.6** |

The unscoped time is the **far half of f_b, recomputed every step outside the force kernels**
(`runtime/_force_scale_levels.py`, per-kernel table `traces/kernels_de4_25m.txt`):
- **`_far_force_scale_sorted_jit._chunk`, about 202 ms.** A chunked scatter of each far pair's monopole term into
  its target node (120 ms over 34 chunks). Inside each chunk, a `searchsorted` that expands the CSR row offsets,
  a while-loop bisection of 748 launches with copies (82 ms).
- **`ancestor_sum_by_level`, about 191 ms.** Two scatters per level over 59 levels, pushing the node sums down
  the tree.
- MemcpyD2D: 40 ms over 1731 launches.

**The fix belongs in plan session S3b, and it follows the budget rule (piggyback on passes that already run):**
- the CSR M2L lanes kernel already walks every far pair row by row. It can accumulate `G M_s / (d^2 + eps^2)` per
  target row with no atomics and no searchsorted;
- the L2L cascade already runs top-down per level. It can carry that scalar to the leaves as one extra
  coefficient.

That removes about 390 ms. The rest, the doubled walk (+20) and the f_b near lane (+22), would leave
`dehnen_error` at about 1.2x the geometric step: the S3b target is 1.3x. The 25M disc's `dehnen_error` 1e-5
step (1.4 s) also carries 2.3x the far pairs.

### Findings that change the plan
- **S1 (correctness):**
  - the differentiable forward's accuracy regression (1.3-2x the production error; rel-L2 1.4e-3 against the
    production forward at 8e6; bisect, plus a pin differentiable == production);
  - `BlockStepFMM`'s frozen lists blow up in the tail of a galaxy from k_max 3 (rebuild per boundary, or a
    drift-margin MAC, or refuse k_max/dt_max beyond a measured displacement);
  - the grad bench's reference block (fixed here).
- **S2/S3:** the ferrers3 near kernel (+5.75 ms at 8e6, +74 ms at 1e8): skip the polynomial for tiles entirely
  beyond the support.
- **S3b:** the far f_b into the M2L and L2L kernels, as above.
- **S8:** the single-force gradient does not fit at 8e6. The production-layout reverses come before any rollout
  gradient work.
- **S4:** Odisseo's 40-1600x is the largest single lever for any Odisseo user. The fixed cost per `integrate()` and
  the missing cell leaves come first.
- **D1:** HOLD is recommended, but the block-step track's priority depends on the production ICs' rung
  distribution. This disc caps it at 1.93x.

## S1 — correctness fixes and test infrastructure (2026-10-15)

**Setup.** Code: jaccpot main `63587f2`, yggdrax `2758fa9`, nornax `c1b9ff6`, Odisseo `jaccpot-integration` `adca4f6`;
jax 0.11.2. Each fix is its own PR (jaccpot #402 MAC geometry + gradient lane, #403 multi-GPU, #404 block-step
drift, #405 tests, #406 nightly, #407 docs + this record; nornax #16; Odisseo #34-#36); the gate ran on a local
merge of all the S1 jaccpot branches. GPU runs used
shared A100s booked through autocvd (`-l`): accuracy and pins only, no timings.

### The differentiable forward was on a different MAC than production (the S0 "regression")
- **Cause: the walk's MAC geometry followed the engine's call history.** Phase 6 (`9912f5b`) made the COM geometry
  every lane's default; `bc62978` (2026-09-14) scoped it to the strict lane through `_strict_fused_mode_active`,
  which only a strict call sets. A fresh engine's `compute_accelerations`, `prepare_state` and the gradient lane
  walked the box (AABB) MAC; the same engine after any strict call walked COM.
- `bench/grad_step_profile.py` builds a NEW solver for the differentiable state, so its gradient lane walked AABB
  against the production eval's COM. The Sept "1.8e-11" was measured on the branch before `bc62978`; it never
  held on main.
- **Fix: the geometry is the tree's.** COM on a static-radix tree expanded about the COMs, AABB on every other tree
  (LBVH presets unchanged); the env override still wins. Only the strict lane's in-scan refreshes take the traced
  hot path, never a public refresh.
- Measured (fp64 direct sum):

| case | AABB (fresh engine on main) | COM (fix) |
| --- | --- | --- |
| S4b config, 2e4, leaf 32, theta 0.6, p4 | 72,430 far, rel-L2 1.16e-3 | 67,120 far, 6.73e-4 |
| 1e5, leaf 64, theta 0.8, p6 | 236,080 far, **1.54e-2** | 262,714 far, 6.36e-4 |
| G1: 2e5 grad-bench point, differentiable vs production | **1.21e-3** | **1.66e-7** (equal lists, 750,196 far) |

  The COM MAC is not the less conservative one on this tree, as `_mac_geometry.py` used to claim: it is more
  accurate with fewer far pairs at 2e4. On the LBVH order sweep (N=224) the box stays better, which is why only
  static radix changes.
- The 1.7e-7 left over is fp32 summation order (it grew 1.3e-11 -> 1.7e-7 through rounds 4, 6 and 7, every
  time with identical list counts).
- **Pin G1** (`bench/dce_pins.py`) checks itself: both call orders within 1e-5 of production. Fails on main.

### The other gradient-lane bugs
- **J1:** the differentiable CSR near field dropped `softening_kernel` and ran ferrers3. At eps 0.05, N=1024:
  plummer 5.96e-2 -> 3.3e-7 off production, wendland_c2 1.02e-2 -> 3.3e-7; plummer d/dpos 0.38 -> 7.6e-4 vs its
  direct-sum gradient.
- **J2:** `strict_run_v2`'s returned state carried the START's far list: `differentiable_accelerations` of it gave
  3.2e-2 vs 1.7e-4 from a fresh prepare (N=1024, 3 steps). The list is now dropped and the gradient path raises
  with the remedy. A second source (an eager public refresh after a strict call returned the input's list:
  4.4e-2) went with the J5 fix: 9.47e-5, equal to production.
- **J8:** the Pallas M2M/L2L cascade and CSR M2L-lane reverses lacked the on-axis (rho = 0) transverse limit: FD
  vs AD 1.6e-6 -> 2.7e-10 on a z-stacked system. Forward and pins S4/S4b unaffected (bitwise).

### Multi-GPU fused lane
- **J3:** the non-CSR M2L (pre-Ampere, `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0`, CPU) could not take the cross
  pairs. Fixed; a 2-device CPU end-to-end arm runs decompose -> setup -> force -> `FusedRollout`: rel-L2 3.643e-3
  flat and CSR-interpreted alike, 3.578e-3 on one device, 329 cross pairs per device.
- **J4:** `dehnen_error` (and `adaptive_error_model='dehnen_paper'`) is refused at every fused multi-GPU entry
  point until S12. The mesh lane keeps it.

### Block steps: frozen lists (J10)
- Opt-in `BlockStepFMM(drift_margin=m)`: radii inflated by `(2+theta)·m` (the tight bound: a centre moves m, a
  radius grows 2m), lists stay symmetric (momentum 1e-18). A measured guard flags drift beyond m at every
  boundary through the overflow flag (bit 5). `suggest_drift_margin`. Default bitwise (S5); one warning at
  k_max >= 3 without a margin.
- 25M disc+bulge prefix at 2e4 (rel-L2 / max): k3 frozen 7.7e-4 / 8.1e-2 (fresh lists 7.9e-4 / 8.2e-2), with
  margin 5.6e-5 / 3.0e-2; k6 frozen 4.9e-2 / 2.43, with margin 4e-15 (the margin makes every pair near: a direct
  sum). At 1e6, k3: frozen 1.30e-2 (S0 reproduced), fresh 7.6e-4, margin 7.1e-5.
- **Cost grows with N** (the drift is fixed, the cells shrink): near pairs x1.56 / x2.82 / x5.64 at 2e4 / 2e5 /
  1e6 for k3. A correctness option; HOLD (S6) with fresh subset trees is the real fix.

### Smaller jaccpot items
- Jerk and time derivatives on the GPU large-N lane raise `NotImplementedError` (they failed in jaxtyping).
- Doc contradictions (review P0.7) fixed; dead `on_grad_path` removed.
- The 11 A100-only Pallas unit failures: one kernel bug (above), one facade annotation, the rest the tests' own.
- conftest: `NORNAX_WORKTREE`; a nornax lacking what the tests import stops the run instead of skipping 5 tests.
- `bench/nightly_pins.py`: the local nightly (S/M/DET/L pins + `gpu_gate.py` on the main heads). Not PR-triggered.
  **Correction (S2): not landed.** PR #406 was not merged (the user deferred the nightly); the script is not on main.

### Odisseo and nornax
- **The leapfrog of every non-strict run was first order in the self-gravity**, not only history runs: the `fast`
  preset, every CPU run and every GPU run below `fmm_large_n_min_particles` kicked both halves with a(x_n).
  Fixed (one extra force per segment): convergence ratio 1.9 -> 4.0. A large_n_gpu history run against the strict
  lane, 2e4, 8 steps, max |dv|: 1.5e-3 on main, 6.1e-5 with the KDK fix, **4.8e-7 with the J5 fix as well**.
- The masked scatter of the active-subset lane dropped particle 0's update when a schedule row had padding (6 of 7
  layouts on an A100).
- The block-step jitted lane raises on the five host-lane options it ignored; the stale "111x dt_max" warning went.
- A CPU CI workflow (Odisseo had none); `sample_position_on_sphere/circle` accept a traced radius.
- nornax: `IntScalar` import for `shooting_node`, and a test that every annotation resolves.

### Gate
| pin | result |
| --- | --- |
| S1, S2, S3, S4, S5, S6 | bitwise vs main `63587f2` (main itself bitwise vs the 2026-10-13 record) |
| S4b | **moves (intended)**: the gradient lane now walks COM. g_mass rel 1.24e-3, g_pos 2.79e-5; vs fp64 direct sum forward 1.18e-3 -> 7.60e-4, g_mass 1.60e-3 -> 9.63e-4 |
| M1, M3 | bitwise |
| M2 | within main's A-vs-A envelope (max 2.4e-7) |
| G1 | passes (1.66e-7); fails on main (1.21e-3) |
| DET1 | bitwise A-vs-A, and bitwise vs S0's `main-5bc4959` |
| DET2 | **not bitwise A-vs-A** (g_mass rel 2.0e-9) -- nor is MAIN today (2.6e-9); S0 had it bitwise |

- **DET2 is bitwise on a free card only.** S0 recorded it on a free GPU 7; today's records on a shared card differ
  at 2-3e-9 relative on main and on the S1 code alike, so it is not an S1 change. The reverse kernels' fp32
  accumulation order follows the scheduling. The plan's bitwise-gradient assumptions (S6 gates, R1) need a free
  card, the deterministic-ops flag, or an atomic-free reverse.
- **CPU:** every jaccpot CI shard (unit, unit-runtime, unit-runtime-mac, integration, characterization,
  mutual-static-device, the distributed and nornax tiers, bench tooling) exits 0 on the merged code; Odisseo's new
  CI selection: 219 passed, 2 xfailed, mesh lane 15 passed.

## S2 — the fused force as an object, the fp64 validation lane, traced parameters (2026-10-15/16)

**Setup.** Code: jaccpot main `f967805` (the base), yggdrax `2758fa9`, nornax `763096e`, Odisseo `8fa0c18`; jax 0.11.2.
The work is a linear stack of local branches on `f967805` (`s2/1` to `s2/13`, table below). GPU runs used shared A100s
booked through autocvd for accuracy, program identity and pins; the L1-L3 timing gate ran on a free GPU 7.

**The user's decisions (2026-10-15):**

| | decision |
| --- | --- |
| traced parameters | **Static by default:** Python floats stay compile-time literals, bitwise by construction. **Traced is opt-in** (arrays as arguments, one compile per run). The GPU A/B in S3 decides whether traced becomes the default. |
| fp64 force | **A validation lane:** selectable and correct, Pallas where the kernels are dtype-generic, XLA elsewhere; time and memory recorded, no budget; the 1e8 row is the largest N that fits. |
| `dehnen_error` | A traced G or eps^2 is refused in S2. |
| compact kernels | The walk's separation floor stays static; a traced eps^2 above the walk's sets the PARAMS overflow bit. eps_max and the f_b normalisation are decided in S10. |
| `accelerations(topology, params)` | Takes no positions or masses (lead's decision): the rebuild already ran M2L and L2L at its positions. S8 adds live overrides. |
| carry | The library default stays `"state"`; S3 decides the preset's carry. |

### What S2 built

| branch | commits | what |
| --- | --- | --- |
| `s2/1-bench-pins` | `6fef4ba` | pins S1p, S1t, S1e, S1x, S1g (`bench/dce_pins.py`) |
| `s2/2-hlo-identity-tool` | `c403015`, `c10d826` | `bench/provider_hlo_identity.py` and its tests |
| `s2/3-fused-topology` | `aeb9da8` | `runtime/fused_topology.py`: `FusedTopology`, the overflow bits, `ForceParams`, `StepParams` |
| `s2/4-fused-provider` | `e1b0035` | `runtime/fused_provider.py`: `FusedForceProvider` (no callers yet) |
| `s2/5-strict-steps-on-provider` | `efd2354` | the scan's step closures on the provider (C2) |
| `s2/6-strict-runners-on-provider` | `a1d8786` | the runners on it: compile keys, re-plan, handle, eval seam (C3) |
| `s2/7-strict-fixes` | `4bb1e5e`, `c3fadd4`, `3acbb09`, `c7ab4e4` | bucket leaves refused on the particle carry; the `dehnen_error` chain; slow markers; the tree-overflow verdict |
| `s2/8-fp64-kernel-sweep` | `e81f9ca` | every fused-lane Pallas kernel at float64 against its XLA lane |
| `s2/9-fp64-validation-lane` | `28a8dfe`, `12ad662` | `precision="fp64"` on `large_n_gpu`; `fused_memory_budget.py --state-dtype / --force-dtype` |
| `s2/10-tree-order-fp64` | `86844a4` | the tree-order carry keeps a float64 state |
| `s2/11-traced-params-runtime` | `9aee0e1` | G and eps^2 as arguments of the fused evaluation |
| `s2/12-traced-params-strict` | `4f7f606`, `475b61e` | `strict_run_v2(force_params=, external_params=, dt=array)`; `fused_memory_budget.py --traced-params` |
| `s2/13-docs-and-record` | | the user-scan test, `docs/fused_provider.md`, this section |

### `FusedForceProvider`: the fused lane's force, public
- **Two halves.** `rebuild(pos, m)` is the tree from a shape template, P2M/M2M, the flat walk, M2L and L2L.
  `accelerations(topology, params)` is the near field (P2P), L2P and the scatter. `step(pos, m, params, aux)` is both,
  with the overflow bits. `jaccpot.FusedForceProvider` and `jaccpot.ForceParams` are exported; the guide is
  [`fused_provider.md`](fused_provider.md).
- **`strict_run_v2` runs on it.** Every refresh, evaluation, verdict, compile key and segment re-plan of the fused scan
  goes through the provider: both carries, the tree order, the particle carry's start program, the eval seam.
- **A caller's integrator gets the lane's bits.** A velocity Verlet written in five lines over `prov.step` (a start
  force, then a `lax.scan` of drift, step, kick) reproduces `strict_run_v2(carry="particles")` bitwise (CPU, N = 512,
  3 steps; `test_strict_runners_on_provider.py`). The masses must be an argument of the caller's jit, as they are of
  the lane's runner: closed over, they are a program constant, XLA compiles a different program, and one float32
  rounding of one force component moved at step 3 (1 of 3,072 state elements, 1.5e-8; steps 1-2 bitwise either way).
  Carrying `x`, `v` apart or as the packed `[N, 2, 3]` state makes no difference. S3 and S4 build on this surface.

### Findings that changed the plan
- **"Refresh" is the whole far field.** The in-scan refresh is tree + P2M/M2M + walk + M2L + L2L; "evaluate" is only
  near P2P + L2P + scatter. So `accelerations` takes no positions: at other positions its far field would be stale.
- **The fp64 force was blocked by one guard.** `_large_n_nearfield.py` refused any working dtype but float32 (the
  preset defaults to float32, so an explicit float64 raised). Every kernel turned out dtype-generic (below), so S2 lifted the guard and routed nothing to XLA.
- **The tree-order carry rounded a float64 state to float32 at every step.** `_advance_tree_order` took the new rows
  from the refresh's `positions_sorted`, which are in the force's dtype; a 1e-12 shift of one particle was lost. It now
  gathers from the carry when the dtypes differ. The float32 programs are unchanged (36/36 identical), and S1t and
  S1p are bitwise.
- **The pin gap.** S1, S6 and DET1 run the state carry only. The particle carry, the tree order, `dehnen_error`, an
  external field and G != 1 had no bitwise pin; C0 added S1p, S1t, S1e, S1x and S1g.
- **`dehnen_error`: chained state-carry calls were not one call.** A call continuing from a returned prepared state
  reseeded f_b from the engine's last eager prepass: the first call's start positions, in that tree's order. 1 + 2
  steps differed from 3 in 1,349 of 3,072 elements (max 5.7e-5; CPU, N = 512). The identity tool's chain variants
  found it. The returned state now holds the scan's last f_b. This moves S1e's state-carry state, whose bench chains
  calls (749,302 of 1,200,000 elements, max 4.27e-4, rel-L2 6.8e-7), as intended; nothing else moves.
- **Bucket leaves cannot run the particle carry.** Its template rebuild reads the eager tree's node ranges and
  children, which the template holds as zeros: the walk's queue overflowed on every step (peak wavefront
  177,147 = 3^11 at N = 512) and the retry failed the same way. It is now refused before any prepare, with the remedy
  (cell leaves, or `carry="state"`, which runs on buckets).
- **A tree overflow inside the scan gave silently wrong forces.** The traced refresh never folded the cell-leaf
  rebuild's own overflow (a leaf over `leaf_capacity`, a tree deeper than the level-loop bound, a level wider than the
  level loops' batch) into the verdict. On the CSR near field nothing else failed. Cells sized for 66 leaves and
  stepped on 105 live ones (N = 512): verdict `True`, force rel-L2 0.546. `OVERFLOW_TREE` now fails the verdict, and
  the call raises naming `leaf_capacity`; it is not re-planned. Cost: +22 StableHLO ops per runner (on a GPU at 2e5,
  +40 compiled HLO instructions in the state runner, +53 in the particle runner and its start program); every value
  is bitwise.

### Gates
**Program identity** (`bench/provider_hlo_identity.py`). It records every jitted program a `strict_run_v2` variant runs
(lowered text, and on a GPU the compiled HLO) and the sha256 of every array: 36 variants on CPU at N = 512 (534 s at
`--jobs 4`), 18 on a GPU at 2e5. A negative control (reassociating the drift) fails it. GPU compares need `--control`:
the eager walk ladder's raw outputs are order-nondeterministic, everything after them bitwise. Triton IR embeds the
kernel source paths, so record base and candidate from the same path.

| step | CPU programs (36) | GPU programs (18) | GPU pins S1, S6, DET1, G1, S1p, S1t, S1e, S1x, S1g |
| --- | --- | --- | --- |
| C2: step closures on the provider | identical | identical | 9/9 bitwise |
| C3: runners on the provider | identical | identical | 9/9 bitwise |
| `dehnen_error` chain fix | 35/36; `dehnen_error_chain_state`'s second call moves | 17/18, the same | 8/9; S1e's state-carry state moves |
| tree-overflow verdict | the runners' text (+22 ops); values bitwise, `needs` 7 -> 8 entries | the same | 9/9 bitwise |
| tree-order fp64 fix | identical | | S1t, S1p bitwise |
| traced parameters, default path | identical | identical | 9/9 bitwise |

**New pins** (`bench/dce_pins.py`, 2e5, each bitwise A-vs-A, 0 elements):

| pin | what | rel-L2 |
| --- | --- | --- |
| S1p | S1 on the particle carry | force and final state bitwise S1's |
| S1t | S1p in tree order | force and final state bitwise S1's |
| S1e | `dehnen_error`, `adaptive_eps` 1e-4, both carries: 3,350,146 far pairs (geometric: 1,876,324) | 1.45e-4 (geometric: 7.76e-4) |
| S1x | a 2e5 prefix of the 25M disc+bulge in its NFW halo (external field traced in the scan), eps 1.5e-3, both carries | 8.24e-4 |
| S1g | G = 4.3e-6, masses divided by G | 7.7637e-4 (= S1) |

**L1-L3** (free GPU 7, A-B-A, 2026-10-16, every row clean; `s2/7` tip `c7ab4e4` against main `f967805`):

| pin | main | `s2/7` tip | main again (A-vs-A) |
| --- | --- | --- | --- |
| L1, 8e6 step | 78.36 ms, 317 B/p | 78.64 ms (+0.35 %), 318 B/p | 78.50 ms (+0.17 %) |
| L2, 1e8 step | 873.27 ms, 257 B/p | 873.41 ms (+0.02 %), 257 B/p | 871.29 ms (-0.23 %) |
| L3, 25M disc step | 238.57 ms, 255 B/p | 238.88 ms (+0.13 %), 255 B/p | 240.30 ms (+0.72 %) |
| L3 force | 100.17 ms, rel-L2 4.179e-4 | 100.49 ms (+0.33 %), 4.179e-4 | 100.21 ms (+0.05 %) |

`compare --envelope` passes L1, L2 and L3: the provider costs nothing measurable. `s2/8`-`s2/12` leave the float32
programs identical (above), so the row stands for them.

### The fp64 validation lane
- **Every fused-lane Pallas kernel runs float64 natively** (A100): the walk (`soa` nodes), the COM radii, the leaf
  P2M, the M2M/L2L and M2L CSR lanes, the L2P, the CSR near field's table and direct kernels. 23 native cases, the
  largest error 6.4e-15 of the XLA reference. **No XLA fallback was needed**; the API is the facade's
  (`precision="fp64"` / `working_dtype=jnp.float64`), and only the guard moved.
- **Accuracy:**

| configuration | float64 | float32 |
| --- | --- | --- |
| CPU, p12, theta 0.4, unsoftened, vs direct sum | 7.93e-10 | 9.20e-6 |
| CPU, exact FMM (theta 1e-3): Verlet vs direct-sum Verlet | 1.11e-15 | 1.43e-4 |
| GPU 2e5, the S1 configuration (truncation-bound) | 7.7656e-4 | 7.7637e-4 |
| GPU 2e5, p8, theta 0.4 | 6.434e-7 | 6.946e-7 |

- **Memory at 2e6** (shared card, `fused_memory_budget.py` defaults, no donation), state / force bits: 32/32 508 B/p,
  64/32 632, 32/64 707, 64/64 820.
- **Compile time grows steeply with the order:** the M2M/L2L cascade kernels' Triton compile takes 140 s at p8 (in
  either dtype); at p12 in float64 it had not finished after 12 minutes. GPU validation at p12 needs
  `JACCPOT_CASCADE_PALLAS=0` (the cascades as XLA level loops).

<!-- LEAD: precision rows -->

### Traced parameters: static against traced
- **API.** `strict_run_v2(dt=float | Array, force_params=ForceParams(G, eps2), external_params=pytree)` and
  `strict_fused_prepared_eval_fn(force_params=)`. Floats stay literals (no extra argument: every program unchanged).
  Arrays ride in one `StepParams`, the last argument of the scan and of the particle carry's start program, after the
  donated ones. Three calls with three different (dt, G, eps2, external) values: one key, one executable, one trace.
- **Refusals before any work:** `dehnen_error` with anything but the solver's own G and softening; an eps^2 beyond the
  walk's under a compact kernel (`OVERFLOW_PARAMS`; `provider.step` sets the bit for a traced caller); external
  parameters without an external field. A particle-carry handle records its force parameters (`force_key`), and a
  call continuing with others evaluates its start force again.
- **GPU A/B (deterministic ops, A-vs-A control 0 elements):**

| pin | traced against static |
| --- | --- |
| S1 | force 0 of 600,000 elements, final state 0 of 1,200,000 |
| S1g | force 0 of 600,000, final state 0 of 1,200,000 |
| L3 force (25M, ferrers3, eps 1.5e-3) | 0 of 75,000,000; rel-L2 4.179e-4 in every arm |

  **On the GPU production path traced is bitwise static.**
- **CPU is not bitwise under ferrers3.** Plummer: bitwise. ferrers3 (the library default): 753 of 1,536 elements on
  the pure-JAX near field and 732 on the CSR lane (interpret mode) differ by one float32 rounding, at most 1.35e-7 of
  the largest component. The lowered programs differ only in the constant becoming a parameter. XLA's algebraic
  simplifier then rewrites `rsqrt(x)^3` into `rsqrt(x) / x` only where it can prove `x = max(r^2, h^2) > 0`, which a
  literal `h^2` gives it and a parameter does not. The tree-order carry with any one traced argument moves 2 of 3,072
  velocity components on CPU (2.4e-7 and 6e-8: 1 and 8 ulps of their own values, the second a kick that nearly
  cancels), with the positions bitwise: the extra argument changes XLA's compilation, not the arithmetic.
- **Linearity and gradients (CPU):** `acc(2G) == 2 acc(G)` bitwise under one compile. `jax.grad` in G against central
  differences: 1e-10; in eps^2: 1e-6 (float64 force, pure-JAX near field). Not verified on the Pallas lane, whose
  production forward has two kernels without a reverse rule.

<!-- LEAD: traced timing -->

### For S3
- **Leaf capacity:** size the cell-leaf capacity automatically, with headroom over the live cells the run will reach,
  and make `OVERFLOW_TREE` re-plannable at the segment start. Today a tree overflow raises (the S2 fix makes it loud
  instead of wrong) and the bench sizes the capacity by hand.
- **Traced parameters as the default:** consider flipping it. On the GPU the traced programs are bitwise the static
  ones (S1, S1g, the 25M force), and a run that varies dt, G or eps then compiles once instead of once per value. On
  CPU, ferrers3 results would move by one float32 rounding (the lane goldens run Plummer, which stays bitwise). The
  traced timing row above decides the cost.
- **The preset's carry** (`"state"` or `"particles"`), deferred to S3 by the user.
