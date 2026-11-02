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
The work is a linear stack on `f967805` (`s2/1` to `s2/13`, table below), published as five stacked PRs:
#408 (`s2/1`-`s2/2`, bench), #409 (`s2/3`-`s2/4`, provider), #410 (`s2/5`-`s2/7`, strict lane on it + fixes),
#411 (`s2/8`-`s2/10`, fp64), #412 (`s2/11`-`s2/13`, traced parameters, docs, this record). GPU runs used shared A100s
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
| `s2/13-docs-and-record` | `17e1cca`, `d1ca5b1`, `7bdd90f`, `4ba7bc2` + the timing rows | the user-scan test, `docs/fused_provider.md`, this section |

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

**Time and memory per mode** (free GPU 7, 2026-10-16, every row clean; the L-pin settings: clipped Plummer, cells of
64, p6, theta 0.8, particle carry, unnamed caps, arena 0.88, no donation; the `s2/12` tip; step = min of 3 x 4 steps;
rel-L2 on 4096 targets):

| state / force | 8e6 step | 8e6 B/p | 8e6 force eval, rel-L2 | 1e8 step | 1e8 peak, B/p |
| --- | --- | --- | --- | --- | --- |
| fp32 / fp32 | 78.7 ms | 325 | 27.8 ms, 4.333e-4 | 872.0 ms | 23.9 GiB, 257 |
| fp64 / fp32 | 81.4 ms (+3 %) | 442 | 28.3 ms, 4.333e-4 | out of memory (a 14.5 GiB request) | |
| fp32 / fp64 | 151.5 ms (1.92x) | 558 | 66.4 ms, 4.339e-4 | 1721.8 ms (1.97x) | 31.9 GiB, 342 |
| fp64 / fp64 | 150.7 ms (1.91x) | 575 | 66.6 ms, 4.339e-4 | out of memory (a 20.9 GiB request) | |

- **The largest N that fits** (same settings): fp64 state with fp32 force 8e7 (722.7 ms, 406 B/p; 9e7 out of memory);
  fp64 / fp64 at least 7e7 (1220.1 ms, 479 B/p; the first size tried). Donation (`--donate-state`) was not tried.
- **fp64 force costs about 2x the step** and 1.3x the memory at 1e8. **An fp64 state costs 3 % of the step** but
  about +117 B/p, which is what stops it below 1e8 on a 40 GB card.
- **At the production configuration fp64 buys no accuracy:** 4.333e-4 against 4.339e-4. The error is the expansion's
  truncation (p6, theta 0.8), not round-off; fp64 matters where the expansion is pushed (p8 theta 0.4: 6.4e-7 against
  6.9e-7) or exact (1.1e-15 against 1.4e-4).

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

**Timing** (free GPU 7, the same settings as the precision rows, `--traced-params`: G, eps^2 and dt as arguments):

| | static | traced |
| --- | --- | --- |
| 8e6 step | 78.7 ms / 79.3 ms (two runs) | 79.3 ms (325 B/p) |
| 1e8 step | 872.0 ms (257 B/p) | 873.2 ms (+0.13 %, 257 B/p) |

Traced parameters cost nothing measurable: the difference is inside the run-to-run spread (the A-vs-A of the same
day's L-pins is 0.2-0.7 %).

### For S3
- **Leaf capacity:** size the cell-leaf capacity automatically, with headroom over the live cells the run will reach,
  and make `OVERFLOW_TREE` re-plannable at the segment start. Today a tree overflow raises (the S2 fix makes it loud
  instead of wrong) and the bench sizes the capacity by hand.
- **Traced parameters as the default:** consider flipping it. On the GPU the traced programs are bitwise the static
  ones (S1, S1g, the 25M force), and a run that varies dt, G or eps then compiles once instead of once per value. On
  CPU, ferrers3 results would move by one float32 rounding (the lane goldens run Plummer, which stays bitwise). The
  traced timing above shows no cost.
- **The preset's carry** (`"state"` or `"particles"`), deferred to S3 by the user.

## S3a — the production runner, `preset="galaxy"`, automatic leaf capacity, the tree re-plan (2026-10-16)

**Setup.** jaccpot main `dd1ac36` (S2 merged), yggdrax main `3299b70` plus `return_num_leaves`; jax 0.11.2. Design and
user decisions: the plan's S3 design note. Timings on free A100s booked through autocvd (no foreign process on the
card); accuracy, identity and pins on shared ones.

**The user's decisions (2026-10-16):**

| | decision |
| --- | --- |
| traced parameters | the **runner's** default (`SegmentConfig.traced_params=True`); `strict_run_v2` keeps static literals |
| `preset="galaxy"` MAC | geometric theta 0.8 (`dehnen_error` only once S3b brings its step within 1.3x) |
| diagnostics | momentum, angular momentum, COM, kinetic energy every segment edge; energy every 10th edge (session 2) |
| ferrers3 tile skip | opt-in unless bitwise |
| restart files, carry | `.npz` + JSON meta (tmp + rename); the preset's carry is `"particles"` (lead, per the design) |

### What S3a built
| branch | what |
| --- | --- |
| yggdrax `feat/s3-return-num-leaves` (yggdrax #101) | the cell template rebuild can return the partition's live leaf count (computed, dropped before) |
| `s3/1-auto-leaf-capacity` (#414) | `TreeConfig(leaf_capacity="auto", leaf_headroom=)`: the live cell leaves counted ON DEVICE, x1.15 in steps of 1024, once, at the first prepare |
| `s3/2-galaxy-preset` (#415) | `preset="galaxy"` (`GalaxySettings`): cells of 64, cml 8, p6, theta 0.8, ferrers3, auto capacity, particle carry; `theta=None` sentinel (0.6 for every other preset) |
| `s3/3-tree-replan` (#416) | a cell-leaf tree that outgrows its leaf capacity, its depth bound or its level width is re-planned at the segment's start (needs 8 -> 11 fields); after a tree re-plan, one walk-only retry more |
| `s3/4-production-runner` (#417) | `ProductionRunner`: K-step `strict_run_v2(carry="particles")` segments chained through the handle; callbacks, snapshots, a jsonl log; `carry_total_acceleration` |
| `s3/5-runner-restart` (#418) | restart files: a fresh process continues a run bitwise |
| `s3/6-runner-bench-docs` (#419) | `fused_memory_budget.py --runner`, pins R1p / R1x / R1r / DET3, `docs/production.md`, this section |

### Findings
- **The leaf capacity is not what a production run outgrows first.** Over a 25M disc+bulge quarter orbit (489 steps)
  the live cell leaves moved +0.26 % (1,502,983 -> 1,506,881): 1.15x headroom lasts thousands of steps. The widest
  tree level grew +10.9 % against its 1.25x bound (extrapolated trip at ~1,700-2,600 steps). The re-plan therefore
  grows the level width and the depth bound as well as the capacity.
- **The device count is ~400x the host count:** 25M 8-14 ms (host 3.2 s), 1e8 27 ms (host 10.9 s), ~25 B/p transient,
  equal on all 72 (IC, N, leaf, cml) rows.
- **`preset="galaxy"` compiles the bench's explicit L-configuration's programs** (CPU 12/12 programs, 84/84 arrays; GPU
  14/14 texts); a theta 0.75 control fails the compare.
- **Two latent faults in a width-only re-plan**, found by the R6 tests: the level batch width is read at trace time and
  is not in any compile key (the runner cache is now dropped on a re-plan), and the Pallas L2L cascade loops the shape's
  registered level count, so a deeper depth bound alone left levels unswept (it is raised with the bound).
- **An external field across segment edges** was not bitwise: a chained call rebuilt `acc0 = acc_self + ext(state)`
  outside the scan (1 of 3,072 components, 1.9e-6, CPU). `strict_run_v2(carry_total_acceleration=True)` (opt-in; the
  runner uses it) carries the scan's own total; it also removes a second compile at segment 2.
- **A ragged last segment recompiles** (`num_steps` is in the key): 53 s at 25M. The runner splits into equal segments
  and warns once on a remainder.

### Gates
- CPU: every CI shard exits 0 on the integration and on each stack branch; `provider_hlo_identity.py` 36/36 identical
  on `s3/1`-`s3/2` (vs S2) and on `s3/4`-`s3/6` (vs the integration); `s3/3` changes the runners' text only (+5 StableHLO
  ops, the `needs` vector 8 -> 11), every value bitwise.
- GPU: pins S1, S6, DET1, G1, S1p, S1t, S1e, S1x, S1g bitwise; R1p (the runner) bitwise to S1p; R1x (external field,
  chained = one call) bitwise; R1r (restart in a fresh process = straight) bitwise; DET3 A-vs-A bitwise. Two cards: M1
  and M3 bitwise, M2 within main's A-vs-A envelope.
- R6: a tree outgrown mid-segment (leaf capacity, or level width) is re-planned and equals a fresh run with the grown
  bounds, bitwise; with unnamed walk caps the walk-only second retry finishes it, bitwise to a fresh run with the final
  caps. The runner driven through a re-plan in its second segment equals one `strict_run_v2` call, bitwise.

### Timing (free GPU 0, every row clean; L-pin settings, particle carry)
| pin | main `dd1ac36` | S3a | main again (A-vs-A) |
| --- | --- | --- | --- |
| L1, 8e6 step | 78.82 ms, 325 B/p | 78.81 ms (-0.01 %) | +1.84 % |
| L2, 1e8 step | 875.18 ms, 257 B/p | 877.23 ms (+0.23 %) | +0.24 % |
| L3, 25M step | 241.92 ms, 256 B/p | 241.04 ms (-0.37 %) | +0.42 % |
| L3 force | 101.03 ms, rel-L2 4.18e-4 | 100.27 ms | -0.61 % |

**Through the runner** (segments of 50 steps, traced parameters, donation; the per-step time includes the edges):

| N | plain `strict_run_v2` | runner | runner, diagnostics off |
| --- | --- | --- | --- |
| 8e6 | 78.5 ms, 324 B/p | 84.1 ms (+7.2 %), 320 B/p | 82.3 ms (+4.8 %) |
| 1e8 | 889.9 ms, 257 B/p | **919.4 ms (+3.3 %)**, 255 B/p | 909.2 ms (+2.2 %) |

**1e8 stays below 1 s per step through the runner.** The edges cost ~0.3 s per segment at 8e6 and ~0.5-1.5 s at 1e8: the
24 B/p host copy `strict_run_v2(donate_state=True)` keeps for its retry, and the diagnostics (~0.5 s per edge at 1e8,
more than four fp64 reductions should take). Session 2 makes both cheaper.

### The 25M disc+bulge through the runner (free GPU 7)
`preset="galaxy"`, eps 1.5e-3, the NFW halo as an external field, dt 5e-4, 500 steps in 10 segments of 50, traced
parameters, a restart file every 5 segments, snapshots at 250 and 500:
- **354 s for the 500 steps**, edges, two restarts and two snapshots included; 246.8 ms/step in the segment calls,
  249.0 ms with the edges. No re-plan, no recompile after the first segment. The automatic capacity resolved to the
  bench's 1,728,512 (live 1,502,983 -> 1,507,252, at most 0.872 of it).
- **A kill and resume** from the step-250 restart in a fresh process ends bitwise on the uninterrupted run (0 of
  1.5e8 elements; both digests matched).
- |dL|/L = 4.97e-7 after 500 steps; peak 5.52 GiB (237 B/p), set by the eager prepare. The mesh lane ran this disc at
  82 s/step on 8 cards (another configuration; this force is the more accurate one).
- **Costs to remove in session 2:** the first segment, 232 s (the eager prepare 96 s and 333 compiles on a loaded host);
  a restore, 101 s, almost all of it a full eager prepare (the caps-only prepare); a restart file is 1.3 GB (52 B/p) and
  takes 3-3.7 s to write.

### For S3 session 2
- the fast-lane O(N) potential and the energy diagnostic (the generic potential path needs a 30.75 GiB buffer at 25M);
- the caps-only prepare (first segment, restore, re-plan);
- cheaper edges (diagnostics, the retry's host copy);
- the 25M production gate with energy, and a 1e8 run through the runner with energy on;
- the ferrers3 tile skip (opt-in unless bitwise).

## S3 session 2 — the energy, the caps-only prepare, the production gate (2026-10-16/17)

Published as seven stacked PRs on main after #421 (the 1-step-segment hotfix): #428 fast-lane potential, #422 runner
energy, #423 caps-only walk, #424 caps-only default, #425 ferrers3 tile skip, #426 int32 list limit, #427 the
autotuner flag and this record.

**Setup.** jaccpot main `392061d` (S3a merged) with #421, yggdrax main `ca3a93c`, jax 0.11.2. Four tracks on
branches of their own (the potential and the energy, the caps-only prepare, the ferrers3 tile skip, the int32 list
limit), merged into one integration for the gates. Timings on free A100s booked through autocvd (no foreign process
on the card in any timed row; the host's load, 18-78, and the busy other cards are flagged on every row); accuracy,
identity and pins on shared cards.

**The user's decisions (2026-10-17):**

| | decision |
| --- | --- |
| ferrers3 tile skip | opt-in (`JACCPOT_NEARFIELD_TILE_SKIP=1`), off by default: bitwise, but +1.8 % per step on the production disc |
| XLA's fusion autotuner | off in production: `--xla_gpu_experimental_enable_fusion_autotuner=false` in the launch environment. The library cannot set it (XLA reads its flags when the backend starts), so the runner recommends it and warns |

### What session 2 built
| branch | what |
| --- | --- |
| `fix/runner-no-one-step-segment` (#421) | a 1-step remainder joins the last segment (5 steps of 2 run as 2 + 3) |
| `s32p/1-fastlane-potential` | `FusedForceProvider.potential`: the fast lane's O(N) potential under a compile key of its own; the near field from the CSR direct kernel's potential lane, the far field from the XLA particle-major L2P's value |
| `s32p/2-runner-energy` | the runner's energy `E = K + W_self + W_ext` (fp64, every 10th edge, `external_potential_fn`); the retry's start copy into pinned host memory, asynchronously; the energy program compiles once |
| `s32p/3-caps-only-walk` | the strict flat walk's eager ladder can stop before its lists; `prepare_state_caps_only` |
| `s32p/4-caps-only-default` | `FusedForceProvider.prepare(caps_only=True)` (the template from `jax.eval_shape`); the particle carry bootstraps, retries and restores caps-only (`JACCPOT_STRICT_CAPS_ONLY_PREPARE=0`: the full prepare); the identity A/B of a library switch |
| `s32p/5-ferrers3-tile-skip` | the ferrers3 tile skip, opt-in |
| `s32p/6-replan-int32-limit` | list re-plans stop at the int32 index range, and a list past it raises a clear error; a wrapped walk count raises instead of looping |
| `s32p/7-autotuner-and-record` | the launch environment (the runner's warning, the benches' flag, `docs/production.md`); this section |

### Findings that changed the plan
- **A scan of length one is other code.** After #419, main's CI failed `2+2+1_vs_5` on one runner. XLA inlines a
  1-step scan into straight-line code and fuses it differently (one fusion against the loop body's three, CPU toy),
  so the remainder step rounded differently on that runner's CPU. #421 gives a 1-step remainder to the last segment;
  `SegmentConfig(steps=1)` is one long call to round-off only.
- **The S3 1e8 gate outgrew the int32 index range because its Morton box stretched.** Run for 451 steps, the S3a
  sequence crashed (main too): the re-plans grew the far list from 302M to 474M and 754M canonical pairs, and
  building the last raised `pair width 754450432 overflows int32`. In the clipped Plummer (softening 1e-7, dt
  1e-2) close encounters fling a few particles out from step ~200; the per-axis box elongates every cell and the
  far need went 402M -> 632M -> 1,005M directed entries by step 401. At 1e7 on the same IC the far count grows 44 %
  by step 451 (47,747,880 -> 69,051,414); with `JACCPOT_CUBIC_BOUNDS=1` the same positions need 37,308,898, **22 %
  fewer than at step 0**.
- **XLA's fusion autotuner made restarts on shared cards non-bitwise.** It times the native against the block-level
  (Triton) emitter for some fusions, so a fresh process on a busy card can compile other code from the same HLO
  (below). Off, every resume and DET2 were bitwise, at no measurable cost: the user's decision above.
- **The energy program compiled twice** (51.6 + 46.1 s at 25M): step 0's state reached it committed (a host copy put
  back with the masses' sharding), the segments' states uncommitted, and jit keeps one executable per commitment.
  Every state now goes to the masses' sharding.
- **The edges' cost at 1e8 was the retry's host copy**, not the diagnostics: a synchronous 2.4 GB `device_get` into
  pageable memory, 0.99-1.38 s per segment, against 13-23 ms for the moments. Pinned and asynchronous it takes
  0.10 s (the first of a run 1.4 s, the pinned allocation); segments 2-5 of 10 steps at 1e8: 9.83-10.57 s ->
  8.79-8.86 s.

### The potential and the energy
- `FusedForceProvider.potential(topology, params)` against an fp64 direct sum: rel-L2 3.4e-4 (CPU, N = 512, Plummer,
  pure-JAX near field), 7.7e-5 (ferrers3, the CSR kernel in interpret mode), 1.02e-5 on an A100 at 2e5
  (`preset="galaxy"`, 4096 targets; the force: 8.0e-4). On a contended A100: 25M 247 ms and 133 B/p of temporaries
  (a force evaluation 220 ms, 143 B/p), 1e8 960 ms and 123 B/p (851 ms, 133 B/p). The generic path it replaces asked
  for a 30.75 GiB buffer at 25M.
- The energy, CPU N = 512: `W_self` within 9.5e-6 of the fp64 direct sum at step 0 (2.6e-6 at step 6); with the
  energy at every edge the runner is one long call bitwise (R4, the same two scan programs), and a fresh process
  continues it bitwise (R5). Its trace leaves the engine's trace-time state as it found it.
- Energy conservation, A100, Plummer 2e5 (eps 0.05), max |dE/E0| over the run: it falls with dt down to a floor of
  4-5e-7.

| dt (steps) | 0.16 (8) | 0.08 (16) | 0.04 (32) | 0.02 (50) | 0.01 (100) | 0.005 (200) |
| --- | --- | --- | --- | --- | --- | --- |
| max \|dE/E0\| | 1.77e-5 | 2.22e-6 | 6.55e-7 | 4.25e-7 | 4.08e-7 | 5.18e-7 |

### The caps-only prepare
The particle carry reads four things of its eager prepare: the template's shapes, the engine's trace-time state
(walk caps, level registry, depth bound, the near field's static caps), the initial verdict and, under
`dehnen_error`, the f_b seed. The caps-only prepare builds the tree and the upward pass and runs the eager walk's
ladder (which records the caps); it places no list, no M2L or L2L, no near payload and no far f_b, and takes the
template from a rebuild under `jax.eval_shape`. The template key, the provider key, the verdict and the trace-time
state equal the full prepare's (CPU: pure-JAX, CSR, `dehnen_error`; A100: 2e5, 8e6, the 25M disc), so it is the
particle carry's default for the bootstrap, the segment retry and the restore. Free A100, L-pin rows, main against
the branch:

| N | prepare peak (B/p) | prepare wall | step |
| --- | --- | --- | --- |
| 8e6 | 325.5 -> 242.3 | 78.8 -> 60.2 s | 78.72 -> 78.18 ms |
| 25M disc | 243.6 -> 184.9 | 76.6 -> 62.2 s | 239.54 -> 239.54 ms |
| 1e8 | 253.0 -> 193.2 | 80.5 -> 63.2 s | 872.55 -> 872.81 ms |

A 25M restore takes 60.4 s (80.0). **The runner's one-card ceiling with donation is 1.92e8 particles at 175.2 B/p**
(31.32 GiB; before: 1.68e8 at 214 B/p, set by the full prepare). What binds now is the tree and upward pass's
transient peak and the walk ladder's queue doublings after it (1.96e8 runs out there). The rows in full:
`docs/fused_memory_2026-10.md`, "S3 session 2".

### The 25M production run (free A100; the integration with the tile skip off)
`preset="galaxy"`, the 25,000,000-particle disc+bulge, the NFW halo as field and potential, ferrers3 eps 1.5e-3, dt
5e-4, 500 steps in 10 segments of 50, traced parameters, `donate_input`, the moments at every edge, the energy every
10th, a restart every 5 segments, snapshots at 250 and 500:
- **238.6 ms/step** in the segment calls (median; min 238.18; the S3a dry run 246.8), 238.8 ms with the edges.
- **`run()` 423.6 s** (S3a 354.1): the stepping 305.7 s, plus the energy program's two compiles (51.6 + 46.1 s,
  the double compile fixed since). The first segment, with the caps-only bootstrap, 198.2 s (S3a 232.0): the prepare
  72.5 s (S3a's eager prepare 95.7 s), 285 compiles in 205.2 s. At host load 18 instead of 78: 161.7 s, a 57.9 s
  prepare.
- **Energy:** E0 = -322.42190 (K0 37.9065, W_self -44.3289, W_ext -315.9995); **dE/E0 = -2.08e-7 at step 500** (the
  run's one due energy edge). With the energy at every edge (150 steps): +6.4e-8, +1.2e-7, +7.4e-9 at steps 50,
  100, 150. A warm energy edge costs 0.268 s (1.12 steps).
- |dL|/|L0| 4.97e-7 at step 500 (max 5.89e-7); COM drift 5.68e-5; KE +7.69 % (the IC's disequilibrium, as in the
  S3 probe). No re-plan, no retry; live cell leaves 1,495,590-1,507,252 of 1,728,512.
- **Peak 4.549 GiB = 195.4 B/p** (S3a 237.1). A restart file (1.30 GB) in 2.8-3.0 s, a snapshot (0.6 GB) in 1.2 s.
- **The final state is bitwise the S3a dry run's** (0 of 1.5e8 elements): the energy, the caps-only prepare and the
  pinned copy change no number.
- **Kill and resume** from the step-250 restart in a fresh process: the digests matched, the final state differs in
  0 of 1.5e8 elements and the step-500 restart file is the same (arrays and meta). Restore 68.9 s before the first
  segment (S3a 100.9), first segment 63.9 s (71.0); `run()` 237.4 s for 250 steps, a 52.8 s energy compile included.

### 1e8 through the runner (free A100; clipped Plummer, segments of 50, the S3a sequence 1 + 50 + 3 x 50 steps)
| row | ms/step (min of 3 calls) | S3a | B/p | edge per segment |
| --- | --- | --- | --- | --- |
| runner, diagnostics and energy | **873.75** | 919.4 | 208.77 | 0.015 s (S3a 0.23-0.31) |
| runner, `--runner-diagnostics-every 0` | 873.84 | 909.2 | 208.77 | 0.002 s |
| plain `strict_run_v2` | 871.69 | 889.9 | 256.77 | - |

**1e8 stays below 1 s per step through the runner, 0.24 % over the plain call.** The energy's first due edge comes
after step 201; E0 took 51.7 s at the first edge, untimed (its compile included). Past ~250 steps this IC stretches
its box (above): with the clamp below the 451-step sequence runs to step 551 without another re-plan, the step
following the box's aspect (874 ms cubic, 1,596 ms stretched at steps 452-501, 1,004 ms at 502-551, once particles
flung along z had made the box cubic again).

### The int32 list limit
- `_interaction_cache.flat_walk_list_limit`: the largest directed list the index range builds, 1,431,306,240 entries
  on the Pallas walk, whose CSR placement addresses 3W slots for a canonical width W (2,146,435,072 for the flat
  walk's interleaved lists). The re-plan's 1.5x headroom and the eager ladder stop there; a list that needs more
  raises a `RuntimeError` before the re-prepare, naming the remedies (theta or `adaptive_eps`, `leaf_size`, more
  GPUs, `JACCPOT_CUBIC_BOUNDS=1`), and the runner adds its last restart. The eager ladder's 2^30 ceilings are gone
  (they refused needs between 2^30/1.5 and the range, which fit). Below the limit every capacity is the old one; the
  checks run on the host, and no traced program changes.
- At the limit the lane is correct: 2e6 forces with the far cap named at the limit are bitwise those at the sized
  caps (0 of 6M elements; A-vs-A 0); one granule past it raises.
- A walk that counted past 2^31 - 1 pairs wrapped its int32 counter negative, and the ladder sized its next rung
  from it and walked again, forever; it raises now. The same wrap writes out of bounds inside the kernel (`slot <
  cap` accepts a negative slot), which only a kernel guard can stop (it changes the traced program; not done).
- **Lifting the limit** (not done; not needed now). What is int32 today:
  - the Pallas walk hard-codes it and bitcasts node ids into float32 records, so an int64 walk would not compile;
  - its counters and slots are int32, and `slot < cap` has no `slot >= 0` guard, so a counter that wrapped would
    write out of bounds (a kernel fix that changes the program);
  - `walk_needs` doubles the far count in int32 (it wraps at 2^30 canonical pairs);
  - `csr_place` uses int32 offsets and float32 row cursors (exact to 2^24 entries per row), and the XLA placement's
    `2W + i` is the 3W bound;
  - `csr_by_target` casts to int32, and the M2L reverse sorts 2W entries with int32 keys.
- **Memory per canonical pair, int32:** far 8 B in the walk, 16 B at the CSR build's peak, 8 B live through M2L;
  near about the same. int64 would roughly double the build and live parts. At W = 1e9 that is 16 GB transient and
  8 GB live for the far list alone, so on a 40 GB card memory binds near today's index limit anyway. With the 1e8
  far list at the limit, XLA's rematerialisation brought the step's programs down to 32.7-37.2 GiB only.
- **The options:**
  1. A route-aware 3W check (2W plus lanes on the Pallas placement) reaches about 1.07e9 canonical pairs and keeps
     the default program, but is safe only with the kernel's `slot >= 0` guard and saturating traced needs, which
     change programs.
  2. int64 offsets behind a static switch touch about 10 files; the walk still caps W.
  3. A lo/hi row split keeps int32 to W < 2^31 but changes the `TargetSortedFarPairs` pytree.
- **For this IC the levers are physical:** a larger softening or a smaller dt, or `JACCPOT_CUBIC_BOUNDS=1`. Beyond
  1.43e9 directed pairs, use multi-GPU.

### The ferrers3 tile skip (opt-in)
With `JACCPOT_NEARFIELD_TILE_SKIP=1`, the CSR direct kernel evaluates a source range's tiles with the exact Newtonian
factors when every leaf of the range clears the target leaf's box by the support (one scalar test per neighbour
entry, from a per-leaf box table), and with the polynomial otherwise. It is bitwise: Triton tests in five regimes
(beyond, straddling, inside, roundings at the box edge, overlapping boxes); with it on, the eight S pins that inherit
the environment and S6 are bitwise `s2-tree`, and the 25M L3 force is bitwise main's (0 of 75,000,000). Free A100,
min of 3 calls x 4 steps, the near kernel from traces:

| | main | tile skip | near kernel |
| --- | --- | --- | --- |
| 8e6 clipped Plummer, softening 1e-7 | 78.43 / 78.51 ms | 76.16 / 75.59 ms | 24.92 -> 22.02 ms |
| 1e8 clipped Plummer, softening 1e-7 | 872.21 / 870.61 ms | 829.16 ms | 311.93 -> 271.26 ms |
| 25M disc+bulge, softening 1.5e-3 | 239.81 / 239.76 ms | **244.01 ms (+1.8 %)** | 91.89 -> 96.88 ms |

At the production softening nearly every neighbour leaf comes within the support, so the test is overhead: off by
default (the user). Two other designs were measured and dropped: a per-tile test on the tile's smallest r^2
(bitwise, but its reduction cost 0.76 of the polynomial's 0.88 ms; 8e6 near 24.79 -> 25.46 ms) and an optimistic
Newtonian pass (3.92 -> 3.82 ms on the microbench).

### The launch environment: XLA's fusion autotuner
- **The effect.** The first 2e6 particles of the 25M disc with its NFW halo, 20 steps, resumed in a fresh process
  from step 10, on busy cards: 4 of 12 resumes bitwise, the other 8 differing in 436,234 of 12M elements (max 4.4e-6).
  The optimised programs of one bitwise and one differing process: the HLO before optimisation identical, after it 7
  fusions switched between the native and the Triton emitter, one of them the NFW field added to the self-gravity.
  With the autotuner off: 8 of 8 bitwise. On the free card the 25M run and its resume had picked the same emitters.
- **DET2** (the gradient on the production flags), one shared card: with the autotuner off 5 of 5 bitwise A-vs-A;
  with it 3 of 5, with at least two independent flips (652 position- and 1147 mass-gradient elements, max 9.5e-7 /
  1.2e-4; and 1 element, 5.8e-11). The first is the DET2 A-vs-A mismatch of the S1 record (647 / 1147).
- **The cost** (free card, interleaved A1 B1 A2 B2, `compare --envelope` passes L1-L3):

| ms (min) | A1 | B1 (off) | A2 | B2 (off) |
| --- | --- | --- | --- | --- |
| L1 step 8e6 | 78.50 | 78.51 | 78.99 | 79.93 |
| L2 step 1e8 | 872.33 | 870.44 | 873.19 | 871.27 |
| L3 step 25M | 239.25 | 239.83 | 239.49 | 239.95 |
| L3 force 25M | 100.32 | 100.86 | 100.67 | 100.71 |

  B/p equal (L3's bimodal step peak aside), rel-L2 and far pairs equal; compile time within the host-load noise.
  Under deterministic ops the flag moved nothing: S1 and S1x bitwise `s2-tree`, S4b bitwise `s2-main-f967805`.
- **The alternative,** a per-fusion autotune cache (`--xla_gpu_per_fusion_autotune_cache_dir`), also made 4 of 4
  resumes bitwise, but it is state: what the first process timed, a directory that must travel with the restart
  files, not invalidated by XLA across versions or devices.
- **Built** (`s32p/7`): `ProductionRunner.run` warns once per process when its state is on a GPU and `XLA_FLAGS` does
  not turn the autotuner off; `fused_memory_budget.py`, `dce_pins.py`'s production-flag pins (L1-L3, DET1-DET3) and
  `provider_hlo_identity.py --device gpu` add the flag unless `XLA_FLAGS` sets it; `docs/production.md`, "Launch
  environment". The flag is experimental: re-check it at every jax upgrade.

### Gates
- **CPU.** Every CI shard on the integration: all pass but one test in `unit` and `py-floor`,
  `test_module_exports_guard` (`record_fused_nearfield_caps` was missing from `_large_n_pipeline.__all__`; fixed in
  `s32p/3`). `test_shards.py check`: the shards cover the 2148 tests. On the stack, each branch passes
  the CPU tests of what it touches, `test_shards.py check`, black, isort and pydoclint. Under
  `JACCPOT_RUNTIME_TYPECHECK=1` 7 caps-only tests fail, as on the track's own tip: the caps-only prepare's stub
  prepared state holds `local_data=None` against its `LocalExpansionData` annotation. CI runs `integration-strict`
  with the runtime type checks off.
- **Program identity** (`provider_hlo_identity.py` against `s32-base-cpu`, main `392061d`): the integration's start
  programs, scans and returns (`--programs compiled_runner`) 36/36 identical; in full, the 18 particle-carry
  variants lose the eager `_flat_walk_lists`, M2L and L2L programs and gain `_near_leaf_counts` (the caps-only
  prepare), and with `JACCPOT_STRICT_CAPS_ONLY_PREPARE=0` all 36 are identical in full. On the stack: the compiled
  runners 36/36 on every branch; in full 36/36 on `s32p/1`-`s32p/3`, and from `s32p/4` on the same 18 variants
  and programs differ as on the integration (on `s32p/7` with the switch off: 36/36 in full).
- **GPU** (the integration, shared cards): S1, S6, DET1, G1, S1p, S1t, S1e, S1x, S1g bitwise `s2-tree`; R1p, R1x, R1r
  bitwise (`s3integ-pins`); DET3 bitwise A-vs-A and to `s3integ-det3`; M1, M3 bitwise, M2 within main's envelope;
  with the tile skip on, S1 and S1x bitwise. On `s32p/7`, where the benches add the autotuner flag: S1, S6, G1,
  S1p, S1t, S1e, S1x, S1g bitwise `s2-tree`; R1p, R1x, R1r bitwise; DET1 and DET3 (now with the autotuner off)
  bitwise A-vs-A and bitwise the integration's records made without it.
- **L1-L3** (free card; main `392061d`, the integration, main again; `compare --envelope`): L2 +0.30 % step, -0.18 %
  peak; L3 force +0.03 % / +0.03 %; L3 step +0.12 % / +2.18 % (limit 4.37 %: the step peak is bimodal, 255.78 or
  261.36 B/p, and the A-vs-A control moves the same): pass. **L1 fails only because its peak DROPPED 2.93 %**
  (325.28 -> 315.74 B/p: the caps-only bootstrap at 8e6) against the symmetric 2 % limit; its step +0.40 %. rel-L2
  4.179e-4 and the lists unchanged. **Re-record the L and DET baselines after the merge**: L1's peak moved by
  design, and the production-flag pins now run with the autotuner off (DET2 lands on other values; DET1 and DET3
  did not move).

### For S3b and S4
- **S3b:** the `dehnen_error` refresh cost (its step 2.9x geometric at 25M) is untouched by this session. Also
  measure the per-axis against the cubic Morton box on the disc over a production run, and consider
  `JACCPOT_CUBIC_BOUNDS=1` for `preset="galaxy"` if the disc behaves like the Plummer (it changes the lists: a
  decision, not bitwise).
- **The Pallas walk's counter wrap** past 2^31 - 1 pairs writes out of bounds in the kernel: a guard there (a traced
  change, gated by identity and pins).
- **S4:** Odisseo's `integrate()` through the runner, with the launch environment's flag in Odisseo's launchers and
  docs; the GPU identity and pin references re-recorded with the flag.
- **Compiles** are now the first segment's cost at 25M (285 programs, 160-200 s on a loaded host) and ~50 s per
  energy program: a persistent compilation cache is the next lever.

## S3b and S4 — `dehnen_error` at 1.12x, the cubic galaxy box, nornax and Odisseo on the runner (2026-10-17/21)

Published as PRs. jaccpot:
- independent of each other: #431 (the walk and near reads), #435 (the cubic galaxy box), #436 (`FusedForceProvider.retry_mode`) and this record;
- stacked: #432 → #433 → #434 (the far f_b), #437 on #434 (an eager cascade compiles once), #438 on #435 (the galaxy preset's MAC; merge it after #434).

nornax: #18 → #19 → #20 (#20 needs jaccpot #436). Odisseo: #37 into `jaccpot-integration`.

**The user's decisions (2026-10-21):**

| | decision |
| --- | --- |
| `preset="galaxy"`'s MAC | `dehnen_error` at `adaptive_eps=1e-4` (was geometric θ 0.8), once S3b met its target: 1.12x the geometric step at 25M for 2.8x lower force error |
| Odisseo's automatic large-N profile | goes to the runner lane with the galaxy settings (an explicit `fmm_preset="large_n_gpu"` keeps the coupler) |
| nornax's single-rung step (lead) | written in the lane's velocity-Verlet association, so a nornax rollout equals the runner bit for bit |

**Setup.** jaccpot main `5f1f85f` (S3 session 2 merged), yggdrax main `ca3a93c`, nornax main `763096e`, Odisseo
`jaccpot-integration` `8fa0c18`, jax 0.11.2. Five tracks on their own branches. S3b's three jaccpot branches were
merged into one integration for the gates (clean merges). Timings on A100s with no other job running: card 2 on
2026-10-17 (the L-pin base), and card 5 on 2026-10-21. On card 5, one idle process of another user (416 MiB, 0 %
util) sat on the card for the timing gates. The rows flag it; main's rows there reproduce the clean L-pin base within 0.3 %. Accuracy, identity and
pins ran on shared cards.

**New references on main `5f1f85f`, recorded with the autotuner flag:**
- L pins `s3b-L-main-5f1f85f-t1` (and `-b`):
  - L1 78.3 ms, 2.352 GiB;
  - L2 873.2 ms, 23.913 GiB;
  - L3 240.8 / 239.4 ms, 6.062 GiB.
  - A-vs-A within the envelope, and within it against `s32-L-main-392061d`.
- DET1-DET3 (`s3b-det-free` / `-b`): bitwise A-vs-A on a free card. On a shared card, DET2 had 1 of 60,000
  position-gradient elements differ (5.8e-11) even with the autotuner off. The DET pins are therefore a free-card
  gate.
- GPU pins `s3b-main` (S1-S6, S1p, S1t, S1e, S1x, S1g, G1) and the CPU identity base `s3b-base-cpu` (36 variants).

### What S3b built
| branch | change |
| --- | --- |
| `s3b/walk-vector-loads` | the Pallas walk reads each eq (16a) table row and each node record with one vector load (it read the `[nodes, 8]` table one column at a time: 16 scalar gathers per pair); the near kernel's f_b lane multiplies by the reciprocal inside Triton's float32 division (`rcp.approx.ftz.f32`, the division's bits for a divisor in [2^-126, 2^126]) |
| `s3b/1-m2l-fb-lane`, `s3b/2-l2l-fb-channel`, `s3b/3-fb-in-downward` | the far half of f_b: each node's own term accumulated in the CSR M2L lanes kernel's launch (one lane accumulator, no atomics), pushed down the tree as one more channel of the Pallas L2L cascade, read at the leaves. Before, a chunked scatter and a searchsorted over every far pair and two scatters per level ran beside the downward pass every step |
| `s3b/galaxy-cubic-bounds` | `TreeConfig.cubic_bounds`; `preset="galaxy"` builds a cubic Morton box (the library default stays per-axis; `JACCPOT_CUBIC_BOUNDS=1/0` overrides) |

### Why `dehnen_error` cost 2.9x
- **The walk doubled from its memory reads, not from more pairs.** At 25M both MACs run 82 rounds, with far lists
  of 93.2M vs 94.8M. A geometric walk that merely loads the eq (16a) table beside itself took 47.6 ms against
  22.9 ms without it.
- **The near f_b lane cost +24 %.** The ferrers3 force is 39 instructions per pair. The f_b lane added 16, half of
  them for one float32 division that Triton compiles with two range tests and rescaling; it now adds 8.
- **The far f_b took about 390 ms.** It went to the separate pass's scatters (120 ms), the CSR row expansion
  (82 ms) and the push-down (191 ms). Moved into launches that already run, it costs about nothing.
- **The old far f_b was not reproducible on a GPU.** Run twice on the same lists, 126,363 of 200,000 values differed
  (max rel 5.2e-7) at 2e5: the scatter-add's order is unspecified without deterministic ops. The new one is bitwise
  A-vs-A, and its own term is closer to an fp64 sum (2.8e-7 against 8.0e-7).

### Gates
- **Lists.**
  - The walk and near changes: far and near lists identical as sorted pairs at 25M and 4e6, both MACs; f_b and
    forces bitwise at 25M.
  - The far f_b: new vs old over 4 steps is 0 list entries at 2e5 and 8e6, where forces and states are bitwise.
    The 25M disc at `dehnen_error` 1e-4 differs by 6-14 far entries out of ~91.5M; the old route against itself
    also differs by 14.
- **Accuracy.** Final-step rel-L2 identical in every arm: 1.5209e-4 (2e5), 1.7198e-4 (8e6), 1.5086e-4 (25M).
- **The integration (all three branches).**
  - CPU identity 36/36.
  - GPU pins 13/13 bitwise, S1e included (3,350,146 far pairs).
  - Every CPU shard rc 0; partition check clean (2,202 tests).
  - Two-card pins M1/M3 bitwise; M2 within main's own run-to-run spread.

### Timing (card 5; main vs the S3b integration, each row twice)
| row | main | S3b |
| --- | --- | --- |
| 25M disc+bulge, geometric step | 241.3 / 239.1 ms | 232.8 / 233.6 ms |
| 25M, `dehnen_error` 1e-4 step | 695.7 / 696.0 ms | **261.2 / 260.5 ms** (1.12x the geometric step) |
| 25M, `dehnen_error` 1e-4 force | 99.3 ms, rel-L2 1.4936e-4 | 99.3-99.5 ms, rel-L2 1.4936e-4 |
| 1e8 clipped Plummer, `dehnen_error` 1e-4 step | 2082 ms, 26.6 GiB | **831.6 / 831.3 ms**, 26.0 GiB (279 B/p) |
| 1e8, geometric step | 871.9 / 871.4 ms | 855.3 / 855.3 ms |
| L1 / L2 / L3 step | 78.3 / 873.2 / 240.8 ms | 76.4 / 855.2 / 232.8 ms (lists equal, peaks equal or lower) |

The S3b target (a `dehnen_error` 1e-4 step within 1.3x geometric at 25M, 1e8 below 1 s) is met.

### The cubic box for `preset="galaxy"`
| case | box aspect | far entries per-axis → cube | rel-L2 | peak |
| --- | --- | --- | --- | --- |
| 25M disc+bulge, step 0 | 1.004 | 94.80M → 94.75M | 4.18e-4 → 4.21e-4 | 210 B/p both |
| the same run, step 500 | 1.012 | 96.80M → 95.78M | 3.85e-4 → 4.19e-4 | equal |
| its disc alone, 21M | 3.01 | 215.9M → 78.9M | 1.02e-3 → 5.28e-4 | 323 → 208 B/p |
| 8e6 clipped Plummer (L1) | 1.004 | 39.09M → 38.95M | 4.33e-4 → 4.30e-4 | equal |

The production disc's box is already a cube: the bulge reaches r = 20 on every axis. The cube costs nothing there and
wins wherever the box is not cubic: a thin disc tilted at random needs 11.8M-39.3M far pairs per-axis, 10.4-11.3M
cubic. The L pins keep the per-axis bench configuration.

### What S4 built
- **nornax** (`s4/1-leapfrog-hooks`, `s4/2-retry-segment`, `s4/3-provider-force`):
  - `leapfrog_kdk_rollout` gets the block rollout's hooks (topology, rebuild, records, checkpoint);
  - an external field that reads the state's time and is kicked per particle rung;
  - `retry_segment`, a host driver that re-plans and re-runs a segment whose capacities overflowed;
  - `FusedProviderForce`, jaccpot's provider step under the leapfrog, carrying only `(f_b, bits, needs)`.
  - The single-rung step is written in the lane's velocity-Verlet association, so it equals the runner bit for bit.
  - jaccpot side: `FusedProviderForce.retry_mode`.
- **Odisseo** (`s4/thin-client`, into `jaccpot-integration`): `integrate()` runs as a thin client of
  `ProductionRunner`, the lane `"fmm_runner"`. It is taken by `fmm_preset="galaxy"`, and by the automatic large-N
  profile (FMM on a GPU, N >= 200,000) whenever the runner can do the run. Snapshots are the state after k·K steps,
  the initial one first, at times k·K·dt. Settings come from `GalaxySettings` (`fmm_galaxy_settings` to change them).
- **jaccpot** `s4/eager-cascade-once`: an eager M2M / L2L lane cascade compiles once per static configuration (every
  eager call recompiled its level loop: 30.7 of 38.2 s of a warm Odisseo call at 1e6). It forwards the far f_b
  channel; it is stacked on the S3b f_b branches.

### S4 gates (card 5)
| gate | result |
| --- | --- |
| nornax rollout == runner | bitwise: A100 2e5, 0 of 1.2M (f32, f64, f32 + NFW halo); CPU N=512, 0 of 3072 (with the halo, 1 velocity component 1 ulp) |
| nornax vs runner, 8e6 | 76.95 / 76.96 vs 77.86 / 77.78 ms; 223.7 vs 244 B/p |
| nornax vs runner, 1e8 | 866.3 / 865.5 vs 874.6 / 874.1 ms; 188.8 vs 208.8 B/p |
| Odisseo `integrate()` vs the harness (L1, 8e6) | 77.77 / 77.78 vs 78.48 / 78.27 ms (0.991 / 0.994); 243.7 vs 315.7 B/p |
| Odisseo fixed cost per call, 8e6 | S0 ~190 s; now ~4-11 s warm, nothing recompiled for new dt, G or halo parameters |
| Odisseo == runner | bitwise (CPU; GPU 2e5 against main and main against itself, 0 of 1.2M / 4.8M) |

`block_kdk_rollout(k_max=0)` costs 2 force evaluations per step on the per-level path and 3 on the fused one; the
single-rung rollout costs 1.

### Findings
- **nornax kept the eager prepare's tree alive** (the engine's topology cache and the caller's `prov, _ =
  prepare(...)`): +11 % peak at 8e6 until it cleared the cache as the runner does. Since then **nornax peaks ~20 B/p
  below the runner** at 8e6 and 1e8. The runner holds something nornax does not, a lever for the one-card ceiling.
- **The first Pallas launch in interpret mode defines a config in jax's tracing context**
  (`pallas_fuser_disable_nested_fuse`), so the next call misses the jit cache once. CPU only; the GPU's Triton
  lowering does not import it.
- **Odisseo's old FMM coupler returned snapshots one step late and mislabelled their times.** It returned the states
  after steps 1, 1+K, ..., labelled `linspace(0, t_end)`; fixed with the runner lane's semantics.

### The galaxy preset on `dehnen_error` (#438)
- **The solver resolves the bench's eq (16a) settings:** eps 1e-4, `adaptive_error_model="dehnen_paper"` and
  `mac_force_scale_mode="paper_fb"`. Their constructor defaults are now `None` ("the preset's"), the way `theta`
  works. Every other preset resolves as before.
- **θ does nothing under `dehnen_error` on the fused lane.** The walk accepts by eq (16a) and its convergence clause;
  the first step's f_b seed walks at θ 0.8. θ 0.3, 0.5 and 0.8 gave identical lists and bitwise states.
- **The geometric MAC back:** `FastMultipoleMethod(preset="galaxy",
  advanced=dataclasses.replace(FastMultipoleMethod(preset="galaxy").advanced, mac_type="dehnen"))`, together with
  `GalaxySettings(mac_type="dehnen", adaptive_eps=None)` for the runner's restart fingerprint.
- **Costs:**
  - 279 against 257 B/p at 1e8;
  - G and the softening are not traced (a new value recompiles);
  - restart files carry f_b;
  - **`compute_accelerations` on the preset (the facade's generic large-N route, not the fused runner) peaks ~6x
    higher**: 4553 against 736 B/p at 2e5.
- **Restart after 2 of 4 segments, fresh process** (`preset="galaxy"` at 2e5 through the runner):
  - With all S3b PRs merged: resume, straight run and control are all bitwise.
  - With #438 alone: 2 of 1.2M elements differ, the same noise as one straight run against another, from the old
    far-f_b scatter.

### For S5 and later
- **The facade's `dehnen_error` route** (`compute_accelerations` on the galaxy preset): ~6x the geometric memory at 2e5.
  The runner and the provider are lean; the facade is not.
- **The runner holds ~20 B/p** that nornax on the same provider does not (8e6 and 1e8): a lever for the one-card ceiling.
- **Open since S3 session 2:**
  - the Pallas walk's counter wrap past 2^31 - 1 pairs;
  - a persistent compilation cache (the first segment's ~285 programs).
- **Odisseo:**
  - `restart_from`;
  - the last ~5-8 s of fixed cost per call (yggdrax's eager tree build, the template's `eval_shape`).
- **Multi-GPU (S12):** the lane reads the cubic box only from `JACCPOT_CUBIC_BOUNDS`.
- **The runner does not check** that its `GalaxySettings`' MAC matches the solver's.

## C1 — the timestep criterion for galaxy discs (2026-10-22)

**[draft]** S0 left one question before the block-step track (S5-S7) resumes: would the orbit-time criterion
`eta sqrt(sqrt(r^2 + eps^2) / |a|)` spread the 25M disc's steps far enough (bound ~40x) to make HOLD worth 10-40x,
where nornax's acceleration criterion `eta sqrt(eps / |a|)` caps it at ~2x? C1 answers it at equal accuracy.

**Setup** (`bench/c1_criterion_calibration.py`; rows, tables and README in `bench/results/c1_criterion_2026-10/`):
- **IC:** 1e5 particles of the shuffled 25M disc+bulge IC (an independent draw from the same distribution
  function, masses rescaled, recentred) in its NFW halo, with the production ferrers3 kernel at eps 1.5e-3.
- **Run length:** t_end = 1.5, one stellar orbit at 2.5 R_d (2 pi R / <v_phi> = 1.47; 1.45 circular periods there).
- **Forces:** a Pallas direct sum on A100s, fp32 pairs with fp64 sums (2e-7), the state in fp64.
- **Schemes:** shared KDK at dt, and HOLD (level = min rung) with nornax's base-step rung assignment, optionally
  refined at a particle's own boundary.
- **Costs** are counted, not timed, as equivalent shared steps:
  - `evals`: particle evaluations of the FMM with one padded-subset evaluation per active level, as S6 plans HOLD;
  - `weighted`: one weighted evaluation per boundary, S0's model;
  - `pairs`: direct-sum pair work.
- **Three configurations:**
  - **A**, the N-body at the production softening;
  - **B**, eps 0.0238, where one particle pulls as weakly against the mean field (0.13 against 23) as at 25M;
  - **S**, the same particles as test particles in a fixed bulge + disc + halo field. S has no encounters and no
    chaos, keeps the production spread, and makes E_i exact.
- **References:** shared steps at 1/16 and 1/32 of the production step (A, S), 2.5e-4 and 1.25e-4 (B).

**Answer: keep the acceleration criterion with shared steps, and double the production step to dt 1e-3** (the
acceleration criterion's dt_min at eta about 0.33; point 5 below). **The orbit-time criterion does not make block
steps worth 10-40x at equal accuracy on this disc.**

**1. The spread is 20x, not 40x, and plain HOLD pays twice the particle-step count.** Probe at t = 0, unchanged at
1e6 and for a second draw; the 1e6 acceleration bound is S0's 25M value:

| criterion | dt / dt_min p1 / p50 / p99 | particle-step bound, k_max 6 | `evals` | `weighted` | `pairs` |
| --- | --- | --- | --- | --- | --- |
| acceleration | 1.5 / 2.9 / 6 | 2.01x (1e6: 1.93x) | 0.95x | 1.95x | 2.11x |
| orbit time | 4.8 / 38 / 178 | 20.6x | 9.9x (k_max 5: 11.9x) | 15.6x | 21.8x |

- The eps under the square root lifts the centre's dt_min: S0's `sqrt(r/|a|)` had a median of 78x and a bound of
  39.7x.
- **HOLD on plain padded subsets evaluates every active level's subset at every boundary**, about
  `2 sum_i 2^rung_i` targets. On the acceleration criterion it is slower than shared dt_min at the same eta, and
  only S5's weighted single pass (with per-rung source multipoles) approaches the bound.

**2. At equal accuracy the spread does not translate into speed** (S, HOLD's gain = the shared cost that reaches the
same score over HOLD's cost; ranges over total energy, the per-particle energy tail dE_i p99 and positions):

| criterion | `evals` | `weighted` | `pairs` |
| --- | --- | --- | --- |
| acceleration, k_max 4, eta 0.05-0.4 | 0.26-0.80x | 0.52-1.59x | 0.61-1.85x |
| orbit time, k_max 6-8, refined, eta 0.025-0.1 | 0.28-1.05x | 0.55-2.03x | 0.86-3.28x |

| S, one stellar orbit | `evals` | `pairs` | max dE/E0 | dE_i p50 | dE_i p99 | bulge dE_i p99 | sigma_z rms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shared 5e-4 (production) | 3009 | 3009 | 2.8e-7 | 1.0e-7 | 1.0e-5 | 8.8e-5 | 9.2e-5 |
| shared 1e-3 | 1505 | 1505 | 1.1e-6 | 4.0e-7 | 5.2e-5 | 5.0e-4 | 2.1e-4 |
| HOLD orbit time eta 0.025, refined | 2285 | 745 | 2.9e-6 | 5.5e-6 | 5.1e-5 | 7.6e-5 | 1.2e-3 |
| HOLD orbit time eta 0.05, refined | 1146 | 382 | 1.8e-5 | 2.4e-5 | 2.1e-4 | 3.0e-4 | 2.2e-3 |
| HOLD acceleration eta 0.2 | 3244 | 1395 | 3.8e-7 | 8.4e-7 | 4.2e-5 | 1.1e-4 | 2.6e-4 |

- **A shared step's accuracy is set by the bulk of the mass, not by its fastest particle.** The shared step that
  matches HOLD-orbit-time at eta 0.025 is about 1e-3, the orbit-time dt_min at eta 0.26, so equal accuracy needs a
  10x smaller orbit-time eta and the 20x spread shrinks to about 2x.
- The orbit-time criterion gives every orbit the same 2 pi / eta steps; a shared step over-resolves the outer disc,
  which holds most of the mass, for free. Leapfrog's per-particle error scales as (dt / T)^2, so the mass-summed
  error is minimised at dt proportional to T^(2/3). The acceleration criterion is about T^(1/2) on a flat rotation
  curve, the orbit-time criterion T^1.
- The orbit-time criterion wins only in the bulge tail (eta 0.025: 1.3x in `evals`, 4.0x in pairs against shared
  5e-4).

**3. In the N-body the orbit-time criterion is blind to close encounters** (A).
- At 1e5 one particle at eps pulls with 32 against a median |a| of 23.
- Every scheme drifts once a particle's step exceeds about 5e-4 (eps / 3, a softened encounter's crossing time).
  Shared steps converge (9.4e-5, 5.7e-6 and 1.3e-7 at 2e-3, 1e-3 and 5e-4).
- The orbit-time criterion saturates near dE/E0 = 1e-4 for every eta <= 0.1 (4.9e-5 at 0.0125).
- It adds spurious per-particle energy diffusion in the disc: +7 % (eta 0.05), +23 % (0.1), +53-59 % (0.2) and
  +164 % (0.4), against at most 2 % for shared steps up to 2e-3 and every acceleration run (shared 4e-3: +5 %).
- The centre and bulge move beyond the 2-5 % converged-run spread only at eta >= 0.2: the 0.5 % radius by +7 % at
  0.2 and +28 % at 0.4, the bulge's 10 % radius by +14 % at 0.4.
- Sigma, sigma_z and A2 hardly discriminate. The hot disc (Q_min 1.6) grows no bar in one orbit, and A2 stays at
  shot noise.
- At 25M encounters are 250x weaker, so this degradation matters for low-N or small-softening runs.
- At equal cost shared steps win in A in every family.

**4. HOLD itself has two error terms shared steps do not** (B, smooth forces):
- **Secular drift from the base-step rung reassignment** (nornax `block_kdk`'s rule). HOLD-acc eta 0.2 drifts to
  5.6e-5; with the rungs frozen at t = 0 it stays bounded at 7.7e-6. Without pair coupling (S) the same
  reassignment is harmless, which makes this the time-asymmetric step selection of Hut, Makino & McMillan (1995),
  amplified because a rung change moves all of a particle's pairs to another level.
- **The slow-fast coupling**, applied on the slow step: even frozen, 7.7e-6 against 8.6e-7 for shared at a
  coarser 4e-3.
- In A, where the rungs must follow the particles, frozen rungs are worse (dyn eta 0.1: 8.0e-4 against 9.6e-5).
- Refining at the particle's own boundary cuts the orbit-time criterion's error 2.5-3x (S, eta <= 0.1; A, eta 0.1:
  9.6e-5 to 1.4e-5).
- Momentum stays at round-off in every HOLD run: pair impulse at most 5.6e-11 of `sum m |v|`, |dL| / |L| at most
  1.2e-10.

**5. The 25M production configuration takes dt 1e-3.**
- **Setup:** jaccpot main f654bd6, `ProductionRunner` with `preset="galaxy"` as is (`dehnen_error` 1e-4, the cubic
  box), on the L3 pin's 25M disc+bulge+NFW IC. Runs at dt 1e-3, 5e-4 and the reference 2.5e-4 to t = 1.5
  (`bench/results/c1_criterion_2026-10/25m/`). Peak 5.5 GiB; 0.258 s per step on a free card inside a segment.

| 25M, one orbit | steps | max dE/E0 | E_i diffusion inner 2 % | Lagrangian radii, worst | Sigma / sigma_z / sigma_R / v_phi, worst bin | A2 diff |
| --- | --- | --- | --- | --- | --- | --- |
| reference 2.5e-4 | 6000 | 7.5e-8 | 1 | -- | -- | -- |
| 5e-4 | 3000 | 6.2e-8 | 1.0002 | 0.54 % (bulge 0.5 %) | 2.2e-4 / 8.8e-5 / 4.5e-5 / 1.7e-4 | 2.2e-6 |
| 1e-3 | 1500 | 1.9e-7 | 0.9965 | 0.23 % (bulge 2 %) | 6.3e-4 / 1.4e-4 / 2.9e-4 / 7.5e-5 | 5.5e-6 |

- **Energy:** the 1e-3 error is a bounded offset (at most -1.8e-7 early, -5.4e-8 at the end), not a drift. The
  reference and 5e-4 sit at the diagnostic's floor of about 1e-7.
- **Structure:** every Lagrangian radius of the disc, the bulge and all baryons (0.5-90 %) is within 0.25 % of the
  reference, against the run's own contraction of the central radii by 1.7-5.4 %. The per-particle energy
  diffusion is unchanged, so the encounters that set the step at 1e5 (point 3) are absent at 25M.
- **Recommendation: production dt 1e-3 for this disc** (half the steps per orbit). Keep the energy diagnostic on
  for runs of many orbits; at lower N or a smaller softening, encounters set the step (point 3).

**Decision input for the plan.**
- The block-step track has no payoff on this disc with either criterion: at equal accuracy at most ~2x in S5's
  weighted pass, and below 1x with the padded subsets S6 planned.
- If it resumes for ICs with a genuinely wider range at fixed accuracy (cusps, a central black hole, mergers,
  smaller softening), it needs:
  - the weighted single pass;
  - a time-symmetric rung selection;
  - refinement at the particle's own boundary;
  - an accuracy gate against shared steps at equal cost, not at equal eta.

## S8 — gradients on the production layout (2026-10-22)

Plan S8 (D6, part 1): the differentiable force runs the production forward's kernels, with reverses that fit at 1e8.
Three tracks, all on main `248f93a` (S3b and S4 merged) and then `f654bd6`: **S8-N** (`s8/near-direct-reverse`), the
near field's reverse on the direct layout; **S8-F** (`s8/l2p-reverse`), the far field's (L2P, P2M, M2L, M2M / L2L)
and the far half's entry points; **S8-W** (`s8/wire-production-grad`, on `s8/integ` = main + both), the wiring of
the differentiable seam onto them, the gates, the memory at 1e8. GPU rows ran on A100s; a row is a timing only when
the card had no other process before and after (said so where it did), otherwise a contended hint.

| target / gate | result |
| --- | --- |
| differentiable forward == production forward | **bitwise**: pin G1 (2e5, both call orders, 0 of 600,000 elements); the gate bench at 8e6 (0 of 24M) and 2.5e7 (0 of 75M); every softening kernel on CPU (Pallas interpret, compiled) |
| FD in float64 through the whole force (near + far) | positions, masses within 1e-6 (directional), `G` 1e-10, `eps2` 1e-6 (CPU interpret, N 1024, cells of 8, p4, ferrers3; the far list target-sorted and symmetric as the GPU walk hands it over) |
| per-softening-kernel parity | plummer, ferrers3, wendland_c2: forward bitwise; the direct near reverse vs a float64 pair sum <= 6.8e-15 (S8-N) |
| no buffer of 16 N floats | the largest at 8e6: the `[nodes, 36]` tables, 9.4 N floats (pow2 leaf capacity); at 1e8 the COM forward's `f64[N, 3]` prefix sums, 6.0 N floats |
| single-force gradient at 1e8 within 36 GB | `jax.grad` (positions, masses, the production leaf capacity): device peak **28.29 GiB (30.4 GB)**, temporaries 13.92 GiB (149 B/p); `value_and_grad`: temporaries 19.15 GiB, ~33.4 GiB (35.9 GB) estimated; out of memory on a free card (the block does not fit beside the prepared state) |
| gradient <= 5x the force eval at 8e6 | `jax.grad` **4.37x** (131.5 / 30.1 ms; 1e8: 4.14x), `value_and_grad` 5.37x (161.5 ms); free card, A-vs-A |
| gradient A-vs-A | bitwise at 8e6 (G1 point and galaxy), 2.5e7 and 1e8 (it differed in 636-1036 mass elements at 8e6 before S8-W's COM reverse) |

### The near field's reverse on the direct layout (S8-N)
- `nearfield_leafpair_csr_sorted_direct_reverse_pallas` (`pallas/nearfield_leafpair_csr_direct_vjp.py`) is the
  adjoint of the production near kernel on its own sorted layout: one program per (leaf, target subtile) walks the
  leaf's row as the forward does. The near list is symmetric (one canonical walk emits (A, B) and (B, A)), so a sink's
  program sees every pair it is part of from both ends and writes its source-side cotangent too: one writer per
  particle, no atomics, no second pass. `G_bar` comes from the mass cotangent by Euler (the force is mass-linear).
- `nearfield_leafpair_csr_sorted_direct_pallas_cvjp` is the production call under a `custom_vjp` (the same
  keywords; its forward IS that call). It differentiates positions (sinks and sources), masses, `softening_sq` and
  `G`; the `dehnen_error` force-scale lane is computed and treated as a constant (it only sets the next walk's
  thresholds); the potential lane is refused.
- Measured (A100, no other process before or after, `bench/near_direct_reverse_gate.py` on captured steps):

  | capture | forward | gradient (table layout's) | temporaries |
  | --- | --- | --- | --- |
  | 8e6 disc (ferrers3, eps 1.5e-3) | 26.8-30.0 ms | 57.6 ms (274.5) | 45.4 B/p, largest 4.00 N floats (1380 B/p) |
  | 8e6 clipped Plummer | 25.5 ms | 54.7 ms (reverse alone) | |
  | 2e5 disc / plummer / wendland_c2 | 1.81 / 1.38 / 2.01 ms | 3.18 / 2.39 / 3.64 ms (15.9 disc) | |
  | 25M disc+bulge, `dehnen_error` 1e-4 (with the force-scale lane) | 106.1 ms | 202.9 ms | 45.4 B/p, device peak 2.43 GiB |

  Float32 against float64 <= 1.9e-6 (the table layout's <= 1.2e-6); A-vs-A bitwise. Gates: central differences of
  the production forward in float64 for positions, masses, eps^2 and G (<= 1.4e-8 relative, every kernel); the
  reverse against `jax.vjp` of a pure-JAX pair sum <= 6.8e-15 (plummer, ferrers3, wendland_c2; rows in pieces,
  padded subtiles, self terms alone, an empty neighbour array).

### The far field's reverse on the production layout (S8-F)


Track S8-F of S8 (gradients on the production layout), branch `s8/l2p-reverse` on main `248f93a` (S3b and S4
merged). It gives the far half of the differentiable forward the production layout and kernels, with reverses that
fit at 8e6 and 2.5e7. S8-W wires the seam onto it (`evaluate_large_n_state_at_positions_and_masses_sorted` still
calls the leaf-major L2P). The operating point is G1's unless stated: Plummer, cells of 64, θ 0.8, p5, fp32,
`mac_type="dehnen"`. GPU rows ran on shared A100s (other users' jobs on the card): **contended hints, not timings**.
The measured quantity is `jax.value_and_grad` of `sum(w * far_acc)` through the far chain alone (P2M, M2M, M2L,
L2L, L2P at the state's frozen lists), jitted.

#### What the differentiable far path ran, against the production refresh
| stage | production | differentiable (before) | effect |
| --- | --- | --- | --- |
| P2M, M2M, M2L, L2L | Pallas through their `custom_vjp` seams | the same kernels and seams | locals bitwise the prepared state's at 2e5, both at the G1 point and at `preset="galaxy"` (`dehnen_error`, p6) |
| far-pair operands | the walk's `TargetSortedFarPairs` as row offsets (a presorted CSR) | one target per pair, expanded in the plan, then a stable sort by target in the M2L | the same values; an extra P-int32 array in the plan and a P-pair sort in every forward |
| box geometry | deferred on the COM walk | computed | dead code in the far chain |
| the `dehnen_error` force-scale twins of M2L / L2L | yes (f_b feeds the next walk) | no | locals bitwise either way |
| L2P | particle-major Pallas (`l2p_real_particles_pallas`) | leaf-major padded jnp (`_evaluate_local_expansions_for_particles`) | 506,531 of 600,000 components differ (max 6.6e-7 relative). Its reverse is NaN at the 13 single-particle leaves (delta = 0 exactly), which made the far chain's whole gradient NaN at 2e5 |
| expansion centres | the COMs of the live positions and masses | the same | |

Two fixes:
- `large_n_farfield_locals_at` now builds the production operands (the far pairs and the geometry;
  `LargeNGradPlan.far_pairs_target_sorted`).
- `large_n_farfield_l2p_at` evaluates as `evaluate_large_n_state` does: one helper both read,
  `_large_n_pipeline.resolve_large_n_l2p`.

With both, the far half at the state's own inputs is the production eval's far field bitwise (A100 at 2e5, and the
CPU interpret test).

#### The reverses
- **L2P** (`l2p_real_particles_pallas_cvjp`, the production call with a reverse):
  - a program owns 32 leaves as the rows of a (32, 4) lane tile;
  - per lane, the coefficient cotangents are the gradient's recurrence contracted with the cotangent
    (`l2p_real_grad_cartesian_transpose`);
  - the delta cotangent is ONE forward-mode jvp of the recurrence (the Hessian is symmetric);
  - the leaf's sums run along the row and are written once by the leaf's program: no N x (p+1)^2 buffer, no
    atomics.
- **P2M**: the same blocked shape, 16 leaves x 8 lanes (the forward's). The (L, W, 4) slot block it replaced was 21
  floats per particle at 2e5 and 33.5 at 8e6.
- **M2L**: the reverse of a presorted list reads the forward's CSR for its by-target pass, one P-pair sort less (343 MiB
  of CUB scratch at 8e6). It expands the per-pair targets for its by-source sort in two passes, one scatter and one
  prefix sum, instead of a binary search per pair (9.8 ms at 8e6).
- **M2M / L2L**: one node per lane in the packed `[nodes, C]` table, the forwards' shape.
  - The coefficient half is `jax.vjp` of `_translate_lane` in its coefficients only.
  - The geometry half comes from the generator of translations. Translating and differentiating commute, so
    `d/d delta <g, T v> = <g, A w>` for M2M (`w = T v`) and `<A^T v, T^T g>` for L2L, with `A` the solid harmonics'
    derivative ladder (the L2P gradient's).
  - This is exact on the axis and at a zero edge: central differences of the pair twins agree to 1e-10 at p 2-6.
  - The level reverse it replaced ran the widest level's grid on every level, on centred `[nodes, 128]` rows.
- **What did not work:** a particle-major L2P reverse, with one particle per lane, each leaf's sums as masked lane
  reductions across the program, and an owner per leaf. At 8e6 it took 9.9-25 ms over 7 block / warp / group
  variants, against 3.94 ms for the blocked reverse and 1.15 ms for the forward (synthetic cell-like layout, p5).

#### Memory and time (shared A100s)
`temp` is the compiled gradient's temporaries (`memory_analysis`). The peak is the process's `peak_bytes_in_use`,
which includes the prepared state (1.66 GB at 8e6, 4.13 GB at 2.5e7). Times are the minima over 3-7 calls.

| | base (`s5/base`) | S8-F |
| --- | --- | --- |
| 2e5: temp / peak | 666 MB / 722 MB, **NaN gradient** | 55.5 MB / 111 MB |
| 2e5: far forward / far gradient | 2.16 / 25.4 ms | 1.54 / 5.55 ms (3.6x) |
| 8e6: temp / peak | **out of memory** (a 41.38 GiB allocation; 44.4 GB requested) | 2.00 GB / 3.79 GB |
| 8e6: far forward / far gradient | -- | 21.3 / 87.3 ms (4.1x; another call on another card: 15.9 / 95.0 ms) |
| 8e6 `preset="galaxy"` (`dehnen_error` 1e-4, p6; 28.5M far pairs in a 179M capacity): temp / peak, forward / gradient | -- | 5.95 GB / 12.9 GB; 21.4 / 146 ms (6.8x) |
| 2.5e7: temp / peak | -- | 4.92 GB / 9.46 GB |
| 2.5e7: far forward / far gradient | -- | 36.9 / 209 ms (5.7x) |

At intermediate states of this work, at 8e6:
- the particle-major L2P / P2M reverses with the level cascade reverse: temp 3.74 GB, far gradient 547 ms (34.5x);
- at 2.5e7 the same: 7.25 GB, 1.13 s.

Kernel time of the far gradient at 8e6 (one trace, a shared card): 95.5 ms. The largest items:
- the M2L reverse, 23.5 ms by source and 16.6 ms by target (40.1 ms);
- an XLA scatter of the COM's reverse (`input_scatter_fusion`, 13.9 ms: yggdrax's mass moments);
- the per-pair target expansion, 9.8 ms (since replaced);
- the M2L forward, 8.2 ms;
- the cascade reverses, 3.1 + 2.6 ms (163 + 2 x 162 ms with the level kernels);
- the L2P reverse 2.0 ms and the P2M reverse 2.0 ms.

**Largest buffers of the compiled far gradient** (XLA buffer assignment):
- base, 2e5:
  - CUB sort scratch, 21.4 floats per particle;
  - the cascade reverse's centred rows `f32[32768, 128]`, 21.0;
  - the P2M block `f32[16384, 64, 4]`, 21.0;
  - the leaf-major L2P's `f32[16384, 64, 3]`, 15.7;
  - ~150 `f32[16384, 64]` residuals.
- S8-F, 8e6 (2.66 GiB in 214 allocations, 1.91 GiB of them preallocated temp):
  - the CUB scratch of the M2L reverse's by-source sort, 343 MiB (11.2 floats per particle);
  - the `[nodes, 36]` coefficient tables, 288 MiB (9.4);
  - the mass moments' float64 prefix sums `f64[N + 1, 3]`, 183 MiB (6.0);
  - the far list's `s32[44M]` arrays, 168 MiB (5.5).
  - Nothing reaches 16 floats per particle, and nothing is per particle and per coefficient.

#### Gates
| gate | result |
| --- | --- |
| L2P reverse vs `jax.vjp` of a pure-JAX L2P (interpret, p 3-8: every occupancy, leaves over many chunks, empty leaves, dead rows) | float64 < 1e-12 and float32 < 2e-5 of the largest component (the test's bounds; float64 measured 3.8e-16). A100 (Triton): float64 <= 7.9e-16, float32 <= 6.0e-7, A-vs-A bitwise |
| L2P at 2e5 in fp32, against float64 truth (`jax.vjp` of the Cartesian L2P) | coefficients / centres / positions rel-L2 1.56e-7 / 1.48e-7 / 1.35e-7 (max 3.4e-7). The leaf-major reverse: 8.5e-7 / 1.2e-5 / 1.2e-5 (max 3.5e-4), and NaN on 13 leaves. A-vs-A bitwise |
| FD in float64 through the whole far chain (P2M, M2M, M2L, L2L and L2P reverses, interpret; N 1024, cells of 8, p4) | directional derivatives in positions and masses within 1e-6; per entry within 4.7e-7 (the finite-difference noise) |
| P2M / M2L / M2M / L2L reverse tests (both halves vs `jax.vjp` of the pure-JAX references; on-axis edges; zero displacements) | pass, interpret and native (27 cascade tests on an A100) |
| CPU program identity vs `s3b-integ-cpu` | 36/36 identical (HEAD, rebased on main `248f93a`) |
| the forward kernels' source lines (L2P, P2M, cascade lanes; their Triton IR embeds the locations) | unchanged |

#### For S8-W and later (S8-W wired the seam, removed the by-source sort and the COM scatter; see below)
- **Wire the seam:** `evaluate_large_n_state_at_positions_and_masses_sorted` should call
  `-G * large_n_farfield_l2p_at(state, large_n_farfield_locals_at(...), positions)`. The G1 pin should then become
  bitwise on the far half.
- **The far reverse's next levers** at 8e6:
  - the M2L reverse's two passes (40 ms; the by-target pass re-runs the pair vjp only for the geometry half);
  - the COM reverse's scatter in yggdrax (14 ms);
  - the by-source sort of the frozen list (one CUB sort per reverse). It runs over the list's capacity, not its live
    count (179M against 28.5M pairs under `dehnen_error` at 8e6). The list is frozen, so its by-source CSR could be
    built once per plan.
- **The far gradient is not bitwise A-vs-A on a shared card:** 2 of 200,000 mass-gradient elements differed at 2e5
  (the positions' were bitwise). The new kernels write every row once (no atomics), and the L2P reverse alone was
  bitwise A-vs-A. The difference is in XLA's part of the chain, most likely the scatters of the COM's reverse; not
  bisected.

### The wiring of the differentiable seam (S8-W)

What the differentiable forward (`differentiable_accelerations` on a `LargeNPreparedState`,
`runtime/_large_n_grad.py`) ran against the production evaluation (`evaluate_large_n_state`), and what it runs now:

| piece | production | before S8-W | S8-W |
| --- | --- | --- | --- |
| near field | the CSR lane's direct layout | the table layout (`differentiable=True` forced it) | the direct layout under its `custom_vjp` (`nearfield/_fast_lane.py` `_csr_direct_near`) |
| far field | the prepare's locals; particle-major Pallas L2P | the recomputed locals; leaf-major jnp L2P | the recomputed locals on the production operands, the production L2P (S8-F's `large_n_farfield_locals_at` / `large_n_farfield_l2p_at`) |
| the sum | `near + far` | `(0 + far) + near` | `near + far` |
| `G`, softening | literals, or traced `ForceParams` | literals (`-float(engine.G)`) | literals, or `force_params=ForceParams(...)` (arrays: differentiable) |
| the topology in a jitted program | an argument | closed over: every array embedded as a constant | an argument: `LargeNGradPlan` is a pytree; the input permutation is the state's `tree.particle_indices`; `differentiable_step_fn` binds both |
| the M2L reverse's by-source pass | -- | a sort of the far list's capacity | the forward's CSR (the list checked symmetric at plan time) |
| the COM reverse | -- | yggdrax's prefix sums transposed: colliding f64 scatters | ancestor sums by pointer doubling, gathers only |

- **Forward bitwise.** Pin G1 (A100, 2e5, deterministic ops): both arms 0 of 600,000 elements differ (rel-L2 0.0),
  with 750,196 far pairs and 252,442 near edges in each. `bench/grad_production_gate.py` (the production eval of the
  state against the differentiable forward jitted with the state as an argument): 0 of 24,000,000 at 8e6 and 0 of
  75,000,000 at 2.5e7. On CPU (Pallas interpret, the compiled forward): bitwise for plummer, ferrers3, wendland_c2,
  the fused lane's force in both call orders, and `dehnen_error` at its frozen lists.
- **Not routed through the `custom_vjp` by default.** Always calling the direct kernel through its `custom_vjp`
  keeps every bit but moves the literal `G` and softening constants in the lowered text: 20 of 36 CPU
  program-identity variants changed (texts only). The default programs (Python-float parameters) call the kernel
  directly; `differentiable=True` or array parameters take the `custom_vjp`. CPU identity of the final tip against
  `s3b-integ-cpu` and the lead's `s8-integ-cpu`: 36/36 identical.
- **`G` and `eps2`.** `differentiable_accelerations(..., force_params=ForceParams(G, eps2))` (engine, facade,
  `differentiable_step_fn`'s optional third argument): central differences in float64 agree to 1e-10 (G, linear) and
  1e-6 (eps2). The provider's `accelerations(topology, params)` with array parameters is differentiable on the
  direct layout too (its d/dG, d/deps2 test now runs on the CSR lane; on the base it failed with pallas_call's "JVP
  with aliasing not supported"). `mac_type="dehnen_error"` refuses any `force_params` but the solver's own
  (`ValueError`, the fused lane's rule): its positions and masses differentiate at its frozen lists (FD in float64
  within 1e-6).
- **What stays outside the rule, and why.** The caller's external field (`external_acceleration_fn`) is outside the
  self-gravity. The near field's overflow and target-block payloads are the pure-JAX rectangle route's; the CSR lane
  never builds them, and the plan refuses a state that has them (unchanged). A near list that is not symmetric is
  refused (the direct reverse reads each pair from both ends); a far list that is not is differentiated with the
  sorting M2L reverse.

**Findings.**
1. A gradient jitted with the prepared state CLOSED OVER embeds every array of the topology in the program as a
   constant (a 10M-element closure is 80 MB of program text, and XLA constant-folds through it). The plan is a pytree
   now and the seam reads the permutation from the state, so both are arguments; `differentiable_step_fn` binds them
   with `functools.partial` (its docstring's "compiled is slower than eager on `large_n_gpu`" was this).
2. The M2L reverse's by-source sort ran over the far list's capacity (542M entries for 361M live pairs at 1e8). The
   walk's lists are symmetric -- checked at 2e5 (G1 and `preset="galaxy"`), 8e6 and 1e8 -- so the by-source pass
   reads the forward's CSR; the plan checks the symmetry with a sort-free hash test, chunked (an exact check would
   be a sort of its own).
3. The COM reverse (yggdrax's prefix sums under autodiff) was two colliding float64 scatters: 16 ms of the 8e6
   gradient and its A-vs-A nondeterminism (3 position and 1036 mass elements of 8e6; 1 and 1492 at 2.5e7; at 2e5 2 of
   200,000, S8-F's open item). With the ancestor-sum reverse the gradient is bitwise A-vs-A at 8e6 and 1e8.
4. On CPU in Pallas interpret mode an EAGER differentiable forward differs from the compiled production one by one
   float32 rounding for the compact kernels (ferrers3 1471, wendland_c2 1334 of 3072 elements; plummer none): eager,
   the `custom_vjp`'s forward is a program of its own whose softening is an argument, which XLA cannot fold into the
   kernel (S2's traced-parameter effect). Compiled, it is bitwise; on the GPU the eager G1 arms are bitwise.
5. With the CPU's XLA L2P (no interpret flag) the far field rounds differently when its locals are computed in the
   same program (23 of 3072 elements at N 1024); not the production lane -- the CPU tests run the L2P in interpret
   mode.
6. `value_and_grad` keeps the forward's arrays through the reverse: +5.2 GiB of temporaries at 1e8 over `jax.grad`
   (+1.3 GiB at 2.5e7). When memory binds, evaluate the loss in a separate forward (one force eval).
7. `XLA_PYTHON_CLIENT_MEM_FRACTION` (default 0.75) caps the allocator with preallocation off too: the 1e8 gradient
   needs 0.95 (the gate bench's runs set it).
8. The direct near reverse's bwd got a literal softening back from its residuals as a numpy scalar under jit, which
   the unit shards' runtime type check refuses (S8-N's rule; fixed in the bwd).

### S8 gates (S8-W's tests and runs)
| gate | result |
| --- | --- |
| forward == production | G1 bitwise (both arms); bench 0 of 24M (8e6), 0 of 75M (2.5e7); CPU per kernel and per call order bitwise (new tests fail on the base: the table layout / leaf-major L2P, 1471+ elements) |
| FD float64, whole force, positions / masses / G / eps2 | 1e-6 / 1e-6 / 1e-10 / 1e-6 (CPU, every kernel's reverse in interpret mode, the symmetric sort-free M2L reverse, the COM ancestor reverse) |
| `dehnen_error` at frozen lists | forward bitwise, FD 1e-6 in positions and masses; other `force_params` refused |
| symmetric reverse, COM reverse (unit) | the sort-free M2L reverse = the sorting one to 1e-13 and `jax.vjp` of the reference to 1e-9, no sort; the COM reverse = autodiff of the prefix sums to 1.9e-15 / 3.9e-15, no float scatter |
| no buffer of 16 N floats | largest: 9.44 N floats at 8e6 (`[nodes, 36]` tables, pow2 capacity; 12.85 with p6's `[nodes, 49]`), 6.0 N floats at 1e8 (the COM forward's `f64[N, 3]`) |
| CPU program identity | 36/36 identical to `s3b-integ-cpu` and `s8-integ-cpu` |
| GPU pins on the tip (`s8w-tip` vs the lead's `s8-integ`, deterministic ops) | S1, S2, S3, S4, S5, S6, S1p, S1t, S1e, S1x, S1g bitwise; **S4b moved by design** (the large-N gradient lane: 57,392 of 60,000 position and 19,156 of 20,000 mass elements, rel-L2 3.5e-7 / 3.2e-7); G1 passes bitwise (its production arm bitwise the reference's) |
| DET (production flags, card 7 free before and after both) | `s8w-det` vs `s8w-det-b`: DET1, DET2, DET3 bitwise A-vs-A. Against `s3b-det-free`: DET1 and DET3 bitwise, **DET2 moved by design** (57,458 position / 19,161 mass elements, rel-L2 3.5e-7 / 3.1e-7): the new DET2 reference is `s8w-det` (+`-b`) |

### S8 memory and time (whole force: `jax.value_and_grad` / `jax.grad` of `sum(w . acc)` in positions and masses, input order; G1 operating point)
| N | leaf capacity | temporaries v&g / grad | device peak | largest array | A-vs-A |
| --- | --- | --- | --- | --- | --- |
| 8e6 | pow2 | 1.98 GiB (265.8 B/p) / 1.71 GiB (229.1 B/p) | 3.65 GiB (v&g) | `[nodes, 36]` tables, 9.44 N floats | bitwise |
| 8e6 `preset="galaxy"` (`dehnen_error` 1e-4, p6, eps 1e-3; 28.5M far pairs in a 179M capacity) | pow2 | 2.48 GiB (332.9 B/p) / 2.01 GiB (270.0 B/p) | 12.06 GiB (the prepare's) | `[nodes, 49]`, 12.85 N floats | bitwise |
| 2.5e7 | pow2 | 5.18 GiB (222.6 B/p) / 4.33 GiB (186.0 B/p) | 9.42 GiB (v&g) | `[nodes, 36]`, 6.04 N floats | bitwise |
| 1e8 | auto | 19.15 GiB (205.6 B/p) / 13.92 GiB (149.4 B/p) | **grad 28.29 GiB (30.4 GB)**; v&g 33.4 GiB estimated (it ran out of memory beside another user's 3.9 GB) | the COM forward's `f64[N, 3]`, 6.0 N floats | bitwise (grad) |

S8-F's far chain alone at 8e6 `preset="galaxy"` needed 5.95 GB of temporaries (the by-source sort over the 179M
capacity); the whole force now 2.48 GiB. Every row: the differentiable forward 0 elements off the production eval.

Time, final tip (`454f1476`), one A100 with no other process before or after any run (card 4, 2026-10-22), each
row twice in fresh processes (min of 7 calls; the force eval is `evaluate_prepared_state`):

| N | force eval | `jax.grad` | `value_and_grad` | peak |
| --- | --- | --- | --- | --- |
| 8e6 (pow2) | 30.10 / 30.25 ms | 131.47 / 131.42 ms (**4.37x / 4.35x**) | 161.53 / 161.54 ms (5.37x / 5.34x) | 3.64 / 3.60 GiB |
| 1e8 (auto capacity) | 347.6 / 346.4 ms | 1437.4 / 1436.5 ms (**4.14x / 4.15x**) | out of memory (below) | 28.29 / 28.21 GiB (grad) |

At 1e8, `value_and_grad` failed on the free card in both runs and again with `TF_GPU_ALLOCATOR=cuda_malloc_async`.
The allocator could not place its 19.15 GiB temporaries as one block beside the prepared state (15.2 GiB in use after
the prepare, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`). `jax.grad` alone (13.92 GiB of temporaries) fits.

Kernel time of the 8e6 value_and_grad (A100 trace, 162.0 ms per call): the near reverse 59.9 ms, the near forward 27.3
(dead under `jax.grad`), the M2L reverse 23.4 by source + 16.5 by target, the M2L forward 8.2, the cascades 10.7,
the XLA scatters 3.7 (17.6 with the prefix-sum COM reverse), the L2P and P2M reverses 2.0 each.

### For S9 and later
- **The time target at 8e6 (5x) holds for `jax.grad` (4.37x, and 4.14x at 1e8), not quite for `value_and_grad`
  (5.37x).** At 1e8, `value_and_grad` does not fit on a 40 GB card: its 19.15 GiB of temporaries do not fit as one
  block beside the prepared state. That needs the forward's arrays released before the reverse, or `jax.grad` plus
  one force eval.
- **The near reverse** is the largest item (59.9 ms, 2.2x its forward): S8-N's levers.
- **The M2L reverse's two passes** read the same CSR now; one launch computing both halves per row would read it
  once (40 ms).
- **The COM forward's float64 prefix sums** are the largest arrays at 1e8 (2.4 GB each, several live); they are the
  production refresh's too.
- **The input-order permutation**: its transposes are XLA scatters (3.7 ms at 8e6); a permutation `custom_vjp`
  would gather both ways.
- **GPU program identity** of the final tip was not recorded (CPU identity, the pins and DET are).

## S11, session 1 — the fused multi-GPU lane on the runner's machinery (2026-10-22/23)

Plan S11: the fused multi-GPU lane gets what the single-card runner has (S2-S3): a shape template instead of
prepared states, one program per segment with the state donated, a segment retry, capacities sized in the library.
Four tracks on main `90ca9f38` (S8 merged): **A** (`s11/mesh-segment-runner`) the template force and the segment
runner; **B** (`s11/cross-capacities`) the cross capacities in the library; **C** (`s11/cross-csr-merge`) the cross
far pairs into the walk's CSR; and in yggdrax `fix/ragged-exchange-clamp`. Then the wiring on top
(`s11/mesh-library-caps`). The old route (`setup_fused_force`, the stacked evaluator, `FusedRollout`) is unchanged
and stays the reference. The single-card lane is untouched: CPU program identity 36/36 against main.

**No card was free this session.** Every two-card row below ran on cards 2 + 4 with other users' processes on them
(card 2 at 98-100 % utilisation during the library-caps rows): correctness and memory are results, times are not.

| target / gate | result |
| --- | --- |
| the template force == the stacked force (A1) | bitwise on two forced CPU devices, every value finite (no template value is read), same plan and walk caps |
| a one-device mesh segment == the single-card particle carry (A2) | bitwise (a user-written scan over `prov.step`, the same explicit box) |
| segments chain, repartitions move only ownership (A2) | 2 + 2 steps == 4, bitwise by id (two devices, cross field on); a repartition before every segment == none, bitwise by id (partition-independent direct force); a carried acceleration left in the old rows is caught |
| a segment retry == a fresh run with the final caps (A3) | bitwise: a leaf capacity of 44 grown to 1024, the walk caps and the cross caps re-planned; and an undersized `recv_node_cap` / `send_particle_cap` |
| a mesh run needs no bench formula and no Odisseo harness env | yes: Plummer and the disc + bulge at 2e5 and 2e6 on two A100s, six segments with repartitions, every flag clean |
| accuracy in class with one card | rel-L2 vs fp64 direct (512 ids), mesh start / end vs one card on the same particles: Plummer 2e6 5.19e-4 / 4.99e-4 vs 5.10e-4 / 5.04e-4; disc 2e6 3.19e-4 / 3.14e-4 vs 3.26e-4 / 2.93e-4 |
| peak per device | Plummer 2e6 (1e6 per card): **412 B/p** on library caps (rerun on a run's final caps), 962 B/p with the bench formulas, 1214-1257 B/p on the old route; between segments 51-76 MB against 297-347 MB |

### The template force and the segment runner (A)
- `jaccpot/distributed/template_force.py`: per device (mesh order, on that device) a caps-only prepare in the GLOBAL
  box, the level registry lifted to the refresh tree over the live rows, plans and walk caps merged and installed,
  `leaf_capacity="auto"` sized once from the largest device; ONE shape template and provider for every device; the
  eager prepares dropped. Inside the `shard_map` each device rebuilds from `provider.like()` in the global box over
  its live rows, with the cross hook between the sweeps; the cross near term is the old route's
  (`fused._add_cross_near_term`, factored out, ops unchanged). Its verdict is the provider's bits plus two for this
  lane (`OVERFLOW_LEVEL_PLAN`, `OVERFLOW_CROSS`) and its needs the walk's and tree's plus the cross needs (B).
- `jaccpot/distributed/segment.py`: `MeshSegmentRunner`. A segment is ONE program, jit(shard_map(lax.scan)), with
  the packed state and the total acceleration donated and the single card's velocity-Verlet association; the start
  force is its own program; the verdict is reduced over the mesh once per segment and read once. Between segments a
  repartition (its own program) routes the velocities, the total acceleration and the ids with the particles. The
  old route ran 3-4 programs and host syncs per step: at 2e6 on two contended cards 68.1 against 25.9 ms per step,
  with the force alone 26.7 against 28.0 ms.
- The retry: needs reduced over the mesh, the engine's walk and tree re-plan, the cross re-plan, caps-only
  re-prepares at the segment's start (a host copy of the donated start state), at most twice.
- The template route honours `TreeConfig(cubic_bounds)` (the galaxy preset's cube); the old route reads only
  `JACCPOT_CUBIC_BOUNDS`. Built with `mac_type="dehnen"`: `dehnen_error` stays refused at every entry point (S12).
- CI: job `test-distributed-segment` (marker `distributed_segment`, two forced devices, path-gated on
  `jaccpot/distributed/` and the runtime files the template and verdict come from).

### The cross capacities in the library (B)
- `CROSS_NEEDS_FIELDS`: the demand of every capacity the default (symmetric) exchange reads, traced next to its
  flags (`hook.needs_sink`, `cross.cross_needs`), exact past its own overflow except where a stopped walk makes it a
  lower bound (two bits say which). `CrossCapacities.for_shard` (a first guess from the leaf capacity), `from_needs`
  (1.5x, the summary clamped to the tree), `replanned` (monotone), `plan_cross_capacities`.
- At 1e6 per card (Plummer, N = 2e6) the library caps sit at 1.50-1.51x need (the summary at 1.15x); the probe's
  formula at 3.5-14x and the rollout gate's at up to 235x (the walk queue). The force program's temporaries per
  device: 523.8 MB (library), 731.0 MB (probe formula), 1207.4 MB (rollout formula).
- **Bug fixed: a padded shard's summary cell spanned all its dead rows.** Padding leaves carry the empty range
  `(n, n - 1)`, so the parent of the last live leaf spans `[that leaf's start, n - 1]`. Under one-leaf summary cells
  without `summary_cell_level` that parent is a cut cell, and the near export laid out every dead row: the force was
  right (zero mass) but `full > W` fired at any width (two A100s, the rollout gate's caps, 2e5 per card, on main).
  Ranges are now clamped to the last live particle.
- An overflowed cross-walk queue doubles on a re-plan: the Pallas walk reports the widest wavefront it admitted,
  which read 252,788 off a 389,120 queue that overflowed (disc 2e5; clean at 4x with a peak of 375,310).
- **The flat M2L's chunk switch reassociates.** On the flat lanes (CPU, pre-Ampere,
  `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0`) `pair_count <= m2l_chunk_size` selects the full-batch or the chunked
  accumulation, and `pair_count` includes the cross list's capacity: crossing it moved 2347 of 6144 components by
  1 ulp. The `FarFieldConfig.m2l_chunk_size` docstring said it did not change the result; corrected.

### The cross far pairs into the CSR (C)
- The walk's far list reaches the downward sweep target-sorted; the cross pairs were appended as COO and the whole
  list sorted again by target (an `s32[P + E]` sort over the far capacity, 20,971,520 entries at 1e6 per card).
  `merge_into_csr` sorts the E cross pairs only and slots each behind its row's own entries. The old sort was stable
  and the CSR M2L kernel sums a row in list order, so the locals are bitwise the old route's: 0 of 6,553,575 M2L
  outputs differ on captured GPU inputs at 1e6 per card. The flat lanes keep the concatenation (their chunked sums
  would reassociate). The list build alone at 1e6 per card, one contended card: 5.4-7.2 -> 1.6-1.9 ms (minimum).
- S12 needs `FarForceScaleInputs` over the imported nodes; the force-scale kernel reads the same presorted list.

### The ragged exchange stays in its buffers (yggdrax)
- `ragged_all_to_all_exchange`'s native (NCCL) path handed XLA the requested sizes unclamped. On a capacity overflow,
  which the cross exchange detects only after it from the returned sizes, it wrote and read out of bounds: on two
  A100s 512 rows written past the output, 1024 rows read from beyond the send buffer (padded-buffer canaries).
  `_clamped_transfer` derives one clamped transfer matrix from the gathered sizes on every device; the requested
  sizes are still returned, and a fitting exchange is unchanged. Merge it before the jaccpot PRs.

### Re-plans from the steps on the trajectory only (fix 8)
- At 8e6 (4e6 per card) the first segment's cross flag fired at step 1. The scan kept stepping on that step's wrong
  force (one device's acceleration 1.5e8 against 49 at the start, NaN by step 4), and the re-plan folded every
  step's needs by max: on steps 5-8 each shard's root had become one summary cell of 4,002,489 particles, so the
  re-plan asked for a leaf width and particle caps of a whole shard (out of memory at 282.7 GiB, or past Triton's
  launch grid). A stopped cross walk was NOT the cause: it only under-reads (pinned, traced and Pallas walks).
- The segment program now folds bits and needs only from steps that started on the trajectory (the failing step
  included); the verdict is mesh-wide, one scalar all-reduce per step. `leaf_width` never exceeds the template's
  leaf size, the particle caps never exceed rows x `leaf_width`, and a run whose tree overflowed re-plans only the
  cross summary (the rest re-measures on a tree that fits).
- Two A100s (card 0 shared, so times are contended), Plummer, library caps, 6 segments x 8 steps, geometric MAC:

  | N | retries | rel-L2 mesh start / end | one card, same particles | peak | step (contended) |
  | --- | --- | --- | --- | --- | --- |
  | 8e6 (4e6 per card) | segment 1: 2 | 4.3568e-4 / 4.3011e-4 | 4.3159e-4 / 4.5346e-4 | 662 B/p (2.65 GB) | 222 ms |
  | 2e7 (1e7 per card) | segments 1 and 6: 1 each | 4.3331e-4 / 3.9118e-4 | 4.4181e-4 / 4.3008e-4 | 644 B/p (6.44 GB) | 1328 ms |

- The cross near pairs jump from 129,544 at the start positions to 360,247 at the first step and fall back to ~150k
  (8e6): not diagnosed (the start sizing's undersizing that the first segment's retries absorb).
- The single-card scans fold their needs the same way (every step); that is fixed on its own branch.

### Not done yet (S11 session 2) and found on the way
- **The runner's multi-GPU mode** (edges, diagnostics, restart, energy: the cross near term has no potential yet)
  and **Odisseo's multi-GPU lane** off the mesh lane (`fmm_mesh` -> the segment runner).
- **Measurements on free cards**: 2/4/8 cards, B/p at production particles per card, weak and strong scaling, the
  8-card ceiling. The small-N rows above are dominated by fixed buffers (disc 2e5: 1437 B/p).
- **The first segment retries once**: one drift step after a decomposition puts leaf cells inside the neighbour's
  domain and roughly doubles the cross near demand (export_near 11,603 -> 22,795, particles 36,231 -> 87,867 at
  Plummer 2e5), which the start sizing (1.5x) cannot see. The grown caps hold for the rest of the run; measuring
  the sizing also at drifted positions would save that recompile.
- yggdrax `let.py` (`import_near_halo`, round A, the old mesh lane) has no flag for more requests than
  `max_recv_leaves`.
- pydoclint's exclude pattern `dist` also matches `jaccpot/distributed/`, so that package is never linted (50
  violations on main).

## S12 — `dehnen_error` on the fused multi-GPU lane (2026-10-23)

Plan S12, the user's decision D2: the multi-GPU lane runs the production MAC (since 2026-10-21 `preset="galaxy"` is
`dehnen_error` at `adaptive_eps=1e-4`), its cost measured. Built on the S11 stack (`s11/7-runner-library-caps`,
PRs #458-#464), branch `s12/mesh-dehnen-error`; yggdrax #103 (the clamped ragged exchange). Only the NEW path, the
template route (`prepare_mesh_force`, `FusedMeshForce`, `MeshSegmentRunner`), runs the criterion; the stacked route
(`setup_fused_force`, `make_fused_force_evaluator`, `fused_force_step`, `FusedRollout`) keeps refusing it.

### What S12 built
| piece | change |
| --- | --- |
| f_b's far half behind a cross list | `_far_force_scale_inputs` returned `None` behind a cross list, and the separate pass then summed the LOCAL far pairs only: the other domains' far field was missing from the next thresholds. The imported sources' masses now ride next to `cross_far`: on the GPU route `FarForceScaleInputs` covers `[local ; imported]` (the CSR M2L lanes kernel reads mass and centre by source index; the merged CSR of S11-C puts the cross pairs in the rows), on the separate pass (CPU) `cross_far_force_scale_own` adds the cross pairs' own terms before the push-down |
| the cross near term's f_b | the leaf-pool CSR kernel gets the sorted kernel's force-scale lane (`with_force_scale`: `G m / (r^2 + eps^2)` in lane 3 instead of the potential, the same launch); `_add_cross_near_term` adds the share to f_b |
| the criterion in the cross walks | the thresholds are computed before the hook (`_call_cross_hook` hands it this step's `FlatWalkCriterion`); each device publishes, behind its summary rows, every published node's eq (16a) table row (G M, `eps * min_b f_b` of its subtree, `s_1..s_p` on the published radius): p + 2 more columns of the one all_gather. `_symmetric_walk` gives the gathered table to `cross_walk_fn`'s criterion walk (the Pallas walk's `error_table`, or yggdrax's `pair_accept=DehnenWalkAccept`), so both devices evaluate every cross pair from the same floats. The far payload carries each node's mass moment (one column). Under the criterion `make_cross_hook(error_criterion=True)` refuses every control exchange and the `aabb` geometry |
| f_b through the runner | the seed: each device's caps-only prepare's `_fused_force_scale_seed`, in input row order. It is local-only, a lower bound of the single card's seed (f_b sums positive terms; this one leaves out the other domains'), so the first force is stricter, never looser: the single card's own contract holds. Then f_b goes in and out of the start program, rides the scan carry, `MeshState` and the repartition payload; a retry restarts from the re-prepare's seed |

The criterion-aware exchange gates stay correct: under the criterion only the symmetric exchange runs (a separate
near or export theta, `JACCPOT_CROSS_SYMMETRIC=0`, tiles, a receiver walk or `max_leaves_per_cell != 1` select a
control whose walks re-test imported rows without a table: refused at construction). No new capacity: the table
columns are a static width of the summary, the mass a column of the far payload, the cross near f_b the same launch
as its force, so `CROSS_NEEDS_FIELDS` still covers everything the criterion path reads.

**Found on the way (GPU only):** the first version shipped the mass as `where(alive[:, None], mass[rows], 0)`, a
`(rows, rows)` broadcast: on two A100s at 2e6 the start program asked for 215 GiB (323 GiB of temporaries at compile
time), and the receiver read every row's mass as the sender's first row's. At the CPU gates' size the payload was
1024 wide, nothing overflowed and every number but the cross far f_b looked right. Fixed before the PR; gate (b) now
checks each imported far row's mass against its owner's published G M (the broadcast fails it: 24 of 27 rows).

### Gates (CPU, two forced devices; N = 2048 uniform cube, leaf 64, p4, eps 1e-4)
| gate | result |
| --- | --- |
| (a) one device == the single card's particle carry with f_b | bitwise: state 0/6144, acceleration 0/3072, f_b 0/1024 differ (N = 1024, 3 steps, same box, the single card's own seed; the mesh's seed is that seed bitwise). A wrongly ordered seed moves 1005/6144 state elements |
| (b) both devices decide every cross pair alike | identical gathered summary (14 columns: 8 + p + 2) and identical pair sets on both devices (186 far, 870 near) |
| (b) eq (16a) on every cross far pair | evaluated independently in float64 on the gathered table: max ratio 0.9995 (< 1), convergence <= 0.43, gap past the floor >= 0.56; all 870 leaf-leaf near pairs fail it. Mutation (geometric cross walks under the criterion): 329/329 far pairs fail it (max ratio 241) |
| (b) f_b | mesh vs one card on the same particles: median 3.7e-3, max 3.9e-2 (both estimators against the exact sum: 4.3e-3 / 3.1e-2 mesh, 6.6e-3 / 4.2e-2 one card); the cross f_b terms dropped: 0.21 / 0.61 against the exact sum, the local-only sum's spread. The lanes-kernel route (interpret) and the separate pass: f_b identical. Every imported far row's mass is its owner's published G M, bitwise |
| (b) force | rel-L2 against the direct sum: mesh 2.63e-5, one card 3.30e-5 (theta 0.8 on the mesh: 3.64e-3); 2 + 2 steps == 4 bitwise with f_b carried |
| (c) a retry under the criterion == a fresh run with the final caps | bitwise, f_b included (leaf capacity 44 outgrown) |
| f_b rides the repartition | a stand-in f_b (id + forces) after 10 x 2 steps with a repartition before every segment: `id + 21` everywhere, == never repartitioned; unrouted, it breaks |
| single-card lane | CPU program identity 36/36 against `s11-stack-cpu` |

The segment job (`test-distributed-segment`) grows from 4 to 8 cases: 1363 s of calls on this host (700 s wall at
`-n 2`), under the job's 40-minute cap at CI's `-n 1` if its runners are no slower than this shared host.

### On two A100s (cards 2 + 4)
`bench/multigpu_segment_gate.py` (`preset="galaxy"`, leaf 64, p6, library cross capacities sized at the start,
segments of 8 steps with a repartition before each, rel-L2 against an fp64 direct sum on 512 fixed ids). Card 4 held
another user's training job (60-100 % utilisation) until ~05:05; the rows after that ran with no other process on
either card ("free"; card 2 shows 427 MiB held but lists no process).

| row | MAC | rel-L2 start / end | one card, same positions | peak B/p | flags |
| --- | --- | --- | --- | --- | --- |
| Plummer 2e6 | `dehnen_error` 1e-4 | 1.324e-4 / 1.636e-4 | 1.767e-4 / 1.672e-4 | 671-676 (475-487 on explicit caps) | 10 segments, 9 repartitions, clean |
| Plummer 2e6 | geometric 0.8 | 5.191e-4 / 4.770e-4 | 5.098e-4 / 4.507e-4 | 662 (412 on explicit caps) | the same |
| disc+bulge 2e6 | `dehnen_error` 1e-4 | 6.212e-5 / 8.200e-5 | 1.026e-4 / 8.859e-5 | 842-846 | 6 segments, 5 repartitions, clean |
| disc+bulge 2e6 | geometric 0.8 | 3.188e-4 / 3.141e-4 | 3.260e-4 / 2.926e-4 | 961-962 | the same |

The Plummer rows' first segment retries once (the S11 drift effect), the disc rows' fifth, on both MACs.
"Explicit caps" reruns on the first run's final cross capacities (`--cross-json`), without the sizing measurement in
the process: the S11 record's 412 B/p, now +15-18 % under the criterion.

**Time, Plummer 2e6 (1e6 per card), free cards, the arms interleaved from one frozen tree:**

| | `dehnen_error` | geometric | ratio |
| --- | --- | --- | --- |
| mesh step (segment wall / 8, steady state) | 32.67 / 32.49 ms | 21.74 / 21.82 ms | 1.50x / 1.49x |
| mesh force alone (start program, min of 9) | 33.21 / 32.91 ms | 22.49 / 22.30 ms | 1.48x |

The lists explain it: at 1e6 per card the criterion's lists are longer than theta 0.8's (local far +26 %, near
+26 %, cross far pairs +55 %, cross near +14 %, the particles shipped equal), as on one card at this N
(`docs/dehnen_error_fused_2026-10.md`: the criterion's lists per particle shrink with N; at 25M they match theta
0.8's, hence 1.12x there).
One card on the same 2e6 particles (free card 2, both arms in one process, 40 interleaved rounds of the jitted
`prov.step`): `dehnen_error` 29.90 / 30.25 ms (min / median), geometric 18.90 / 19.01 ms, **1.58x**. The mesh's 1.49x
is the single card's ratio at this N, not a cost of its own; at 25M on one card it was 1.12x.

**8e6 and 2e7 (4e6 and 1e7 per card).** At first 8e6 did not run on either MAC: the re-plan folded the needs of
the steps AFTER a failed one, whose force had gone to NaN, and asked for a whole shard (4,002,489 particles) as a
leaf width. Fixed in S11 (fix 8: the re-plan reads only the steps that started on the trajectory; a segment retries
while each re-run fits more steps). On it, Plummer, `dehnen_error` 1e-4, library caps, 6 segments x 8 steps with a
repartition before every segment after the first, cards 0 + 1 (both shared with other jobs at 50-100 %, so no time
here is a result):

| N | retries (segment) | bits | rel-L2 mesh start / end | one card, same particles | peak |
| --- | --- | --- | --- | --- | --- |
| 8e6 | 1 (1), 1 (6) | 0 every segment | 1.3188e-4 / 1.5194e-4 | 1.6033e-4 / 1.5855e-4 | 672-676 B/p (2.70 GB) |
| 2e7 | 1 (3) | 0 every segment | 1.2802e-4 / 1.9313e-4 | 1.8742e-4 / 1.8110e-4 | 633-642 B/p (6.42 GB) |

The geometric MAC on the same code (S11 fix 8's rows): 8e6 4.36e-4 / 4.30e-4, 2e7 4.33e-4 / 3.91e-4, peaks 662 and
644 B/p. The large-N time ratio needs free cards.

### Open after S12
- The large-N time ratio of the criterion on the mesh (8e6 and 2e7 ran on shared cards).
- Four cards (none free this session) and timed disc rows (card 4 was shared during them: the disc's 307 against
  361 ms per step is contended and is not a result).
- The per-device seed's far half is the single card's `far_force_scale_sorted` (a scatter-add): like the single
  card's seed, not bitwise reproducible on a GPU; every later f_b is (the lanes kernel and the cascade).
- The runner's multi-GPU mode (S11 session 2) takes the criterion with no further change to the lane.

## S11, session 2 — the fused multi-GPU lane in the runner (2026-10-23/24)

Plan S11, session 2: the production runner's multi-GPU mode, Odisseo off the old mesh lane, and what session 1 left
open (the first segment's retry, the cross near volume, the free-card measurements). On main `e82389b5` (S11 session
1, S12 and #469 merged), in PR order: **E** the mesh potential, **D** the runner's multi-GPU mode, **F** sizing at the
drift and the same-step retry rule, **G** the 2-GPU force bitwise run to run, **I** migration to fixed cuts, **N** the
mesh's memory (from track **M**'s stage-by-stage diagnosis); in Odisseo **H** its multi-GPU lane on the runner; **P**
the close-out baselines on main; **T** the free-card measurements. The single-card lane is untouched throughout: CPU
program identity 36/36 against `s11s2-main-cpu` after every track.

| target | result |
| --- | --- |
| the runner on several cards | `ProductionRunner(mesh=MeshRunConfig(...))`: segments, edges, diagnostics with the energy, restart files, the retry, by-id input and output; CPU gates bitwise (the mode == the segment runner by hand, chained == one run, a fresh-process restart == the straight run) |
| the energy on the mesh | `FusedMeshForce.potential_shard`: phi rel-L2 1.4e-6 against fp64 direct (`dehnen_error`, 2e6 on two A100s), W against one card 2.6e-8 (Plummer) and 9.4e-8 (disc + bulge) |
| no first-segment retry | the cross capacities sized also at the first segment's drift: 0 retries on every gate row (Plummer 2e5 / 8e6, disc 2e5 / 2e6, both MACs) |
| bitwise run to run on GPUs | the cross pairs in (target, source) order, their rows summed in a fixed order: the 2-device start force bitwise in one process and across processes (Plummer 4e5 / 2e6, both MACs), a restart's state bitwise; open: ~8.5k of 6.9M components on the disc 2e6, from the single card's near kernel (not changed) |
| the cross near volume | migration to fixed cuts (`migrate=True`, opt-in): Plummer 8e6 on two cards 344-537k -> ~97k cross near pairs, 3.4-4.0M -> 318k particles shipped per step, 7.4k migrants; the force 219-262 -> 51-54 ms (contended cards) |
| the mesh's memory | the cross near term on the local near field's direct layout, the sizing measured on small buffers: per device a compact Plummer at 2^19 per card 1,474 -> 619 MB, Plummer 8e6 662 -> 342 B/p, the disc + bulge 2e6 796 -> 766 MB |
| Odisseo | `fmm_preset="galaxy"` + `fmm_mesh=True` + `MeshRunnerOptions` runs the runner's multi-GPU mode; `MeshOptions` keeps the old lane |
| free-card scaling | track T (below) |

No timing in the subsections below is a free-card result unless it says so: the cards were shared with other users'
jobs for most of the session.

### The mesh potential (E)
The runner's multi-GPU mode reports the energy (`E = K + W_self + W_ext`), and the mesh had no potential: the cross
near kernel's fourth lane carried f_b (S12) or nothing. Branch `s11s2/mesh-potential`.

| piece | change |
| --- | --- |
| the cross near pairs' potential | `cross_near_acceleration(with_potential=True)`: the leaf-pair kernel's fourth lane in its default meaning (`-G m psi`, accumulated beside the force, as the sorted kernel's `with_potential`), which the force drops; exclusive with `with_force_scale`. The kernel is unchanged. `fused._add_cross_near_potential` adds it in input row order from the pool and pairs the force's term reads (`_cross_near_pool`, factored out of `_add_cross_near_term`) |
| `template_force.mesh_potential_shard` | the single card's energy program (`ProductionRunner.potential_energy`) on the template route: the force's rebuild (the global box, the cross hook between the sweeps, under `dehnen_error` eq (16a)'s thresholds from the given f_b, so the lists are the force's), `FusedForceProvider.potential` (local near lane 4, the far L2P value of the local expansions, which hold the cross far pairs' M2L: the force's L2P reads the same locals), plus the cross near pairs' potential; zero on padding rows; the bits and needs of the rebuild (`_shard_verdict`, factored out of `mesh_force_shard` with `_shard_rebuild`) |
| `FusedMeshForce.potential_shard(positions, masses, live, ids, force_scale=None) -> (phi, bits)` | the force protocol method `MeshSegmentRunner.potential_energy` calls inside its `shard_map`, like `shard`. No capacity of its own |

The runner's energy program (`MeshSegmentRunner.potential_energy`) calls it per device inside its `shard_map`, as
its start program calls `shard`: `phi, bits = force.potential_shard(state[:, 0], masses, min(count[0], cap), ids)`,
with `force_scale=<the carried f_b>` under the criterion (required: the rebuild refuses without it), under
`force.context()`, `force.before_trace()` before a fresh trace, the engine's trace-time state kept around the trace
(`production._trace_state_kept`, as the single card's energy program). `phi` is in the positions' dtype and zero on
padding rows, so `W_self = 1/2 psum(sum(float64(m) * float64(phi)))` needs no mask; the energy is valid when
`overflow_ok` holds for the bits OR-ed over the mesh (`segment._mesh_verdict`) and every count fits its shard.

No force program changes: the template route's start and segment programs (geometric and `dehnen_error`) and the
stacked route's force program, lowered on two forced CPU devices (`as_text(debug_info=False)`), are text-identical to
main `e82389b5` (an A-vs-A control on main is identical too). No single-card file is touched.

Gates (`tests/distributed/test_mesh_potential.py`: two forced CPU devices, N = 2048 uniform cube, leaf 64 with leaf
capacity 128, p4, eps 1e-2 ferrers3; 386 s of calls for the four cases on this host, in `test-distributed-segment`):

| gate | geometric (theta 0.8) | `dehnen_error` 1e-4 |
| --- | --- | --- |
| one device == the single card's potential, same box (N = 1024) | bitwise | bitwise, from the same seed f_b (the mesh's seed is the single card's, bitwise) |
| phi vs the fp64 direct sum (all pairs, the same kernel), rel-L2 / max: mesh | 1.862e-4 / 1.118e-3 | 1.010e-6 / 1.125e-5 |
| one card on the same particles | 1.878e-4 / 1.113e-3 | 1.275e-6 / 1.249e-5 |
| `W = 1/2 sum m phi` vs direct: mesh / one card (bound) | 1.76e-5 / 1.94e-5 (1e-4) | 1.13e-8 / 1.84e-8 (1e-6) |
| mutation: W without the cross near term / without the cross far pairs | 7.1e-2 / 3.3e-1 | 3.5e-1 / 4.5e-2 |
| the potential's lists | -- | its needs equal the force's at the same f_b, and move with it (4x f_b) |

Two A100s (`bench/multigpu_potential_gate.py`, `preset="galaxy"`, leaf 64, p6, library cross caps; cards 0 + 7, both
shared with other users' jobs at 45-100 % utilisation: the times are contended and are not results). phi against an
fp64 direct sum on 512 fixed ids; one card on the same positions (its own start force's f_b); temporaries from
`memory_analysis()` per device; the peak is the process's, set by the setup's cross sizing (the start force's
program), and the potential did not raise it:

| row | phi rel-L2 / max, mesh | one card | W mesh vs one card | temporaries potential / force | peak per device | potential / force, median (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Plummer 2e6, `dehnen_error` | 1.408e-6 / 8.88e-6 | 1.536e-6 / 8.72e-6 | 2.6e-8 | 390.3 / 371.3 MB | 671-676 MB | 125.8 (107.3) / 165.3 (126.3) ms |
| disc + bulge 2e6, `dehnen_error` | 1.359e-6 / 8.48e-6 | 1.477e-6 / 8.04e-6 | 9.4e-8 | 675.2 / 682.6 MB | 842-847 MB | 635 (597) / 745 (629) ms |
| Plummer 2e6, geometric 0.8 | 6.739e-6 / 4.146e-5 | 7.083e-6 / 1.454e-4 | 2.2e-7 | 294.0 / 308.9 MB | 662 MB | 219 (206) / 200 (164) ms |

Every potential's bits were clean and its padding rows zero; no start force retried.

**Found on the way (GPU only): the mesh force is not bitwise reproducible run to run.** A-vs-A on two A100s (cards 0 +
6, then 0 + 7; Plummer 2e6, `dehnen_error`, the same program on the same inputs twice): 2,664,694 of 6,900,000
acceleration components differ (max 4.8e-7 at |a| <= 21.4, ulps) and 455,887 of 2,300,000 potentials (max 3.0e-7);
with `--xla_gpu_deterministic_ops=true` 2,706,687 and 482,140. Without the cross hook (each device's own particles)
both are bitwise (0 differ); without the cross near term the potential still differs (472,909). The symmetric cross
walk (Pallas) returns the same pair SET in a different ORDER each run (507,778 far and 73,840 near pairs per device;
1,012,559 of the 1,015,556 live far slots and 146,730 of the 147,680 near ones hold another pair), and the stable sort
that slots the cross pairs into the CSR rows keeps that order, so the M2L (and the cross near kernel) sum each row in
a run-dependent order. Not the potential's: it inherits it from the rebuild. A canonical order (sorted by target, then
source) should make both reproducible (not tried: it changes the force program). On CPU the lane's bitwise gates hold
(S11 A2/A3, S12 (a)-(c)).

### The runner's multi-GPU mode (D)
`ProductionRunner(solver, ..., mesh=MeshRunConfig(make_mesh(n)))` (and `production.run(..., mesh=...)`) runs the
single card's loop on `MeshSegmentRunner`: segments, edges and callbacks (`SnapshotWriter`, `DiagnosticsLog`, a
truthy return stops), the diagnostics with the energy (`MeshSegmentRunner.potential_energy`: the force's
`potential_shard` per device, fp64 sums, a program of its own whose trace leaves the engine as it found it), restart
files and their continuation, the segment runner's retry, the plan record (caps, cross caps, repartition stats).
Input and output by id (`rollout.decompose` with `mesh_shard_capacity(N, ndev)` = ceil(ceil(N / ndev) x 1.15) rows
per device; `RunResult.state` gathered by id on the host; `RunResult.handle` the segment runner, so a continuation in
memory is bitwise); `dt` and the external field's parameters traced, G and the softening the force's own
(`force_params` must equal them); edges gather by id only when asked (`MeshSegmentEdge.state_host()` / `.gather()`).
A restart on the mesh that wrote it rebuilds the identical layout and plan (leaf capacity, walk caps, depth bound,
cross caps, level floor and plan, template shapes; the force key's digest checked); on another mesh it decomposes the
by-id state (`restored["bitwise"] = False`). Backend: `jaccpot/distributed/mesh_runner.py`; the single-card path is
unchanged (CPU program identity 36/36). Branch `s11s2/runner-mesh-mode`; docs: `production.md` "Multi-GPU mode".

| gate (CPU, two forced devices) | result |
| --- | --- |
| the mode == `MeshSegmentRunner` driven by hand (direct force; the fused force with a retried segment) | bitwise by id at every edge |
| chained runs (`restart_from=RunResult`) == one run | bitwise |
| a restart file continued in a FRESH process, same mesh (direct force; fused force from a file written after a retried segment) | bitwise, digests matching |
| restored on one device | allclose 1e-12, `bitwise = False` |
| moments and energy against the host fp64 sums of the gathered state | 1e-12 (the summation order only) |
| energy diagnostic on / off | bitwise |
| a stop callback; a retry shown on its edge; a failure names the last restart file | yes |

On two shared A100s (Plummer 2e6, the galaxy preset, 3 segments of 8 steps with repartitions, every diagnostic):
flag-clean but the first segment's single retry (fixed by F), rel-L2 1.793e-4 / 1.638e-4 at steps 8 / 24 against the
fp64 direct sum, 616-619 B/p per device. Its restart continued in a fresh process differed from the straight run in
165,799 of 12,000,000 state elements -- as much as a second continuation from the same file did: the lane was not
run-to-run deterministic on GPUs (found here; G fixed it). Odisseo's multi-GPU lane runs on this mode (H, below).

### Sizing at the drift and the same-step retry rule (F)
**The cross near "jump" was physics, and one number was mis-recorded.** Per-step needs (1-step segments, two
contended A100s, Plummer, theta 0.8) -- cross near pairs / particles shipped:

| state | 2e5 | 8e6 |
| --- | --- | --- |
| start (zero drift) | 11,603 / 36,231 | 129,544 / 430,677 |
| step 1 | 16,569 / 70,250 | 360,247 / 3,743,931 |
| step 2 | 17,257 / 72,153 | 424,282 / 3,838,602 |
| step 3 | 20,438 / 87,867 | 621,796 / 3,927,219 |
| step 4 | 22,234 / 87,232 | 399,351 / 3,828,308 |
| later | 20,077 (step 6) | 337,626-504,898 near, 1.85-3.86M shipped (steps 6-12) |

The start program at each step's positions gives exactly the scan's needs (no start-vs-scan difference), and a host
model (yggdrax's numpy cell leaves, COM and exact radius, the geometric MAC) reproduces the force's near pairs within
2 %. Mechanism: a decomposition leaves exact domains; one drift carries boundary particles into Morton cells the
OTHER domain fills densely, where the owner's tree, seeing only its stragglers, keeps one coarse leaf (8e6: a leaf of
19 particles, radius 3.07, 2.25 from the centre -- the box is thousands wide -- forms 194,255 of the 361,797 near
pairs after one drift, and device 0 ships 3.74M of its 4.0M particles). Only the start force runs at zero drift, so
sizing at 1.5x the start undersizes every later force. Corrections: S11's "after one drift step" 2e5 values (22,795 /
87,867) are the first segment's maxima at steps 5 and 3; at 8e6 the needs do not fall back to ~150k, they stay at
2.6-4.8x the start. The coarse leaves themselves are what I removes.
- **Drift sizing:** `cross_caps="auto"` also measures at x + s v dt for s = 1, 2, 4, ... up to the first segment's
  length (1 + ceil(log2 k) forces, once per runner) and sizes from the larger needs. First segments retry 0 times
  on Plummer 2e5 (main: segments 1 and 3 once each), the disc 2e5 (main: none) and 2e6 (main: segment 5 once) and
  Plummer 8e6 (main, fix 8's record: segment 1 twice), both MACs; final caps 1.12-1.50x the run's maximum need (main 1.01-1.62x); peaks +0.8 % at most on the
  run, -7 % to +0.3 % on the final caps; rel-L2 equal to ~1e-5 relative.
- **The same-step lower-bound rule** (#469's, now on the mesh): a re-run that failed at the same step as the attempt
  before is retried again while its needs grow past what that attempt's LOWER BOUNDS said (the local walk's
  `lower_bound` / `queue_overflow`, the cross walk's `walk_lower_bound`, the exchange's `pairs_lower_bound`). Test: a
  cross walk that stops at its first overflow is retried 10 -> 20 -> 40 -> 80 -> 160 pairs at one step and then equals
  a fresh run, bitwise (main: RuntimeError at "1 of 4 steps fitted").

### Bitwise run to run on two GPUs (G)
D's finding: the same compiled start force on the same state, three times in one process, differed in 612,657 of
1,380,000 components on two A100s (one device: bitwise; deterministic ops did not help). Stage by stage (every
intermediate of the start force, 3 runs): the tree, the multipoles, the gathered summary, both exchanged payloads,
the walk COUNTS and the CSR row offsets were bitwise; the symmetric Pallas cross walk returned the same pair SETS in
another ORDER each run (227,075 of 344,064 far and 39,311 of 61,440 near slots moved; atomic cursors), and
`merge_into_csr` and the cross near sort, both by target only and stable, kept that order inside each row, so the
CSR M2L and the cross near kernel summed rows in a run-dependent order. A second mechanism (`dehnen_error`, after the
first fix): 43-69 components in rows of three or more 64-leaf chunks, whose partials the table kernel reduced with a
`segment_sum` (a GPU scatter-add).
- **Fixes:** the cross far list and the near sink in (target, source) order (`cross._in_pair_order`, one composite-key
  sort); the cross near kernel's chunk partials summed in one fixed order (`ordered_rows=True`: first chunks gathered,
  the rest by one segmented `associative_scan`; rows of one or two chunks keep their bits). N's direct layout keeps
  both (its long rows' pieces in fixed order, `ordered_rest=True`).
- **Gates** (`bench/multigpu_force_repeat.py`, production flags): the start force 0 differing at Plummer 4e5 geometric
  and 2e6 `dehnen_error` (0 of 6.9M accelerations, 0 of 2.3M f_b), in one process and across three (main on the same
  cards: 605,714-622,174 of 1.38M and 2,591,046-2,694,456 of 6.9M). D's restart gate (2e6, 3 x 8 steps): the continued
  state bitwise (0 of 12M, both MACs; main 166,939), the carried acceleration 1-9 of 6M (max 3.0e-8). A test reverses
  the walk's lists and requires the same force (main: 3,191 of 6,144 differ); the CPU walk emits in one fixed order,
  so the GPU check is the bench.
- **Cost** (paired in one process, base / both fixes): +0.046 ms at 4e5 geometric (ratio 1.015), +0.228 ms at 2e6
  `dehnen_error` (1.005) on near-idle cards; +2.8 % / +1.9 % on contended ones; mostly the pair sort.
- **Left:** the 1-9 accelerations and the disc's 8.5k of 6.9M (N, below) come from outside the cross terms: the
  single card's direct near kernel scatter-adds the pieces of rows past 2 x 1024 entries in atomic order (bitwise with
  `JACCPOT_NEARFIELD_DIRECT_ROW_LIMIT=1e8`: suggestive, not proven). A single-card default-path change, not made.

### Migration to fixed cuts (I)
Track F found what sets the cross near volume: INTRUDER leaves. A decomposition leaves exact Morton domains; one
drift later a domain's boundary particles sit in cells its neighbour fills, the owner's tree does not subdivide
there (a few of its particles), and one coarse sparse leaf fails the MAC against much of the neighbour's core (8e6
Plummer, two cards: one 19-particle leaf of radius 3.07 made 194,255 of 361,797 cross near pairs). It recurs after
every drift that follows a cut. Branch `s11s2/mesh-migration`.

**Phase 1, the host model** (`bench/multigpu_migration_model.py`: per-device cell leaves of the galaxy preset --
leaf 64, level >= 8, codes in the force's cube -- after `s` drift steps `x0 + s dt v0`, leaf pairs failing the
geometric MAC at theta 0.8 with COM centres and exact radii; it reproduces the cards' 131,199 / 359,944 at 8e6 against
129,544 / 360,247). Cross near pairs per device at drift 0 / 1 / 2 / 4 / 8, two devices; "shipped" the most
particles a device sends over steps 1-8; "migrants" the rows that change device per step:

| rule | Plummer 2e5 | Plummer 8e6 | disc + bulge 2e6 |
| --- | --- | --- | --- |
| today: the cut kept, per-axis box | 11.8k / 16.3k / 17.1k / 22.5k / 21.0k; 87k shipped | 131k / 360k / 420k / 400k / 623k; 3.90M shipped | 49.2k / 80.1k / 90.4k / 102k / 119k; 382k shipped |
| the cut kept, the decomposition in the cube | 8.8k / 30.1k / 33.1k / 39.7k / 37.1k | 95.0k / 1.37M / 1.12M / 1.44M / 1.57M | 39.0k / 100k / 112k / 115k / 130k |
| migration, keys in the per-axis box of the drifted positions (a) | 11.8k at every drift; 37k; 170-211 migrants | 131k / 131k / 138k / 206k / 138k; 1.75M | 49.0-49.4k; 167k; 3.3-3.4k |
| migration, keys in the per-axis box frozen at the cut | 11.9k-15.0k | 131k-143k; 530k | 70.0k-71.8k |
| **migration, keys in the force's cube of the drifted positions (b)** | **8.76k-8.82k at every drift; 28.4k; 189-233 migrants** | **95.1k-95.3k; 310k; 7.35k-7.52k migrants** | **38.6k-39.0k; 139k; 3.28k-3.41k migrants** |
| migration, the cube frozen at the cut | 10.8k-17.5k | 95k-746k (erratic); 3.92M | 70.8k-78.3k |
| no migration, intruders as one-particle leaves (c) | 12.1k-15.9k; 51k | 141k-251k; 549k | 70.7k-133k |
| no migration, intruders grouped by the owner's leaf cell (c) | 12.2k-28.1k | 147k-508k | 254k-633k |

Four and eight devices (most pairs on one device; drift 0 / 1-8), today against (b): Plummer 2e5 on four 12.1k /
20.3k-27.9k against 10.3k / 10.2k-10.3k, on eight 10.1k / 15.2k-25.3k against 7.18k / 7.17k-7.19k; disc 2e6 on four
46.7k / 78.1k-118k against 34.0k / 34.0k-34.2k.

**Decision: (b).** Migration to the fixed cuts after every drift, the keys in the FORCE's own box at that force
(the cube of the drifted live positions, `global_mesh_bounds` with the force's box shape), the cuts chosen in that
box. It holds the zero-drift level of a cube decomposition at every drift -- below today's start (95k against
131k at 8e6) -- for 0.09-0.17 % of the particles moved per step, and the margin grows with the device count. A
box frozen at the cut is not enough: a box that moves by a fraction of a finest cell moves the cut by that much and
the particles in the sliver are intruders again (the frozen cube reached 746k at 8e6). (c) refines the symptom and
still ships the neighbour's leaves around every intruder; it loses to (b) everywhere and would change the
single-card tree build.

**What was built** (opt-in; the default programs unchanged):

| piece | what |
| --- | --- |
| `jaccpot/distributed/migration.py` | `migrate_shard`: after the drift each live row whose code (force's box) lies outside its device's splitter range goes to the owner, through an unconditional ragged exchange of the leavers (buffer `migration_cap` rows each way); the need (the most rows a device sends or receives) is read off the all-gathered send matrix BEFORE anything moves, so it is exact and replicated, and a migration that does not fit the buffer (`OVERFLOW_MIGRATION`) or a shard (`OVERFLOW_SHARD`) is declined on every device -- nothing moves. Every per-row array moves bitwise, grouped by dtype: state, the carried TOTAL acceleration, mass, id, f_b. `splitters_shard` (the cuts of an exact-range ownership), `balanced_splitters_shard` (even cuts read off the ranks), `migration_need_shard` |
| `MeshSegmentRunner(migrate=True, migration_cap="auto")` | the step is drift -> migrate -> force -> kick, in the scan; counts, masses and ids ride the carry. The constructor repartitions once into the force's box (`_repartition_shard(cubic=...)`) unless a restart passes the cuts; later repartitions are a rebalance by rank plus a migration (`sfc_repartition`'s sample sort cannot rebalance exact ranges -- the disc's counts drifted 999,037 / 1,000,963 -> 1,012,142 / 987,858 over 12 steps through two repartitions that moved 0 rows -- and its cap-sized exchange was the run's peak). `"auto"` sizes the buffer from one step's drift (2x, pow2, >= 256 rows); the cross sizing's drifted probes are migrated before they are measured. A declined migration ends the trajectory before its force; `OVERFLOW_MIGRATION` re-sizes the buffer from the exact need and re-runs the segment (outside the force's retry policy); `OVERFLOW_SHARD` raises. `SegmentReport.migrated` / `migration_need` |
| `MeshRunConfig(migrate=..., migration_cap=...)` | the runner's mode; the restart file keeps the cuts (uint64) and the buffer, a same-mesh restore hands them back (bitwise layout and cuts); the edges' plan record carries the buffer and the segment's migrated rows |
| `bench/multigpu_segment_gate.py --migrate` | the two-card gate on the migrating runner |

**Gates.**

| gate | result |
| --- | --- |
| a migrating run == one that never migrates nor repartitions (`DirectMeshForce`, N = 256, 10 segments, a repartition before each) | bitwise by id (positions, velocities, acceleration, masses); rows moved every segment; exact Morton domains at every edge; the acceleration left in the old rows breaks it (mutation) |
| f_b rides the migration | `id + 21` after 21 forces, and bitwise against the never-migrating run |
| an undersized buffer (1 row) | declined, re-sized from the exact need, == a run started with the final buffer, bitwise (direct force, and the fused force on two forced CPU devices) |
| the rebalance | counts N / 2 each on exact domains, the run unchanged bitwise |
| the exchange on crafted rows | order, arrays and padding as specified; the buffer and shard declines move nothing and report the exact need |
| the production mode | == the non-migrating mode bitwise; its restart keeps the cuts (a restore that chose its own ends on another layout: caught) |
| the fused force on two forced CPU devices (N = 2048 cube at 30x its speeds) | flag-clean, exact domains, acceleration within 5e-3 of the direct sum, cross near pairs and shipped particles at most the non-migrating run's |
| the default (migrate=False) programs | the template route's start, segment and repartition programs (geometric and `dehnen_error`) and the direct force's, lowered on two forced CPU devices: text-identical to `s11s2/integ` 700bcbe6 (8/8); single-card CPU program identity 36/36 against `s11s2-main-cpu` |

The fast cases (10, ~30 s) are in the distributed tier; the two fused ones (~4.5 min of compilation) in a new job,
`test-distributed-segment-migration` (shard `distributed-segment-migration`), not in `test-distributed-segment`.

**On two A100s** (`bench/multigpu_segment_gate.py`, `preset="galaxy"`, leaf 64, p6, library cross caps sized at the
start and the first segment's drift; rel-L2 against an fp64 direct sum on 512 fixed ids). No pair of cards was free:
every row shared its cards with other users' or tracks' processes (0-100 % utilisation), so the times below are
CONTENDED and the arms were interleaved; correctness, needs and memory are results. "Today" is `migrate=False` on
this branch (its programs are integ's). The per-step probes and the first gate rows ran before the rank rebalance
(their migrating repartitions moved 0 rows, see above); the gate rows below ran on `33e512aa`, the last caps rows
on `29177658` (the rebalance's rows donated).

Per step (geometric theta 0.8, one-step segments, a repartition every 4 steps; cross near pairs / particles a
device ships, the most over the mesh):

| row | start force | steps 1-12, near pairs | shipped | migrants per step (rows per device) |
| --- | --- | --- | --- | --- |
| Plummer 8e6, today | 129,544 / 430,677 | 337,626-621,796 | 1,849,830-3,927,219 | -- |
| Plummer 8e6, migrate | 94,409 / 303,828 | 94,370-94,732 | 304,308-306,178 | 7,285-7,475 (3,647-3,787) |
| disc + bulge 2e6, today | 2,884,947 / 406,885 | 2,939,483-3,340,622 | 475,149-556,832 | -- |
| disc + bulge 2e6, migrate | 2,842,523 / 379,397 | 2,481,213-2,839,335 | 377,618-379,904 | 3,196-3,417 (2,110-2,312) |

The host model has the Plummer rows (95.1-95.3k pairs, 306-310k shipped, 7.35-7.52k migrants) but not the disc's
absolute numbers: it has no separation floor (the cross walks send every pair closer than the compact kernel's
support to the near field, `make_cross_hook(separation_floor=...)`; eps = 2.7e-2 here, 1e-7 for Plummer), which makes
most of the disc's cross near pairs. Its migrants (3.28-3.41k) match. The library's sizing follows: Plummer 8e6's
final `send_particle_cap` 5,891,072 rows today (more than a shard's 4.6M) against 456,704 migrating,
`export_near_cap` 932,864 against 142,336.

The S12 gate configuration (`dehnen_error` 1e-4, 6 segments of 8 steps, a repartition before every segment after
the first); every segment of every row flag-clean with no retry:

| row | rel-L2 start / end | cross near pairs per segment | shipped | peak B/p, library caps / explicit final caps |
| --- | --- | --- | --- | --- |
| Plummer 8e6, today | 1.3188e-4 / 1.5193e-4 (S12: 1.3188e-4 / 1.5194e-4) | 344,210-537,214 | 3,377,081-3,996,574 | 671.5-673.6 / 470.5-475.0 |
| Plummer 8e6, migrate | 1.3459e-4 / 1.5075e-4 | 97,317-97,905 | 316,748-319,056 | 705.4-711.5 / **415.1-428.8** |
| disc + bulge 2e6, today | 6.2111e-5 / 8.1998e-5 (S12: 6.212e-5 / 8.200e-5) | 3,263,939-3,711,530 | 557,699-636,739 | 842.4-846.4 / 791.3-798.3 |
| disc + bulge 2e6, migrate | 5.8287e-5 / 7.9486e-5 | 2,839,325-3,306,801 | 379,960-413,203 | 867.7-875.7 / 783.3-799.6 |

The library-caps peak is the setup's (the sizing measurement in the process; and the constructor's repartition into
the cube: 1.38-1.43 GB per device after the setup against 1.23-1.29 GB today, on explicit caps); on explicit caps the migrating Plummer peak is its segment program
(1.66-1.69 GB per device; on `33e512aa`, before the rebalance donated its rows, the rebalance set it at 1.76 GB,
439.0-439.8 B/p), today's the start force (1.75-1.80 GB). The disc's peak does not move: its cross field is mostly the
real boundary through the disc plane, not intruders.

Counts: today's repartitions do not keep the balance -- 3,997,511 / 4,002,489 drifted to 3,970,414 / 4,029,586 over
five (Plummer 8e6), 999,058 / 1,000,942 to 984,856 / 1,015,144 (disc), 0.1-0.4 % of a shard per repartition: the
sample sort's bias (above) on nearly exact ranges. The migrating rebalance holds Plummer within 160 rows of N / 2 at every edge after the first; the disc's
fixed cuts drift ~0.8 % per 8 steps (it rotates through them) and every edge evens them again (6.6-8.0k rows sent).

Time (contended, explicit caps, interleaved off / on / off / on, Plummer 8e6 `dehnen_error`): the force alone (start
program at the final state, min of 5) 262.2 / 249.2 / 223.7 / 219.3 ms today against 51.2 / 53.8 / 54.1 / 53.6 ms
migrating; steady segments 108-266 ms per step today (growing with the cross volume) against 56-63 ms. The disc 2e6: force alone
199.4 / 217.2 ms against 219.0 / 177.2 ms -- no difference outside the contention. A free-card A/B (and four to eight
cards, which the model says gain more) is still to do.

**Recommendation.** Make `migrate=True` the multi-GPU mode's default (`MeshRunConfig`) once a free-card A/B confirms
the time: on the Plummer rows it removes the intruder field outright (near pairs 3.5-5.5x and particles shipped
10.6-12.6x fewer per segment, the force 4.1-5.1x faster on contended cards, the peak 9-12 % lower) for ~0.1 % of the
rows moved per step, at the same accuracy; on the disc it ships 30-35 % fewer particles and makes 2-17 % fewer near
pairs per segment, with no measurable time or peak change on contended cards. Found on the way, for the default path:
the repartition's sample sort (yggdrax `_choose_pivots`: the same number of samples from every device whatever its
count) cannot rebalance nearly exact ranges and lets the imbalance grow 0.1-0.4 % of a shard per repartition.

### Mesh memory (N)
Track M measured what set the two-device mesh's per-device peak (stage probes: the allocator's peak after every
setup stage, `memory_analysis()` of every compiled program) on H's IC -- 2^20 particles of a Plummer sphere truncated
at r < 10, `preset="galaxy"`, eps 1e-3, so 2^19 per card in 164,418 live cell leaves (~3 particles per 64-slot
leaf) -- and the disc + bulge at 2e6: (1) the cross sizing's measurement program, compiled at
`CrossCapacities.for_shard`'s per-leaf first guess (12-113x the needs on H's IC), which F's drift probes reran, and
(2) in every program the cross near term's leaf-major tile pool (`_cross_near_pool` plus the imported (R, W)
tiles, the table kernel's mask and chunk partials over ALL local leaf slots and every received row: >95 %
padding). Branch `s11s2/mesh-memory`.

| piece | change |
| --- | --- |
| the cross near term on the direct layout | `cross.cross_near_acceleration_direct`: the cross pairs on `nearfield_leafpair_csr_sorted_direct_pallas` (the local near field's kernel) over ONE table, the device's sorted particles then the flat receive buffer, each leaf a range of it (local leaves from the tree's node ranges, imported rows from the exchange's row starts and counts); the local leaves the only targets (`num_target_leaves`), no self term (`include_self=False`; the kernel's two new keywords, defaults unchanged), stored into particle order. The exchange leaves the flat import in the near sink, no W-wide tiles. Both force routes and E's potential take it (`fused._cross_near_lanes`); the table stays for `JACCPOT_NEARFIELD_ACCUM=wide` / `JACCPOT_NEARFIELD_LAYOUT=table` (as the local near field) and the control exchanges |
| the sizing measured on small buffers | `CrossCapacities.for_measurement`: what a measurement only COUNTS at one granule -- the rows and particles (presence maps; exact past their own overflow), and on the Pallas walk (counts past a full list) its far and near lists; the summary at the tree's bound; the queue (its need is a lower bound past an overflow) at least the local walk's widest wavefront x peers. `cross.size_cross_capacities`: a measurement whose `walk_lower_bound` and `pairs_lower_bound` are clear sizes every capacity (1.5x), overflowed or not; a lower bound grows only the walk (2x, the near list to at least 4 pairs per live leaf: room for the drift probes) and the summary and measures again. `FusedMeshForce` sizes with it at the start and at the drift, the drift probes on the last measurement's program. A far or near pair naming a row past the receive buffer is dropped (its source index read past the imported multipoles). The stacked route keeps `for_shard` + `plan_cross_capacities` |
| the eager geometry cache | `prepare_mesh_force` clears `engine._geometry_reuse_entry` (the last device's box geometry stayed through every segment: 12 MB on H's IC) |
| a fresh run's programs | `rollout.decompose`'s jit(shard_map) and `MeshState.from_shard_state`'s packer are built once per (mesh, capacity, options) / sharding: a second run of one runner lowered and compiled them again |
| `migrate=True` decomposes into the force's box | `rollout.decompose(cubic=...)`; the mode reads the cuts off that ownership (`migration.ownership_splitters`) and hands them to the runner, which no longer repartitions in its constructor |

**Before / after** (two A100s, cards shared with other users' jobs at 80-100 % utilisation: memory is a result,
times are not; one segment of 8 steps after the library's sizing; base = `s11s2/integ` 6b9cda95, N = this branch's
last commit). MB per device;
"peak" the process's high-water mark after the stage:

| stage | H's IC, base | H's IC, N | disc + bulge 2e6, base | disc 2e6, N |
| --- | --- | --- | --- | --- |
| decompose + pack | 112.7 / 112.7 | 112.7 / 112.7 | 224.3 / 224.3 | 224.3 / 224.3 |
| per-device caps-only prepares | 485.2 / 486.3 | 484.9 / 381.2 | 704.1 / 646.8 | 700.4 / 644.6 |
| start sizing (`plan_cross`) | 1443.0 / 1457.3 | 589.9 / 589.9 | 762.5 / 765.9 | 732.2 / 730.7 |
| drift sizing (`plan_cross_drift`) | 1459.8 / 1474.0 | 606.6 / 606.6 | 796.1 / 793.5 | 765.8 / 764.2 |
| start force + segment (final peak) | **1459.8 / 1474.0** | **618.6 / 618.6** | **796.1 / 793.5** | **765.8 / 764.2** |
| final B/p | 2784 / 2811 | **1180 / 1180** | 796 / 794 | **766 / 764** |
| measurement programs, temporaries | 1404.6 (one) | 517.8, 551.5 | 658.7, 658.7, 658.7, 683.2 | 594.8, 652.9 |
| start / segment program | 722.0 / 724.8 | 580.2 / 528.7 | 662.7 / 661.9 | 660.7 / 644.8 |
| compiles of the mesh programs | 3 | 4 | 6 | 4 |
| in use between segments, device 1 | 52.6 | 38.4 | 82.7 | 77.8 |

The sized capacities are the same, value for value, as the base's (H: far 840,704, near 55,296, rows 64,512,
particles 96,256 at the start; 1,186,816 / 94,208 / 70,656 / 106,496 after the drift): the measurements on small
buffers read the same needs. One card holds H's 2^20 particles at a 660 MB peak and the disc's 2e6 at 1389 MB (track
M); the mesh's per-device peak at half the particles is now 94 % and 55 % of those.

What each capacity costs the measurement (H's IC, the direct layout, temporaries of the measure program):
`for_shard`'s guess 792.4 MB; its rows and particles at a granule 710.1; the walk's lists at their sized widths too
540.1; the sized force 568.9. On top of the sized force: the far list at 8x +141.2 MB (23-24 B a pair: it feeds the
CSR M2L's merge), the near list at 8x +2.8, the queue at 8x +0, the rows at 8x +12.6. The disc: the far list at 8x
+407.7 over its sized 651.2. Every variant read the same needs bit for bit; the base's disc measurement took four
rounds (its queue 4.5 a priori slots per leaf against 15 needed), now two (the queue from the local walk's widest
wavefront).

**Gates.** `bench/multigpu_segment_gate.py` (6 segments of 8 steps, a repartition before every segment after the
first, library cross capacities, rel-L2 against an fp64 direct sum on 512 fixed ids; the base and N runs on the
same contended cards, interleaved); every segment of every row flag-clean, and no row's first segment retried (F's
guarantee):

| row | rel-L2 start / end, base | N | peak per device, base | N |
| --- | --- | --- | --- | --- |
| Plummer 2e5, `dehnen_error` | 1.1475e-4 / 1.5896e-4 | 1.1475e-4 / 1.5896e-4 | 79.0-79.1 MB | 78.9-79.4 MB |
| Plummer 2e5, geometric 0.8 | 7.6106e-4 / 8.1967e-4 | 7.6106e-4 / 8.1967e-4 | 78.5 MB | 44.9 MB |
| disc + bulge 2e5, `dehnen_error` | 3.5781e-5 / 3.5205e-5 | 3.5781e-5 / 3.5211e-5 | 146.5-147.2 MB | 146.5 MB |
| disc + bulge 2e5, geometric | 2.1453e-4 / 1.6699e-4 | 2.1453e-4 / 1.6699e-4 | 145.6 MB | 145.6 MB |
| disc + bulge 2e6, `dehnen_error` | 6.2112e-5 / 8.1999e-5 | 6.2110e-5 / 8.1999e-5 | 817.4-824.6 MB (818-824 B/p) | 787.2 MB (787 B/p) |
| disc + bulge 2e6, geometric | 3.1878e-4 / 3.1406e-4 and 3.1407e-4 (two runs) | 3.1878e-4 / 3.1407e-4 | 1005.0 MB (1005 B/p) | 717.3 MB (718 B/p) |
| Plummer 8e6, geometric | 4.3568e-4 / 4.3011e-4 (track F's run) | 4.3568e-4 / 4.3011e-4 | 2647.6 MB (662 B/p, F's) | 1365.6 MB (**342 B/p**) |

The base's Plummer 8e6 row ran out of memory three times on the shared cards (2-4.7 GB free); its numbers are track
F's run of the same configuration on F's branch (integ without G and I; the same final cross capacities, value for
value). N's 8e6 measured twice at the start (the base once) and not again at the drift (2x the walk, the near list
at 4 per leaf): five force programs compiled against the base's four. The 2e5 rows' peaks are the setup's eager
prepares, not the force programs. CPU (two forced devices): gate A1 (the
template route == the stacked route, bitwise), the segment, segment-criterion, segment-migration and criterion
jobs, the distributed tier, the single-card runner file and the guards pass; CPU program identity 36/36 against
`s11s2-main-cpu` (the kernel's three new keywords default to the traced program it always was).

**The longest cross row** (entries of one local target leaf's cross CSR row, every cross near term of a segment of 8
steps, both devices): H's IC 102 (no row reaches the direct kernel's 1024-entry `DIRECT_ROW_LIMIT`); the disc + bulge
2e6 **4,225**, up to 926 rows per force past 1024 entries and 355 past 2048. So the disc's cross term runs the
kernel's `_rest` pieces, and a row past 2048 entries has two or more pieces scatter-added onto the same outputs --
in the atomics' order: the cross term passes `ordered_rest=True` (the kernel's new keyword, default off: a row's
pieces summed by one segmented scan onto its last piece and added once). Run to run (`bench/multigpu_force_repeat.py`,
the start force three times in one process and once in another, two A100s): Plummer 4e5, geometric and
`dehnen_error`, bitwise (0 of 1,380,000 components, across processes too: G's gate holds). The disc + bulge 2e6
(`--ic disc`, new, `dehnen_error`) is NOT bitwise, and was not on the base either: base 8,561 / 8,169 of 6,900,000
acceleration and 2,493 / 2,412 of 2,300,000 f_b components differ from the first run (max 7.6e-6 / 3.1e-5), N without
the ordered pieces 9,786 / 9,304 and 2,595 / 2,483, N 8,620 / 8,414 and 2,237 / 2,411; the control (one run's buffers
read twice) 0 every time. So the ordered pieces remove the direct layout's own share, and what is left predates N and
lies outside the cross near term (G's gate ran Plummer only): not diagnosed here.

**Programs a fresh run compiles again** (one runner, `run()` three times; GPU, a Plummer sphere of 131,072, the galaxy
preset): base run 2: 27 compiles -- the decomposition's jit(shard_map), the state packer's `jax.jit(lambda)`, 16
jit(while), 4 jit(wrapped), 2 jit(cond), and the measure, start and segment programs (run 1's start force retried
on the LOCAL walk -- its near-edge capacity 131,072 -> 1,048,576, which stays on the engine -- so run 2's programs
have other walk caps and cross capacities; run 3 adds no program key); run 3: 24. The single card's second and
third runs compile 11 each (8 jit(while), 2 jit(wrapped), 1 jit(cond)): the caps-only prepare's eager calls --
yggdrax's `_propagate_extents` `lax.while_loop` (from `_interaction_cache._build_flat_walk_artifacts_strict_streamed`),
the P2M and the f_b seed's near kernel as eager `pallas_call`s, that kernel's rest-piece `lax.cond` -- which the
mesh runs once per device. N removes the two mesh-only ones (N's run 3: 22 compiles, the prepares' 16 + 4 + 2 alone,
60.6 s; run 2 again recompiled the four force programs after run 1's local retry); the prepare's eleven are the single
card's own (and yggdrax's) and are left as they are. CPU: a second and third run of the mode on the direct force
lower no program (`tests/distributed/test_production_runner_mesh.py`).

**`migrate=True`'s setup** (H's IC): the mode decomposes in the force's cube and reads the cuts off it (a program of
4.8 MB temporaries); the base decomposed (48.4 MB) and then repartitioned again in the runner's constructor (44.0 MB
temporaries, 28.9 MB outputs per device), which is gone. At H's size the setup's peak is the caps-only prepares'
(484-489 MB), so it does not move there (final peaks 620.4 / 608.1 MB migrating, base 1471.6 at the start sizing);
track I's +0.13 GB per device was at 4e6 per card.

**Open.** At 2^19 per card the start program's 580 MB is still close to one card's for twice the particles (450 MB
temporaries, 660 MB peak on H's IC). Its largest temporaries now (XLA's buffer assignment): the local tree's
per-node arrays -- 378,879 nodes for 2^19 particles on this compact IC, its multipoles and locals f32[378879, 49] at
74 MB a copy, several live, the local walk's far list s32[14,680,064] at 59 MB a copy -- then the cross far half: the
[local; imported] multipoles f32[449535, 49] (88 MB) and the all-gathered summary f32[2, 378879, 16] (48.5 MB, ndev x
one device's block: it grows with the mesh). The cross near term's whole table is f32[709492, 4] (11.4 MB). The
single card's `_rest` scatter-add (its default) is not changed. The eleven eager-prepare programs per device per
run (above). A drift whose walk lists outgrow the measurement's (2x the far pairs, 4 near pairs per leaf) costs a
third measurement. The disc's remaining run-to-run differences (above).

### Odisseo's multi-GPU lane on the runner (H)
Odisseo's `fmm_mesh` lane ran the OLD `distributed/fmm.py` force (three dispatches per step, no potential). Odisseo
branch `s11s2/fmm-mesh-runner` (against `jaccpot-integration`) adds lane `"fmm_mesh_runner"`: `fmm_preset="galaxy"`
with `fmm_mesh=True` and an `odisseo.mesh_runner_coupling.MeshRunnerOptions(ndev=..., repartition_every=1,
cap_headroom=None, force=None)` in `config.mesh_options` runs `ProductionRunner(mesh=MeshRunConfig(...))` as the
single-card `fmm_runner` lane runs the single-card runner (the solver and runner kept between calls, the external
field as one pair of traced functions, snapshots from the edges, the final state a host array in input order). A
`MeshOptions` keeps the old lane, unchanged. Gradients, block steps, active schedules, `step_callback`, adaptive
steps and a G or softening other than the solver's are refused with the reason. 28 tests (two forced CPU devices,
`DirectMeshForce`, 12.6 s): `integrate()` == the runner called directly, bitwise; a second call reuses the runner;
snapshots, energy and angular momentum against host fp64 sums; the old lane still reached. One `integrate()` on two
A100s (Plummer r < 10, 2^20 particles, NFW halo, 16 steps): positions rel-L2 6.6e-8 against the single-card lane after
16 steps (a second mesh run differs from the first by 2.4e-7 at most), self-gravity rel-L2 1.800e-4 (one card
1.749e-4), angular momentum drift 1.4e-6. Its measurement on that compact IC found the setup peak N removed.

### Close-out on merged main (P)
Baselines re-recorded on main e82389b5 (everything through S12 and #469), card 1 (only another user's idle 416 MiB
server on it, which flags every row `contaminated`; the numbers are clean):

| pin | result |
| --- | --- |
| L1 8e6 / L2 1e8 / L3 25M disc step (tag `s11s2-L-main-e82389b`, control `-b`) | 76.66 / 864.03 / 235.21 ms; L2 256.77 B/p; L3 force 100.21 ms, rel-L2 4.1791e-4: within the envelope of `s3b-L-integ-g5`, 1-2 % faster than `s3b-L-main-5f1f85f-t1` (the S3b walk speed-ups) |
| DET1-DET3 (`s11s2-det-free`, `-b`) | bitwise A-vs-A; DET2 bitwise `s8w-det` |
| single-card GPU pins S1, S6, S1p, S1t, S1e, S1x, S1g, G1, R1p, R1x, R1r (`s11s2-pins-main`) | bitwise against `s12pins-main` / `nf-tip2` / `s8w-tip` |
| two-card M1, M3 / M2 (`s11s2-m-main`, `-b`) | bitwise / within M2's own run-to-run spread |
| CPU program identity (`s11s2-main-cpu`, 36 variants) | 36/36 against #469's tip; against `s11-stack-cpu` only #469's documented scan-program text |

### Free-card measurements (T)
Cards 0, 1, 5, 6 with no other user's job on them (another user's idle 416 MiB server sits on every card), 2026-10-23
/24, from a frozen tree at the integration of E, D, F, I and G -- BEFORE N, so every peak below that includes the
library's sizing is N's "base" (the measurement program at the per-leaf first guess); the explicit-capacity reruns are
the force's own footprint. `bench/multigpu_segment_gate.py`, 8 segments of 8 steps, a repartition before every segment
after the first, rel-L2 against an fp64 direct sum on 512 ids; the step is the median segment wall / 8 over segments
4-8, two interleaved repetitions (a / b) per arm. A100-PCIE-40GB, no NVLink.

**Migration on / off** (step ms, mean of the two repetitions; the ratio on / off):

| config | 2 cards | 4 cards |
| --- | --- | --- |
| Plummer 8e6, `dehnen_error` | 164.2 / 63.7 (0.39) | 170.1 / 98.5 (0.58) |
| Plummer 8e6, geometric | 204.1 / 53.1 (0.26) | 144.4 / 63.3 (0.44) |
| Plummer 1.6e7, `dehnen_error` / geometric | | 486.2 / 188.0 (0.39); 408.6 / 312.6 (0.77) |
| disc + bulge 2e6, `dehnen_error` / geometric | 187.3 / 189.8 (1.01); 180.2 / 179.0 (0.99) | 192.9 / 188.4 (0.98); 176.2 / 172.5 (0.98) |
| disc + bulge 8e6, `dehnen_error` / geometric | | 431.8 / 372.0 (0.86); 366.5 / 309.5 (0.84) |
| disc + bulge 25M, `dehnen_error` | 249.8 / 197.0 (0.79) | 583.1 / 301.5 (0.52) |

Migration is never slower beyond the repetitions' spread, and on every Plummer row and the large discs it is 1.2-3.8x
faster: the cross near pairs and particles shipped per step fall to the start's level (Plummer 1.6e7 on four cards:
828-924k -> 172-173k near pairs, 5.7-6.9M -> 596-599k shipped, 28.4-28.5k migrants), and the non-migrating arms'
retries on four cards (up to 5 per run) are gone. The disc 2e6's cross volume is set by its separation floor, not by
intruders, so it gains nothing there. Accuracy is unchanged (rel-L2 at the end within the repetitions: e.g. Plummer
1.6e7 `dehnen_error` 1.7550e-4 off, 1.6222e-4 on). The peak rises 1-6 % with migration (the buffer and the routed
rows).

**Strong scaling** (step ms; one card = the single-card runner on the same particles with the L3 flags,
`fused_memory_budget.py`, and the gate's one-device mesh; the mesh rows migrate):

| N, MAC | 1 card | 2 cards | 4 cards |
| --- | --- | --- | --- |
| disc + bulge 25M, `dehnen_error` | 258.8 (runner) / 249.8 (1-device mesh) | 197.0 (1.27-1.31x) | 293.4-309.6 (0.81-0.88x) |
| disc + bulge 25M, geometric | 231.3 (runner) / 226.5 | 173.0 (1.31-1.34x) | 189.4 (1.20-1.22x) |
| Plummer 8e6, `dehnen_error` | 76.9 (1-device mesh) | 63.2-64.2 (1.20-1.22x) | 96.8-100.2 (0.77-0.79x) |
| Plummer 8e6, geometric | 66.0 | 52.8-53.4 (1.24-1.25x) | 62.5-64.0 (1.03-1.06x) |

**Weak scaling** (4e6 per card, Plummer, `dehnen_error`, migrating): 46.8 ms on one device, 63.2-64.2 on two (73 %),
187.7-188.3 on four (25 %).

On this node the mesh buys CAPACITY, not speed beyond two cards: every card-to-card byte crosses PCIe (the
cross near term ships ~1M particles per step at 25M on four cards, plus the summary all-gather and two ragged
exchanges per step). The scaling question belongs on a node with NVLink (HoreKa's four-A100 nodes), on dedicated
cards.

**Memory per device** (B per particle of the device's share):

| row | library sizing (pre-N) | explicit capacities (the force's own) |
| --- | --- | --- |
| Plummer 8e6, two cards | 705-713 | 448-470 |
| disc + bulge 25M, two / four cards | 755-765 / 1377-1380 | 436-437 / 498-499 |
| Plummer 1.6e7, four cards, off / on | 1229-1230 / 1267-1269 | 717-737 / 640-644 |
| Plummer 4e7, four cards (1e7 per card) | 1793-1805 | 580-603 |
| Plummer 8e7, four cards (2e7 per card) | 1626-1643 (32.9 GB per card) | 516-526 (10.5 GB per card) |
| disc + bulge 8e7, four cards | 1406-1412 | -- |

The library-sizing column is what N's measurement on small buffers removes (N's own rows: Plummer 8e6 on two cards
662 -> 342 B/p). At 8e7 on four cards the step is 906.6 ms (one segment) and the force alone 832.8-852.8 ms (`dehnen_error`). On
the explicit-capacity footprint a card holds ~6.9e7 particles (0.9 x 40 GB / 520 B): four cards ~2.8e8, eight ~5.5e8 -- an extrapolation, past one
card's 1.68e8 (round 4); the post-N library-sizing peak at those counts is still to be measured.

### A cell leaf wider than the leaf size (found 2026-10-29)
Plummer at 4e7, 8e7 and 1.2e8 on four cards (1e7-3e7 per card; the stack tip, cards shared, memory only) stopped
with `OVERFLOW_CROSS` and "re-plan grew nothing": a near row shipped 65, 144 and 133 particles against a leaf width
of 64, and fix 8's structural bound (`_structural_bounds`: the leaf width at most the leaf size) kept the re-plan at
64. The bound was wrong: yggdrax keeps a cell at the finest Morton level as a leaf whatever it holds, and the
Plummer draw's outliers make the box 35,148 wide at 8e7, so a finest cell is 0.017 across and holds up to 121
particles in the core (51,245 such cells hold more than 64; 54 at most at 4e7 before any drift). The width is now
bounded by 64 x the leaf size (`_FINEST_CELL_WIDTH_FACTOR`), which still keeps a degenerate tree's whole-shard
"leaf" (the NaN root of 4,002,489) out. Fix 8's sentence above ("`leaf_width` never exceeds the template's leaf
size") holds below ~1e7 particles per card only. Below that the same four-card runs: Plummer 8e6 531-533 B/p, the
25M disc + bulge 503 B/p per device, flag-clean (post-N sizing). With the fix (four shared cards, `dehnen_error`,
3 segments of 4 steps, migrating, library sizing):

| Plummer | per card | peak per device | retries (the width grown) | widest row | rel-L2 start / end |
| --- | --- | --- | --- | --- | --- |
| 4e7 | 1e7 | 481-492 B/p (4.8-4.9 GB) | 1 | 65 | 9.97e-5 / 1.85e-4 |
| 8e7 | 2e7 | 423-427 B/p (8.5 GB) | 1 | 156 | 2.48e-3 / 4.32e-3 |
| 1.2e8 | 3e7 | 432-437 B/p (13.0 GB) | 2 | 149 | 2.09e-3 / 2.57e-3 |

Every segment flag-clean. A card holds ~8e7 particles at that footprint (0.9 x 40 GB / 430 B): four cards ~3.3e8,
eight ~6.7e8 (an extrapolation). The rel-L2 at 8e7 and 1.2e8 is NOT the mesh's: ONE card on the same 8e7 particles
(the gate's single-card lane on the mesh run's positions) gives 4.60e-3 / 3.90e-3 (214 B/p, 650 ms per force). The
fused lane's accuracy on this draw -- a box of 35,148 for a sphere of scale radius 1, softening 1e-7 -- falls 25x
between 4e7 and 8e7 on one card too; the production IC (the 25M disc + bulge) is at 1.3e-4. Open, not diagnosed.

**Resolved 2026-10-30 (yggdrax #104, jaccpot #483 / #485, and the bound back at the leaf size).** Both findings
were the same defect: the finest-level cells over 64 were leaves the local lane's leaf tables (64 wide) silently
truncated -- particles past the 64th in such a leaf got no near field on one card as on four. yggdrax #104 cuts
a finest-level cell into leaf-size chunks, so no cell leaf is wider than the leaf size, and `_structural_bounds`
is back at fix 8's `leaf_width <= leaf_size` (`_FINEST_CELL_WIDTH_FACTOR` removed; a wider need now names a
degenerate tree or a yggdrax older than #104 in the "re-plan grew nothing" message). The one-card 8e7 force on
the same particles: 4.60e-3 / 3.90e-3 -> 1.72e-4 / 1.45e-4. The four-card rows again (cards 3, 5, 6, 7, shared
with an idle context only; same command):

| Plummer | per card | peak per device | retries | widest row | rel-L2 start / end |
| --- | --- | --- | --- | --- | --- |
| 4e7 | 1e7 | 481-492 B/p (4.8-4.9 GB) | 0 | 64 | 9.97e-5 / 1.85e-4 |
| 8e7 | 2e7 | 424-427 B/p (8.5 GB) | 0 | 64 | 1.04e-4 / 1.69e-4 |
| 1.2e8 | 3e7 | 416-420 B/p (12.5-12.6 GB) | 0 | 64 | 9.71e-5 / 1.83e-4 |

Every segment flag-clean with no retry; the accuracy no longer falls with N.

### Open after S11 session 2
- `MeshRunConfig.migrate` is `True` by default after T's A/B (the stack's last PR); `MeshSegmentRunner`'s own
  `migrate` stays `False` (the low-level runner, its gates' reference).
- Run to run on GPUs: the single card's direct near kernel scatter-adds the pieces of rows past 2 x 1024 entries in
  atomic order (the 2-GPU residual above, and the disc's 8.5k of 6.9M); `ordered_rest=True` exists, off by default:
  turning it on for the single card is a change of its default path.
- The repartition without migration does not rebalance nearly exact ranges (yggdrax's `_choose_pivots` takes equal
  samples per device): the imbalance grows 0.1-0.4 % of a shard per repartition (Plummer 8e6). The migrating mode
  rebalances by rank instead.
- The all-gathered summary grows with the device count (f32[ndev, nodes, 16]: 48.5 MB at two devices on the compact
  IC); the eager caps-only prepare's eleven small programs per device per fresh run are the single card's own.
- Eight cards (four were free at most); multi-node (D5).
- nornax `retry_segment` has no progress rule (deferred to S9: the provider's aux would need the steps an attempt
  fitted). Odisseo's old `MeshOptions` lane: kept until its users move (user decision).
