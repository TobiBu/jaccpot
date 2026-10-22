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
