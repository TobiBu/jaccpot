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
