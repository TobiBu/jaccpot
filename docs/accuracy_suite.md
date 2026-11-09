# The accuracy verification suite (plan A4)

The suite is how later work proves it kept jaccpot accurate. It has four parts:

* a **reference force** that is the same number on any device count;
* **standard problems** with fixed analyses;
* **convergence ladders** run on HoreKa;
* a **baseline + check** mode that later tracks must pass.

Plan: `~/.claude/plans/multinode-accuracy-dev-plan.md`, section A4. The targets are decision U4. The record of the
build is `docs/production_2026-11.md`, `## A4`.

| piece | where |
| --- | --- |
| reference force, scoring, runner adapters | `jaccpot/verification/reference.py` |
| result schema, `record` / `check` | `jaccpot/verification/baseline.py` |
| standard problems, analyses, CLI | `bench/accuracy/` (`suite.py`, `problems.py`, `analysis.py`, `ics.py`, `ladders.py`, `local_tier.sh`) |
| galaxy ICs (agama) | Odisseo `tools/a4_galaxy_ics.py` |
| HoreKa jobs | `bench/slurm/a4/` (README there) |
| tests | `tests/unit/verification/`, `tests/distributed/test_reference_force_devices.py`, `tests/distributed/test_multiprocess_reference.py`, `bench/tests/test_accuracy_suite.py` |

## 1. The reference force

```python
from jaccpot.verification import ForceLaw, reference_force, score_force, select_targets, gather_by_id

law = ForceLaw(G=1.0, softening=1.5e-3, kernel="ferrers3")     # or force_law_of(solver, handle)
tids = select_targets(N, 4096, seed=0)                          # sorted particle ids: a function of (N, K, seed)
ref = reference_force(positions, masses, ids, tids, law, mesh=mesh)   # rows P(axis)-sharded, or on one device
prod, _ = gather_by_id(acceleration, ids, tids, mesh=mesh)            # the production force at the same ids
print(score_force(prod, ref))     # rel_l2, p50, p90, p99, max, mean, num_targets, nonfinite
```

There are runner adapters: `score_single_result(result, solver)` and `score_mesh_result(result, solver,
external_acceleration=...)`. Both score a `ProductionRunner` result at its final state without gathering the
particles.

**Why the reference is exact in any order.** Each source-target pair term is computed in fp64. The terms are:

* `m g(r) (x_s - x_t)` for the acceleration;
* `-m psi(r)` for the potential.

The pair factors are `jaccpot.softening.masked_pair_factors`, the production law for every kernel.

Each term is rounded onto a fixed-point grid, one grid per target and component. The grid step is `2^-86` of
the largest term for that target and component. The largest term is a `pmax` over all devices, and a max is
exact in any order. The rounded integers are below `2^85`. They are split into three 30-bit limbs and summed as
`int64`.

Integer addition is associative, so the sum is the same bits whatever the order: one card, N cards, N processes,
any row order, any padding, any split of the sources into groups (e.g. an escaper tail). After the sum, the limbs
are carried on the host in Python integers and rounded once to fp64.

The result is the exactly rounded sum of the fp64 pair terms, to within `N * 2^-86 * max|term|`. That is below
the round-off of a plain fp64 direct sum.

Only the `K` target rows are gathered. Each device finds its rows by binary search, and the rows travel as an
`all_gather` of `[K, 4]` with a found-count. The sources stay where they are.

On each device the work is a scan over fixed-size chunks of its own rows. The chunk length depends on `K` only,
so every device count runs the same per-device program.

Cost: two passes over all `N * K` pairs, about 3x an fp64 direct sum on those pairs. At `K = 4096` and
`N = 1e8` that is about 4e11 pair terms per pass, which makes it a verification step, not a per-step
diagnostic.

**Gated** (CPU):

* It equals `jaccpot.runtime.reference.direct_sum` to < 1e-14 for every kernel (and Newton), and an
  extended-precision sum to < 4e-16.
* It is bitwise identical across 1/2/4 forced devices with device-dependent row orders and padding.
* It is bitwise identical between 2 and 4 processes x 1 device and 1 process x N devices (gloo, M2's harness).
* The runner's multi-GPU mode with `DirectMeshForce` scores < 1e-13 against it.

**Why it replaces the probes.** The earlier rel-L2 probes summed a subsample in an order set by the
decomposition. Two probe settings of the same run differed by 41 % (memory `rel-l2-probe-not-comparable`). Here
the target set and the sum are both fixed by `(N, K, seed)`. Two runs are comparable whenever `N`, `K` and the
seed match, and `check` enforces that.

## 2. The standard problems

`python -m bench.accuracy.suite run PROBLEM --out DIR [--tier local|horeka-1card|horeka-node] [options]`

| problem | IC | duration (default) | snapshot analysis | key metrics |
| --- | --- | --- | --- | --- |
| `plummer`, `hernquist`, `nfw` | isotropic Eddington sphere, `G = M = 1`, scale 1 (`bench/accuracy/ics.py`; NFW with agama's `exp(-(r/10)^2)` cutoff) | 100 `t_dyn` (`t_dyn = sqrt(r50^3 / GM)`), dt = `min(t_dyn / 64, 0.05 sqrt(eps / a(eps)))` with `a(eps) = G M(<eps) / eps^2` (Plummer 6400 steps; the Hernquist and NFW cusps ~32k / ~31k) | Lagrangian radii (1/10/25/50/75/90 %), sigma between them | `lagr.max_drift_r10/r50/r90`, `sigma.max_drift`, `energy.max_abs_dE`, `momentum.max_dP`, `force.*` |
| `disc_heating` | the production disc + bulge + live halo (`a4_galaxy_ics.py galaxy`, Q ~ 2: stable) | 1 Gyr (6.707), dt 1e-3 | `sigma_z^2`, `sigma_R^2`, `z_rms` in 0.5-1 / 1-2 / 2-3 R_d | `heat.sigz2_growth_k`, `heat.sigz2_rate_k` (and R), `energy.max_abs_dE`, `angmom.max_dL`, `force.*` |
| `bar` | the bar-unstable disc (`a4_galaxy_ics.py bar`: no bulge, Q 1.2) | 3 Gyr (20), dt 1e-3 | `run_bar.py`'s bar probe (A2(R), m=2 phase, z_rms in 0.4 kpc annuli) | `bar.t_bar_myr`, `bar.t_peak_myr`, `bar.a2max_peak`, `bar.t_buckle_myr`, `bar.omega_p_end`, `bar.length_end_kpc` |
| `merger` | two production galaxies on the 2026-10-31 parabolic orbit (`a4_galaxy_ics.py merger`) | t = 10, dt 1e-3 | shrinking-sphere centre per component | `merger.peri1_t/kpc`, `apo1_*`, `peri2_*`; with `--reference`: `merger.sep_max_dev_kpc`, `merger.peri1_dt` |

Every run scores the force against the reference at `--targets` ids (2048 by default) at three points: the first
edge with a force, every `--score-every` segments, and the last edge. It writes these files:

* `probe.jsonl`: one row per edge, with the diagnostics, the analysis when due and the score;
* `config.json`;
* `result.json`.

**Reproducing the 2026-10-31 runs** (`suite.py analyse-existing bar|merger [LABEL=]DIR[:DIR]...`) from their
`probe.jsonl`:

* **The bar ladder.** `t_bar`, the peak time and amplitude, the end bar length and the pattern speed all equal
  `ladder_final.json` exactly. The buckling time is the time of the steepest rise of the R < 2 kpc thickness,
  from a 1.0-wide sliding fit. It reads 2.74 / 2.67 / 1.74 / 1.47 Gyr at 1e8 / 4e7 / 1e7 / 2.5e6. The record is
  2.74 / 2.68 / 1.73 / 1.47.
* **The merger.** First pericentre 9.04 kpc at t 1.307 in all three dt runs (the record: 9.1 kpc at t 1.30).
  Second pericentre 0.57 kpc at t 7.61 (the record: t 7.6, 0.6 kpc).

## 3. The convergence ladders

`bench/accuracy/ladders.py`: `suite.py ladder NAME --tier T` prints one command per rung.

| ladder | axis | 1 card | node (4 GPUs, `--devices 4`) |
| --- | --- | --- | --- |
| `plummer_n`, `hernquist_n`, `nfw_n` | N | 1e5, 1e6, 1e7 | 1e7, 4e7 (Plummer only) |
| `plummer_dt` | dt = t_dyn/32, /64, /128 (reference /128) | 1e6 | - |
| `plummer_mac`, `disc_mac` | geometric theta 0.5 / 0.7 / 0.8, `dehnen_error` tol 1e-3 / 1e-4 / 3e-5 | 1e6 / 1.6e7 | disc at 6.4e7 |
| `plummer_precision`, `disc_precision` | `fp32`, `fp64state`, `fp64`, `legacy` (A3) | 1e6 / 1.6e7 | disc at 6.4e7 |
| `disc_n` | N | 1e6, 4e6, 1.6e7, 6.4e7 | 6.4e7, 2e8 |
| `disc_dt` | dt 2e-3, 1e-3, 5e-4 (reference 5e-4) | 1.6e7 | - |
| `bar_n` | N (theta 0.7 as the 2026-10-31 ladder; `dehnen_error` at the two smallest) | 2.5e6, 1e7, 4e7 | 1e8 |
| `merger_n` | N | 1.25e7, 2.5e7, 5e7 | 1e8, 2e8 |
| `merger_dt` | dt 2e-3, 1e-3, 5e-4 (reference 5e-4: `summarize` scores the orbit against it) | 2.5e7 | 1e8 |
| `merger_mac` | theta 0.7, 0.8, tol 1e-4 | 2.5e7 | - |

The galaxy ladders take per-component prefixes of one IC at the largest N, so the rungs are nested subsamples.

**The precision axis.** It is keyed by a runner setting in `bench/accuracy/suite.py`'s `PRECISIONS`. `fp32`
drifts summed (`x + (v dt + 0.5 a dt**2)`, the default since A3); `legacy` is the same fp32 state with the drift
evaluated left to right (`GalaxySettings(drift_order="legacy")`, every run before A3), which drifts 7x (energy)
and 24x (angular momentum) more on the merger probe (`docs/production_2026-11.md`, A3). A3's plan U2 (a), an fp64
anchor per leaf, was dropped: the drift order closes the gap at no memory cost. An entry set to `None` (a slot)
makes its rung exit 3 (reported as skipped).

## 4. Baseline and check

```bash
python -m bench.accuracy.suite record DIR/result.json BASELINE.json     # --force to replace
python -m bench.accuracy.suite check  NEW/result.json BASELINE.json     # exit 1 on FAIL; --json for the rows
bash bench/accuracy/local_tier.sh OUT IC_DIR [BASELINE_DIR]              # every problem locally (+ checks)
```

**Schema** (`jaccpot.accuracy/1`):

```text
schema, problem, tier,
config {N, ic_seed, dt, t_end, mac, theta, tol, precision, softening, kernel, targets, target_seed, ...},
comparable [config keys a check requires EQUAL],
metrics {flat name -> float},
tolerances {metric -> {kind, value, why}},
series {time series, never checked},
provenance {commit, host, platform, devices, processes, jax, date, wall_s, slurm_job, xla_flags}
```

A check first requires every `comparable` key to be equal. A different N, dt, MAC, precision, target count or
seed is **not comparable**, and the check fails without comparing a number. This is the subsampled-probe lesson
built in. Then every metric the baseline has a tolerance for must hold:

| kind | passes when |
| --- | --- |
| `abs` | `|new - base| <= value` |
| `rel` | `|new - base| <= value * |base|` |
| `ratio` | `new <= value * base` (an error may not grow by more than `value`) |
| `ceiling` | `new <= value` (an absolute target) |
| `floor` | `new >= value` |

A missing metric fails, and so does a non-finite value.

**The tolerances and why** (`bench/accuracy/problems.py`). They come in two layers.

*Regression* tolerances apply at every tier. A re-run of the same configuration must land within run-to-run
noise. N-body runs are chaotic, so two GPU runs that differ by one ulp (the `dehnen_error` level under
production flags) diverge microscopically, and only statistics are reproducible:

* force rel-L2 at the first edge within 1.25x, and at the end within 1.5x;
* p99 at the end within 1.5x;
* energy and angular-momentum drift within 2x;
* momentum drift within 10x (it is a round-off random walk);
* Lagrangian radii and dispersions within 3-5 points;
* disc sigma^2 growth within 5 points;
* bar peak within 0.05, peak time within 150 Myr (the bar onset is a shot-noise seed: two independent 1e7
  realisations differed by ~300 Myr);
* merger first pericentre within 0.02 time units and 0.3 kpc.

*Target* tolerances apply at the HoreKa tiers only. They are plan U4's ceilings:

* `force.p50_end <= 1e-4` for `dehnen_error` (the median force error);
* `energy.max_abs_dE <= 1e-4` and `angmom.max_dL <= 1e-5` over 1 Gyr in the disc (bar: scaled to 3 Gyr);
* `momentum.max_dP <= 1e-8`. That is the fp32 force's round-off: M1 measured ~3e-10 over 1000 steps, and the
  ceiling is that x30 for longer runs.

The p99 is recorded and held by ratio, never by an absolute ceiling (U4: "p99 recorded").

**What later tracks must hold.** Record the local baselines once (`bench/accuracy/baselines/local/`, from
`local_tier.sh`) and the HoreKa baselines from the first suite run. A change passes when `check` passes at
every tier it can run. A deliberate accuracy change re-records the baseline in the same PR, with the reason in
the PR body.

## 5. Tiers

| tier | where | sizes | what it is for |
| --- | --- | --- | --- |
| CI | GitHub runners (CPU) | N <= 512, `K` <= 64 | the reference's correctness and its bitwise identity across 1/2 forced devices and 2 processes; schema/check; the tooling |
| local | one free card (autocvd) or the CPU | 8192 (spheres), 1e5-2e5 (galaxies), short durations | functional: every problem end to end, a baseline, gate 3's pass/fail demonstration |
| `horeka-1card` | one A100 per rung (`bench/slurm/a4/card1.sbatch`, array) | up to 6.4e7 | the ladders on one card; times recorded |
| `horeka-node` | one exclusive node, 4 A100 (`node.sbatch`) | up to 2e8 | the runner's multi-GPU mode; the same reference, so its force scores compare with the card's |

Local timings are not evidence (shared cards). The HoreKa results record `wall_s` per rung.
