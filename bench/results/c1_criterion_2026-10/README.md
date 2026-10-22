# C1: the timestep criterion for galaxy discs, shared steps vs HOLD, 2026-10-22

Record: `docs/production_2026-10.md`, section "C1". Harness: `bench/c1_criterion_calibration.py` (its docstring
defines every scheme, criterion and cost proxy).

## Setup

- **IC:** particles `[0, 1e5)` of the shuffled production IC `/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz`
  (agama self-consistent disc + Hernquist bulge, seed 7): an independent draw of 83,344 disc and 16,656 bulge
  particles from the production distribution function. Masses rescaled by 25,165,824 / 1e5 (7.2e-5 each), the
  mass-weighted centre (offset 2.7e-3) and drift (0.018) removed. The NFW halo (M 100, r_s 2) is the production
  IC's analytic external field. G = 1, the production code units.
- **Softening:** the production kernel, `ferrers3` with Plummer-equivalent eps 1.5e-3 (support h = 3.69e-3), in
  both the forces and the energy.
- **Why N = 1e5:** a direct sum costs N^2 per force: about 46 ms per full pass at 1e5 on one A100, so the converged
  reference (48,000 steps) took 37 min and its half-step twin 75 min. 1e6 would be 100x that.
  At 1e5 a particle is 250x heavier than in production: one particle at eps pulls with G m / eps^2 = 32, more than
  the median |a| of 23. Close encounters are therefore far stronger than at 25M, which makes this a harsh test of
  a criterion that ignores them.
- **Run length:** t_end = 1.5, one azimuthal period of the disc's stars at 2.5 R_d (2 pi R / <v_phi> = 1.47; the
  disc is hot, sigma_R = 1.65 and <v_phi> = 2.58 against v_c = 3.6 there), 1.45 circular periods at 2.5 R_d, about
  2.5 at R_d and many more in the bulge.
- **Direct sum:** one Pallas kernel (`make_pair_pass`): fp32 pair arithmetic on fp32 positions, each 16 x 64-source
  chunk summed in fp32 and the chunks in fp64. A force is good to 2e-7 (median 2e-8) against fp64. The pair terms of
  (i, j) and (j, i) are bitwise opposite, so momentum is limited by the sums' round-off only. The integrator state
  and the energy diagnostic (an fp64 XLA direct sum of the ferrers3 potential) are fp64. fp32 pairs suffice because
  every scheme's integration error is at or above 4e-9 in dE/E0 (the reference's), and the fp32 force is a
  conservative force evaluated at positions rounded by 6e-8.
- **Cards:** A100-PCIE-40GB through autocvd, at most two at once; every row is a counted cost, not a timing, so
  contention does not enter any number. jax 0.11.2 (`/export/scratch/tbuck/jax0112-venv`), jaccpot `s5/base`
  (fff3155) + this branch (only `jaccpot.softening` is used from the library).
- **Three configurations**, the same 1e5 particles in each:
  - **A**, the task's: the production softening (1.5e-3), direct-sum self-gravity.
  - **B**, eps 0.0238 = 1.5e-3 x sqrt(25,165,824 / 1e5): the softening at which one particle at eps pulls as weakly
    against the mean field (0.13 against 23) as at 25M with 1.5e-3. It isolates encounters, but it also softens
    the centre so much that the timestep spread collapses (orbit-time bound 3.65x).
  - **S**, a fixed field (`--static`): the IC's particles as test particles in the NFW halo + a Hernquist bulge
    softened at 1.5e-3 + a Miyamoto-Nagai disc (M 6, b 0.03, a + b = 2.15 R_d / sqrt 2). There are no encounters
    and no chaos, the spread is the production one (orbit-time bound 17.4x), and E_i is an exact invariant, so
    per-particle errors against the reference are meaningful. It is the configuration closest to a 25M disc,
    where encounters are negligible. The IC is not in exact equilibrium in this field (sigma_z changes by up to 21 %
    in the inner disc over the run), which does not matter: every score is against the converged run in the same
    field.
- **References:** shared KDK at dt 3.125e-5 (A, S; 1/16 of the production step) and 2.5e-4 (B), each repeated at
  half the step as `ref2`. In A, ref2 vs ref is not yet saturated by chaos (r_1 % differs by 0.23 % against 1.5-1.8 %
  between converged shared runs at 1.25e-4 and 2.5e-4), so A's noise floor is the range over ref2 and those two
  runs. B and S are not chaotic at this level (ref2 vs ref: sigma_z 3e-6 and 3e-7).

## Schemes, criteria and costs

- **shared:** KDK with one fixed dt. A shared run at dt is the shared dt_min of either criterion at eta =
  dt / tau_min: tau_min(acc) = 2.78e-3, tau_min(dyn) = 3.87e-3 at t = 0 (A; S 2.9e-3 and 4.1e-3; B 0.0189 and
  0.0240). The production step 5e-4 is the acceleration criterion at eta 0.18.
- **HOLD:** level = min rung of a pair (the map S6 builds). Rungs come from the criterion at every base step from
  the total force (nornax `block_kdk`'s rule), with the fastest particle of t = 0 on rung k_max. `+ref` lets a
  particle move to a finer rung at its own interior boundary. `frozen` keeps the t = 0 rungs (`--rungs-once`, a
  diagnostic). The base step is t_end / n_base with n_base a multiple of 16 (the output cadence). That caps dt_max
  at 0.094 and sets the finest rung slightly below the criterion's dt_min.
- **Criteria:** `acc` = eta sqrt(eps / |a|); `dyn` = eta sqrt(sqrt(r^2 + eps^2) / |a|), r from the halo centre;
  `dyn+floor 0.2` = dyn with no step below 0.2 t_eps, where t_eps = sqrt(eps^3 / G M(<eps)) = 2.47e-3 is the
  dynamical time at the softening length and 0.2 t_eps = 4.95e-4 the production shared step. The floor stops the
  orbit-time criterion from resolving the softened core more finely than production does, which removes its
  finest rungs.
- **Costs, as equivalent shared steps** (a shared run costs `steps + 1` in every column):
  - `evals`: particle evaluations of the FMM as S6 plans HOLD, one fused evaluation of the padded subset S_m of
    every active level (the level-m kick needs F(S_m) - F(S_{m+1})), plus both base-boundary passes;
  - `weighted`: one weighted evaluation of S_floor per boundary (S0's model; needs per-rung source multipoles);
  - `pairs`: direct-sum pair work. `padded` (each level padded to its largest subset of the run) is 1.00-1.19x
    of `evals` and is in the `tables_*.md` files.

## 1. The spread and what it bounds (`probes/`, t = 0)

dt / dt_min percentiles p1 / p50 / p99, then per k_max: particle-step bound / `evals` / `weighted` / `pairs`
speed-up of HOLD over shared dt_min at the SAME eta (the rung histogram's ceiling, not an accuracy comparison).

| IC | criterion | p1 / p50 / p99 | k_max 4 | k_max 6 | k_max 8 |
| --- | --- | --- | --- | --- | --- |
| N 1e5 seed 1 | acc | 1.5 / 2.9 / 6 | 2.01 / 0.89 / 1.78 / 1.92 | 2.01 / 0.95 / 1.95 / 2.11 | 2.01 / 0.98 / 1.99 / 2.16 |
| N 1e5 seed 1 | dyn | 4.8 / 38.4 / 178 | 13.85 / 11.33 / 7.42 / 7.98 | 20.60 / 9.88 / 15.59 / 21.77 | 20.68 / 9.48 / 19.14 / 29.54 |
| N 1e5 seed 2 | acc | 1.4 / 2.8 / 6 | 1.98 / 0.88 / 1.76 / 1.89 | 1.98 / 0.93 / 1.92 / 2.07 | 1.98 / 0.97 / 1.96 / 2.13 |
| N 1e5 seed 2 | dyn | 4.5 / 38.6 / 176 | 13.83 / 11.29 / 7.42 / 7.97 | 20.49 / 9.82 / 15.52 / 21.72 | 20.57 / 9.43 / 19.04 / 29.44 |
| N 1e6 | acc | 1.3 / 2.6 / 5 | 1.93 / 0.86 / 1.73 / 1.85 | 1.93 / 0.91 / 1.88 / 2.03 | 1.93 / 0.95 / 1.92 / 2.08 |
| N 1e6 | dyn | 4.7 / 38.6 / 176 | 13.84 / 11.31 / 7.42 / 7.98 | 20.54 / 9.85 / 15.55 / 21.75 | 20.62 / 9.46 / 19.08 / 29.51 |
| fixed field (S) | acc | 1.2 / 2.7 / 6 | 1.83 / 0.82 / 1.64 / 1.89 | 1.83 / 0.86 / 1.78 / 2.08 | 1.83 / 0.90 / 1.82 / 2.13 |
| fixed field (S) | dyn | 3.8 / 35.9 / 170 | 12.80 / 9.59 / 7.11 / 7.93 | 17.37 / 8.21 / 13.66 / 20.45 | 17.42 / 8.05 / 16.31 / 27.13 |
| eps 0.0238 (B) | acc | 1.1 / 1.7 / 4 | 1.11 / 0.51 / 1.04 / 1.13 | 1.11 / 0.53 / 1.09 / 1.19 | 1.11 / 0.55 / 1.10 / 1.21 |
| eps 0.0238 (B) | dyn | 1.1 / 6.2 / 29 | 3.65 / 1.66 / 2.97 / 3.97 | 3.65 / 1.69 / 3.46 / 4.89 | 3.65 / 1.77 / 3.60 / 5.19 |

- The spreads do not depend on N or the draw: 1e6 reproduces 1e5, and 1e6's acceleration bound (1.93x) is S0's
  25M value.
- The softened orbit-time criterion spreads half as far as S0's `sqrt(r/|a|)`: median 38x dt_min instead of 78x,
  bound 20.6x instead of 39.7x. The eps in sqrt(r^2 + eps^2) lifts the centre's dt_min.
- **HOLD on plain padded subsets pays every active level's subset at every boundary:** about 2 sum_i 2^rung_i
  evaluations, twice the particle-step count. On the acceleration criterion that is 0.86-0.98x, a LOSS against
  shared dt_min at the same eta; on the orbit-time criterion 9.5-11.9x. Only the weighted single pass (S0's model)
  approaches the bound.

## 2. S, the fixed field: orbit integration at the production spread (`scores_S.json`, `tables_S.md`)

Costs as equivalent shared steps. Then max |dE/E0| (the total energy), the per-particle |dE_i / E_i| median
and 99th percentile (all particles, then the bulge), the position error against the reference (90th percentile, in
R_d) and the RMS relative sigma_z(R) difference, all at t_end = 1.5.

| run | evals | weighted | pairs | max dE/E0 | dE_i p50 | dE_i p99 | bulge p99 | dx p90 / R_d | sigma_z rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ref2 vs ref | | | | 1.2e-08 | 9.9e-11 | 9.9e-09 | 2.6e-06 | 1.8e-07 | 2.9e-07 |
| shared dt 1.25e-4 | 12001 | 12001 | 12001 | 2.8e-08 | 6.3e-09 | 6.3e-07 | 2.5e-05 | 3.6e-06 | 1.8e-06 |
| shared dt 2.5e-4 | 6001 | 6001 | 6001 | 7e-08 | 2.5e-08 | 2.5e-06 | 4.9e-05 | 1.5e-05 | 2.8e-05 |
| **shared dt 5e-4 (production)** | 3009 | 3009 | 3009 | 2.8e-07 | 1e-07 | 1e-05 | 8.8e-05 | 6e-05 | 9.2e-05 |
| shared dt 1e-3 | 1505 | 1505 | 1505 | 1.1e-06 | 4e-07 | 5.2e-05 | 0.0005 | 0.00025 | 0.00021 |
| shared dt 2e-3 | 753 | 753 | 753 | 3.7e-06 | 1.6e-06 | 0.00049 | 0.0056 | 0.001 | 0.00023 |
| shared dt 4e-3 | 385 | 385 | 385 | 0.0001 | 6.5e-06 | 0.006 | 0.029 | 0.0044 | 0.00059 |
| HOLD acc k4 eta 0.05 | 12901 | 6437 | 5489 | 3.7e-08 | 6e-08 | 1.8e-06 | 2.7e-05 | 2.9e-05 | 8.6e-07 |
| HOLD acc k4 eta 0.1 | 6459 | 3228 | 2761 | 1.2e-07 | 2.4e-07 | 8.4e-06 | 5.2e-05 | 0.00013 | 4e-05 |
| HOLD acc k4 eta 0.2 | 3244 | 1627 | 1395 | 3.8e-07 | 8.4e-07 | 4.2e-05 | 0.00011 | 0.00063 | 0.00026 |
| HOLD acc k4 eta 0.4 | 1644 | 830 | 708 | 1.4e-05 | 2.9e-06 | 0.00037 | 0.00063 | 0.0034 | 0.002 |
| HOLD dyn k6 eta 0.025 | 1920 | 1150 | 744 | 8.1e-06 | 6.1e-06 | 9.6e-05 | 0.00017 | 0.0018 | 0.0016 |
| HOLD dyn k6 eta 0.05 | 954 | 584 | 382 | 5.4e-05 | 2.9e-05 | 0.00055 | 0.00099 | 0.0083 | 0.0065 |
| HOLD dyn k6 eta 0.1 | 478 | 293 | 191 | 0.00033 | 0.00015 | 0.0032 | 0.0072 | 0.041 | 0.013 |
| HOLD dyn k6 eta 0.025 +ref | 2285 | 1163 | 745 | 2.9e-06 | 5.5e-06 | 5.1e-05 | 7.6e-05 | 0.0015 | 0.0012 |
| HOLD dyn k6 eta 0.05 +ref | 1146 | 596 | 382 | 1.8e-05 | 2.4e-05 | 0.00021 | 0.0003 | 0.0061 | 0.0022 |
| HOLD dyn k6 eta 0.1 +ref | 590 | 304 | 192 | 0.00013 | 0.00012 | 0.001 | 0.0014 | 0.027 | 0.0072 |
| HOLD dyn k6 eta 0.2 +ref | 311 | 158 | 97 | 0.00093 | 0.00068 | 0.006 | 0.0086 | 0.14 | 0.012 |
| HOLD dyn k6 eta 0.4 +ref | 168 | 83 | 50 | 0.0077 | 0.0047 | 0.054 | 0.086 | 0.82 | 0.015 |
| HOLD dyn k4 eta 0.05 +ref | 892 | 1071 | 941 | 1.1e-05 | 7.5e-06 | 0.0002 | 0.00027 | 0.0043 | 0.0015 |
| HOLD dyn k8 eta 0.05 +ref | 1452 | 538 | 289 | 3e-05 | 2.9e-05 | 0.00023 | 0.0003 | 0.0063 | 0.002 |
| HOLD dyn+floor0.2 k6 eta 0.025 +ref | 2442 | 895 | 554 | 4e-06 | 6.1e-06 | 5.6e-05 | 0.00011 | 0.0015 | 0.00051 |
| HOLD dyn+floor0.2 k6 eta 0.05 +ref | 1285 | 509 | 302 | 2.3e-05 | 2.7e-05 | 0.00023 | 0.00031 | 0.0061 | 0.0017 |

**HOLD's gain at equal accuracy** (`table`: the shared cost that reaches the HOLD run's score, interpolated
log-log along the shared runs, over the HOLD run's cost; `--` where the score lies outside the shared runs' range):

| run | score | evals | weighted | pairs |
| --- | --- | --- | --- | --- |
| HOLD acc k4 eta 0.05 | max dE/E0 / dE_i p99 / dx p90 | 0.75 / 0.55 / 0.33 | 1.51 / 1.10 / 0.67 | 1.77 / 1.29 / 0.78 |
| HOLD acc k4 eta 0.2 | max dE/E0 / dE_i p99 / dx p90 | 0.80 / 0.51 / 0.29 | 1.59 / 1.01 / 0.59 | 1.85 / 1.18 / 0.68 |
| HOLD dyn k6 eta 0.025 +ref | max dE/E0 / dE_i p99 / dx p90 | 0.38 / 0.66 / 0.28 | 0.74 / 1.31 / 0.55 | 1.16 / 2.04 / 0.86 |
| HOLD dyn k6 eta 0.05 +ref | max dE/E0 / dE_i p99 | 0.48 / 0.85 | 0.91 / 1.63 | 1.42 / 2.54 |
| HOLD dyn k6 eta 0.1 +ref | dE_i p99 | 1.05 | 2.03 | 3.21 |
| HOLD dyn k8 eta 0.05 +ref | max dE/E0 / dE_i p99 | 0.34 / 0.65 | 0.92 / 1.76 | 1.71 / 3.28 |
| HOLD dyn+floor0.2 k6 eta 0.025 +ref | max dE/E0 / dE_i p99 / dx p90 | 0.30 / 0.60 / 0.26 | 0.83 / 1.65 / 0.72 | 1.34 / 2.66 / 1.16 |

- **At equal accuracy the orbit-time criterion's 17-20x spread buys 0.28-1.05x with plain padded subsets,
  0.55-2.03x with the weighted pass and 0.86-3.28x in pair work** (refined, eta 0.025-0.1; eta 0.2 reaches
  1.24 / 2.45 / 3.98x, but only on a per-particle tail of 6e-3, 600x the production step's). The best case is the per-particle energy tail
  (dE_i p99). On total energy and positions HOLD loses in the FMM proxies.
- **Why the gain collapses.** A shared step's accuracy is set by the bulk of the mass, not by its fastest particle.
  The shared step that matches HOLD-dyn at eta 0.025 is dt ~ 1e-3, which is the orbit-time criterion's dt_min at
  eta 0.26. Matching it therefore needs a 10x smaller orbit-time eta, and the 20x spread becomes about 2x. Put
  differently, the orbit-time criterion spends the same number of steps on every orbit (2 pi / eta: 250 at
  0.025), while a shared step over-resolves the outer disc, which holds most of the mass, for free.
  - Leapfrog's energy error per particle scales as (dt / T)^2. Minimising the mass-summed error at fixed cost gives
    dt proportional to T^(2/3); the acceleration criterion is about T^(1/2) on a flat rotation curve, the
    orbit-time criterion T^1.
  - The orbit-time criterion wins where it should: the bulge tail. At eta 0.025 +ref its bulge p99 is 7.6e-5 at
    2285 evals / 745 pairs, against shared 5e-4's 8.8e-5 at 3009 (1.3x / 4.0x).
- **The acceleration criterion's spread is too narrow to pay for HOLD's overhead:** 0.26-0.80x in evals, up to
  1.85x in pairs.
- `+ref` (refinement at the particle's own boundary) cuts the orbit-time criterion's energy error 2.5-3x at
  eta <= 0.1 and 1.8x at 0.2; at 0.4 it is worse (7.7e-3 against 4.6e-3).
- The floor at the production step (eta 0.025 +ref) saves 23-26 % of the weighted and pair work (895 / 554 against
  1163 / 745) for a 1.4x larger total-energy and bulge-tail error, and costs 7 % more `evals`.
- sigma_z(R) shows the same order: within 0.06 % of the reference for every shared step up to 4e-3 and 0.26 % for
  HOLD-acc up to eta 0.2. For HOLD-dyn it is 0.12-0.22 % at eta <= 0.05 with refinement, 0.7-1.3 % at 0.1 and
  1.2-1.9 % at 0.2-0.4 (scatter, not heating: the signed mean stays within +-0.5 %). At about equal cost (dyn 0.05 +ref,
  1146 evals, against shared 1e-3, 1505) that is a 10x larger sigma_z error.

## 3. A, the N-body at the production softening: encounters decide (`scores_A.json`, `tables_A.md`)

Costs; max |dE/E0|; per-particle energy diffusion of the disc (RMS of E_i(t_end) - E_i(0)) over the reference's;
RMS relative sigma_z(R) and Sigma(R) differences; the 0.5 % Lagrangian radius and the bulge's 10 % radius,
relative to the reference; max |A2(t) - A2_ref(t)| (disc, R < 3 R_d).

| run | evals | weighted | pairs | max dE/E0 | E diffusion | sigma_z | Sigma | r(0.5 %) | bulge r(10 %) | A2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ref2 vs ref | | | | 9.8e-09 | 1.00 | 0.0085 | 0.015 | +0.019 | -0.006 | 0.00085 |
| shared dt 1.25e-4 | 12001 | 12001 | 12001 | 1.2e-08 | 1.00 | 0.003 | 0.0085 | +0.038 | -0.003 | 0.00061 |
| shared dt 2.5e-4 | 6001 | 6001 | 6001 | 4.9e-08 | 0.99 | 0.0028 | 0.0067 | -0.050 | -0.016 | 0.00085 |
| **shared dt 5e-4** | 3009 | 3009 | 3009 | 1.3e-07 | 1.00 | 0.0042 | 0.016 | -0.004 | +0.009 | 0.00082 |
| shared dt 1e-3 | 1505 | 1505 | 1505 | 5.7e-06 | 1.00 | 0.0052 | 0.034 | +0.009 | +0.003 | 0.0011 |
| shared dt 2e-3 | 753 | 753 | 753 | 9.4e-05 | 1.00 | 0.019 | 0.0098 | +0.033 | +0.028 | 0.0018 |
| shared dt 4e-3 | 385 | 385 | 385 | 0.00033 | 1.05 | 0.015 | 0.017 | +0.242 | +0.090 | 0.00067 |
| HOLD acc k4 eta 0.05 | 12308 | 6173 | 5706 | 8e-07 | 1.00 | 0.011 | 0.021 | +0.001 | +0.006 | 0.00054 |
| HOLD acc k4 eta 0.1 | 6236 | 3131 | 2888 | 1.7e-06 | 1.00 | 0.0072 | 0.0057 | +0.007 | +0.003 | 0.00063 |
| HOLD acc k4 eta 0.2 | 3118 | 1566 | 1444 | 1.3e-05 | 1.00 | 0.0057 | 0.0089 | -0.007 | -0.002 | 0.00081 |
| HOLD acc k4 eta 0.4 | 1631 | 823 | 748 | 2.9e-05 | 1.01 | 0.024 | 0.013 | +0.037 | -0.000 | 0.0016 |
| HOLD dyn k6 eta 0.025 | 1647 | 1048 | 745 | 8.8e-05 | 1.01 | 0.0068 | 0.0095 | -0.021 | +0.028 | 0.0019 |
| HOLD dyn k6 eta 0.05 | 824 | 525 | 373 | 7.7e-05 | 1.07 | 0.0049 | 0.0087 | +0.013 | -0.016 | 0.0016 |
| HOLD dyn k6 eta 0.1 | 412 | 263 | 187 | 9.6e-05 | 1.23 | 0.022 | 0.025 | +0.037 | -0.020 | 0.002 |
| HOLD dyn k6 eta 0.2 | 206 | 132 | 94 | 0.0011 | 1.59 | 0.017 | 0.019 | +0.069 | -0.014 | 0.0019 |
| HOLD dyn k6 eta 0.4 | 100 | 65 | 47 | 0.0042 | 2.64 | 0.033 | 0.049 | +0.277 | +0.141 | 0.0052 |
| HOLD dyn k6 eta 0.0125 +ref | 3789 | 2092 | 1472 | 4.9e-05 | 1.00 | 0.0052 | 0.018 | -0.005 | +0.035 | 0.00093 |
| HOLD dyn k6 eta 0.025 +ref | 1893 | 1059 | 745 | 9.3e-05 | 1.01 | 0.0048 | 0.012 | +0.035 | +0.006 | 0.0014 |
| HOLD dyn k6 eta 0.05 +ref | 961 | 534 | 373 | 0.0001 | 1.07 | 0.013 | 0.0089 | +0.015 | -0.007 | 0.0017 |
| HOLD dyn k6 eta 0.1 +ref | 493 | 272 | 188 | 1.4e-05 | 1.23 | 0.021 | 0.014 | +0.008 | -0.024 | 0.0022 |
| HOLD dyn k6 eta 0.2 +ref | 260 | 141 | 95 | 0.00097 | 1.53 | 0.01 | 0.039 | -0.060 | -0.040 | 0.0041 |
| HOLD dyn+floor0.2 k6 eta 0.0125 | 2883 | 1404 | 1020 | 2.3e-06 | 0.99 | 0.021 | 0.0091 | +0.018 | +0.013 | 0.0011 |
| HOLD dyn+floor0.2 k6 eta 0.025 | 1600 | 785 | 537 | 2.4e-05 | 1.02 | 0.0058 | 0.013 | +0.002 | +0.003 | 0.001 |
| HOLD dyn k6 eta 0.05 frozen | 778 | 505 | 365 | 0.00029 | 1.13 | 0.016 | 0.03 | -0.019 | -0.009 | 0.0025 |
| HOLD dyn k6 eta 0.1 frozen | 389 | 253 | 183 | 0.0008 | 1.38 | 0.012 | 0.026 | +0.045 | +0.022 | 0.0029 |

(`tables_A.md` has every run, the `k4`/`k8` and refined-acc variants, sigma_R, v_phi and the bulge median.)

- **Energy is set by close encounters at the softening length.** Every scheme drifts linearly once a particle's
  step exceeds about 5e-4 (eps / 3, the crossing time of a softened encounter). Shared steps converge steeply:
  9.4e-5, 5.7e-6 and 1.3e-7 at 2e-3, 1e-3 and 5e-4. The orbit-time criterion does not see encounters, so its
  drift saturates near 1e-4 for every eta <= 0.1 (8.8e-5 at eta 0.025, 7.7e-5 at 0.05, 9.6e-5 at 0.1; 4.9e-5 even
  at 0.0125 with refinement). HOLD-acc keeps 8.4 % of its particles on steps of 4-16 dt_fine and drifts 8e-7 at
  eta 0.05, against 4.9e-8 for shared 2.5e-4.
- **The orbit-time criterion heats particle energies.** The disc's per-particle energy diffusion exceeds the
  reference's by 7 % (eta 0.05), 23 % (0.1), 53-59 % (0.2) and 164 % (0.4). Every shared step up to 2e-3 and every
  HOLD-acc run stays within 2 %. The extra diffusion sits in the outer galaxy: at r 1-3 the RMS change is 0.36
  against 0.21 for the reference at eta 0.1.
- **The centre and the bulge** move only at the coarsest steps. The 0.5 % radius grows 24-28 % for shared 4e-3 and
  dyn eta 0.4, and the bulge's 10 % radius 9-14 %, against a 2-5 % spread between converged runs. Everything at
  dyn eta <= 0.1 or shared dt <= 2e-3 is within that spread.
- **Disc structure hardly discriminates.** sigma_z and Sigma are within about 3x the floor for every run except
  dyn eta 0.4. The A2 difference reaches 5-6x the floor at dyn eta 0.2-0.4, still at A2's own shot-noise level
  (the reference's A2 is 0.0004-0.012). No bar grows over one orbit of this hot disc (Q_min 1.6).
- **Frozen rungs are worse, not better** (dyn eta 0.1: 8.0e-4 frozen against 9.6e-5 reassigned): in A the rungs
  must follow the particles.
- **At equal cost HOLD loses in every family.** In max dE/E0, HOLD-acc reaches 0.18-0.62x in `evals` and
  0.38-1.35x in pairs. HOLD-dyn reaches 0.4-2.5x in `evals` and 1.0-6.5x in pairs, but only at energy errors of
  1.4e-5 to 1e-4, 100-700x the production step's 1.3e-7.

## 4. B, the production encounter strength with a soft centre (`scores_B.json`, `tables_B.md`)

| run | evals | weighted | pairs | max dE/E0 | sigma_z | Sigma |
| --- | --- | --- | --- | --- | --- | --- |
| ref2 vs ref | | | | 6.4e-09 | 2.7e-06 | 0 |
| shared dt 5e-4 | 3009 | 3009 | 3009 | 4.4e-08 | 9.3e-06 | 0.00011 |
| shared dt 2e-3 | 753 | 753 | 753 | 2.7e-07 | 0.00064 | 0.0011 |
| shared dt 4e-3 | 385 | 385 | 385 | 8.6e-07 | 0.0033 | 0.0047 |
| shared dt 8e-3 | 193 | 193 | 193 | 2.5e-06 | 0.01 | 0.0036 |
| HOLD acc k3 eta 0.1 | 1667 | 854 | 751 | 2.9e-07 | 0.00077 | 0.0021 |
| HOLD acc k3 eta 0.2 | 708 | 379 | 331 | 5.6e-05 | 0.001 | 0.0034 |
| HOLD acc k3 eta 0.2 frozen | 672 | 363 | 324 | 7.7e-06 | 0.0018 | 0.0043 |
| HOLD acc k1 eta 0.2 | 686 | 576 | 539 | 4.6e-06 | 0.0022 | 0.0028 |
| HOLD acc k1 eta 0.2 frozen | 681 | 574 | 536 | 6e-06 | 0.0027 | 0.0039 |
| HOLD dyn k5 eta 0.05 | 802 | 411 | 292 | 0.00011 | 0.0037 | 0.002 |
| HOLD dyn k5 eta 0.1 | 398 | 212 | 153 | 0.00051 | 0.0032 | 0.017 |
| HOLD dyn k5 eta 0.1 +ref | 499 | 220 | 155 | 0.0004 | 0.003 | 0.0088 |
| HOLD dyn k5 eta 0.2 | 200 | 107 | 77 | 0.0022 | 0.008 | 0.052 |

- With smooth forces, shared steps conserve energy even at dt 8e-3 (2.5e-6, 193 steps). Every HOLD run with
  rungs reassigned at the base step drifts upwards linearly: dyn eta 0.05 gains +1.1e-4 at 4x shared 8e-3's cost.
- **The drift comes from the base-step rung reassignment interacting with HOLD's pair splitting.** With the rungs
  frozen at t = 0, HOLD-acc eta 0.2 stays bounded (at most 7.7e-6) instead of drifting to 5.6e-5. In S, which has no
  pair coupling, the same reassignment left HOLD-acc within 1.4x of shared at the same finest step. This is the time-asymmetric step
  selection of a variable-step leapfrog (Hut, Makino & McMillan 1995), made worse because a rung change moves every
  pair of the particle to another level.
- **Even frozen, HOLD's slow-fast coupling costs accuracy:** 7.7e-6 against 8.6e-7 for shared at the larger step
  4e-3. In HOLD both partners of a slow-fast pair get their kick on the slow step.

## 5. Recommendation

- **Galaxy discs keep the acceleration criterion and the production shared step** (dt 5e-4, the acceleration
  criterion's dt_min at eta 0.18). The orbit-time criterion does not make block steps worth 10-40x on this disc at
  equal accuracy:
  - in S, the orbit-integration test at the production spread, HOLD-dyn buys 0.28-1.05x in particle evaluations
    on plain padded subsets (the S6 plan), 0.55-2.03x with one weighted pass per boundary and 0.86-3.28x in pair
    work (refined, eta 0.025-0.1), and it loses on total energy and positions in both FMM proxies;
  - in the N-body tests A and B, shared steps win at equal cost in every family.
- **HOLD's speed-up over shared steps at equal accuracy** (S, the range over total energy, the per-particle energy
  tail and positions):

  | criterion | plain padded subsets (`evals`) | weighted pass | pair work |
  | --- | --- | --- | --- |
  | acceleration, k_max 4, eta 0.05-0.4 | 0.26-0.80x | 0.52-1.59x | 0.61-1.85x |
  | orbit-time, k_max 6-8, refined, eta 0.025-0.1 | 0.28-1.05x | 0.55-2.03x | 0.86-3.28x |

- **What the orbit-time criterion degrades:**
  - **Close encounters** (A): it does not see them. Its energy drift saturates near 1e-4 whatever its eta, and it
    adds 7-164 % of spurious per-particle energy diffusion in the disc (eta 0.05-0.4). At 25M the encounters are
    250x weaker, so this matters most at low N or small softening.
  - **The centre and bulge** move beyond the converged-run spread only at eta >= 0.2: the 0.5 % radius by +7 %
    (A, eta 0.2) and +28 % (eta 0.4); in S, the bulge's 10 % radius by -13 % at eta 0.4 with refinement.
  - **Disc structure** (Sigma, sigma_z, A2): no systematic heating at any eta. The sigma_z error in S is 10x
    shared's at equal cost.
  - **Momentum:** HOLD keeps the pair impulse at round-off in every run (at most 5.6e-11 of sum m |v|; |dL|/|L| at
    most 1.2e-10 in A and B).
- **For the block-step track (S5-S7)**, if it is resumed for ICs with a genuinely wider range at fixed accuracy
  (cusps, a central black hole, mergers, a much smaller softening):
  - the weighted single pass (S5's weight tables plus per-rung source multipoles) is required; padded subsets pay
    about twice the particle-step count;
  - the base-step rung reassignment needs a time-symmetric selection, or HOLD drifts secularly in a smooth N-body
    (B: 5.6e-5 against a bounded 7.7e-6 with frozen rungs);
  - with long base steps, rungs must also refine at the particle's own boundary (A, dyn eta 0.1: 9.6e-5 to
    1.4e-5);
  - HOLD's slow-fast coupling error is about 9x shared's even with frozen rungs (B).
- **Not decided here:** whether 25M could run at a coarser shared step. In S, shared 1e-3 holds the per-particle
  energy tail at 5.2e-5 and the bulge tail at 5e-4 over the run; at 1e5 (A) it drifts 5.7e-6 through encounters,
  which are 250x weaker at 25M. A 25M fused-lane energy run at 5e-4 against 1e-3 would settle it.

## Files

| path | what |
| --- | --- |
| `rows/*.json` | every run: config, the cost counters, the 17-point series (E, K, W, L, P, pair impulse, A2, Lagrangian radii); HOLD rows add the rung histogram per base step. Prefixes: none = A, `B_`, `S_` |
| `scores_{A,B,S}.json` | `score` output: per-run scores against `ref`, `noise_floor` = `ref2` against `ref`, the reference's profiles at t = 0 and t_end |
| `tables_{A,B,S}.md` | `table` output: costs (also `padded`), all scores, the equal-accuracy gains |
| `probes/` | `probe` at t = 0: N 1e5 (seeds 1, 2), 1e6, the fixed field, eps 0.0238 |
| `logs/` | one log per run, the probes and the sweeps (card, start/end, nvidia-smi before and after) |
| `scripts/` | the run lists `sweep{1,2,B,S}.txt` with the drivers `sweep.sh` / `launch.sh` and `sweepB2.txt` (the frozen-rung check in B) |

The per-particle end states (`x`, `v`, `E_i`; 83 npz files, 825 MB) are kept outside the repo in
`/export/scratch/tbuck/c1_criterion_2026-10/`; `score` needs them.
