# Dehnen's error criterion on the fused lane (2026-10-09)

Branch `feat/dehnen-error-fused`; yggdrax #89 (`pair_accept` on `dual_tree_walk_mutual`). Modules:
`jaccpot/runtime/_walk_criterion.py`, `jaccpot/runtime/_force_scale_levels.py`. Tests:
`tests/unit/runtime/test_walk_criterion.py`, `tests/unit/runtime/test_force_scale_levels.py`,
`tests/unit/runtime/test_force_scale_evaluation.py`, `tests/unit/operators/test_nearfield_force_scale_lane.py`,
`tests/unit/runtime/test_mutual_walk_pallas.py` (eq16a cases), `tests/integration/test_strict_run_v2_dehnen_error.py` (GPU).

## What changed

`mac_type="dehnen_error"` (Dehnen 2014 eq 16a: a far pair is accepted when its estimated force error is below
`eps * min_b f_b` in both directions) now runs on the strict fused lane, including every traced step of
`strict_run_v2`. Before, the eager prepare fell back to the yggdrax dual walk with a pair policy, which ran out of
memory at 2e6 particles with leaf 64, and the traced refresh refused the criterion.

```python
FastMultipoleMethod(
    preset="large_n_gpu", ...,
    advanced=FMMAdvancedConfig(..., mac_type="dehnen_error"),
    adaptive_eps=1e-4,                  # the relative force-accuracy target
    adaptive_error_model="dehnen_paper",
    mac_force_scale_mode="paper_fb",    # f_b, eq 16b (the fused lane always uses it)
)
```

The geometric default (`mac_type="dehnen"`) is unchanged: every new switch is static and off, and the traced
program holds no new array.

## The criterion inside the walk

`dehnen_walk_table` builds, once per walk, a `[nodes, W]` table (`W` a power of two):
- `G M`;
- the node threshold `eps * min_b f_b`;
- the normalised multipole powers `s_n = P_n / (M rho^n)`, `n = 1..p` (eq 12).

`dehnen_pair_accept` evaluates eq (15) in the gather-free form
`E = 8 max(rho)/(rho_A + rho_B) * sum_n C(p,n) s_n a^n b^(p-n)`, with `a = rho_A/r` and `b = rho_B/r`. The sum is
term by term the paper's, and well scaled in float32 at any node size because `s_n` lies in `[0, 1]`. The test is
symmetrised (both directions), as `adaptive_pair_policy` does on the self walk.

The same elementwise function runs in two places:
- inside the Pallas walk (`mutual_walk_pallas(error_order=p, error_table=...)`);
- in yggdrax's XLA walk (`pair_accept=DehnenWalkAccept(p)`).

The radii are the walk's own centre-of-mass radii, a valid and tighter bound than the general path's `"com"` policy
radii. The softening floor applies on top, which the single-GPU general path never did.

Parity: the Pallas walk, the XLA walk and the dual walk running `adaptive_pair_policy` (fed the walk's radii) emit
identical far and near sets.

## The force scale, step to step

`f_b = sum_a G m_a / (|x_a - x_b|^2 + eps^2)` (eq 16b, `eps` the Plummer-equivalent softening for every kernel) is a
by-product of each step's force. Nothing extra is walked.

- **Near half:** the direct CSR near-field kernel accumulates it in its fourth output lane
  (`with_force_scale=True`, one multiply-add per pair; exclusive with the potential).
- **Far half:** each far pair `A -> B` contributes `G M_A / ((|c_A - c_B| + rho_B)^2 + eps^2)`, a lower bound. It
  is computed right after the walk, because the refresh drops the far list before the evaluation.
  - The pairs are walked in chunks of 2^22, so no temporary is list-sized.
  - The result is pushed down the tree (`ancestor_sum_by_level`) and read at each particle's leaf.
- **Carry:** `evaluate_large_n_state(..., return_force_scale=True)` returns `(acc, f_b)`. The scan carries `f_b`, a
  sixth slot of the state carry and a seventh of the particle carry, and so does the `StrictParticleCarry` handle.
- **Thresholds:** the next refresh sorts `f_b` by its new tree, takes each leaf's minimum, then the subtree minimum
  (`subtree_min_by_level`), and uses `eps * min` as the thresholds.
- **Tree reductions:** both are level-order passes over the tree's own level tables, as M2M walks them. They replace
  the general path's serial loops over internal nodes (~4e5 sequential scatters at 25M).
- **First step:** the eager prepare seeds `f_b` the same way (`_fused_force_scale_seed`): one geometric flat walk at
  theta 0.8 (`mac_force_scale_prepass_theta` overrides it), the near lane and the far monopoles. That replaces the
  general `paper_fb` prepass, whose theta-0.5 walk plus downward sweep put the 2e6 disc's prepare at 1.6 kB/particle.
  The seed errs low, so the first step is slightly stricter, never looser.

Values only ever move along true parent edges. A capacity-padded cell partition's dead internal nodes list children
they do not own, and the first push-down counted through them.

## Results (A100, untimed, p6, leaf 64)

2e6 disc, ferrers3 at softening 1.5e-3, theta 0.8; error against an fp64 direct sum of the same kernel:

| MAC | rel-L2 | p99.9 | max da/f | far pairs | near leaf pairs | prepare peak |
| --- | --- | --- | --- | --- | --- | --- |
| geometric theta 0.8 | 6.1e-4 | 5.4e-3 | 8.4e-3 | 7.4M | 2.3M | 207 B/p |
| geometric theta 0.6 | 1.0e-4 | 6.9e-4 | 1.0e-3 | 16.8M | 5.5M | |
| geometric theta 0.5 | 3.3e-5 | 2.3e-4 | 4.6e-4 | 29.1M | 9.2M | |
| dehnen_error eps 1e-4 | 1.6e-4 | 1.1e-3 | 3.0e-3 | 10.2M | 3.0M | 250 B/p |
| dehnen_error eps 1e-5 | 2.1e-5 | 1.2e-4 | 4.9e-5 | 24.0M | 7.8M | 378 B/p |

At eps 1e-5 the criterion beats theta 0.5 on every error measure, on the worst particle by 10x, with 15-18% fewer
lists: Dehnen 2014's narrow distribution, reproduced on the disc. Its prepare peak is the criterion walk's own lists;
the force-scale work adds nothing on top. Timings at 1e8 are pending (no free card).

The full 25,165,824-particle disc+bulge IC, same settings (`bench/fused_memory_budget.py --n 25165824 --ic disc`);
peak is the prepare's, per particle:

| MAC | rel-L2 | p99.9 | max | max da/f | far pairs | near leaf pairs | peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| geometric theta 0.8 | 4.1e-4 | 3.2e-3 | 1.2e-2 | 1.1e-2 | 95.6M | 37.9M | 205 B/p |
| geometric theta 0.5 | 2.0e-5 | 1.7e-4 | 4.7e-4 | 4.4e-4 | 389.6M | 123.4M | 430 B/p |
| dehnen_error eps 1e-4 | 1.5e-4 | 8.7e-4 | 1.3e-3 | 4.9e-4 | 93.9M | 37.7M | 219 B/p |
| dehnen_error eps 1e-5 | 2.3e-5 | 1.3e-4 | 1.9e-4 | 3.9e-5 | 216.6M | 81.7M | 305 B/p |

- **eps 1e-4 against theta 0.8:** the same lists (98-99%), with 2.8x lower rel-L2, 9x lower max error and 22x lower
  max da/f. At 2e6 the same eps needed 38% more far pairs; per particle the criterion's lists shrink with N
  (eps 1e-5: 12.0 far pairs per particle at 2e6, 8.6 at 25M), while the geometric MAC's stay flat (3.7, 3.8).
- **eps 1e-5 against theta 0.5:** a rel-L2 16% higher and a lower tail (p99.9 0.76x, max 0.41x, max da/f 0.09x), from
  56% of the far pairs and 66% of the near pairs, at 71% of the memory.
- At 1e8 these peaks are 22 GB (eps 1e-4) and 31 GB (eps 1e-5) on a 40 GB A100. eps 1e-4's lists match the geometric
  MAC's, so its step should cost about the geometric step plus eq 16a in the walk; eps 1e-5 has about 2.2x the lists.
  Both are inferences from list volumes, not timings.

Rollout conservation, 1e6 particles of the same disc+bulge with its NFW halo as the external field:
- `strict_run_v2` with the particle carry: 800 steps of dt 5e-4, t = 0.4, about 0.6 of an orbit at the disc scale
  radius.
- Every mark is measured with one meter, a geometric theta-0.4 potential (median 1.5e-7 off a direct sum).
- `|dE|` is the largest energy excursion over the marks. `W` is the self-gravity energy; the halo holds 88% of the potential energy.
- `|dL|` is the angular momentum change (the halo exerts no torque) over `sum m |r x v|`.
- The imbalance is `|sum m a| / sum m |a|` of each MAC's own force.

| MAC | max \|dE\|/\|W\| | \|dL\| at t = 0.4 | max imbalance |
| --- | --- | --- | --- |
| geometric theta 0.8 | 1.3e-6 | 7.1e-7 | 1.4e-8 |
| geometric theta 0.5 | 2.1e-7 | 6.4e-8 | 5.3e-9 |
| dehnen_error eps 1e-4 | 7.4e-7 | 5.5e-8 | 7.0e-9 |
| dehnen_error eps 1e-5 | 2.9e-7 | 5.1e-8 | 4.3e-9 |

- The carried force scale and the step-to-step lists inject no drift.
- eps 1e-4 cuts theta 0.8's energy excursion 1.8x and conserves angular momentum 13x better.
- eps 1e-5 sits with theta 0.5 at what is likely this setup's floor (time step, float32 state, meter).
- Every MAC conserves momentum to round-off: the mutual walk's lists are symmetric.

GPU rollout tests:
- both carries run the criterion in the traced steps and agree;
- the carried `f_b` matches a fresh evaluation at the same positions (median within 5%);
- the final force error is under half the geometric MAC's at the same theta.

## Known limits

- **The error tail at outliers.** On an unclipped Plummer sphere the criterion leaves a worst particle at ~900 eps.
  - The cause is sink-side truncation: an outskirts particle sits at the edge of every node above it, and the
    errors of the local expansions add coherently.
  - eq 16a cannot see this, because splitting a source makes each piece pass.
  - A sink cap plus cell-to-particle (M2P) interactions at leaf sinks removes it in simulation (870 -> 26 eps). That
    is the pkdgrav3 design, and it is not built yet.
  - On the disc the worst particle is ~5 eps.
- The tree-order carry (`JACCPOT_STRICT_CARRY_ORDER=tree`) is off with the criterion: input order only.
- The multi-GPU fused lane does not run the criterion.
- The general (non-fused) path is unchanged. Its policy still ignores the softening floor on one GPU.
