# MESC: escapers on the mesh at 1e8 on HoreKa (prepared, not run)

Track MESC of the multi-node accuracy plan (wave 3): `ProductionRunner(mesh=MeshRunConfig(...),
escapers=EscaperPolicy(r_escape=80))` on the FULL unclipped 1e8 Sgr IC (100,000,001 rows, a 45.7 Mpc
box) as N processes x 1 GPU, against the clipped control (escapers off, every particle beyond 80
pulled onto r = 80, same N). Local gates (CPU, 2 shared A100s at 2.2e6): `docs/production_2026-11.md`,
section MESC.

## What you run

H0's environment (`bench/slurm/h0/`: `site.env`, the venv, `setup_env.sh` once). Stage the IC first
(4.4 GB):

```bash
mkdir -p $H0_ROOT/data
# from the development machine:
#   /export/scratch/tbuck/mn-scratch/a2/sgr/sgr_light_1e8_noclip.npz -> $H0_ROOT/data/
bash bench/slurm/mesc/submit.sh --dry-run
bash bench/slurm/mesc/submit.sh            # node1 and node2; or: submit.sh node1
```

Results: `$H0_ROOT/mesc/<run id>/<job>/SUMMARY.md` and every row's `result_p<rank>.json`. Send back
`tar czf mesc.tgz -C $H0_ROOT/mesc/<run id> .` (small). A resubmission into the same run
(`MESC_RUN=$H0_ROOT/mesc/<run id> bash bench/slurm/mesc/submit.sh node1`) skips the finished rows.

## The jobs

| job | nodes x tasks | rows (`bench/mesh_escaper_timing.py`, 6 segments x 10 steps, dt 1e-3, energy every edge) | `--time` |
| --- | --- | --- | --- |
| `node1` | 1 x 4, exclusive | `off_dehnen`, `esc_dehnen`, `off_dehnen_error`, `esc_dehnen_error`, `esc_dehnen_e10` (energy every 10th edge) | 3:00 |
| `node2` | 2 x 4, exclusive | `off_*`, `esc_*` for both MACs | 2:30 |

`dehnen` is the geometric MAC at theta 0.7 (the Sgr production run's); `dehnen_error` the galaxy
preset's. Compare `step_s` (median of segments 2-6) and the peak B/p per card between `esc_*` and
`off_*`; `edge_s` holds the edges' energies, the escaper membership and, when the split changes, the
two ledger energies and the re-cut repartition (`RunResult.escapers["recuts"]`).

What to look for:
- `esc_*` runs without a `TreeDepthError` (the escapers are not in the tree); escaper counts ~1.8e5
  (the single card's 182,878 -> 182,889 over 15 steps);
- peak B/p per card close to `off_*` (the tail is ~1.8e5 / N rows of ~60 B);
- the edge cost of a changing split: `esc_dehnen` vs `esc_dehnen_e10` vs `off_dehnen`.

If a row OOMs with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, rerun it with `PREALLOCATE=true` (ESC's 1e8
single-card note: A1's BFC placement sensitivity, not the escapers).
