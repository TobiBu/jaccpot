# M1: momentum of the multi-GPU runner at scale, local vs cross part (prepared, not run)

Track M1 of the multi-node accuracy plan (`~/.claude/plans/multinode-accuracy-dev-plan.md`, jaccpot #173). The local
record (`docs/production_2026-11.md`, `## M1`) found NO cross-domain momentum gap: every cross pair is evaluated on
both of its devices from bitwise-identical shipped multipoles, centres and particles, as one card evaluates each of
its own pairs once per direction, so the cross part of `sum m a` cancels as well as the local part (fp64, forced
CPU devices: both ~1e-17 relative; fp32 on two A100s: both ~4e-9). No reaction-return transport was built, so there
is no option to time on/off. These jobs check the finding where the local runs could not: the production preset at
1e7-1e8, 4 GPUs on NVLink and 8 GPUs over two nodes, one process per GPU.

## What you run

H0's environment (`bench/slurm/h0/`: `site.env`, `setup_env.sh` once, `provenance.py`). No staged data: the jobs
draw a bulge + disc IC themselves (seeded; `--ic plummer` for a Plummer sphere).

```bash
bash bench/slurm/m1/submit.sh --dry-run
bash bench/slurm/m1/submit.sh            # node1 and node2; or: submit.sh node1
```

Results: `$H0_ROOT/m1/<run id>/<job>/SUMMARY.md` and every arm's `result.json`. Send back
`tar czf m1.tgz -C $H0_ROOT/m1/<run id> .` (small).

## The jobs

| job | nodes x tasks | arms (each 2 x 100 steps, dt 1e-3, migration on) | `--time` |
| --- | --- | --- | --- |
| `node1` | 1 x 4, exclusive | `one_<mac>`: 1 GPU, 1e7 (the one-card yardstick); `mp4_<mac>_1e7`: 4 x 1 GPU at 2.5e6/GPU, both MACs; `mp4_dehnen_error_1e8`: 2.5e7/GPU | 2:00 |
| `node2` | 2 x 4, exclusive | `mp8_<mac>_1e7`: 8 x 1 GPU at 1.25e6/GPU, both MACs; `mp8_dehnen_error_1e8`: 1.25e7/GPU | 1:30 |

Per arm (`bench/mesh_cross_momentum.py`): `momentum_drift` at every edge, and at the start and the end the force with
and without the cross hook on the same state: `|sum m a| / sum m |a|` of the total, of the local part, of the cross
part alone, and the cross part's share of `sum m |a|` (every number a mesh-order fold of per-device fp64 rows).
Why a job of its own: `bench/slurm/m2g/` (`multiprocess_gate.py`) records `momentum_drift` at every edge but not the
per-step local/cross split, which needs the force evaluated twice (with and without the cross hook) on the same state.
Walls per edge and peak bytes are recorded too (dedicated nodes: they count), but nothing here compares an option.

**What to look for:** the cross part's residual at the local part's level and the total at the `one_<mac>` level
(fp32 round-off: ~1e-8 per step); the drift a random walk of that size (~sqrt(steps) x dt x 1e-8). A cross residual
orders of magnitude above the local one at scale would reopen the reaction-return design (the M1 report's
alternatives (a)/(c)).
