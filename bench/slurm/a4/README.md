# A4: the accuracy verification suite on HoreKa (prepared, not run)

Track A4 of the multi-node accuracy plan (`~/.claude/plans/multinode-accuracy-dev-plan.md`). These jobs run the
standard problems' convergence ladders (`bench/accuracy/ladders.py`) in two tiers:

* **`horeka-1card`**: one A100 per rung, an array task on a quarter node;
* **`horeka-node`**: one exclusive node per rung, the runner's multi-GPU mode on its 4 A100s.

Every rung scores the production force against the exact-order reference force
(`jaccpot.verification.reference_force`). That reference is the same number on any device count, so a node
rung's force score compares directly with a card rung's. Every rung also records its wall time; these are
dedicated nodes, so the times count. The suite itself is described in `docs/accuracy_suite.md`.

## What you run

H0's environment does the setup: `bench/slurm/h0/site.env` (account, partition, `H0_ROOT`), then
`setup_env.sh` once.

A4 needs two more values in `bench/slurm/a4/site.env`:

* `A4_AGAMA_PY`: a python with agama, for the galaxy ICs (H0's venv has none);
* `A4_ODISSEO`: an Odisseo checkout with `tools/a4_galaxy_ics.py`.

```bash
bash bench/slurm/a4/submit.sh --dry-run                # every sbatch command, with its dependencies
bash bench/slurm/a4/submit.sh                          # ics -> one array per ladder and tier -> summary
bash bench/slurm/a4/submit.sh --tier 1card --only disc_n,merger_dt   # a subset
A4_RUN=$H0_ROOT/a4/<run id> bash bench/slurm/a4/submit.sh --no-ics   # resume: rungs with a result are skipped
```

Results land in `$A4_RUN/results/<tier>/<ladder>/<rung>/`. Each rung directory holds `result.json`,
`probe.jsonl`, `config.json`, `run.log` and `provenance.json`. Two summaries are written at the end:
`$A4_RUN/SUMMARY-horeka-1card.md` and `$A4_RUN/SUMMARY-horeka-node.md`.

Send back `tar czf a4.tgz --exclude='*.npz' -C $A4_RUN .`, which is small. The lead records the results as the
HoreKa baselines (`suite.py record`). From then on, set `A4_BASELINES` and every rung is checked against its
baseline (`check.txt` beside `result.json`).

## The jobs

| job | resources | what |
| --- | --- | --- |
| `ics` | `cpuonly`, 1 node, 4 h | the galaxy ICs at the ladders' largest N, generated once into `A4_IC`: `galaxy.npz` 2e8, `bar.npz` 1e8, `merger.npz` 2 x 1e8. Skipped when they exist. About 1-1.5 h (the 2026-10-31 1.6e8 disc took 21 min). |
| `card1` (array per ladder) | 1 A100, 19 cores, 125 GB | one rung per task (`suite.py ladder NAME --index $SLURM_ARRAY_TASK_ID`) |
| `node` (array per ladder) | 1 exclusive node | one rung per task, `--devices 4` |
| `summary` | `cpuonly`, 30 min, `afterany` on every array | `suite.py summarize all` per tier, plus the checks and the skipped rungs |

The ladders, with their `--time` (the longest rung x ~1.5, from about 1 s/step at 1e8 on one A100 and linear
scaling in N above ~1e6):

| ladder | 1 card: rungs (wall) | node: rungs (wall) |
| --- | --- | --- |
| `plummer_n` / `hernquist_n` / `nfw_n` | 1e5, 1e6, 1e7, 6400 steps each (2 h) | Plummer 1e7, 4e7 (2 h) |
| `plummer_dt` | 1e6 at t_dyn/32, /64, /128 (2 h) | - |
| `plummer_mac` | 1e6: theta 0.5 / 0.7 / 0.8, tol 1e-3 / 1e-4 / 3e-5 (2 h) | - |
| `plummer_precision` | 1e6: fp32, fp64state, fp64, anchored (3 h) | - |
| `disc_n` | 1e6, 4e6, 1.6e7, 6.4e7, 1 Gyr = 6707 steps (3 h) | 6.4e7, 2e8 (6 h) |
| `disc_dt` | 1.6e7 at dt 2e-3, 1e-3, 5e-4 (3 h) | - |
| `disc_mac` | 1.6e7, six MAC rungs (3 h) | 6.4e7 (6 h) |
| `disc_precision` | 1.6e7, four precisions (6 h) | 6.4e7 (12 h) |
| `bar_n` | 2.5e6, 1e7, 4e7 at theta 0.7, and 2.5e6, 1e7 at `dehnen_error`; 3 Gyr = 20000 steps (10 h) | 1e8 at theta 0.7 and `dehnen_error` (16 h) |
| `merger_n` | 1.25e7, 2.5e7, 5e7; t = 10, 10000 steps (6 h) | 1e8, 2e8 (12 h) |
| `merger_dt` | 2.5e7 at dt 2e-3, 1e-3, 5e-4 (8 h) | 1e8 (16 h) |
| `merger_mac` | 2.5e7: theta 0.7, 0.8, tol 1e-4 (6 h) | - |

**Budget.** The 1-card tier has 53 rungs. They are mostly short: about 60-80 GPU-hours expected, which is
15-20 node-hours, charged per GPU. The node tier has 21 rungs, mostly 0.5-2 h on dedicated cards (the `--time` limits are generous): about 25-40 node-hours expected, at most ~200 if every rung hit its limit.

To stay inside a smaller budget:

* run `--tier 1card` first;
* in the node tier, run `disc_n`, `merger_n` and `bar_n` first. They answer U4 at production N on one node.

The `anchored` precision rungs exit at once (reported as skipped) until A3's runner setting is filled into
`bench/accuracy/suite.py` `PRECISIONS`.

## What to look for (`SUMMARY-*.md`)

* **Force.** `force.p50_end` <= 1e-4 for every `dehnen_error` rung (U4), with p99 recorded. The `*_mac`
  ladders show the error against the MAC parameter, and the node rungs should equal the card rungs at equal N.
* **Disc conservation.** `energy.max_abs_dE` <= 1e-4 and `angmom.max_dL` <= 1e-5 over 1 Gyr for `disc_*`
  (U4).
* **Momentum.** `momentum.max_dP` at round-off on 1 card and on 4 cards alike.
* **Disc heating.** The heating curve against N (`disc_n`: `heat.sigz2_growth_k` should fall about as 1/N
  until the softening floor).
* **The bar.** Buckling time against N (`bar_n`; locally 1.47 / 1.73 / 2.68 / 2.74 Gyr at 2.5e6 / 1e7 / 4e7 /
  1e8 at theta 0.7).
* **The merger.** Orbit convergence in dt and N (`merger_dt`: `merger.sep_max_dev_kpc_vs_ref` against dt 5e-4).

## Local dry run (gate 4)

`bash bench/slurm/a4/submit.sh --local-smoke` runs every job's exact command lines on this host's CPU.

* Every ladder rung is shrunk to a few hundred particles and a few steps (`A4_SMOKE=1`).
* The node tier runs on 4 forced host devices.
* The IC job runs at 1e-4 scale (only when `A4_AGAMA_PY` and `A4_ODISSEO` are set).
* The summary job runs at the end.

The local wrappers set `A4_PY` (e.g. `/export/scratch/tbuck/mn-scratch/bin/py-a4`). `bench/tests/test_accuracy_suite.py`
checks `bash -n` on every script, the dry run's sbatch commands, and that every rung's command line parses.
