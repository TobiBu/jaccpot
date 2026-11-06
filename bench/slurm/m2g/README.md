# M2G: the multi-process runtime on HoreKa (prepared, not run)

Track M2G of the multi-node accuracy plan (`~/.claude/plans/multinode-accuracy-dev-plan.md`): the
multi-GPU runner (`ProductionRunner(mesh=MeshRunConfig(...))`) as one process per GPU under
`jax.distributed`, against one process driving every GPU of the node. The local gate (2 processes x
1 GPU == 1 process x 2 GPUs, bitwise, on shared PCIe A100s) is in `docs/production_2026-11.md`,
section "M2 (GPU part)". These jobs are its HoreKa half: the same gate at 4 GPUs on NVLink, the
timing on a dedicated node, and the first run over two nodes.

## What you run

The environment is H0's (`bench/slurm/h0/`): its `site.env` (account, partition, `H0_ROOT`), its
venv and pinned siblings (`setup_env.sh`, once), its `provenance.py`. No staged data: the jobs draw
their Plummer spheres themselves (seeded).

```bash
git checkout m2/multiprocess-gpu         # (or the branch the lead merged it into)
bash bench/slurm/h0/setup_env.sh         # once, if H0's setup has not run in this H0_ROOT
bash bench/slurm/m2g/submit.sh --dry-run
bash bench/slurm/m2g/submit.sh           # node1 and node2; or: submit.sh node1
```

Results: `$H0_ROOT/m2g/<run id>/<job>/SUMMARY.md`, with every arm's `result_p<rank>.json` (walls per
run and per edge, peaks, the SLURM / CUDA environment each task saw) and every `compare_*.json`.
Send back `tar czf m2g.tgz --exclude='*.npy' -C $H0_ROOT/m2g/<run id> .` (a few MB).

## The jobs

| job | nodes x tasks | what | `--time` |
| --- | --- | --- | --- |
| `node1` | 1 x 4, exclusive | collectives probe (1 process x 4 GPUs, 4 x 1); `dehnen_error` and `dehnen` at 2e6 per GPU (8e6), 3 segments x 4 steps, migration, energy at every edge: `sp_a`, `mp_a`, `sp_b`, `mp_b` (A-vs-A both arms), `mpbind_a` (per-task GPU binding); `det_*`: `dehnen_error` with `--xla_gpu_deterministic_ops=true`, both arms twice. Bitwise compares; timings (repeat 1 of 2 is warm) | 3:30 |
| `node2` | 2 x 4, exclusive | collectives probe at 8 processes; both MACs at 2e6 per GPU (1.6e7), each twice (A-vs-A across nodes); the barrier probe across nodes (below) | 2:00 |

Each `srun` step is skipped when its outputs exist, so a resubmission into the same run
(`M2G_RUN=$H0_ROOT/m2g/<run id> bash bench/slurm/m2g/submit.sh node1`) resumes.

### How each process gets its GPU (read from jax 0.11.2's code)

`multiprocess.initialize()` with no arguments calls `jax.distributed.initialize()`, whose cluster
detection (`jax/_src/clusters/slurm_cluster.py`) needs `SLURM_JOB_ID`, `SLURM_STEP_NODELIST`,
`SLURM_NTASKS`, `SLURM_PROCID` and `SLURM_LOCALID`. It takes the coordinator from the first host of
`SLURM_STEP_NODELIST` (port `JAX_COORDINATOR_PORT`, else one derived from the job id: the jobs set a
new port per step), the process count from `SLURM_NTASKS`, the index from `SLURM_PROCID`, and
(`jax/_src/clusters/cluster.py`) `local_device_ids = [SLURM_LOCALID]` unless the caller or
`JAX_LOCAL_DEVICE_IDS` named the devices. `local_device_ids` becomes `jax_cuda_visible_devices`:
ordinals AFTER `CUDA_VISIBLE_DEVICES`.

- **No per-task binding** (`--ntasks-per-node=4 --gres=gpu:4`, the default arms): every task sees
  the node's four cards; jax's `[SLURM_LOCALID]` gives task `l` card `l`. Correct as is.
- **Per-task binding** (`--gpus-per-task=1 --gpu-bind=single:1`, the `mpbind` arm): each task sees ONE
  card, as device 0, and jax's `[SLURM_LOCALID]` names a device that does not exist from local rank
  1 on. `initialize()` detects this case (SLURM step, nothing named the devices, `CUDA_VISIBLE_DEVICES`
  one card: `multiprocess.slurm_bound_device_ids`) and passes `local_device_ids=[0]`.

The default jobs use the first (no binding): NCCL sees every card of the node from every process,
and the jobs do not depend on how a site configures device cgroups. The `mpbind` arm checks the
second; both are checked locally with a simulated SLURM step.

### The barrier probe (node2, last step)

jax 0.11 turned `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl` on by default; with
XLA's NCCL ragged fallback (the production flags select it) that deadlocks in one process on two
PCIe cards (jaccpot #398, yggdrax #95). The production flag set turns it off. The probe runs the
collectives with it ON across nodes under `timeout 300` (rc 124 = it hung), for M4's re-check.

## Not here

Per-rank restarts (M3: restart files stay refused under several processes), the exchange redesign
(M4), the scaling matrix (M5).
