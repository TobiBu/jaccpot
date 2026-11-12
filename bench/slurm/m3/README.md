# M3 on HoreKa: sharded checkpoint I/O, elastic restores, walltime chaining

Track M3 of the multi-node accuracy plan (`docs/production.md`, "Sharded checkpoints"; the record is
`docs/production_2026-11.md`, `## M3`). Two jobs of the HoreKa bundle run it (`bench/slurm/h0/submit_all.sh
--only m3_node1,m3_node4`; both in the default set), plus a self-resubmitting chain (`chain.sbatch`) that the
bundle does not submit.

The venv needs orbax: `bench/slurm/h0/requirements.txt` pins orbax-checkpoint 0.12.9 and its closure (the
versions the local functional runs imported), so `setup_env.sh` installs it with the rest. The preflight's
`m3` entry writes a tiny checkpoint from the node's 4 processes and restores it on 1 GPU.

## What each step measures (`bench/m3_checkpoint_io.py`; one `result_<cmd>_p<rank>.json` per process)

| job / step | processes x GPUs | what |
| --- | --- | --- |
| `m3_node1/write_mp` | 4 x 1 | a 1e8 Plummer (each process draws only its ids), one 2-step segment, then 2 writes of a sharded checkpoint: **write seconds per rank, the bytes of its own shards, its OCDBT database's bytes on disk**; the 2nd write replaces the 1st (the atomic swap is in its time) |
| `m3_node1/restore_mp` | 4 x 1 | the same count: each process reads its own shards (sha256 checked); 2 restores (the 1st includes the restore's compiles) and one continued segment |
| `m3_node1/restore_sp` | 1 x 4 | the same device count from one process (bitwise to `restore_mp`: same layout) |
| `m3_node1/restore_half` | 2 x 1 | elastic 4 -> 2: the re-sharding restore, host compaction, the repartition program |
| `m3_node1/restore_one` | 1 x 1 | elastic 4 -> 1 GPU (1e8 on one 40 GB card in the mesh mode: if it does not fit, the row records the error) |
| `m3_node1/chain_a`, `chain_b` | 4 x 1 | the chaining mechanics: `chain_a`'s deadline is its start, so it stops at its first edge with a checkpoint (`STATUS=continue`); `chain_b` continues from the newest committed checkpoint to the target (`STATUS=done`) |
| `m3_node4/write_mp` | 16 x 1 (4 nodes) | **1e9**: 6.25e7 per card; write seconds and bytes per rank as above |
| `m3_node4/restore_mp` | 16 x 1 | the same count |
| `m3_node4/restore_3nodes` | 12 x 1 | elastic 16 -> 12 (8.3e7 per card) |
| `m3_node4/restore_2nodes` | 8 x 1 | elastic 16 -> 8 (1.25e8 per card: past what one 40 GB card holds in the mesh mode; expected to fail with a device OOM, recorded either way) |

Every row also has the peak host RSS per process (`resource.getrusage`) and the devices' peak bytes.
`SUMMARY.md` in each job directory is the table (`bench/slurm/m3/summarize.py`). What fits at 1e9: 16 GPUs
(4 nodes) is the default because the mesh mode at 1.25e8 per 40 GB card (2 nodes x 4) is above the per-card
ceiling measured so far (MESC: 460 B/p per card at 5e7, i.e. ~8.7e7 per 40 GB card); 12 cards (8.3e7 per
card) is the smallest elastic target in the job expected to fit.

**Host memory.** The write and the restore are `O(N/P)` per process (one shard at a time; the elastic
compaction of a process's own blocks). The run's own end is not: a `RunResult` gathers the final state by id
on EVERY process (M2's `host_value`), `O(N)` each. The bench replaces that gather with a stub above 2e8
(`--lazy-result-above`), so the 1e9 rows measure the I/O, not that gather. An open item for M5.

## Walltime chaining (`chain.sbatch`, for M5)

```bash
M3_CHAIN_DIR=$WS/chain M3_N=1e8 M3_TARGET_STEPS=2000 M3_EVERY=5 \
  sbatch -A $H0_ACCOUNT -p accelerated bench/slurm/m3/chain.sbatch
```

Each job first submits its successor (`afterany` on itself: it runs however this job ends -- finished, out of
time, a lost node, a crash), then continues from `latest_checkpoint("$M3_CHAIN_DIR/ck_{step:08d}")` (the newest
COMMITTED one; a write killed mid-way is never chosen) and checkpoints every `M3_EVERY` segments; when its own
end (`squeue %e`) is `M3_MARGIN` seconds (900) plus two segments away, every process agrees to stop, writes one
more checkpoint and exits. At the target it writes `DONE` and cancels its successor; a successor that finds no
new checkpoint since its predecessor started stops the chain instead of looping. `M3_CHAIN_MAX` (20) bounds it.

## Local smoke (CPU, no SLURM)

```bash
SMOKE=1 H0_LOCAL_ENV=<an env file: H0_ROOT, H0_PY_OVERRIDE, H0_YGGDRAX/NORNAX/MIMIRAX, H0_EXTRA_PYTHONPATH with
  orbax, H0_PREFLIGHT_ONLY=m3> bash bench/slurm/h0/submit_all.sh --only m3_node1,m3_node4
```

2026-11-11 (jaccpot a2d2cb98, orbax 0.12.9): preflight `m3` ok; `m3_node1` 7/7 steps ok (1168 s), `m3_node4`
4/4 ok (740 s, 4 local processes standing in for 4 nodes x 1), `chain_a` `continue` -> `chain_b` `done`.
