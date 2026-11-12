# The HoreKa bundle: one checkout, one setup, every timing job

This directory is THE HoreKa guide. One checkout of jaccpot `main`, one `setup_env.sh`, one
`submit_all.sh` runs every timing job the multi-node accuracy plan
(`~/.claude/plans/multinode-accuracy-dev-plan.md`) and the production plan owe to HoreKa, on dedicated
nodes with the exact local software pins. Track H0 built it (2026-11-04, single-node baseline); track H1
(2026-11-10) moved it to the mains of 2026-11-10 and folded in every other HoreKa job (the per-track
directories `bench/slurm/{m1,m2g,mesc,a3,a4}/` and the stand-alone `bench/slurm/*.sbatch` now run as jobs of
this bundle). Timings come only from HoreKa; the local cards are for functional and accuracy work.

## What you run

On the **development machine** (once; the data is not in any repo):

```bash
cd ~/jaccpot            # any checkout of main
bash bench/slurm/h0/stage_data.sh USER@horeka.scc.kit.edu:/path/to/H0_ROOT            # 12.4 GB, everything
# or: bash bench/slurm/h0/stage_data.sh --no-big USER@...:/path/to/H0_ROOT            # 575 MB; jobs escaper,
#     mesc_node1/2 and a1 then skip (they need the 1e8 Sgr IC / the two 1e8 restarts under data/big/)
```

On a **HoreKa login node**:

```bash
git clone git@github.com:TobiBu/jaccpot.git && cd jaccpot     # main
ws_allocate h0 60 && ws_find h0                 # a workspace for H0_ROOT (~30 GB with the big data)
$EDITOR bench/slurm/h0/site.env                 # H0_ACCOUNT, H0_ROOT (and H0_MODULES if no python3.12),
                                                # or export H0_ACCOUNT / H0_ROOT in the shell instead
bash bench/slurm/h0/setup_env.sh                # once: venv, pinned siblings, harness, data check (~5 min)
bash bench/slurm/h0/submit_all.sh --list        # the job table
bash bench/slurm/h0/submit_all.sh --dry-run     # look at the sbatch commands and the worst-case node-hours
bash bench/slurm/h0/submit_all.sh               # the default set: preflight first, the rest afterok on it
bash bench/slurm/h0/collect.sh                  # after the jobs: one tarball + SUMMARY.md
```

A subset: `submit_all.sh --only preflight,pins,scaling_a` (the preflight is always added). The A4 accuracy suite
is not in the default set (it needs `A4_AGAMA_PY`, a python with agama, in `bench/slurm/a4/site.env`, and its
own budget, below): `submit_all.sh --only a4` or `--all`. Options for it go through `A4_ARGS`
(e.g. `A4_ARGS="--tier 1card"`). If a job runs out of time or fails, resubmit into the same run: rows already
written are skipped.

```bash
H0_RUN_ID=<run id> bash bench/slurm/h0/submit_all.sh --only scaling_b
```

**Send back** `$H0_ROOT/h0-results-<run id>.tar.gz` from `collect.sh`: every job's JSON rows, logs, provenance
headers, compare verdicts and summaries, a4's included, without position dumps, `.npy` / `.npz` arrays or
profiler traces (`H0_COLLECT_TRACES=1` adds those). A few MB. Its `SUMMARY.md` is the readable part.

## The software

| what | where it comes from |
| --- | --- |
| Python packages | `requirements.txt`: the local venv's exact pins (jax / jaxlib / CUDA-12 plugin + pjrt 0.11.2, the `nvidia-*` 12.9 wheels, numpy 2.5.3, ...), installed `--no-deps` into a plain `python -m venv` (no `--system-site-packages`: that once resolved the CUDA plugin to another environment's `.so`); `setup_env.sh` checks every version and that the plugin is the venv's. Unchanged by H1 (every sibling main's `pyproject.toml` is satisfied) |
| jaccpot | this checkout (main) |
| yggdrax, nornax, mimirax | cloned at the commits in `pins.env` (the mains of 2026-11-10: yggdrax f722c24, nornax 30be376, mimirax 1101a6e), imported through `PYTHONPATH` (no editable installs, so an A-B arm is a path swap) |
| yggdrax before #104 | `src/yggdrax-pre104`: yggdrax main with `yggdrax/_cell_partition.py` from aaf8d6d (the only change to that file since is #104: the arm isolates its extra cummax) |
| Odisseo | cloned at `pins.env`'s `jaccpot-integration` commit a0e8848, for job a4's IC tool only (`tools/a4_galaxy_ics.py`); nothing imports it |
| Odisseo's bench harness | `harness/benchmark_multigpu/{common,codes}` (vendored; the ICs, the fp64 direct sum, the contention monitor the benches import) |
| the data | `stage_data.sh` from the development machine, checked against `data.sha256` (below) |

No CUDA module: jax's wheels bring their CUDA libraries, `ptxas` and NCCL; a loaded CUDA module can shadow them.
The node needs an NVIDIA driver new enough for CUDA 12.9 (the preflight runs a GPU op). Every job sets the
production flag `--xla_gpu_experimental_enable_fusion_autotuner=false` and no persistent compilation cache (a
warm cache changes a process's peak memory, i.e. the B/p rows). After a `git pull` that changed `pins.env`,
rerun `setup_env.sh` (it moves the clones).

**The data** (`stage_data.sh`, one step for every job):

| file under `$H0_ROOT/data/` | size | read by |
| --- | --- | --- |
| `disk_bulge_25m.npz` (the 25M disc + bulge IC) | 573 MB | pins (L3), scaling_*, s1b, grad, mgrad, a1 (err rows), preflight |
| `reference/*.npy` (fp64 direct sums of the harness) | 2 MB | pins (L3), the matrices' accuracy columns |
| `big/sgr_light_1e8_noclip.npz` (the 1e8 light-Sgr merger, UNCLIPPED, A2) | 2.8 GB | escaper, mesc_node1/2 |
| `big/bar_restart_3.npz` (the 1e8 bar disc at step 30: the #487 stall) | 4.4 GB | a1 |
| `big/merger_restart_136.npz` (the 1.06e8 merger at t 1.36) | 4.7 GB | a1 |

Jobs m1_*, m2g_*, auto and forward draw their ICs themselves (seeded); a4 generates its galaxy ICs (agama, its
IC job on the CPU partition) into `$H0_ROOT/a4/ics` once.

## The jobs

Every job takes **exclusive** nodes of `H0_PARTITION` (4x A100-40GB, NVLink), all of their host memory and its
own `--time`. The `preflight` job runs first; the others start `afterok` on it (`H0_SERIAL=1`: also one after
the other). Each job writes `provenance.json` first (commits and status of every checkout, package versions,
the CUDA plugin's path, host, CPU, GPU model / driver / clocks / NVLink topology, the environment) and a `DONE`
file last; a job whose staged data is missing writes `SKIPPED` and ends at once. Jobs with several tasks per
node start their multi-process steps with `srun` inside the allocation (one process per GPU, every task sees
the node's four cards and jax gives task `l` card `l`: M2G's launch pattern); their single-process steps run in
the batch step.

| job | nodes x tasks | what it measures | rows | expected wall | `--time` |
| --- | --- | --- | --- | --- | --- |
| `preflight` | 1 x 4 | the environment (4 GPUs, the venv's plugin, a GPU op, srun), then one tiny row (2e5) of every entry point below (16 entries, one per job family; the multi-process ones as srun steps); a broken entry is recorded in `preflight/status/` and its job exits at once | 16 entries | 45-60 min | 2:00 |
| `pins` | 1 x 1 | **L1** (8e6 clipped Plummer step), **L2** (1e8 step), **L3** (25M disc + bulge step + force rel-L2) on one card (`bench/dce_pins.py`), an **A-vs-A control**, and yggdrax **#104's extra cummax at 1e8** (L2 on main (A) vs the pre-#104 partition (B), A B A B A); `dce_pins compare` verdicts | 9 records | ~1.3 h | 3:00 |
| `auto` | 1 x 1 | `cell_min_level="auto"` vs the galaxy preset's level 8 on mimirax's S10 bench (`seg_<N>_auto_adj` / `seg_<N>_adj`, `pre_<N>_fc1_auto` / `pre_<N>_fc1`) at 1e5, 1e6, 1e7, each pair A B B A | 24 | ~1.5 h | 3:00 |
| `s1b` | 1 x 1 | S1b's particle-major L2P, Mosaic GPU vs Triton-Pallas (`s1b_mosaic_l2p.sbatch`): 2e6 and 2.5e7 disc, 1e8 Plummer | 3 sizes x 3 kernels | ~1 h | 2:00 |
| `grad` | 1 x 1 | S9's gradient timings (`grad_rollout_timing.sbatch`): S8's force gradient at 8e6 / 2.5e7 / 1e8, the rollout gradient (step / nested / offload) vs its forward at 8e6, 2.5e7 (T = 1, 16) and 1e8 (T = 1, 4, 16) | 33 | 4-5 h | 9:00 |
| `scaling_a` | 1 x 1 | the #480 matrix (`multigpu_scaling.sbatch`): **strong** (25M disc + bulge, Plummer 8e6, both MACs, 1 / 2 / 4 cards, the single-card lane beside the one-device mesh), **weak** (4e6 per card), **memory** per card up to the **4-card ceiling** (1e7 ... 7e7 per card; stops at the first failure) | ~26 | ~3.5 h | 8:00 |
| `scaling_mp` | 1 x 4 | the matrix's **mp tier** (new, H1): the production multi-GPU path as **N processes x 1 GPU** (`jaccpot.distributed.multiprocess`) next to **1 process x N GPUs**, same bench (`multiprocess_gate.py run`, the production runner, migration on), 1 / 2 / 4 cards: Plummer 8e6 and the 25M disc + bulge (both MACs), Plummer 4e6 per card; interleaved sp / mp, 2 repetitions; the mp / sp step ratio per row | 50 | ~3 h | 6:00 |
| `scaling_b` | 1 x 1 | the #480 matrix's migration on / off tier: 2 and 4 cards, Plummer 8e6 / 1.6e7, disc 2e6 / 8e6 / 25M, both MACs, 2 interleaved repetitions | 64 | ~6.5 h | 12:00 |
| `forward` | 1 x 1 | forward mode round 2 (`forward_mode_timing.sbatch`): jvp, jacfwd K = 1/2/4/8, directions per pass, tangent groups g = 1/2/4, value_and_grad, the per-seam tangents-per-program sweep at 2.5e7; forward / jvp / jacfwd at 1e8; the satellite Jacobian demonstrator (T = 200 at 2.5e7 and 1e8). Decides the default g / tpp | 21 | 8-10 h | 16:00 |
| `mgrad` | 1 x 1 | WP-G's multi-GPU reverse gradients (`multigpu_grad.sbatch`): weak (2e5-1.6e7 per card on solo / 1 / 2 / 4), strong, memory (to the 4-card ceiling), check (accuracy vs the single card and fp64) | 51 | ~7.5 h | 12:00 |
| `escaper` | 1 x 1 | ESC's escaper cost at 1e8 on ONE card (`escaper_timing.sbatch`): the unclipped Sgr IC with escapers on / off / on-but-none-escape / energy every 10th edge, both MACs; preallocated BFC (ESC: 1e8 on one card needs it) | 8 | 1.5-2 h | 4:00 |
| `a1` | 1 x 1 | A1's dehnen_error stall (`a1_dehnen_error_stall.sbatch`): the 1e8 bar disc from its step-30 restart (10 steps) and the 1.06e8 merger (5 steps) with the fix (today's default), the force-scale floor and the geometric MAC at theta 0.7; the err rows (8e6 / 2.5e7 / 1e8, floor off / on) | 12 | ~2-2.5 h | 6:00 |
| `a3` | 1 x 1 | A3's summed drift (the default) vs the legacy order (`a3/position_precision_timing.sbatch`): the runner at 1e8 and 2.6e5, both MACs, A-B-A-B; A3's gate 3 (summed within +10 % at 1e8) | 16 | ~2 h | 4:00 |
| `m1_node1` | 1 x 4 | M1: momentum of the multi-GPU runner, local vs cross part (`m1/node1.sbatch`): 1 GPU at 1e7 (the yardstick), 4 x 1 GPU at 1e7 (both MACs) and 1e8 | 5 | ~1 h | 2:00 |
| `m1_node2` | 2 x 4 | M1 over two nodes: 8 x 1 GPU at 1e7 (both MACs) and 1e8 | 3 | ~45 min | 1:30 |
| `m2g_node1` | 1 x 4 | M2G (`m2g/node1.sbatch`): 4 processes x 1 GPU vs 1 x 4 at 8e6, both MACs, A-vs-A, per-task binding, deterministic arms; collectives probe; bitwise compares + timing | 14 steps | ~2 h | 3:30 |
| `m2g_node2` | 2 x 4 | M2G over two nodes: 8 x 1 GPU (both MACs, A-vs-A), collectives at 8, the #398 barrier probe across nodes | 6 steps | ~1 h | 2:00 |
| `mesc_node1` | 1 x 4 | MESC (`mesc/node1.sbatch`): escapers on the mesh, the full unclipped 1e8 Sgr IC as 4 x 1 GPU vs the clipped control, both MACs + energy every 10th edge | 5 | 1.5-2 h | 3:00 |
| `mesc_node2` | 2 x 4 | MESC over two nodes: 8 x 1 GPU, both MACs, on / off | 4 | ~1.25 h | 2:30 |
| `m3_node1` | 1 x 4 | M3's sharded checkpoints (`bench/m3_checkpoint_io.py`, `bench/slurm/m3/README.md`): a 1e8 Plummer written by 4 x 1 GPU twice (the second replaces the first); restored on the same 4 processes, on 1 process x 4 GPUs (same count: bitwise), elastically on 2 x 1 GPU and on 1 GPU, each with one continued segment; the walltime-chaining mechanics (a run stopped by its deadline, continued from its last checkpoint). Per rank: write / restore seconds, shard and disk bytes, peak host RSS, device peak | 7 steps | ~1 h | 2:00 |
| `m3_node4` | 4 x 4 | M3 at **1e9**: written by 16 x 1 GPU (6.25e7 per card), restored on the same 16 and elastically on 12 (3 nodes) and 8 (2 nodes, 1.25e8 per card: past the mesh's per-card ceiling on 40 GB, recorded either way) | 4 steps | ~1 h | 2:00 |
| `a4` | arrays | the A4 accuracy suite (delegated to `bench/slurm/a4/submit.sh`, its README has the ladders): an IC job (CPU partition, agama), one job array per ladder and tier (1-card tier: one A100 per rung; node tier: one exclusive node per rung), the summary; afterok the preflight | 53 + 21 rungs | see below | per ladder |

The per-track READMEs (`bench/slurm/{m1,m2g,mesc,a4,m3}/README.md`) say what each arm measures and what to look
for; their scripts are what these jobs run.

## The node-hour budget

**The default set: ~60 node-hours expected, at most ~117 if every job hit its limit** (`submit_all.sh
--dry-run` prints the worst case for what it would submit). The expected walls come from the same rows on the
local shared A100s and from each job's own estimate (compile and setup dominate the multi-GPU rows: 2-10 min
each); HoreKa's dedicated cards should be faster, not slower. Per job, expected (worst): forward 9 (16), mgrad
7.5 (12), scaling_b 6.5 (12), grad 4.5 (9), scaling_a 3.5 (8), scaling_mp 3 (6), a1 2.5 (6), a3 2 (4), m2g 4
(7.5), mesc 4.25 (8), m1 2.5 (5), escaper 1.75 (4), auto 1.5 (3), pins 1.3 (3), s1b 1 (2), preflight 1 (2), m3 5 (10: m3_node4 is four nodes).

**A4 on top** (`--only a4`): the 1-card tier ~60-80 GPU-hours = 15-20 node-hours (charged per GPU), the node tier
~25-40 node-hours expected, ~200 at worst (its README: run `A4_ARGS="--tier 1card"` first; in the node tier
`disc_n`, `merger_n`, `bar_n` first).

**What to drop first** (each line keeps what the next decisions need):

1. `scaling_b` (6.5): the migration on / off tier repeats track T's question on NVLink;
2. `s1b` (1): the Mosaic question is settled (keep Triton; S1b);
3. `grad` (4.5): S9's timings, informative, no pending decision;
4. `mgrad`'s memory tier (`TIERS=weak,strong,check`; the job has no switch: resubmit `--only mgrad` with
   `TIERS` exported) and `forward`'s 1e8 rows.

A core set for a small budget (~25 node-hours): `--only scaling_a,scaling_mp,m2g_node1,m2g_node2,a3,forward,auto,pins`
(the H0 gate, the production multi-GPU path, the multi-node bring-up, A3's gate 3, the two pending default
decisions).

## Reading `SUMMARY.md`

1. **Provenance and job status**: one line per job (status done / skipped / INCOMPLETE / not run, node, GPU,
   driver, jax, the four commits). An `INCOMPLETE` job ran out of time or failed: its log is
   `logs/h0-<job>-<id>.out`. Then the preflight's entry verdicts.
2. **Pins**: every L record; the A-vs-A spread (expect <= 0.3 %, as locally); the #104 cost at 1e8 (A mean vs B
   mean; it matters only beyond the A-vs-A spread; `dce_pins`' own envelope is max(3 %, 2x the spread)); **7.**
   the sanity row: HoreKa's L pins over the local ones (`local_reference.json`, refreshed by H1 at these pins).
   B/p and L3's rel-L2 must agree to the digits shown (the same program on the same IC); the walls are a
   ratio (different cards: SXM vs PCIe, and the local cards are shared).
3. **Scaling**: strong-scaling speed-ups, weak efficiency, B/p and GiB per card per memory row, the first memory
   row that failed (the 4-card ceiling lies below it), migration on / off ratios.
4. **Grad** and **S1b**: the benches' own summaries. **auto**: per N and program, level 8 vs auto and their
   ratio (< 1: auto is faster); whether auto regresses anywhere at 1e5-1e7 decides it as the galaxy preset's
   default (a pending user decision).
5. **8. The jobs folded in by H1**: scaling_mp's table (sp and mp rows, the mp / sp step ratio), forward's and
   mgrad's own summaries, the escaper / a1 / a3 rows, the M1 / M2G / MESC jobs' `SUMMARY.md`, a4's
   `SUMMARY-horeka-*.md`.

`rows.jsonl` has one JSON object per row with its job's provenance; the lead compares those with the local
numbers.

## Checking the bundle without a node

```bash
bash bench/slurm/h0/submit_all.sh --dry-run          # the sbatch commands
SMOKE=1 bash bench/slurm/h0/submit_all.sh            # every job's commands on the CPU at N = 2048
```

`SMOKE=1` runs each job script locally, one after the other, with the same arguments at a tiny N (`SMOKE_N` /
`DCE_PINS_SMOKE_N` / `SMOKE_N_PER_GPU`, the matrices' `--smoke-n`, a tiny random subsample of the Sgr IC
(`tiny_ic.py`) and a tiny runner restart for A1; reps cut where they only repeat a row, one row per distinct
argument pattern where a row costs a gradient compile), on forced CPU devices, through
`smoke_site/sitecustomize.py` (the fused lane on the CPU as jaccpot's lane goldens run it). The multi-process
jobs run their steps at two processes on gloo: `smoke_bin/srun` (`local_srun.py`) starts a step's tasks as
local processes with a SLURM step's variables, so jax's own SLURM detection joins them (one-node jobs with 2
"GPUs" per node, two-node jobs with 1). Locally it runs with `H0_LOCAL_ENV=bench/slurm/h0/local_smoke.env`.

It checks arguments and plumbing, not numbers, and it is slow: every Pallas kernel runs in interpret mode, so a
forward row costs minutes and a gradient row 20-60 min of XLA:CPU compile at up to ~50 GB of host memory
(`H0_SMOKE_INTERPRET=all`, set by the gradient jobs). The rollout-gradient `step` / `nested` / 1e8 `offload`
rows and forward's `vgrad` do not finish on XLA:CPU (50-156 GB compiles, track H0): the GPU preflight runs them at
2e5 before any node-hours. Run it sequentially (the default; `--only <jobs>` runs a subset): in parallel, the
gradient compiles were OOM-killed and an XLA:CPU collective timed out under the load.

The local GPU preflight (functional only: the local cards are shared, no timing claims) runs `jobs/preflight.sh`
through `on-free-card` with `H0_LOCAL_SRUN=1` (the same local `srun`), the one-card entries on one card and the
multi-process entries on two (`H0_GPUS_PER_NODE=2`, `H0_PREFLIGHT_ONLY=...`). Tracks H0 and H1 record their runs
in their reports and in `docs/production_2026-11.md`.

## Files

| file | role |
| --- | --- |
| `site.env` | every site-specific value (account, partition, workspace, modules, python) |
| `pins.env`, `requirements.txt`, `data.sha256` | the exact sibling commits, Python packages, data |
| `setup_env.sh` | builds `$H0_ROOT` once (rerun after `pins.env` changed) |
| `stage_data.sh` | copies the data from the development machine (`--no-big`: the small part) |
| `env.sh` | the run environment every job sources (`h0_paths`, `h0_provenance`, the data paths) |
| `submit_all.sh` | submits the jobs (`--list`, `--dry-run`, `--only`, `--all`, `H0_SERIAL`, `H0_RUN_ID`, `SMOKE=1`) |
| `jobs/*.sh` | the job scripts (`_prologue.sh`: environment, provenance, the preflight gate, the data check) |
| `collect.sh`, `summarize.py` | the tarball and `SUMMARY.md` / `rows.jsonl` |
| `provenance.py` | the provenance header |
| `local_reference.json` | the local single-card L pins the sanity row compares with |
| `harness/` | the vendored Odisseo bench harness |
| `local_srun.py`, `smoke_bin/srun` | a SLURM step as local processes (the CPU smoke, the local GPU preflight) |
| `tiny_ic.py` | a tiny random subsample of the Sgr IC (smoke, preflight) |
| `smoke_site/`, `local_smoke.env` | the CPU smoke |

The old entry points: `bench/slurm/{m1,m2g,mesc}/submit.sh` forward to `submit_all.sh --only <track>_node1,<track>_node2`
(the preflight runs first); `bench/slurm/a4/submit.sh` still works on its own and is what job a4 calls; the
stand-alone `bench/slurm/*.sbatch` and `bench/slurm/a3/*.sbatch` still run as they did (each says which bundle job
runs it).
