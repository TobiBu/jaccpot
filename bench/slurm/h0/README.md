# H0: the HoreKa single-node baseline bundle

Track H0 of the multi-node accuracy plan (`~/.claude/plans/multinode-accuracy-dev-plan.md`, 2026-11-03).
One clone, three commands, and every timing the plan owes to HoreKa gets run on dedicated nodes with
the exact local software pins. Every later scaling claim is compared against these rows.

## What you run

On the **development machine** (once; the data is not in any repo):

```bash
cd ~/jaccpot-h0     # or any checkout of h0/horeka-bundle
bash bench/slurm/h0/stage_data.sh USER@horeka.scc.kit.edu:/path/to/H0_ROOT    # 575 MB
```

On a **HoreKa login node**:

```bash
git clone git@github.com:TobiBu/jaccpot.git && cd jaccpot && git checkout h0/horeka-bundle
ws_allocate h0 60 && ws_find h0                 # a workspace for H0_ROOT
$EDITOR bench/slurm/h0/site.env                 # H0_ACCOUNT, H0_ROOT (and H0_MODULES if no python3.12)
bash bench/slurm/h0/setup_env.sh                # once: venv, pinned siblings, harness, data check
bash bench/slurm/h0/submit_all.sh --dry-run     # look at the sbatch commands
bash bench/slurm/h0/submit_all.sh               # submits every job with dependencies
bash bench/slurm/h0/collect.sh                  # after the jobs: one tarball + SUMMARY.md
```

**Send back** `$H0_ROOT/h0-results-<run id>.tar.gz` (a few MB: every JSON row, log, provenance header
and compare verdict, without the position dumps). Its `SUMMARY.md` is the readable part.

If a job runs out of time or fails, resubmit into the same run: rows already written are skipped.

```bash
H0_RUN_ID=<run id> bash bench/slurm/h0/submit_all.sh --only scaling_b
```

## The software

| what | where it comes from |
| --- | --- |
| Python packages | `requirements.txt`: the local venv's exact pins (jax / jaxlib / CUDA-12 plugin + pjrt 0.11.2, the `nvidia-*` 12.9 wheels, numpy 2.5.3, ...), installed `--no-deps` into a plain `python -m venv` (no `--system-site-packages`: that once resolved the CUDA plugin to another environment's `.so`); `setup_env.sh` checks every version and that the plugin is the venv's |
| jaccpot | this checkout (`h0/horeka-bundle` = main 96122f63 + `bench/` and `docs/` only: library code unchanged) |
| yggdrax, nornax, mimirax | cloned at the commits in `pins.env` (yggdrax ac875003, nornax dcf7796, mimirax 23dcdd0: the mains of 2026-11-03), imported through `PYTHONPATH` (no editable installs, so an A-B arm is a path swap) |
| yggdrax before #104 | `src/yggdrax-pre104`: yggdrax main with `yggdrax/_cell_partition.py` from aaf8d6d (the one file #104 changed) |
| Odisseo's bench harness | `harness/benchmark_multigpu/{common,codes}` (vendored from Odisseo `bench/multi-gpu` be3bbd9; the ICs, the fp64 direct sum, the contention monitor the benches import) |
| the 25M disc + bulge IC, the fp64 reference cache | `stage_data.sh` from the development machine; `data.sha256` |

No CUDA module: jax's wheels bring their CUDA libraries, `ptxas` and NCCL; a loaded CUDA module can
shadow them. The node needs an NVIDIA driver new enough for CUDA 12.9 (the preflight runs a GPU op).
Every job sets the production flag `--xla_gpu_experimental_enable_fusion_autotuner=false` and no
persistent compilation cache (a warm cache changes a process's peak memory, i.e. the B/p rows).

## The jobs

Every job takes one **exclusive** node of `H0_PARTITION` (4x A100-40GB, NVLink), all of its host
memory, and its own `--time`. The `preflight` job runs first; the others start `afterok` on it (with
`H0_SERIAL=1` also one after the other). Each job writes `provenance.json` first (commits and status
of every checkout, package versions, the CUDA plugin's path, host, CPU, GPU model / driver / clocks /
NVLink topology, the environment, the IC's sha256) and a `DONE` file last.

| job | what it measures | rows | expected wall | `--time` |
| --- | --- | --- | --- | --- |
| `preflight` | the environment on a GPU node (4 GPUs, the venv's plugin, a GPU op), then one tiny row (2e5) of every entry point below with its job's exact arguments; a broken entry is recorded in `preflight/status/` and its job exits at once | 5 entries | 15-20 min | 0:45 |
| `pins` | **L1** (8e6 clipped Plummer step), **L2** (1e8 step), **L3** (25M disc + bulge step + force rel-L2) on one card (`bench/dce_pins.py`), with an **A-vs-A control** (the same code twice) and **yggdrax #104's extra cummax at 1e8**: L2 with yggdrax main (A) against the pre-#104 partition (B), interleaved A B A B A; `dce_pins compare` verdicts | 9 records | ~1.3 h | 3:00 |
| `auto` | `cell_min_level="auto"` against the galaxy preset's level 8 on mimirax's S10 bench (`seg_<N>_auto_adj` / `seg_<N>_adj`, `pre_<N>_fc1_auto` / `pre_<N>_fc1`) at 1e5, 1e6, 1e7, each pair A B B A | 24 | ~1.5 h | 3:00 |
| `s1b` | S1b's particle-major L2P, Mosaic GPU vs Triton-Pallas (`bench/slurm/s1b_mosaic_l2p.sbatch`): 2e6 and 2.5e7 disc, 1e8 Plummer, the L2P alone and a full force step, interleaved rounds | 3 sizes x 3 kernels | ~1 h | 2:00 |
| `grad` | S9's gradient timings (`bench/slurm/grad_rollout_timing.sbatch`): S8's single-force gradient at 8e6 / 2.5e7 / 1e8, the rollout gradient (step / nested / offload) against its forward at 8e6 and 2.5e7 (T = 1, 16) and 1e8 (T = 1, 4, 16) | 33 | 4-5 h | 9:00 |
| `scaling_a` | the #480 matrix (`bench/slurm/multigpu_scaling.sbatch`): **strong** scaling (25M disc + bulge and Plummer 8e6, both MACs, 1 / 2 / 4 cards, the single-card runner beside the one-device mesh), **weak** scaling (4e6 per card), **memory** per card up to the **4-card ceiling** (1e7 ... 7e7 per card; stops at the first failure) | ~26 | ~3.5 h | 8:00 |
| `scaling_b` | the #480 matrix's migration on / off tier: 2 and 4 cards, Plummer 8e6 / 1.6e7, disc 2e6 / 8e6 / 25M, both MACs, 2 interleaved repetitions | 64 | ~6.5 h | 12:00 |

**Node-hours: ~19 expected (charged for the wall actually used), at most 37.75 if every job hit its
limit.** The expected walls come from the same rows on the local shared A100s (setup and compile
dominate the scaling rows: 2-10 min each, from track T's logs; S9 estimated its matrix at 4-5 h);
HoreKa's dedicated cards should be faster, not slower. To stay inside a smaller budget, drop
`scaling_b` first (`--only preflight,pins,auto,s1b,grad,scaling_a`, ~12.5 node-hours): the strong,
weak and memory tiers are what H0's gate needs; the migration on / off tier repeats track T's question
on NVLink.

## Reading `SUMMARY.md`

1. **Provenance and job status**: one line per job (node, GPU, driver, jax, the four commits). A job
   `INCOMPLETE` ran out of time or failed: its log is `logs/h0-<job>-<id>.out`.
2. **Pins**: every L record; the A-vs-A spread (expect <= 0.3 %, as locally); **5.** the #104 cost
   at 1e8 (A mean vs B mean; it matters only beyond the A-vs-A spread, and `dce_pins`' own envelope
   is max(3 %, 2x the spread)); **7.** the sanity row: HoreKa's L pins over the local ones (different
   cards: SXM vs PCIe, so a ratio, not equality, is expected; B/p and L3's rel-L2 must agree to the
   digits shown, they are the same program on the same IC).
3. **Scaling**: strong-scaling speed-ups (vs the one-device mesh and the single-card runner), weak
   efficiency, B/p per card and GiB per card for each memory row, the first memory row that failed
   (the 4-card ceiling lies below it), migration on / off ratios. The matrix's own table (all columns:
   retries, cross near pairs, particles shipped, migrants per step) is in the collapsed block.
4. **Grad** and **S1b**: the benches' own summaries.
5. **auto**: per N and program, level 8 vs auto and their ratio (< 1: auto is faster). The question
   is whether auto regresses anywhere at 1e5-1e7 (the S10 follow-ups' condition for making it the
   galaxy preset's default).

`rows.jsonl` has one JSON object per row with its job's provenance; the lead compares those with the
local numbers.

## Checking the bundle without a node

```bash
bash bench/slurm/h0/submit_all.sh --dry-run          # the sbatch commands
SMOKE=1 bash bench/slurm/h0/submit_all.sh            # every job's commands on the CPU at N = 2048
```

`SMOKE=1` runs each job script locally, one after the other, with the same arguments at a tiny N
(`SMOKE_N` / `DCE_PINS_SMOKE_N`, the scaling matrix's `--smoke-n`; reps cut where they only repeat a
row), on forced CPU devices, through `smoke_site/sitecustomize.py` (the fused lane on the CPU as
jaccpot's lane goldens run it). Locally it ran with `H0_LOCAL_ENV=bench/slurm/h0/local_smoke.env`.
It checks arguments and plumbing, not numbers, and it is slow: every Pallas kernel runs in interpret
mode, so a forward row costs minutes and a gradient row 20-60 min of XLA:CPU compile at up to ~50 GB of
host memory (`H0_SMOKE_INTERPRET=all`, set by the grad and auto jobs). Run it sequentially (the default;
`--only <jobs>` runs a subset): run in parallel, the gradient compiles were OOM-killed and an XLA:CPU
collective timed out under the load. The 2026-11-04 run on the development machine is recorded in
track H0's report.

## Files

| file | role |
| --- | --- |
| `site.env` | every site-specific value (account, partition, workspace, modules, python) |
| `pins.env`, `requirements.txt`, `data.sha256` | the exact sibling commits, Python packages, data |
| `setup_env.sh` | builds `$H0_ROOT` once |
| `env.sh` | the run environment every job sources (`h0_paths`, `h0_provenance`) |
| `submit_all.sh` | submits the jobs (`--dry-run`, `--only`, `H0_SERIAL`, `H0_RUN_ID`, `SMOKE=1`) |
| `jobs/*.sh` | the job scripts (`_prologue.sh`: environment, provenance, the preflight gate) |
| `collect.sh`, `summarize.py` | the tarball and `SUMMARY.md` / `rows.jsonl` |
| `provenance.py` | the provenance header |
| `stage_data.sh` | copies the IC and the fp64 references from the development machine |
| `local_reference.json` | the local single-card L pins the sanity row compares with |
| `harness/` | the vendored Odisseo bench harness |
| `smoke_site/`, `local_smoke.env` | the CPU smoke |
