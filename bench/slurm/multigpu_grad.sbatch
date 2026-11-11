#!/bin/bash
# WP-G G5: reverse-mode gradients of the fused multi-GPU force on ONE dedicated node
# (bench/multigpu_grad_matrix.py -> bench/multigpu_grad_gate.py, one fresh process per row).
# Written for a HoreKa accelerated node (4x A100-40GB, NVLink); adapt the partition, the
# environment and the paths below. One process drives every card of a row: no srun per GPU.
#
#   sbatch bench/slurm/multigpu_grad.sbatch                    # the whole matrix (51 rows, ~7.5 h est.)
#   TIERS=check sbatch bench/slurm/multigpu_grad.sbatch        # a subset; a resubmission continues where
#                                                              # the last stopped (RETRY_FAILED=1: failed rows too)
#   python bench/multigpu_grad_matrix.py --out-dir $OUT --summary   # the table again
#
# On HoreKa this runs as job "mgrad" of the bundle (bench/slurm/h0/README.md: one checkout, one setup, every job).
# ROWS (a regex over the tags), SMOKE_N (every row at that N), MAX_CARDS: the bundle's preflight and smoke.
#
# Tiers (bench/multigpu_grad_matrix.py's docstring has the rows): weak (2e5 ... 1.6e7 per card on solo / 1 /
# 2 / 4 cards), strong (1.6e7 Plummer + disc, 5e7 Plummer), memory (2e7 ... 5e7 per card on 4 cards, jax.grad
# only, stops at the first failure), check (accuracy at 2.6e5 vs the single card and an fp64 direct sum).
# PRESET=galaxy (default: dehnen_error at p6) or geometric (mac dehnen p5, G and eps2 differentiated).
#
# Multi-node (NOT in scope yet): the gradient runs one process over the node's cards. Across nodes the force
# needs the multi-process mesh (jax.distributed, one process per node or per card, the NCCL cross-process
# ragged exchange validated first: WP-G design section 5, after #503); the gate would then run under srun
# with --ntasks-per-node and build its mesh from jax.devices() of every process. Not wired here.
#
#SBATCH --job-name=jaccpot-mesh-grad
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), jaccpot (this checkout), yggdrax main.
# With the H0 bundle installed (bench/slurm/h0/README.md), H0_ENV=1 takes its venv and paths instead.
JACCPOT=${JACCPOT:-$PWD}                                  # this jaccpot checkout
if [ "${H0_ENV:-0}" = 1 ]; then
  # shellcheck source=h0/env.sh
  source "$JACCPOT/bench/slurm/h0/env.sh"
else
  # source ~/envs/jaccpot-jax0112/bin/activate
  export BENCH_DIR=${BENCH_DIR:-$HOME/Odisseo/benchmark_multigpu}   # Odisseo's ICs (the Plummer sampler)
  export DISC_IC=${DISC_IC:-$HOME/ic/disk_bulge_25m.npz}           # from /export/scratch/tbuck/odisseo_ic/
  export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
  unset JAX_COMPILATION_CACHE_DIR   # a warm cache changes a process's peak memory (the B/p columns)
fi
OUT=${OUT:-${WORKSPACE:-$HOME}/mesh-grad-$(date +%Y%m%d)}
PY=${PY:-python}

# the allocation's cards: SLURM sets CUDA_VISIBLE_DEVICES; the driver takes subsets of it
echo "node $(hostname) cards ${CUDA_VISIBLE_DEVICES:-unset} out $OUT"
mkdir -p "$OUT"
nvidia-smi topo -m > "$OUT/nvidia_topo.txt" 2>&1 || true   # record the NVLink topology
cd "$JACCPOT"
git rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true

$PY bench/multigpu_grad_matrix.py \
    --out-dir "$OUT" \
    --tiers "${TIERS:-weak,strong,memory,check}" \
    --preset "${PRESET:-galaxy}" \
    --reps "${REPS:-5}" \
    --disc-ic "$DISC_IC" \
    ${MEMORY_PER_CARD:+--memory-per-card "$MEMORY_PER_CARD"} \
    ${ROW_TIMEOUT:+--timeout "$ROW_TIMEOUT"} \
    ${RETRY_FAILED:+--retry-failed} \
    ${ROWS:+--rows "$ROWS"} \
    ${SMOKE_N:+--smoke-n "$SMOKE_N"} \
    ${MAX_CARDS:+--max-cards "$MAX_CARDS"}
