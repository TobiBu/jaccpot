#!/bin/bash
# The fused multi-GPU lane's scaling matrix on ONE dedicated node (bench/multigpu_scaling_matrix.py).
# Written for a HoreKa accelerated node (4x A100-40GB, NVLink); adapt the partition, the
# environment and the paths below. One process drives every card of the node: no srun per GPU.
#
#   sbatch bench/slurm/multigpu_scaling.sbatch                 # the whole matrix (~90 rows, hours)
#   TIERS=ab,strong sbatch bench/slurm/multigpu_scaling.sbatch # a subset; a resubmission
#                                                              # continues where the last stopped
#SBATCH --job-name=jaccpot-mesh-scaling
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax >= 0.11.2 (CUDA 12), jaccpot (this checkout), yggdrax main
# source ~/envs/jaccpot-jax0112/bin/activate
JACCPOT=${JACCPOT:-$PWD}                                  # this jaccpot checkout
export BENCH_DIR=${BENCH_DIR:-$HOME/Odisseo/benchmark_multigpu}   # Odisseo's ICs + fp64 reference
DISC_IC=${DISC_IC:-$HOME/ic/disk_bulge_25m.npz}           # copied from /export/scratch/tbuck/odisseo_ic/
OUT=${OUT:-${WORKSPACE:-$HOME}/mesh-scaling-$(date +%Y%m%d)}

# the allocation's cards: SLURM sets CUDA_VISIBLE_DEVICES; the driver takes subsets of it
echo "node $(hostname) cards ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi topo -m || true                                # record the NVLink topology

cd "$JACCPOT"
python bench/multigpu_scaling_matrix.py \
    --out-dir "$OUT" \
    --tiers "${TIERS:-ab,strong,weak,memory}" \
    --reps "${REPS:-2}" \
    --disc-ic "$DISC_IC"
# the single-card timing pins on the same node, for the record's 1-card column:
# python bench/dce_pins.py record --pins L1,L2,L3 --tag horeka-L --root "$OUT/pins"
