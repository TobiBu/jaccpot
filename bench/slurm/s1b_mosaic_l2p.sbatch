#!/bin/bash
# S1b: the particle-major L2P forward, Triton-Pallas vs Mosaic GPU (bench/s1b_mosaic_l2p.py).
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment
# and the paths below. ONE card of an exclusive node (the other three idle, so nothing
# shares the card or its PCIe/NVLink); the A-vs-A control is Triton compiled twice.
#
#   sbatch bench/slurm/s1b_mosaic_l2p.sbatch                  # 2e6, 2.5e7 (disc), 1e8 (Plummer)
#   SIZES="2000000" REPS=9 sbatch bench/slurm/s1b_mosaic_l2p.sbatch
#
# Per size, one process: the L2P alone (on the prepared state's own inputs) and one full
# force step (FusedForceProvider.step: rebuild + accelerations), each compiled once per
# kernel, then timed in interleaved rounds (--reps rounds x --inner calls, median per
# round). The rows also carry the compiled temp bytes and the bitwise diff vs Triton.
#SBATCH --job-name=jaccpot-s1b-mosaic-l2p
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), jaccpot (this checkout, branch
# s1b/mosaic-l2p), yggdrax main
# source ~/envs/jaccpot-jax0112/bin/activate
JACCPOT=${JACCPOT:-$PWD}                                  # this jaccpot checkout
DISC_IC=${DISC_IC:-$HOME/ic/disk_bulge_25m.npz}           # copied from /export/scratch/tbuck/odisseo_ic/
OUT=${OUT:-${WORKSPACE:-$HOME}/s1b-mosaic-l2p-$(date +%Y%m%d)}
SIZES=${SIZES:-"2000000 25000000 100000000"}
KERNELS=${KERNELS:-pallas,mosaic,mosaic_select}
mkdir -p "$OUT"

# one card of the exclusive node: the first SLURM gave us
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i 0 --query-gpu=name,clocks.max.sm,power.limit --format=csv || true

cd "$JACCPOT"
git rev-parse HEAD
for n in $SIZES; do
  # the disc IC holds 25.2M particles; above that, the clipped Plummer sphere
  ic=disc
  [ "$n" -gt 25000000 ] && ic=plummer
  python bench/s1b_mosaic_l2p.py \
      --mode time --n "$n" --ic "$ic" --disc-ic "$DISC_IC" \
      --kernels "$KERNELS" \
      --reps "${REPS:-7}" --inner "${INNER:-5}" \
      --out "$OUT/rows.jsonl" 2>&1 | tee "$OUT/n${n}.log"
done
echo "rows: $OUT/rows.jsonl"
