#!/bin/bash
#SBATCH --job-name=m1-node1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=m1-node1-%j.out
# M1 on ONE node: momentum of the multi-GPU runner, local vs cross part, 200 steps (2 x 100, dt 1e-3),
# bulge + disc IC, galaxy preset (dehnen_error) and the geometric MAC (dehnen):
#   one_<mac>     1 GPU (the yardstick: the one-card level), 1e7
#   mp4_<mac>_1e7 4 processes x 1 GPU at 2.5e6 per GPU
#   mp4_<mac>_1e8 4 processes x 1 GPU at 2.5e7 per GPU (dehnen_error only: the production MAC)
#   sbatch -A $H0_ACCOUNT -p accelerated bench/slurm/m1/node1.sbatch     (bench/slurm/m1/submit.sh does it)
export M1_JOB=node1
# shellcheck source=_prologue.sh
source "${M1_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m1/_prologue.sh"
ONE=(--ntasks=1 --nodes=1 --cpus-per-task="$CPN" --gres=gpu:1)
MP=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
for mac in dehnen_error dehnen; do
  arm "one_$mac" "${ONE[@]}" -- --arm sp --mac "$mac" --n-per-gpu 1e7 --segments 2
  arm "mp4_${mac}_1e7" "${MP[@]}" -- --arm mp --mac "$mac" --n-per-gpu 2.5e6 --segments 2
done
arm mp4_dehnen_error_1e8 "${MP[@]}" -- --arm mp --mac dehnen_error --n-per-gpu 2.5e7 --segments 2
m1_done
