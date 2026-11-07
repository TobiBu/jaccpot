#!/bin/bash
#SBATCH --job-name=m1-node2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=m1-node2-%j.out
# M1 across TWO nodes: 8 processes x 1 GPU, momentum local vs cross part over 1000 steps (as node1),
# 1.25e6 per GPU (1e7, both MACs) and 1.25e7 per GPU (1e8, dehnen_error). node1's one_<mac> rows are
# the yardstick.
export M1_JOB=node2
# shellcheck source=_prologue.sh
source "${M1_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m1/_prologue.sh"
NT=$((2 * GPN))
MP=(--ntasks="$NT" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
for mac in dehnen_error dehnen; do
  arm "mp8_${mac}_1e7" "${MP[@]}" -- --arm mp --mac "$mac" --n-per-gpu 1.25e6
done
arm mp8_dehnen_error_1e8 "${MP[@]}" -- --arm mp --mac dehnen_error --n-per-gpu 1.25e7
m1_done
