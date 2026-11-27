#!/bin/bash
#SBATCH --job-name=m4-node2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=m4-node2-%j.out
# M4 across TWO nodes: 8 processes x 1 GPU (NVLink inside a node, InfiniBand between), the same arms as node1
# at the same per-GPU sizes (weak scaling 4 -> 8 GPUs: node1's rows are the yardstick):
#   d1e7_<exchange>_<cuts>   1.25e6 per GPU (1e7)
#   d2e8_all_gather_count / d2e8_let_work   2.5e7 per GPU (2e8)
export M4_JOB=${M4_JOB:-node2}   # the HoreKa bundle's job sets m4_node2
# shellcheck source=_prologue.sh
source "${M4_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m4/_prologue.sh"
NT=$((2 * GPN))
MP=(--ntasks="$NT" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
SEG=(--segments "${M4_SEGMENTS:-3}")
for ex in all_gather let; do
  for cuts in count work; do
    arm "d1e7_${ex}_${cuts}" "${MP[@]}" -- --arm mp --mac dehnen_error --exchange "$ex" --cuts "$cuts" \
      --n-per-gpu "$(nn 1.25e6)" "${SEG[@]}"
  done
done
arm d2e8_all_gather_count "${MP[@]}" -- --arm mp --mac dehnen_error --exchange all_gather --cuts count \
  --n-per-gpu "$(nn 2.5e7)" "${SEG[@]}"
arm d2e8_let_work "${MP[@]}" -- --arm mp --mac dehnen_error --exchange let --cuts work \
  --n-per-gpu "$(nn 2.5e7)" "${SEG[@]}"
m4_done
