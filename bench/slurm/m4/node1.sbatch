#!/bin/bash
#SBATCH --job-name=m4-node1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=m4-node1-%j.out
# M4 on ONE node: 4 processes x 1 GPU, the LET exchange vs all_gather and work cuts vs count cuts
# (bench/m4_exchange_cuts.py; README.md has what each arm measures). Bulge + disc IC, galaxy preset
# (dehnen_error), 3 segments x 20 steps, a rebalance before every segment after the first:
#   d1e7_<exchange>_<cuts>   2.5e6 per GPU, the four combinations
#   d1e8_all_gather_count / d1e8_let_work   2.5e7 per GPU (production per card), the default vs both switches
export M4_JOB=${M4_JOB:-node1}   # the HoreKa bundle's job sets m4_node1
# shellcheck source=_prologue.sh
source "${M4_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m4/_prologue.sh"
MP=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
SEG=(--segments "${M4_SEGMENTS:-3}")
for ex in all_gather let; do
  for cuts in count work; do
    arm "d1e7_${ex}_${cuts}" "${MP[@]}" -- --arm mp --mac dehnen_error --exchange "$ex" --cuts "$cuts" \
      --n-per-gpu "$(nn 2.5e6)" "${SEG[@]}"
  done
done
arm d1e8_all_gather_count "${MP[@]}" -- --arm mp --mac dehnen_error --exchange all_gather --cuts count \
  --n-per-gpu "$(nn 2.5e7)" "${SEG[@]}"
arm d1e8_let_work "${MP[@]}" -- --arm mp --mac dehnen_error --exchange let --cuts work \
  --n-per-gpu "$(nn 2.5e7)" "${SEG[@]}"
m4_done
