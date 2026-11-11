#!/bin/bash
#SBATCH --job-name=mesc-node2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=mesc-node2-%j.out
# MESC over TWO nodes: the full unclipped 1e8 Sgr IC as 8 processes x 1 GPU with escapers, against the
# clipped control, both MACs (as node1; 6 segments x 10 steps, energy at every edge).
export MESC_JOB=${MESC_JOB:-node2}   # the HoreKa bundle's job sets mesc_node2
# shellcheck source=_prologue.sh
source "${MESC_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/mesc/_prologue.sh"
NT=$((2 * GPN))
MP=(--ntasks="$NT" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
for mac in dehnen dehnen_error; do
  row "off_${mac}" "${MP[@]}" -- --arm mp --row off --mac "$mac"
  row "esc_${mac}" "${MP[@]}" -- --arm mp --row esc --mac "$mac"
done
mesc_done
