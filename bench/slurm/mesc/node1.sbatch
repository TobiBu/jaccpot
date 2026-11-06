#!/bin/bash
#SBATCH --job-name=mesc-node1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=mesc-node1-%j.out
# MESC on ONE node: the full unclipped 1e8 Sgr IC as 4 processes x 1 GPU with escapers (r_escape 80),
# against the clipped control (escapers off, r > 80 pulled onto 80, same N). Both MACs (geometric
# theta 0.7, the Sgr production run's; and the galaxy preset's dehnen_error), 6 segments x 10 steps,
# energy at every edge; plus the escaper row with the energy every 10th edge (the edge cost without
# the per-edge energy). Step time, edge time, peak bytes per card.
export MESC_JOB=node1
# shellcheck source=_prologue.sh
source "${MESC_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/mesc/_prologue.sh"
MP=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
for mac in dehnen dehnen_error; do
  row "off_${mac}" "${MP[@]}" -- --arm mp --row off --mac "$mac"
  row "esc_${mac}" "${MP[@]}" -- --arm mp --row esc --mac "$mac"
done
row esc_dehnen_e10 "${MP[@]}" -- --arm mp --row esc --mac dehnen --energy-every 10
mesc_done
