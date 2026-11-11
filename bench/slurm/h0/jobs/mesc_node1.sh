#!/bin/bash
# HoreKa job "mesc_node1": MESC's escapers on the mesh on ONE node (bench/slurm/mesc/node1.sbatch, its rows unchanged;
# bench/slurm/mesc/README.md has what each arm measures). Multi-process steps: srun inside this allocation.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight mesc
h0_need_data "$SGR_IC"
h0_track_run MESC
export MESC_JACCPOT=$H0_JACCPOT
if [ "${SMOKE:-}" = 1 ]; then
  MESC_IC=$OUT/sgr_tiny.npz
  [ -s "$MESC_IC" ] || "$H0_PY" "$H0_DIR/tiny_ic.py" "$SGR_IC" "$MESC_IC" 4096
  export MESC_IC MESC_EXTRA="--segments 2 --steps 2"
fi
bash bench/slurm/mesc/node1.sbatch
h0_done
