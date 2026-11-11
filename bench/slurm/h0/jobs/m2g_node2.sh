#!/bin/bash
# HoreKa job "m2g_node2": M2G's multi-process runtime on TWO nodes (bench/slurm/m2g/node2.sbatch, its rows unchanged;
# bench/slurm/m2g/README.md has what each arm measures). Multi-process steps: srun inside this allocation.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight m2g
h0_track_run M2G
export M2G_JACCPOT=$H0_JACCPOT
if [ "${SMOKE:-}" = 1 ]; then
  export M2G_N_PER_GPU=1024
fi
bash bench/slurm/m2g/node2.sbatch
h0_done
