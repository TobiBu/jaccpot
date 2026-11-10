#!/bin/bash
# HoreKa job "m1_node1": M1's momentum local vs cross on ONE node (bench/slurm/m1/node1.sbatch, its rows unchanged;
# bench/slurm/m1/README.md has what each arm measures). Multi-process steps: srun inside this allocation.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight m1
h0_track_run M1
export M1_JACCPOT=$H0_JACCPOT
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N_PER_GPU=1024 M1_SEGMENT_STEPS=2
fi
bash bench/slurm/m1/node1.sbatch
h0_done
