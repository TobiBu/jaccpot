#!/bin/bash
# HoreKa job "m4_node1": M4's LET exchange and work cuts (bench/slurm/m4/node1.sbatch, its rows unchanged;
# bench/slurm/m4/README.md has what each arm measures). Multi-process steps: srun inside this allocation.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight m4
h0_track_run M4
export M4_JACCPOT=$H0_JACCPOT
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N_PER_GPU=1024 M4_SEGMENT_STEPS=2 M4_SEGMENTS=2
fi
bash bench/slurm/m4/node1.sbatch
h0_done
