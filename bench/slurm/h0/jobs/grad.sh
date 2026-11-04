#!/bin/bash
# H0 job "grad" (~4-5 h): S9's gradient timings (bench/slurm/grad_rollout_timing.sbatch, unchanged rows):
# S8's single-force gradient and the rollout gradient against its forward at 8e6, 2.5e7 and 1e8, one card.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight grad
if [ "${SMOKE:-}" = 1 ]; then
  # one row per distinct argument pattern (the rows left out differ from one of these in N alone):
  # CPU interpret mode costs ~20 min per gradient row
  export SMOKE_N=2048 H0_SMOKE_INTERPRET=all
  export ROWS=${ROWS:-'^(s8_8e6_galaxy|s8_8e6|s8_1e8|r_8e6_T1_fwd|r_8e6_T1_dfwd|r_8e6_T1_grad_step|r_8e6_T16_grad_nested|r_8e6_T16_grad_offload|r_1e8_T4_grad_offload|r_1e8_T16_grad_offload)$'}
fi
PY=$H0_PY OUT=$OUT/timing bash bench/slurm/grad_rollout_timing.sbatch
h0_done
