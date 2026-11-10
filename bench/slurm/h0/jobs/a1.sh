#!/bin/bash
# HoreKa job "a1" (~2-3 h): A1's dehnen_error stall (#487) timed on ONE card (bench/slurm/a1_dehnen_error_stall.sbatch):
# the 1e8 bar disc from its step-30 restart (10 steps: the stall's segment) and the 1.06e8 merger from t 1.36
# (5 steps), each with the fix (net: today's default), the force-scale floor and the geometric MAC at theta 0.7;
# the err rows (fused_memory_budget at 8e6 / 2.5e7 / 1e8, floor off and on). A1 is merged: the pre-A1 base arm and
# the L pins (job pins) are not run. Restore rows on cuda_async (A1: BFC could not place the 14.9 GiB block).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight a1
export ROWS=${ROWS:-'^((stall|merger)_(net|floor|geo07)|err_.*)$'}
if [ "${SMOKE:-}" = 1 ]; then
  # a tiny restart of the production runner (dehnen_error, the galaxy settings), one row per pattern
  R=$OUT/tiny_restart.npz
  h0_tiny_restart "$R" 4096
  export STALL_RESTART=$R MERGER_RESTART=$R STALL_STEPS=2 MERGER_STEPS=1 SMOKE_N=2048
  export ROWS='^(stall_net|stall_floor|merger_geo07|err_8e6_net|err_2p5e7_floor)$'
else
  h0_need_data "$A1_BAR_RESTART" "$A1_MERGER_RESTART"
  export STALL_RESTART=$A1_BAR_RESTART MERGER_RESTART=$A1_MERGER_RESTART
fi
PY_BASE=$H0_PY PY_A1=$H0_PY JACCPOT_A1=$H0_JACCPOT JACCPOT_BASE=$H0_JACCPOT OUT=$OUT/stall \
  bash bench/slurm/a1_dehnen_error_stall.sbatch
h0_done
