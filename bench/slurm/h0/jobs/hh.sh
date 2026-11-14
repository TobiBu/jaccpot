#!/bin/bash
# HoreKa job "hh" (~1.5-2 h): HH2a's target_points cost against the galaxy step at 1e8 on ONE card
# (bench/slurm/hh/target_points_timing.sbatch): step_with_targets at 1e4 / 1e5 / 1e6 stars, both MACs, per-point
# M2P and per-group local far fields; the back-reaction at all 1e8 galaxy points; the full target_points
# back-reaction at 1e7 for comparison.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight hh
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N=4096 SMOKE_M=256 SMOKE_ARGS="--leaf 16 --order 4 --reps 1"
  export ROWS=${ROWS:-'^(step_err_1e8|targets_err_(m2p|local)_1e8_1e4|back_m2p_1e8_1e5|backfull_1e7_1e5)$'}
fi
PY=$H0_PY JACCPOT_HH=$H0_JACCPOT OUT=$OUT/timing bash bench/slurm/hh/target_points_timing.sbatch
h0_done
