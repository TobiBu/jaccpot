#!/bin/bash
# HoreKa job "forward" (~8-10 h): forward mode round 2's timings and the demonstrator
# (bench/slurm/forward_mode_timing.sbatch, unchanged rows), one card: jvp / jacfwd K = 1..8, directions per pass,
# tangent groups, value_and_grad, the per-seam tangents-per-program sweep at 2.5e7; forward / jvp / jacfwd at 1e8;
# the 8 x 7 satellite Jacobian over T = 200 steps at 2.5e7 and 1e8.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight forward
if [ "${SMOKE:-}" = 1 ]; then
  # one row per distinct argument pattern (the others differ in N or K alone); value_and_grad (t_*_vgrad) is
  # the rollout gradient whose XLA:CPU compile H0 could not complete (50-156 GB): the GPU preflight runs it
  export SMOKE_N=2048 H0_SMOKE_INTERPRET=all FWD_EXTRA=--cpu-interpret DEMO_STEPS=3 REPS=1
  export ROWS=${ROWS:-'^(t_2p5e7_forward|t_2p5e7_jvp|t_2p5e7_jacfwd_k2|t_2p5e7_jacfwd_k8_dpp2|t_2p5e7_jacfwd_k8_group2|t_2p5e7_jvp_group1|tpp_2p5e7|demo_2p5e7_check_T4|demo_1e8_T200_dpp1)$'}
fi
PY=$H0_PY JACCPOT=$H0_JACCPOT OUT=$OUT/timing bash bench/slurm/forward_mode_timing.sbatch
h0_done
