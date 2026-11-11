#!/bin/bash
# HoreKa job "mgrad" (~7.5 h): WP-G's reverse-mode gradients of the fused multi-GPU force
# (bench/slurm/multigpu_grad.sbatch -> bench/multigpu_grad_matrix.py, unchanged rows): weak, strong, memory
# (jax.grad only, up to the 4-card ceiling) and check (accuracy vs the single card and an fp64 direct sum).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight mgrad
if [ "${SMOKE:-}" = 1 ]; then
  # one row per distinct argument pattern: solo, the mesh at 2 cards, memory (grad only, BFC preallocated),
  # the accuracy check, the geometric MAC's G / eps2 check (each a gradient compile of 20-60 min on XLA:CPU).
  # The check runs on Plummer: a 2048-row prefix of the disc is nearly all-near and overflows the flat walk's
  # near-pair cap the cross formula sizes from N (check_D runs at its own N, 262,144, on the GPUs)
  export SMOKE_N=2048 REPS=1 MEMORY_PER_CARD=2e7 H0_SMOKE_INTERPRET=all
  export ROWS=${ROWS:-'^(weak_P2e5pc_solo|weak_P2e5pc_2c|memory_P2e7pc_2c|check_P_s0_2c|check_P_geo_p3_2c)$'}
  export MAX_CARDS=2 ROW_TIMEOUT=14400   # an XLA:CPU gradient compile outlasts the 1 h row timeout
fi
PY=$H0_PY JACCPOT=$H0_JACCPOT OUT=$OUT/matrix bash bench/slurm/multigpu_grad.sbatch
h0_done
