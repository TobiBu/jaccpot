#!/bin/bash
# H0 job "grad" (~4-5 h): S9's gradient timings (bench/slurm/grad_rollout_timing.sbatch, unchanged rows):
# S8's single-force gradient and the rollout gradient against its forward at 8e6, 2.5e7 and 1e8, one card.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight grad
[ "${SMOKE:-}" = 1 ] && export SMOKE_N=2048
PY=$H0_PY OUT=$OUT/timing bash bench/slurm/grad_rollout_timing.sbatch
h0_done
