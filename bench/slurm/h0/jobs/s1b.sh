#!/bin/bash
# H0 job "s1b" (~1 h): S1b's particle-major L2P, Triton-Pallas vs Mosaic GPU (bench/slurm/s1b_mosaic_l2p.sbatch):
# 2e6 and 2.5e7 (disc), 1e8 (clipped Plummer), one card, interleaved rounds, A-vs-A = Triton compiled twice.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight s1b
[ "${SMOKE:-}" = 1 ] && export SIZES=2048 REPS=1 INNER=1
OUT=$OUT/l2p bash bench/slurm/s1b_mosaic_l2p.sbatch
h0_done
