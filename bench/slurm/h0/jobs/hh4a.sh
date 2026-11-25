#!/bin/bash
# HoreKa job "hh4a" (~2-3 h): HH4a item 4 on ONE card (bench/slurm/hh/hybrid_clusters_runs.sbatch WHAT=share): the
# O(K M^2) internal direct sum's share of a slow step, fixed n_sub 16 vs adaptive (Aarseth) internal steps, at
# M = 1e3 / 1e4 / 1e5 (galaxy 1e6, HH4a's local rows) and 1e5 / 150 x 1e3 (galaxy 1e8); coupling fmm vs none at 1e5.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight hh4a
export CUDA_VISIBLE_DEVICES=$FIRST_CARD
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N=4096 SMOKE_K=2 SMOKE_M=32 SMOKE_ARGS="--mac geo --leaf 16 --order 4 --cluster-leaf 16 --reps 1 --reserve 64"
  export ROWS=${ROWS:-'^(share_1e6_K1_1e3|share_1e6_K1_1e5_none)$'}
fi
WHAT=share PY=$H0_PY JACCPOT_HH=$H0_JACCPOT OUT=$OUT/share bash bench/slurm/hh/hybrid_clusters_runs.sbatch
h0_done
