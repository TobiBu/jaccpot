#!/bin/bash
# HoreKa job "hh3" (~2-3 h): HH3a's K-cluster slow step against the galaxy step at 1e8 on ONE card
# (bench/slurm/hh/hybrid_clusters_timing.sbatch): K = 10 x 1e4 and 150 x 1e3 clusters, coupling fmm vs none; the
# grouped back-reaction alone; the coupling alone at 150 x 1e3 / 1e4. nornax: the bundle's (pins.env).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight hh3
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N=4096 SMOKE_K=3 SMOKE_M=64 SMOKE_ARGS="--leaf 16 --order 4 --reps 1 --mac geo"
  export ROWS=${ROWS:-'^(galaxy_1e8|slow_1e8_K10_1e4_fmm|back_1e8_K150_1e3|coupling_1e6_K150_1e3)$'}
fi
PY=$H0_PY JACCPOT_HH=$H0_JACCPOT OUT=$OUT/timing bash bench/slurm/hh/hybrid_clusters_timing.sbatch
h0_done
