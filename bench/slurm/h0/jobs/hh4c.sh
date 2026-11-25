#!/bin/bash
# HoreKa job "hh4c" (~3-4 h): HH4c's clusters on the mesh on ONE node, one process over 1 / 2 / 4 cards
# (bench/slurm/hh/hybrid_clusters_runs.sbatch WHAT=mesh): the single-card ClustersRunnerScheme vs
# MeshClustersRunnerScheme, strong (1e7) and weak (1e7 per card) with 10 x 1e4 stars, the production 1e8 + 150 x 1e3
# over 4 and 2 cards, and 2e8 over 4 (5e7 per card: a ceiling probe, its failure is recorded).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight hh4c
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N=2048 SMOKE_K=3 SMOKE_M=32
  export SMOKE_ARGS="--mac geo --leaf 16 --order 4 --cluster-leaf 16 --reserve 64 --segments 2 --steps 2"
  export ROWS=${ROWS:-'^(card1_1e7_K10_1e4|mesh2_1e7_K10_1e4)$'}
fi
WHAT=mesh PY=$H0_PY JACCPOT_HH=$H0_JACCPOT OUT=$OUT/mesh bash bench/slurm/hh/hybrid_clusters_runs.sbatch
h0_done
