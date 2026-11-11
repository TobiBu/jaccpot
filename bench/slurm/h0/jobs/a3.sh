#!/bin/bash
# HoreKa job "a3" (~2 h): A3's summed drift (the default) against the legacy order, one card
# (bench/slurm/a3/position_precision_timing.sbatch): the runner at 1e8 and 2.6e5, geometric and dehnen_error MAC,
# A-B-A-B (legacy, summed, legacy_b, summed_b). A3's gate 3: summed within +10 % of legacy at 1e8.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight a3
if [ "${SMOKE:-}" = 1 ]; then
  export SMOKE_N=2048 ROWS=${ROWS:-'^(geo|err)_1e8_(legacy|summed)$'}
fi
PY=$H0_PY JACCPOT_A3=$H0_JACCPOT OUT=$OUT/timing bash bench/slurm/a3/position_precision_timing.sbatch
h0_done
