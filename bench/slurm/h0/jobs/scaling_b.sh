#!/bin/bash
# H0 job "scaling_b" (~6.5 h): the #480 matrix's "ab" tier -- migration on / off, interleaved repetitions,
# two and four NVLink cards, Plummer 8e6 / 1.6e7, disc + bulge 2e6 / 8e6 / 25M, both MACs (64 rows).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight scaling
export TIERS=ab REPS=${REPS:-2}
if [ "${SMOKE:-}" = 1 ]; then
  # one MAC: the geometric rows differ from these in --mac alone (the other jobs' smokes run it)
  export SMOKE_N=2048 REPS=1 MATRIX_ROWS=${MATRIX_ROWS:-'^ab_.*_de_[24]c_(on|off)_a$'}
fi
OUT=$OUT/matrix bash bench/slurm/multigpu_scaling.sbatch
h0_done
