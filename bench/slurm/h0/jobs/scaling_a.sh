#!/bin/bash
# H0 job "scaling_a" (~3.5 h): the #480 multi-GPU scaling matrix's strong, weak and memory tiers on all four
# NVLink cards of one exclusive node (bench/slurm/multigpu_scaling.sbatch -> bench/multigpu_scaling_matrix.py,
# one fresh process per row). The memory tier climbs past #480's 4e7 per card (to 7e7: the extrapolated
# ~6.9e7 explicit-capacity ceiling) and stops at the first row that fails: that row is the 4-card ceiling.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight scaling
export TIERS=strong,weak,memory MEMORY_PER_CARD=${MEMORY_PER_CARD:-1e7,2e7,3e7,4e7,5e7,6e7,7e7} REPS=${REPS:-2}
if [ "${SMOKE:-}" = 1 ]; then export SMOKE_N=2048 MEMORY_PER_CARD=1e7; fi
OUT=$OUT/matrix bash bench/slurm/multigpu_scaling.sbatch
h0_done
