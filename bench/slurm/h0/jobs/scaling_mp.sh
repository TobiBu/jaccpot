#!/bin/bash
# HoreKa job "scaling_mp" (~4 h): the multi-GPU matrix's mp tier (bench/multigpu_scaling_matrix.py --tiers mp):
# the production multi-GPU path as N processes x 1 GPU (jaccpot.distributed.multiprocess, M2's runtime; srun steps
# with M2G's launch pattern) next to one process driving the same N GPUs, on 1 / 2 / 4 NVLink cards: Plummer 8e6
# and the 25M disc + bulge (both MACs), Plummer 4e6 per card (dehnen_error); bench/multiprocess_gate.py run, the
# production runner with migration on (MeshRunConfig's default); interleaved sp / mp, 2 repetitions (50 rows).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight scaling_mp
export TIERS=mp REPS=${REPS:-2}
if [ "${SMOKE:-}" = 1 ]; then
  # one row per launch pattern and IC at two processes (the 4-card rows differ in the count alone)
  export SMOKE_N=2048 REPS=1
  export MATRIX_ROWS=${MATRIX_ROWS:-'^mp_(P8e6_de_1c_sp|P8e6_de_2c_(sp|mp)|P4e6pc_de_2c_mp|D2.5e7_(de|geo)_2c_mp)_a$'}
fi
OUT=$OUT/matrix bash bench/slurm/multigpu_scaling.sbatch
h0_done
