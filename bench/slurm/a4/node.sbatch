#!/bin/bash
#SBATCH --job-name=a4-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=a4-node-%A_%a.out
# A4 tier "horeka-node": ONE rung of $A4_LADDER per array task on one exclusive node, the runner's
# multi-GPU mode as one process driving the node's 4 A100s (--devices 4; the reference force is the
# same number on any device count, so a node rung's force score compares with a card rung's).
export A4_TIER=horeka-node
# shellcheck source=_prologue.sh
source "${A4_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/a4/_prologue.sh"
a4_task "${A4_LADDER:?}" "$A4_TIER" "${SLURM_ARRAY_TASK_ID:-0}"
