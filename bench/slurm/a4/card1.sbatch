#!/bin/bash
#SBATCH --job-name=a4-card1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=19
#SBATCH --time=08:00:00
#SBATCH --output=a4-card1-%A_%a.out
# A4 tier "horeka-1card": ONE rung of the ladder $A4_LADDER per array task, one A100 (a quarter of a
# node, not exclusive). submit.sh sets --array, --time, --mem and the environment.
#   A4_LADDER=disc_n A4_RUN=... sbatch --array=0-3 bench/slurm/a4/card1.sbatch
export A4_TIER=horeka-1card
# shellcheck source=_prologue.sh
source "${A4_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/a4/_prologue.sh"
a4_task "${A4_LADDER:?}" "$A4_TIER" "${SLURM_ARRAY_TASK_ID:-0}"
