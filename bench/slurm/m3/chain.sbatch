#!/bin/bash
#SBATCH --job-name=m3-chain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=m3-chain-%j.out
# M3/M5: a run longer than one job's walltime, as a chain of jobs that each continue from the newest committed
# checkpoint and resubmit themselves (bench/slurm/m3/README.md).
#
#   M3_CHAIN_DIR=$WS/chain M3_N=1e8 M3_TARGET_STEPS=2000 sbatch -A $H0_ACCOUNT -p accelerated \
#       bench/slurm/m3/chain.sbatch
#
# Each job FIRST submits its successor (afterany on itself: it runs whether this job finishes, runs out of time,
# loses a node or crashes), then runs bench/m3_checkpoint_io.py chain toward M3_TARGET_STEPS with a checkpoint
# every M3_EVERY segments and a deadline M3_MARGIN seconds before its own end; when the run reaches its target
# it writes $M3_CHAIN_DIR/DONE and cancels the successor. A successor that finds DONE exits at once; one that
# finds no new checkpoint since its predecessor's start (no progress) stops the chain instead of looping.
# M3_CHAIN_MAX (default 20) bounds the number of jobs.
set -euo pipefail
: "${M3_CHAIN_DIR:?set M3_CHAIN_DIR (a workspace directory)}"
JACCPOT=${M3_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/env.sh
source "$JACCPOT/bench/slurm/h0/env.sh"
cd "$JACCPOT"
mkdir -p "$M3_CHAIN_DIR/logs"
if [ -f "$M3_CHAIN_DIR/DONE" ]; then echo "chain done before: $(cat "$M3_CHAIN_DIR/DONE")"; exit 0; fi
link=${M3_CHAIN_LINK:-1}
max=${M3_CHAIN_MAX:-20}
newest() { ls -1d "$M3_CHAIN_DIR"/ck_* 2> /dev/null | grep -v '\.' | sort | tail -1 || true; }
before=$(newest)
if [ "$link" -gt 1 ] && [ "$before" = "$(cat "$M3_CHAIN_DIR/last_start" 2> /dev/null || true)" ]; then
  echo "chain: no checkpoint since the last job started ($before): stopping (inspect $M3_CHAIN_DIR/logs)" >&2
  exit 1
fi
echo "$before" > "$M3_CHAIN_DIR/last_start"
next=
if [ "$link" -lt "$max" ]; then
  next=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
    --export="ALL,M3_CHAIN_LINK=$((link + 1)),M3_JACCPOT=$JACCPOT" "$JACCPOT/bench/slurm/m3/chain.sbatch")
  echo "chain link $link: successor $next"
fi
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
deadline=$(date -d "$end" +%s)
GPN=${H0_GPUS_PER_NODE:-4}
export JAX_COORDINATOR_PORT=$((21000 + SLURM_JOB_ID % 20000))
srun --ntasks=$((SLURM_NNODES * GPN)) --ntasks-per-node="$GPN" --gres=gpu:"$GPN" \
  "$H0_PY" bench/m3_checkpoint_io.py chain --n "${M3_N:-1e8}" --steps "${M3_STEPS:-10}" \
  --pattern "$M3_CHAIN_DIR/ck_{step:08d}" --target-steps "${M3_TARGET_STEPS:?}" --every "${M3_EVERY:-5}" \
  --deadline "$deadline" --margin "${M3_MARGIN:-900}" --out "$M3_CHAIN_DIR/logs/link_$link"
if [ "$(cat "$M3_CHAIN_DIR/logs/link_$link/STATUS" 2> /dev/null)" = done ]; then
  date -Is > "$M3_CHAIN_DIR/DONE"
  [ -n "$next" ] && scancel "$next" || true
  echo "chain done at link $link"
fi
