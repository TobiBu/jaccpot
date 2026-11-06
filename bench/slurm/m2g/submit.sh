#!/bin/bash
# M2G: submit the HoreKa jobs (account / partition from bench/slurm/h0/site.env; run H0's setup_env.sh once).
#   bash bench/slurm/m2g/submit.sh [--dry-run] [node1] [node2]     (default: both)
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JACCPOT=$(cd "$DIR/../../.." && pwd)
# shellcheck source=../h0/site.env
source "$JACCPOT/bench/slurm/h0/site.env"
DRY=0; JOBS=()
for a in "$@"; do case $a in --dry-run) DRY=1 ;; node1|node2) JOBS+=("$a") ;; *) echo "unknown $a" >&2; exit 2 ;; esac; done
[ ${#JOBS[@]} -gt 0 ] || JOBS=(node1 node2)
if [ "$DRY" = 0 ]; then
  for v in H0_ACCOUNT H0_ROOT; do [ "${!v}" != CHANGE_ME ] || { echo "set $v in bench/slurm/h0/site.env" >&2; exit 2; }; done
  [ -f "$H0_ROOT/setup.done" ] || { echo "run bench/slurm/h0/setup_env.sh first" >&2; exit 2; }
fi
RUN=${M2G_RUN:-$H0_ROOT/m2g/$(date +%Y%m%d-%H%M%S)}
for job in "${JOBS[@]}"; do
  cmd=(sbatch -A "$H0_ACCOUNT" -p "$H0_PARTITION" --mem="$H0_MEM" $H0_SBATCH_EXTRA
       --chdir="$JACCPOT" --output="$RUN/logs/%x-%j.out"
       --export=ALL,M2G_JACCPOT="$JACCPOT",M2G_RUN="$RUN" "$DIR/$job.sbatch")
  echo "${cmd[*]}"
  if [ "$DRY" = 0 ]; then mkdir -p "$RUN/logs"; "${cmd[@]}"; fi
done
echo "# results: $RUN/<job>/SUMMARY.md (send back: tar czf m2g.tgz --exclude='*.npy' -C $RUN .)"
