#!/bin/bash
#SBATCH --job-name=a4-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=a4-summary-%j.out
# A4: one table per ladder and tier (SUMMARY.md, summary.json) after every array has ended.
set -euo pipefail
A4_JACCPOT=${A4_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
export A4_TIER=summary
# shellcheck source=_prologue.sh
source "$A4_JACCPOT/bench/slurm/a4/_prologue.sh"
for tier in horeka-1card horeka-node; do
  root=$A4_RUN/results/$tier
  [ -d "$root" ] || continue
  {
    echo "# A4 accuracy suite, tier $tier ($(date -Is))"
    echo
    "$A4_PY" -m bench.accuracy.suite summarize all --tier "$tier" --root "$root" --json "$root/summary.json"
    echo "## checks against the baselines"
    find "$root" -name check.txt -exec sh -c 'printf "%s: %s\n" "$1" "$(tail -1 "$1")"' _ {} \; | sort
    echo "## skipped / failed rungs"
    find "$root" -name SKIPPED | sort
  } > "$A4_RUN/SUMMARY-$tier.md"
done
date -Is > "$A4_RUN/DONE"
echo "=== summary: $A4_RUN/SUMMARY-*.md"
