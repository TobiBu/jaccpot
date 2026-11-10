#!/bin/bash
# M1: these jobs are part of the HoreKa bundle now (bench/slurm/h0/README.md: one checkout, one setup_env.sh, one
# submit_all.sh). This forwarder keeps the old command working; it submits through the bundle, so the bundle's
# preflight job runs first (the m1 jobs start afterok on it) and the results land in $H0_ROOT/runs/<run id>/m1_node*/.
#   bash bench/slurm/m1/submit.sh [--dry-run] [node1] [node2]     (default: both)
#   = bash bench/slurm/h0/submit_all.sh [--dry-run] --only m1_node1,m1_node2
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY=(); JOBS=()
for a in "$@"; do case $a in --dry-run) DRY=(--dry-run) ;; node1|node2) JOBS+=("m1_$a") ;; *) echo "unknown $a" >&2; exit 2 ;; esac; done
[ ${#JOBS[@]} -gt 0 ] || JOBS=(m1_node1 m1_node2)
exec bash "$DIR/../h0/submit_all.sh" "${DRY[@]}" --only "$(IFS=,; echo "${JOBS[*]}")"
