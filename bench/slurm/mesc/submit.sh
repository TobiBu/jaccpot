#!/bin/bash
# MESC: these jobs are part of the HoreKa bundle now (bench/slurm/h0/README.md: one checkout, one setup_env.sh, one
# submit_all.sh). This forwarder keeps the old command working; it submits through the bundle, so the bundle's
# preflight job runs first (the mesc jobs start afterok on it) and the results land in $H0_ROOT/runs/<run id>/mesc_node*/.
#   bash bench/slurm/mesc/submit.sh [--dry-run] [node1] [node2]     (default: both)
#   = bash bench/slurm/h0/submit_all.sh [--dry-run] --only mesc_node1,mesc_node2
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRY=(); JOBS=()
for a in "$@"; do case $a in --dry-run) DRY=(--dry-run) ;; node1|node2) JOBS+=("mesc_$a") ;; *) echo "unknown $a" >&2; exit 2 ;; esac; done
[ ${#JOBS[@]} -gt 0 ] || JOBS=(mesc_node1 mesc_node2)
exec bash "$DIR/../h0/submit_all.sh" "${DRY[@]}" --only "$(IFS=,; echo "${JOBS[*]}")"
