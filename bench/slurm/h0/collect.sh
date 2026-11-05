#!/bin/bash
# H0 bundle: after the jobs, one tarball + SUMMARY.md to send back.
#
#   bash bench/slurm/h0/collect.sh                 # the newest run under $H0_ROOT/runs
#   bash bench/slurm/h0/collect.sh RUN_DIR         # a given run (or a SMOKE run under $H0_ROOT/smoke)
#   H0_COLLECT_TRACES=1 bash ...                   # also the profiler traces (grad; can be GBs)
#
# Writes RUN_DIR/SUMMARY.md and RUN_DIR/rows.jsonl (bench/slurm/h0/summarize.py), then
# $H0_ROOT/h0-results-<run id>.tar.gz with every JSON row, log, provenance header, compare verdict and the
# summary -- but not the position dumps (dump_*) and .npz arrays (GBs, and nothing the summary reads).
set -euo pipefail
H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$H0_DIR/env.sh"
RUN=${1:-$(ls -d "$H0_ROOT"/runs/*/ 2> /dev/null | sort | tail -1)}
RUN=${RUN%/}
[ -d "$RUN" ] || { echo "collect: no run directory (${RUN:-none})" >&2; exit 2; }
echo "collect: $RUN"
if [ -f "$RUN/jobs.txt" ] && command -v sacct > /dev/null; then
  ids=$(awk '{print $2}' "$RUN/jobs.txt" | paste -sd,)
  sacct -j "$ids" --format=JobID,JobName%20,State,Elapsed,Timelimit,NodeList,ExitCode > "$RUN/sacct.txt" 2>&1 || true
  cat "$RUN/sacct.txt"
fi
# the scaling matrices' own summaries (their python imports jaccpot: the run environment)
for m in "$RUN"/scaling_*/matrix; do
  [ -d "$m" ] && { "$H0_PY" "$H0_JACCPOT/bench/multigpu_scaling_matrix.py" --out-dir "$m" --summary > /dev/null || true; }
done
"$H0_PY" "$H0_DIR/summarize.py" "$RUN"
name=$(basename "$RUN")
tarball=$H0_ROOT/h0-results-$name.tar.gz
excl=(--exclude='dump_*' --exclude='*.npz' --exclude='*.npy')
[ "${H0_COLLECT_TRACES:-}" = 1 ] || excl+=(--exclude='traces')
tar -czf "$tarball" "${excl[@]}" -C "$(dirname "$RUN")" "$name"
echo "collect: $tarball ($(du -h "$tarball" | cut -f1)); send it back with $RUN/SUMMARY.md"
