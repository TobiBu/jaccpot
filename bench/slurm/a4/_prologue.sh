# A4 jobs' first lines (sourced): H0's environment (site.env, pins, the venv), A4's site values,
# provenance, the task runner.
# shellcheck shell=bash
set -euo pipefail
A4_JACCPOT=${A4_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
if [ "${A4_LOCAL_SMOKE:-}" = 1 ]; then
  # the local dry run (submit.sh --local-smoke): the caller's python, CPU, no H0 venv
  : "${A4_PY:?A4_LOCAL_SMOKE needs A4_PY}"
  H0_PY=$A4_PY
  h0_provenance() { mkdir -p "$1"; echo '{"smoke": true}' > "$1/provenance.json"; }
else
  # shellcheck source=../h0/env.sh
  source "$A4_JACCPOT/bench/slurm/h0/env.sh"
  A4_PY=$H0_PY
fi
# shellcheck source=site.env
source "$A4_JACCPOT/bench/slurm/a4/site.env"
A4_RUN=${A4_RUN:?A4_RUN (set by submit.sh)}
A4_OUT=$A4_RUN/results/${A4_TIER:-horeka-1card}
export A4_PY A4_IC A4_OUT A4_RUN
mkdir -p "$A4_OUT"
cd "$A4_JACCPOT"
echo "=== A4 ${A4_LADDER:-job} task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none})"
# run as the HoreKa bundle's job "a4": stop at once when its preflight found the entry broken
if [ -n "${A4_PREFLIGHT:-}" ] && [ -f "$A4_PREFLIGHT" ] && [ "$(cat "$A4_PREFLIGHT")" != ok ] \
  && [ "${A4_TIER:-}" != summary ]; then
  echo "A4: the bundle's preflight found entry a4 broken ($(cat "$A4_PREFLIGHT")); not spending the node" >&2
  exit 3
fi

# a4_task LADDER TIER INDEX: one rung; skipped when its result exists (a resubmit resumes); then a
# check against the baseline when A4_BASELINES has one. Exit 0 also for the A3 'anchored' slot (rc 3).
a4_task() {
  local ladder=$1 tier=$2 index=$3 cmd out rung rc=0
  cmd=$("$A4_PY" -m bench.accuracy.suite ladder "$ladder" --tier "$tier" --root "$A4_OUT" \
        --ic-dir "$A4_IC" --index "$index")
  out=$(sed -E 's/.*--out ([^ ]+).*/\1/' <<<"$cmd")
  rung=$(basename "$out")
  mkdir -p "$out"
  if [ -s "$out/result.json" ]; then echo "skip $ladder/$rung (done)"; return 0; fi
  h0_provenance "$out"
  echo "--- $ladder/$rung: $cmd ($(date -Is))"
  local t0=$SECONDS
  eval "$cmd" > "$out/run.log" 2>&1 || rc=$?
  if [ "$rc" = 3 ]; then
    echo "    SKIPPED (A3's anchored state not available yet)"; echo skipped > "$out/SKIPPED"; return 0
  elif [ "$rc" != 0 ]; then
    echo "    FAILED rc $rc after $((SECONDS - t0)) s (tail of $out/run.log):"; tail -30 "$out/run.log"
    return "$rc"
  fi
  echo "    ok $((SECONDS - t0)) s"
  local base=${A4_BASELINES:+$A4_BASELINES/$tier/$ladder/$rung.json}
  if [ -n "$base" ] && [ -s "$base" ]; then
    "$A4_PY" -m bench.accuracy.suite check "$out/result.json" "$base" --json "$out/check.json" \
      > "$out/check.txt" 2>&1 && echo "    check PASS" || echo "    check FAIL (see $out/check.txt)"
  fi
}
