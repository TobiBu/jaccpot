#!/bin/bash
# The A4 local functional tier: every standard problem at its local size (problems.py, tier "local") on
# ONE device -- a free card (select it with autocvd, e.g. mn-scratch's on-free-card) or the CPU (slow:
# the fused lane's CPU route runs the Pallas kernels in interpret mode, ~2 s/step at N = 1024).
#
#   bash bench/accuracy/local_tier.sh OUT_DIR IC_DIR [BASELINE_DIR]
#
# OUT_DIR/<problem>/result.json per problem; with BASELINE_DIR, `check` against BASELINE_DIR/<problem>.json
# (OUT_DIR/<problem>/check.txt; the exit code counts the failures). IC_DIR holds the galaxy ICs
# (galaxy.npz, bar.npz, merger.npz from Odisseo tools/a4_galaxy_ics.py at >= the local sizes).
# Env: A4_PY (python; default python), A4_PROBLEMS (default: all six), A4_EXTRA (extra `run` args).
set -uo pipefail
OUT=${1:?OUT_DIR}; ICS=${2:?IC_DIR}; BASE=${3:-}
PY=${A4_PY:-python}
JACCPOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$JACCPOT" || exit 2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
declare -A IC=([disc_heating]=galaxy.npz [bar]=bar.npz [merger]=merger.npz)
fails=0
for p in ${A4_PROBLEMS:-plummer hernquist nfw disc_heating bar merger}; do
  args=(-m bench.accuracy.suite run "$p" --tier local --out "$OUT/$p")
  [ -n "${IC[$p]:-}" ] && args+=(--ic "$ICS/${IC[$p]}")
  # shellcheck disable=SC2206
  args+=(${A4_EXTRA:-})
  echo "--- $p: $PY ${args[*]} ($(date -Is))"
  t0=$SECONDS
  if "$PY" "${args[@]}" > "$OUT.$p.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"
    if [ -n "$BASE" ] && [ -s "$BASE/$p.json" ]; then
      if "$PY" -m bench.accuracy.suite check "$OUT/$p/result.json" "$BASE/$p.json" > "$OUT/$p/check.txt" 2>&1; then
        echo "    check PASS"
      else
        echo "    check FAIL ($OUT/$p/check.txt)"; fails=$((fails + 1))
      fi
    fi
  else
    echo "    FAILED after $((SECONDS - t0)) s:"; tail -20 "$OUT.$p.log"; fails=$((fails + 1))
  fi
done
echo "local tier: $fails failure(s)"
exit "$fails"
