# H0 bundle: every job's first lines (sourced). submit_all.sh exports H0_JACCPOT, H0_RUN and H0_JOB.
# shellcheck shell=bash
set -euo pipefail
: "${H0_JACCPOT:?submit with bench/slurm/h0/submit_all.sh}" "${H0_RUN:?}" "${H0_JOB:?}"
# shellcheck source=../env.sh
source "$H0_JACCPOT/bench/slurm/h0/env.sh"
OUT=$H0_RUN/$H0_JOB
mkdir -p "$OUT"
echo "=== H0 job $H0_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, smoke ${SMOKE:-0})"
echo "    cards ${CUDA_VISIBLE_DEVICES-unset}; out $OUT"
h0_provenance "$OUT"
cd "$H0_JACCPOT"
# the first card of the exclusive node, for the one-card jobs (the others stay idle)
FIRST_CARD=${CUDA_VISIBLE_DEVICES%%,*}
# h0_done: the job's done-file (collect.sh lists the jobs without one)
h0_done() { date -Is > "$OUT/DONE"; echo "=== H0 job $H0_JOB done $(date -Is)"; }
# h0_need_preflight ENTRY: stop at once when the preflight job found ENTRY broken
h0_need_preflight() {
  local f=$H0_RUN/preflight/status/$1
  if [ -f "$f" ] && [ "$(cat "$f")" != ok ]; then
    echo "H0: preflight found entry '$1' broken ($(cat "$f")); not spending the node on it" >&2
    exit 3
  fi
}
