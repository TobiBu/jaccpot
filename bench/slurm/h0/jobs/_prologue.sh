# HoreKa bundle: every job's first lines (sourced). submit_all.sh exports H0_JACCPOT, H0_RUN and H0_JOB.
# shellcheck shell=bash
set -euo pipefail
: "${H0_JACCPOT:?submit with bench/slurm/h0/submit_all.sh}" "${H0_RUN:?}" "${H0_JOB:?}"
# shellcheck source=../env.sh
source "$H0_JACCPOT/bench/slurm/h0/env.sh"
OUT=$H0_RUN/$H0_JOB
mkdir -p "$OUT"
echo "=== H0 job $H0_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-local}, smoke ${SMOKE:-0})"
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
# h0_need_data FILE...: a job that reads staged big data (stage_data.sh without --no-big) records SKIPPED and
# ends (DONE) when a file is missing, instead of failing on the node
h0_need_data() {
  local f missing=()
  for f in "$@"; do [ -s "$f" ] || missing+=("$f"); done
  [ ${#missing[@]} = 0 ] && return 0
  echo "H0: job $H0_JOB skipped: missing ${missing[*]} (bench/slurm/h0/stage_data.sh without --no-big)" | tee "$OUT/SKIPPED"
  h0_done
  exit 0
}
# h0_track_run NAME: the per-track scripts (bench/slurm/{m1,m2g,mesc}) write into this job's directory
h0_track_run() { export "$1_RUN=$H0_RUN" "$1_JOB=$H0_JOB"; }
# h0_tiny_restart OUT_NPZ N: a tiny ProductionRunner restart (clipped Plummer, dehnen_error at 1e-4, three steps),
# the stand-in for A1's 1e8 restarts in the CPU smoke and the preflight (bench/fused_memory_budget.py --runner)
h0_tiny_restart() {
  [ -s "$1" ] && return 0
  "$H0_PY" bench/fused_memory_budget.py --n "$2" --ic plummer_clipped --caps unnamed --no-analysis --skip-eval \
    --mac-type dehnen_error --adaptive-eps 1e-4 --steps 2 --reps 1 --runner --runner-restart "$1" \
    --runner-restart-at 3 --out "$1.json" > "$1.log" 2>&1 || { echo "h0_tiny_restart: failed (see $1.log)" >&2; return 1; }
  [ -s "$1" ] || { echo "h0_tiny_restart: no restart written (see $1.log)" >&2; return 1; }
}
