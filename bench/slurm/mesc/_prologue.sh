# MESC jobs' first lines (sourced): H0's environment (site.env, pins, the venv), provenance, helpers.
# shellcheck shell=bash
set -euo pipefail
MESC_JACCPOT=${MESC_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/env.sh
source "$MESC_JACCPOT/bench/slurm/h0/env.sh"
MESC_RUN=${MESC_RUN:-$H0_ROOT/mesc/${SLURM_JOB_ID:-local}}
MESC_IC=${MESC_IC:-$SGR_IC}   # env.sh: $H0_ROOT/data/big/ (stage_data.sh)
OUT=$MESC_RUN/${MESC_JOB:?}
mkdir -p "$OUT"
echo "=== MESC job $MESC_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-?})"
[ -s "$MESC_IC" ] || { echo "missing IC $MESC_IC (see bench/slurm/mesc/README.md)"; exit 2; }
h0_provenance "$OUT"
cd "$MESC_JACCPOT"
ENTRY="$H0_PY bench/mesh_escaper_timing.py"
GPN=${H0_GPUS_PER_NODE:-4}
CPN=${SLURM_CPUS_ON_NODE:-76}
# MESC_EXTRA: arguments for every row (the CPU smoke's tiny IC runs fewer segments / steps)
MESC_EXTRA=${MESC_EXTRA:-}
MESC_PORT=${MESC_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
next_port() { # MESC_EXTRA: arguments for every row (the CPU smoke's tiny IC runs fewer segments / steps)
MESC_EXTRA=${MESC_EXTRA:-}
MESC_PORT=$((MESC_PORT + 1)); export JAX_COORDINATOR_PORT=$MESC_PORT; }
# row NAME SRUN_ARGS... -- ENTRY_ARGS...: one srun step, skipped when its result exists (resubmit resumes)
row() {
  local name=$1; shift
  local srun_args=() entry_args=()
  while [ "$1" != -- ]; do srun_args+=("$1"); shift; done; shift
  entry_args=("$@")
  if [ -s "$OUT/$name/result_p0.json" ]; then echo "skip $name (done)"; return 0; fi
  next_port
  echo "--- $name: srun ${srun_args[*]} -- ${entry_args[*]} (port $MESC_PORT) $(date -Is)"
  local t0=$SECONDS
  # shellcheck disable=SC2086
  if srun "${srun_args[@]}" $ENTRY "${entry_args[@]}" $MESC_EXTRA --ic "$MESC_IC" --out "$OUT/$name" \
      > "$OUT/$name.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"; grep '^\[done\]' "$OUT/$name.log" || true
  else
    echo "    FAILED rc $? after $((SECONDS - t0)) s (tail of $OUT/$name.log):"; tail -20 "$OUT/$name.log"
  fi
}
mesc_done() { $H0_PY bench/slurm/mesc/summarize.py "$OUT" > "$OUT/SUMMARY.md" || true; date -Is > "$OUT/DONE"; echo "=== done $(date -Is)"; }
