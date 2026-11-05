# M2G jobs' first lines (sourced): H0's environment (site.env, pins, the venv), provenance, helpers.
# shellcheck shell=bash
set -euo pipefail
M2G_JACCPOT=${M2G_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/env.sh
source "$M2G_JACCPOT/bench/slurm/h0/env.sh"
M2G_RUN=${M2G_RUN:-$H0_ROOT/m2g/${SLURM_JOB_ID:-local}}
OUT=$M2G_RUN/${M2G_JOB:?}
mkdir -p "$OUT"
echo "=== M2G job $M2G_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-?})"
h0_provenance "$OUT"
cd "$M2G_JACCPOT"
GATE="$H0_PY bench/multiprocess_gate.py"
GPN=${H0_GPUS_PER_NODE:-4}
CPN=${SLURM_CPUS_ON_NODE:-76}
# every step its own coordinator port (jax derives one port from SLURM_JOB_ID otherwise)
M2G_PORT=${M2G_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
next_port() { M2G_PORT=$((M2G_PORT + 1)); export JAX_COORDINATOR_PORT=$M2G_PORT; }
# arm NAME SRUN_ARGS... -- GATE_ARGS...: one srun step, skipped when its outputs exist (resubmit resumes)
arm() {
  local name=$1; shift
  local srun_args=() gate_args=()
  while [ "$1" != -- ]; do srun_args+=("$1"); shift; done; shift
  gate_args=("$@")
  if [ -s "$OUT/$name/result_p0.json" ] || [ -s "$OUT/$name/collectives.json" ]; then
    echo "skip $name (done)"; return 0
  fi
  next_port
  echo "--- $name: srun ${srun_args[*]} -- ${gate_args[*]} (port $M2G_PORT) $(date -Is)"
  local t0=$SECONDS
  if srun "${srun_args[@]}" $GATE "${gate_args[@]}" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"
  else
    echo "    FAILED rc $? after $((SECONDS - t0)) s (tail of $OUT/$name.log):"; tail -20 "$OUT/$name.log"
  fi
}
cmp_arms() {  # A B
  $H0_PY bench/multiprocess_gate.py compare "$OUT/$1" "$OUT/$2" --json "$OUT/compare_${1}_vs_$2.json" \
    > /dev/null 2>&1 || echo "(compare $1 $2 failed)"
}
m2g_done() { $H0_PY bench/slurm/m2g/summarize.py "$OUT" > "$OUT/SUMMARY.md" || true; date -Is > "$OUT/DONE"; echo "=== done $(date -Is)"; }
