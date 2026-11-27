# M4 jobs' first lines (sourced): H0's environment (site.env, pins, the venv), provenance, helpers.
# shellcheck shell=bash
set -euo pipefail
M4_JACCPOT=${M4_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/env.sh
source "$M4_JACCPOT/bench/slurm/h0/env.sh"
M4_RUN=${M4_RUN:-$H0_ROOT/m4/${SLURM_JOB_ID:-local}}
OUT=$M4_RUN/${M4_JOB:?}
mkdir -p "$OUT"
echo "=== M4 job $M4_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-?})"
h0_provenance "$OUT"
cd "$M4_JACCPOT"
BENCH="$H0_PY bench/m4_exchange_cuts.py"
GPN=${H0_GPUS_PER_NODE:-4}
# SMOKE_N_PER_GPU / M4_SEGMENT_STEPS: the CPU smoke's tiny arms (bench/slurm/h0/submit_all.sh SMOKE=1)
nn() { echo "${SMOKE_N_PER_GPU:-$1}"; }
M4_STEPS=(${M4_SEGMENT_STEPS:+--steps "$M4_SEGMENT_STEPS"})
CPN=${SLURM_CPUS_ON_NODE:-76}
# every step its own coordinator port (jax derives one port from SLURM_JOB_ID otherwise)
M4_PORT=${M4_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
next_port() { M4_PORT=$((M4_PORT + 1)); export JAX_COORDINATOR_PORT=$M4_PORT; }
# arm NAME SRUN_ARGS... -- BENCH_ARGS...: one srun step, skipped when its result exists (resubmit resumes)
arm() {
  local name=$1; shift
  local srun_args=() bench_args=()
  while [ "$1" != -- ]; do srun_args+=("$1"); shift; done; shift
  bench_args=("$@")
  if [ -s "$OUT/$name/result.json" ]; then echo "skip $name (done)"; return 0; fi
  next_port
  echo "--- $name: srun ${srun_args[*]} -- ${bench_args[*]} (port $M4_PORT) $(date -Is)"
  local t0=$SECONDS
  if srun "${srun_args[@]}" $BENCH "${bench_args[@]}" "${M4_STEPS[@]}" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"
  else
    echo "    FAILED rc $? after $((SECONDS - t0)) s (tail of $OUT/$name.log):"; tail -20 "$OUT/$name.log"
  fi
}
m4_done() { $H0_PY bench/slurm/m4/summarize.py "$OUT" > "$OUT/SUMMARY.md" || true; date -Is > "$OUT/DONE"; echo "=== done $(date -Is)"; }
