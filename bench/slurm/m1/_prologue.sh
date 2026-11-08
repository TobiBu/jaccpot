# M1 jobs' first lines (sourced): H0's environment (site.env, pins, the venv), provenance, helpers.
# shellcheck shell=bash
set -euo pipefail
M1_JACCPOT=${M1_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/env.sh
source "$M1_JACCPOT/bench/slurm/h0/env.sh"
M1_RUN=${M1_RUN:-$H0_ROOT/m1/${SLURM_JOB_ID:-local}}
OUT=$M1_RUN/${M1_JOB:?}
mkdir -p "$OUT"
echo "=== M1 job $M1_JOB on $(hostname) $(date -Is) (slurm job ${SLURM_JOB_ID:-none}, nodes ${SLURM_JOB_NODELIST:-?})"
h0_provenance "$OUT"
cd "$M1_JACCPOT"
BENCH="$H0_PY bench/mesh_cross_momentum.py"
GPN=${H0_GPUS_PER_NODE:-4}
CPN=${SLURM_CPUS_ON_NODE:-76}
# every step its own coordinator port (jax derives one port from SLURM_JOB_ID otherwise)
M1_PORT=${M1_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}
next_port() { M1_PORT=$((M1_PORT + 1)); export JAX_COORDINATOR_PORT=$M1_PORT; }
# arm NAME SRUN_ARGS... -- BENCH_ARGS...: one srun step, skipped when its result exists (resubmit resumes)
arm() {
  local name=$1; shift
  local srun_args=() bench_args=()
  while [ "$1" != -- ]; do srun_args+=("$1"); shift; done; shift
  bench_args=("$@")
  if [ -s "$OUT/$name/result.json" ]; then echo "skip $name (done)"; return 0; fi
  next_port
  echo "--- $name: srun ${srun_args[*]} -- ${bench_args[*]} (port $M1_PORT) $(date -Is)"
  local t0=$SECONDS
  if srun "${srun_args[@]}" $BENCH "${bench_args[@]}" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"
  else
    echo "    FAILED rc $? after $((SECONDS - t0)) s (tail of $OUT/$name.log):"; tail -20 "$OUT/$name.log"
  fi
}
m1_done() { $H0_PY bench/slurm/m1/summarize.py "$OUT" > "$OUT/SUMMARY.md" || true; date -Is > "$OUT/DONE"; echo "=== done $(date -Is)"; }
