# M3's HoreKa steps (sourced by bench/slurm/h0/jobs/m3_*.sh after the H0 prologue: $OUT, $H0_PY).
# m3_step NAME SRUN_ARGS... -- BENCH_ARGS...: one srun step of bench/m3_checkpoint_io.py into $OUT/NAME,
# skipped when its outputs exist (a resubmitted job resumes); a failed step is recorded, the job goes on.
# shellcheck shell=bash
M3_PORT=${M3_PORT:-$((21000 + ${SLURM_JOB_ID:-0} % 20000))}
m3_step() {
  local name=$1; shift
  local srun_args=()
  while [ "$1" != -- ]; do srun_args+=("$1"); shift; done; shift
  if compgen -G "$OUT/$name/result_*_p0.json" > /dev/null; then echo "skip $name (done)"; return 0; fi
  M3_PORT=$((M3_PORT + 1)); export JAX_COORDINATOR_PORT=$M3_PORT
  echo "--- $name: srun ${srun_args[*]} -- $* $(date -Is)"
  local t0=$SECONDS
  if srun "${srun_args[@]}" "$H0_PY" bench/m3_checkpoint_io.py "$@" --out "$OUT/$name" > "$OUT/$name.log" 2>&1; then
    echo "    ok $((SECONDS - t0)) s"
  else
    echo "    FAILED rc $? after $((SECONDS - t0)) s (tail of $OUT/$name.log):"; tail -20 "$OUT/$name.log"
    echo "$name failed" >> "$OUT/FAILED"
  fi
}
# m3_summary: one table of every step's per-rank rows ($OUT/SUMMARY.md)
m3_summary() { "$H0_PY" bench/slurm/m3/summarize.py "$OUT" > "$OUT/SUMMARY.md" || true; }
