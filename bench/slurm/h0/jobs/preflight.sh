#!/bin/bash
# HoreKa job "preflight" (~45-60 min): the environment on a GPU node, then ONE tiny row of every entry point the
# other jobs call, with their arguments at 2e5 particles (SMOKE: 2048, CPU). Exits non-zero only when the
# environment itself is broken (every other job depends on it with afterok); a broken entry point is recorded in
# status/<entry> and its job stops at once instead of burning its nodes. An entry whose staged big data is
# missing is recorded "skipped (no data)" (its job skips too).
#
# H0_PREFLIGHT_ONLY=pins,m2g,...: only these entries (the development machine's GPU preflight runs the one-card
# entries on one card and the multi-process ones on two: H0_GPUS_PER_NODE=2, H0_LOCAL_SRUN=1).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
mkdir -p "$OUT/status" "$OUT/logs"
N=200000
[ "${SMOKE:-}" = 1 ] && N=2048
GPN=${H0_GPUS_PER_NODE:-4}
CPN=${SLURM_CPUS_ON_NODE:-$(nproc)}
MP=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
PORT=$((21000 + ${SLURM_JOB_ID:-$$} % 20000))

# 1. the environment (fatal)
"$H0_PY" - <<'PYENV' 2>&1 | tee "$OUT/logs/env.log"
import os, sys
import jax, jaccpot, yggdrax, nornax, mimirax
smoke = os.environ.get("SMOKE") == "1"
print("python", sys.version.split()[0], "jax", jax.__version__)
for m in (jaccpot, yggdrax, nornax, mimirax):
    print(f"  {m.__name__:8s} {os.path.dirname(m.__file__)}")
    root = {"jaccpot": "H0_JACCPOT", "yggdrax": "H0_YGGDRAX", "nornax": "H0_NORNAX", "mimirax": "H0_MIMIRAX"}
    want = os.path.realpath(os.environ[root[m.__name__]])
    assert os.path.realpath(m.__file__).startswith(want), f"{m.__name__} not from {want}"
devs = jax.devices()
print("devices", devs)
if not smoke:
    import jax_plugins.xla_cuda12 as plugin
    if os.environ.get("H0_PY_OVERRIDE") is None:
        assert plugin.__file__.startswith(sys.prefix), f"CUDA plugin outside the venv: {plugin.__file__}"
    want = int(os.environ.get("H0_GPUS_PER_NODE", "4"))
    assert [d.platform for d in devs] == ["gpu"] * want, f"want {want} GPUs, got {devs}"
    import jax.numpy as jnp
    x = jnp.arange(1 << 20, dtype=jnp.float32)
    assert float(jnp.sum(x)) > 0
    print("a GPU op ran")
PYENV
[ "${PIPESTATUS[0]}" = 0 ] || { echo "preflight: the environment is broken (see $OUT/logs/env.log)"; exit 1; }
nvidia-smi topo -m > "$OUT/logs/topology.txt" 2>&1 || true
command -v srun > /dev/null && echo "srun: $(command -v srun)" | tee "$OUT/logs/srun.txt"

# 2. one tiny row per entry point (recorded, not fatal)
wanted() { [ -z "${H0_PREFLIGHT_ONLY:-}" ] || [[ ",$H0_PREFLIGHT_ONLY," == *",$1,"* ]]; }
entry() {  # name command...
  local name=$1; shift
  wanted "$name" || return 0
  echo "--- preflight entry $name: $*"
  local start=$SECONDS
  PORT=$((PORT + 7))
  if JAX_COORDINATOR_PORT=$PORT "$@" > "$OUT/logs/$name.log" 2>&1; then
    echo ok > "$OUT/status/$name"
  else
    echo "failed rc=$?" > "$OUT/status/$name"
  fi
  echo "    $(cat "$OUT/status/$name") in $((SECONDS - start)) s"
}
skip_entry() {  # name reason
  wanted "$1" || return 0
  echo "skipped ($2)" > "$OUT/status/$1"
  echo "--- preflight entry $1: skipped ($2)"
}
fwd_entry() {  # the forward job's program kinds, incl. the reverse (value_and_grad; GPU only) and the demonstrator
  local T=bench/forward_mode_timing.py D=bench/forward_mode_demonstrator.py X=()
  [ "${SMOKE:-}" = 1 ] && X=(--cpu-interpret)
  env XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "$H0_PY" $T "${X[@]}" --n "$N" \
    --steps 2 --program jacfwd --k 2 --reps 1 --out "$OUT/forward/jacfwd.json" \
    && env XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JACCPOT_FORWARD_TANGENT_GROUP=1 \
      "$H0_PY" $T "${X[@]}" --n "$N" --steps 2 --program jacfwd --k 2 --reps 1 --out "$OUT/forward/group.json" \
    && { [ "${SMOKE:-}" = 1 ] || env XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "$H0_PY" $T \
      --n "$N" --steps 2 --program vgrad --reps 1 --out "$OUT/forward/vgrad.json"; } \
    && env XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "$H0_PY" $D "${X[@]}" --n "$N" \
      --check --steps 2 --out "$OUT/forward/demo.json"
}
a1_entry() {  # a tiny restart, then the stall bench on it (the restore rows' allocator)
  h0_tiny_restart "$OUT/a1/tiny_restart.npz" "$N" \
    && env XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=false "$H0_PY" \
      bench/dehnen_error_stall.py --restart "$OUT/a1/tiny_restart.npz" --steps 2 --arm error --out "$OUT/a1/stall.json"
}
a4_entry() {  # one shrunk rung of a sphere ladder (no IC needed), as a4's tasks run it
  local cmd
  export A4_PY=$H0_PY   # the rung's command line names $A4_PY (bench/slurm/a4/_prologue.sh exports it)
  cmd=$(A4_SMOKE=1 "$H0_PY" -m bench.accuracy.suite ladder plummer_n --tier horeka-1card --root "$OUT/a4" --index 0) \
    && A4_SMOKE=1 eval "$cmd"
}
mkdir -p "$OUT/forward" "$OUT/a1" "$OUT/a4"
entry pins env DCE_PINS_SMOKE_N=$((N < 8192 ? 8192 : N)) "$H0_PY" bench/dce_pins.py record --pins L1,L3 --tag preflight --root "$OUT/pins"
entry scaling env SMOKE_N=$N TIERS=weak REPS=1 OUT="$OUT/scaling" bash bench/slurm/multigpu_scaling.sbatch
entry scaling_mp env SMOKE_N=$N TIERS=mp REPS=1 MATRIX_ROWS='^mp_(P8e6_de_1c_sp|P8e6_de_[24]c_mp|D2.5e7_de_[24]c_(sp|mp))_a$' \
  OUT="$OUT/scaling_mp" bash bench/slurm/multigpu_scaling.sbatch
# the rollout gradient's step / nested modes: on the GPU only (their XLA:CPU compiles need 50-156 GB, track H0)
GRAD_ROWS='^(s8_8e6|r_8e6_T1_fwd|r_8e6_T1_grad_step|r_8e6_T16_grad_nested|r_8e6_T1_grad_offload)$'
[ "${SMOKE:-}" = 1 ] && GRAD_ROWS='^(s8_8e6|r_8e6_T1_fwd)$'
entry grad env SMOKE_N=$N H0_SMOKE_INTERPRET=all ROWS="$GRAD_ROWS" PY="$H0_PY" OUT="$OUT/grad" \
  bash bench/slurm/grad_rollout_timing.sbatch
entry s1b env SIZES=$N REPS=1 INNER=1 OUT="$OUT/s1b" bash bench/slurm/s1b_mosaic_l2p.sbatch
entry auto env H0_SMOKE_INTERPRET=all CUDA_VISIBLE_DEVICES="$FIRST_CARD" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  "$H0_PY" "$H0_MIMIRAX/bench/s10_fused_shooting.py" --task timing --n "$((N < 20000 ? N : 20000))" --steps 5 \
  --reps 1 --segments 2 --only seggrad --start-force adjoint --cell-min-level auto --out "$OUT/auto_seg.json"
entry forward env CUDA_VISIBLE_DEVICES="$FIRST_CARD" H0_SMOKE_INTERPRET=all bash -c "$(declare -f fwd_entry); N=$N OUT=$OUT SMOKE=${SMOKE:-} H0_PY=$H0_PY fwd_entry"
# the smoke: an XLA:CPU gradient compile outlasts the matrix's 1 h row timeout on a loaded host
MGRAD_TIMEOUT=
[ "${SMOKE:-}" = 1 ] && MGRAD_TIMEOUT=14400
entry mgrad env SMOKE_N=$N REPS=1 MAX_CARDS="$GPN" ROW_TIMEOUT="$MGRAD_TIMEOUT" H0_SMOKE_INTERPRET=all ROWS='^(weak_P2e5pc_(solo|2c|4c)|check_P_s0_2c)$' \
  PY="$H0_PY" OUT="$OUT/mgrad" bash bench/slurm/multigpu_grad.sbatch
entry a3 env CUDA_VISIBLE_DEVICES="$FIRST_CARD" "$H0_PY" bench/fused_memory_budget.py --n "$N" --caps unnamed --prealloc 0.88 \
  --no-analysis --ic plummer_clipped --skip-eval --steps 4 --reps 1 --runner --mac-type dehnen_error --adaptive-eps 1e-4 \
  --drift-order legacy --env JACCPOT_STRICT_CARRY=particles --out "$OUT/a3.json"
entry m1 srun "${MP[@]}" "$H0_PY" bench/mesh_cross_momentum.py --arm mp --mac dehnen_error --n-per-gpu "$((N / GPN))" \
  --segments 1 --steps 2 --out "$OUT/m1"
entry m2g bash -c "srun ${MP[*]} $H0_PY bench/multiprocess_gate.py collectives --arm mp --out $OUT/m2g/col_mp \
  && JAX_COORDINATOR_PORT=$((PORT + 3)) srun ${MP[*]} $H0_PY bench/multiprocess_gate.py run --arm mp --n-per-gpu $((N / GPN)) \
  --repeats 1 --out $OUT/m2g/mp"
entry a4 env CUDA_VISIBLE_DEVICES="$FIRST_CARD" bash -c "$(declare -f a4_entry); N=$N OUT=$OUT H0_PY=$H0_PY a4_entry"
# M3: a sharded checkpoint written by the node's processes and restored elastically by one (orbax in the venv)
entry m3 bash -c "JAX_COORDINATOR_PORT=$((PORT + 5)) srun ${MP[*]} $H0_PY bench/m3_checkpoint_io.py write --n $N --steps 1 --reps 1 \
  --ckpt $OUT/m3/ck --out $OUT/m3/write && JAX_COORDINATOR_PORT=$((PORT + 6)) srun --ntasks=1 --gres=gpu:1 $H0_PY \
  bench/m3_checkpoint_io.py restore --ckpt $OUT/m3/ck --steps 1 --reps 1 --out $OUT/m3/restore"
if [ -s "$SGR_IC" ]; then
  if wanted escaper || wanted mesc; then
    "$H0_PY" "$H0_DIR/tiny_ic.py" "$SGR_IC" "$OUT/sgr_tiny.npz" "$N" > "$OUT/logs/tiny_ic.log" 2>&1 || true
  fi
  entry escaper env CUDA_VISIBLE_DEVICES="$FIRST_CARD" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    "$H0_PY" bench/escaper_timing.py --ic "$OUT/sgr_tiny.npz" --arm esc --r-escape 80 --mac dehnen_error --steps 2 \
    --segments 2 --out "$OUT/escaper.json"
  entry mesc srun "${MP[@]}" "$H0_PY" bench/mesh_escaper_timing.py --arm mp --row esc --mac dehnen_error --ic "$OUT/sgr_tiny.npz" \
    --segments 2 --steps 2 --out "$OUT/mesc"
else
  skip_entry escaper "no data: $SGR_IC"
  skip_entry mesc "no data: $SGR_IC"
fi
if [ -s "$A1_BAR_RESTART" ] && [ -s "$A1_MERGER_RESTART" ]; then
  entry a1 env CUDA_VISIBLE_DEVICES="$FIRST_CARD" bash -c "$(declare -f a1_entry h0_tiny_restart h0_with_cpu_platform); \
    h0_with_cpu_platform; N=$N OUT=$OUT H0_PY=$H0_PY a1_entry"
else
  skip_entry a1 "no data: the 1e8 restarts under $H0_DATA/big"
fi
# the matrices and the others swallow a row's failure: look at their own records too
for m in scaling scaling_mp mgrad; do
  grep -lE '"rc": ([1-9-]|"timeout")' "$OUT"/$m/*.smi.txt > /dev/null 2>&1 && echo "failed (a row's rc != 0)" > "$OUT/status/$m"
done
grep -q FAILED "$OUT/logs/grad.log" 2> /dev/null && echo "failed (a row FAILED)" > "$OUT/status/grad"
# s1b records a kernel's failure in its row ("errors") and exits 0; on the CPU every kernel fails by design
# (it times the GPU kernels themselves), so only a GPU run is judged by it
if [ "${SMOKE:-}" != 1 ] && grep -q '"errors": {' "$OUT/s1b/rows.jsonl" 2> /dev/null; then
  echo "failed (a kernel errored: $OUT/s1b/rows.jsonl)" > "$OUT/status/s1b"
fi
for e in "$OUT"/status/*; do echo "$(basename "$e"): $(cat "$e")"; done | tee "$OUT/status.txt"
h0_done
