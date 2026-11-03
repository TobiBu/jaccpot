#!/bin/bash
# H0 job "preflight" (~20 min): the environment on a GPU node, then ONE tiny row of every entry point the
# other jobs call, with their exact arguments at 2e5 particles (SMOKE: 2048, CPU). Exits non-zero only when
# the environment itself is broken (every other job depends on it with afterok); a broken entry point is
# recorded in status/<entry> and its job stops at once instead of burning its node.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
mkdir -p "$OUT/status" "$OUT/logs"
N=200000
[ "${SMOKE:-}" = 1 ] && N=2048

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

# 2. one tiny row per entry point (recorded, not fatal)
entry() {  # name command...
  local name=$1; shift
  echo "--- preflight entry $name: $*"
  local start=$SECONDS
  if "$@" > "$OUT/logs/$name.log" 2>&1; then echo ok > "$OUT/status/$name"; else echo "failed rc=$?" > "$OUT/status/$name"; fi
  echo "    $(cat "$OUT/status/$name") in $((SECONDS - start)) s"
}
entry pins env DCE_PINS_SMOKE_N=$N "$H0_PY" bench/dce_pins.py record --pins L1,L3 --tag preflight --root "$OUT/pins"
entry scaling env SMOKE_N=$N TIERS=weak REPS=1 OUT="$OUT/scaling" bash bench/slurm/multigpu_scaling.sbatch
entry grad env SMOKE_N=$N ROWS='^(s8_8e6|r_8e6_T1_fwd|r_8e6_T1_grad_offload)$' PY="$H0_PY" OUT="$OUT/grad" \
  bash bench/slurm/grad_rollout_timing.sbatch
entry s1b env SIZES=$N REPS=1 INNER=1 OUT="$OUT/s1b" bash bench/slurm/s1b_mosaic_l2p.sbatch
entry auto env CUDA_VISIBLE_DEVICES="$FIRST_CARD" XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  "$H0_PY" "$H0_MIMIRAX/bench/s10_fused_shooting.py" --task timing --n "$((N < 20000 ? N : 20000))" --steps 5 \
  --reps 1 --segments 2 --only seggrad --start-force adjoint --cell-min-level auto --out "$OUT/auto_seg.json"
# the scaling sbatch and the others swallow a row's failure: look at their own records too
grep -l '"rc": [1-9]' "$OUT"/scaling/*.smi.txt > /dev/null 2>&1 && echo "failed (a row's rc != 0)" > "$OUT/status/scaling"
grep -q FAILED "$OUT/logs/grad.log" && echo "failed (a row FAILED)" > "$OUT/status/grad"
for e in "$OUT"/status/*; do echo "$(basename "$e"): $(cat "$e")"; done | tee "$OUT/status.txt"
h0_done
