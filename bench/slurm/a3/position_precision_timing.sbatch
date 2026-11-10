#!/bin/bash
# A3 (multi-node plan, U2): the summed drift (the default since A3) against the legacy order, timed on ONE card of
# a dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths below.
# One process at a time, on the allocation's first card only. Resumable: a row whose JSON exists is skipped.
#
#   sbatch bench/slurm/a3/position_precision_timing.sbatch                 # every row (~2 h, rough)
#   ROWS='^geo_1e8' sbatch bench/slurm/a3/position_precision_timing.sbatch # the rows whose tag matches a regex
#
# Every row is bench/fused_memory_budget.py --runner (the scan through jaccpot.production.ProductionRunner, one
# segment per call; L2's operating point: clipped Plummer, cells of 64, cell_min_level 8, p6, theta 0.8,
# unnamed caps, arena 0.88, step = min of 3 calls of 4 steps) with --drift-order:
#   legacy    x + v dt + 0.5 a dt**2 left to right (every run before A3; GalaxySettings(drift_order="legacy"))
#   summed    x + (v dt + 0.5 a dt**2) (the default since A3)
# A-B-A-B: legacy, summed, legacy_b, summed_b -- each arm's two samples are its own A-vs-A control (the run's
# noise envelope). A3's gate 3: summed's 1e8 step within +10 % of legacy's (expected: equal; no array is added).
#
# Rows (tag = <mac>_<N>_<arm>):
#   geo_*   the geometric Dehnen MAC at theta 0.8 (the bench's default; the L2 pin's operating point)
#   err_*   mac_type dehnen_error at adaptive_eps 1e-4 (preset="galaxy"'s MAC)
#   at N = 1e8 and 2.6e5 (2**18).
#
# On HoreKa this runs as job "a3" of the bundle (bench/slurm/h0/README.md: one checkout, one setup, every job).
#
#SBATCH --job-name=jaccpot-a3-position-precision
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), yggdrax main and THIS jaccpot checkout (main: A3 merged)
PY=${PY:-python}
JACCPOT_A3=${JACCPOT_A3:-$PWD}
OUT=${OUT:-${WORKSPACE:-$HOME}/a3-position-precision-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
# SMOKE_N: every row at this N (bench/slurm/h0/submit_all.sh SMOKE=1, a CPU test of the commands)
nn() { echo "${SMOKE_N:-$1}"; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"

echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$JACCPOT_A3" rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true
F="$JACCPOT_A3/bench/fused_memory_budget.py"
COMMON="--caps unnamed --prealloc 0.88 --no-analysis --ic plummer_clipped --skip-eval --steps 4 --reps 3 --runner"
ERR="--mac-type dehnen_error --adaptive-eps 1e-4"

ROW_LIST=()
for n in 1e8 2p6e5; do
  case $n in 1e8) N=$(nn 100000000) ;; 2p6e5) N=$(nn 262144) ;; esac
  for mac in geo err; do
    case $mac in geo) M="" ;; err) M="$ERR" ;; esac
    for arm in legacy summed legacy_b summed_b; do
      ROW_LIST+=("${mac}_${n}_${arm}|--n $N $COMMON $M --drift-order ${arm%_b}")
    done
  done
done

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag args <<< "$row"
  [[ "$tag" =~ $ROWS ]] || continue
  [ -f "$OUT/rows/$tag.json" ] && { echo "skip $tag"; continue; }
  echo "=== $tag $(date -Is)"
  (cd "$OUT" && $PY "$F" $args --env JACCPOT_STRICT_CARRY=particles --out "$OUT/rows/$tag.json") \
    > "$OUT/logs/$tag.log" 2>&1 || echo "FAILED $tag"
done

# summary: step (min / median, ms) and peak (GiB, B/p) per row
$PY - "$OUT/rows" <<'EOF'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    d = json.loads(p.read_text())
    n = d.get("n") or d.get("config", {}).get("n") or 0
    peak = d.get("final_peak_gib") or d.get("peak_after_scan_gib") or 0.0
    st = d.get("step_ms") or {}
    bpp = peak * 2**30 / n if n else float("nan")
    print(f"{p.stem:28s} step min {st.get('min', float('nan')):9.1f} ms median {st.get('median', float('nan')):9.1f} ms"
          f"  peak {peak:6.2f} GiB {bpp:6.1f} B/p")
EOF
