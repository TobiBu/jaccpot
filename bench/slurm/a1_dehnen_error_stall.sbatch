#!/bin/bash
# A1 (#487): the dehnen_error stall and its two fixes, timed on ONE card of a dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environments and the paths
# below. One process at a time, on the allocation's first card only. Resumable: a row whose JSON exists is
# skipped, so a resubmission continues where the last one stopped.
#
#   sbatch bench/slurm/a1_dehnen_error_stall.sbatch                   # every row (~3-4 h, rough; base stall rows
#                                                                      # dominate: up to ~450 s per step)
#   ROWS='^stall_' sbatch bench/slurm/a1_dehnen_error_stall.sbatch     # the rows whose tag matches a regex
#
# Three code arms, two checkouts:
#   base   jaccpot main (PY_BASE): the O(n^2) long-row rank -- the stall as reported
#   net    the A1 branch (PY_A1), floor off: the long CSR rows merge-sorted (the safety net, default)
#   floor  the A1 branch with mac_force_scale_floor 0.5 (the fix; opt-in unless the gates passed)
# and the geometric MAC at theta 0.7 (geo07) as the fallback the plan compares against.
#
# Rows:
#   stall_<arm>        the 1e8 bar-unstable disc from restart_3 (step 30), 10 steps (31-40: the stall's
#                      segment 4 on main, 205-231 s per 10 steps against 9.3 s before it)
#   merger_<arm>       the 1.06e8 major merger from restart_final_136 (t 1.36, after first pericentre), 5 steps
#   L1-L3 base / a1    bench/dce_pins.py's timing pins (geometric MAC: the CSR build's default path), A-B-A
#   err_<N>_<arm>      fused_memory_budget.py at the L sizes under dehnen_error, floor off and on: the floor's
#                      own cost where it does not bind (8e6 clipped Plummer, 25M disc, 1e8 clipped Plummer)
#
# The restart files (copy them to $DATA first, ~4.7 GB each):
#   /export/scratch/tbuck/bar-2026-10-31/stall_a/restart_3.npz          -> $DATA/bar_restart_3.npz
#   /export/scratch/tbuck/merger-2026-10-31/s1_main/restart_final_136.npz -> $DATA/merger_restart_136.npz
#
#SBATCH --job-name=jaccpot-a1-dehnen-error-stall
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: pythons with jax 0.11.2 (CUDA 12) and yggdrax main; jaccpot main in PY_BASE's env, the A1 branch in
# PY_A1's (JACCPOT_A1: that checkout, whose bench/ scripts both arms run)
PY_BASE=${PY_BASE:-python}
PY_A1=${PY_A1:-python}
JACCPOT_A1=${JACCPOT_A1:-$PWD}
JACCPOT_BASE=${JACCPOT_BASE:-$JACCPOT_A1/../jaccpot-main}
DATA=${DATA:-${WORKSPACE:-$HOME}/a1-data}
OUT=${OUT:-${WORKSPACE:-$HOME}/a1-dehnen-error-stall-$(date +%Y%m%d)}
ROWS=${ROWS:-.}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs" "$OUT/pins"
git -C "$JACCPOT_A1" rev-parse HEAD > "$OUT/jaccpot_a1_commit.txt" 2>/dev/null || true
git -C "$JACCPOT_BASE" rev-parse HEAD > "$OUT/jaccpot_base_commit.txt" 2>/dev/null || true
# both arms run the A1 checkout's bench script from $OUT, so `import jaccpot` is each env's own
cp "$JACCPOT_A1/bench/dehnen_error_stall.py" "$OUT/"
S="$OUT/dehnen_error_stall.py"
F="$JACCPOT_A1/bench/fused_memory_budget.py"

# tag | python | arguments (the row's JSON is $OUT/rows/<tag>.json)
ROW_LIST=()
for case in stall merger; do
  case $case in
    stall) R=$DATA/bar_restart_3.npz; K=10 ;;
    merger) R=$DATA/merger_restart_136.npz; K=5 ;;
  esac
  ROW_LIST+=(
    "${case}_base|$PY_BASE|$S --restart $R --steps $K --arm error"
    "${case}_net|$PY_A1|$S --restart $R --steps $K --arm error"
    "${case}_floor|$PY_A1|$S --restart $R --steps $K --arm floor --floor 0.5"
    "${case}_geo07|$PY_A1|$S --restart $R --steps $K --arm geo07"
  )
done
ERR="--caps unnamed --prealloc 0.88 --no-analysis --mac-type dehnen_error --adaptive-eps 1e-4 --skip-eval --steps 4 --reps 3"
for n in 8e6 2p5e7 1e8; do
  case $n in
    8e6) N=8000000; IC="--ic plummer_clipped" ;;
    2p5e7) N=25000000; IC="--ic disc --softening 1.5e-3 --softening-kernel ferrers3" ;;
    1e8) N=100000000; IC="--ic plummer_clipped" ;;
  esac
  ROW_LIST+=(
    "err_${n}_net|$PY_A1|$F --n $N $IC $ERR --env JACCPOT_STRICT_CARRY=particles --out $OUT/rows/err_${n}_net.json"
    "err_${n}_floor|$PY_A1|$F --n $N $IC $ERR --mac-force-scale-floor 0.5 --env JACCPOT_STRICT_CARRY=particles --out $OUT/rows/err_${n}_floor.json"
  )
done

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag py args <<< "$row"
  [[ "$tag" =~ $ROWS ]] || continue
  [ -f "$OUT/rows/$tag.json" ] && { echo "skip $tag"; continue; }
  echo "=== $tag $(date -Is)"
  case $tag in
    err_*) (cd "$OUT" && $py $args) > "$OUT/logs/$tag.log" 2>&1 || echo "FAILED $tag" ;;
    *) (cd "$OUT" && $py $args --out "$OUT/rows/$tag.json") > "$OUT/logs/$tag.log" 2>&1 || echo "FAILED $tag" ;;
  esac
done

# L1-L3, A-B-A: base, A1, base again (the envelope's control), each in its own checkout
if [[ "L_pins" =~ $ROWS ]] || [ "$ROWS" = "." ]; then
  for tag in a1-base a1-tip a1-base-b; do
    [ -d "$OUT/pins/$tag" ] && { echo "skip pins $tag"; continue; }
    case $tag in a1-tip) py=$PY_A1; wt=$JACCPOT_A1 ;; *) py=$PY_BASE; wt=$JACCPOT_BASE ;; esac
    (cd "$wt" && $py bench/dce_pins.py record --tag "$tag" --pins L1,L2,L3 --root "$OUT/pins") \
      > "$OUT/logs/pins_$tag.log" 2>&1 || echo "FAILED pins $tag"
  done
  (cd "$JACCPOT_A1" && $PY_A1 bench/dce_pins.py compare a1-base a1-tip --pins L1,L2,L3 \
      --envelope a1-base-b --root "$OUT/pins") > "$OUT/logs/pins_compare.log" 2>&1 || true
  cat "$OUT/logs/pins_compare.log"
fi
echo "done $(date -Is)"
