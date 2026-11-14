#!/bin/bash
# HH2a (hybrid Hermite session 2): target_points cost against the galaxy step at 1e8, timed on ONE card of a
# dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths below.
# One process per row, on the allocation's first card only. Resumable: a row whose JSON exists is skipped.
#
#   sbatch bench/slurm/hh/target_points_timing.sbatch                     # every row (~1.5-2 h, rough)
#   ROWS='^targets_err' sbatch bench/slurm/hh/target_points_timing.sbatch # the rows whose tag matches a regex
#
# Every row is bench/target_points_timing.py (one compiled program, warm 1, min/median of 3 calls; the IC of the
# HH GPU runs: Hernquist galaxy of N, Plummer cluster of M stars at R = 1; galaxy preset, leaf 64, p6):
#   step_<mac>_1e8                  the galaxy provider's step (rebuild + force): the base
#   targets_<mac>_<far>_1e8_<M>     step_with_targets with M = 1e4 / 1e5 / 1e6 stars as targets (galaxy -> stars;
#                                   the frontier walk; far = m2p per point or local per group)
#   back_<far>_1e8_1e5              the cluster's field at all 1e8 galaxy points (evaluate_back_reaction: the root
#                                   multipole outside r_in, the frontier walk inside; the cluster's rebuild included)
#   backfull_1e7_1e5                the same through target_points over ALL points (the comparison; 1e7 only)
# <mac>: geo (geometric theta 0.6) / err (dehnen_error 1e-4). The target cost is targets_* minus step_*.
#
# On HoreKa this runs as job "hh" of the bundle (bench/slurm/h0/README.md: one checkout, one setup, every job).
#
#SBATCH --job-name=jaccpot-hh-target-points
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), yggdrax and THIS jaccpot checkout (HH2a merged)
PY=${PY:-python}
JACCPOT_HH=${JACCPOT_HH:-$PWD}
OUT=${OUT:-${WORKSPACE:-$HOME}/hh-target-points-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
# SMOKE_N / SMOKE_M: every row at this N / M (bench/slurm/h0/submit_all.sh SMOKE=1, a CPU test of the commands)
nn() { echo "${SMOKE_N:-$1}"; }
mm() { echo "${SMOKE_M:-$1}"; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"

echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$JACCPOT_HH" rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true
F="$JACCPOT_HH/bench/target_points_timing.py"
N8=$(nn 100000000)
N7=$(nn 10000000)

ROW_LIST=()
for mac in geo err; do
  ROW_LIST+=("step_${mac}_1e8|--n $N8 --m $(mm 10000) --mode step --mac $mac")
  for far in m2p local; do
    for m in 1e4 1e5 1e6; do
      case $m in 1e4) M=$(mm 10000) ;; 1e5) M=$(mm 100000) ;; 1e6) M=$(mm 1000000) ;; esac
      ROW_LIST+=("targets_${mac}_${far}_1e8_${m}|--n $N8 --m $M --mode targets --mac $mac --far $far")
    done
  done
done
for far in m2p local; do
  ROW_LIST+=("back_${far}_1e8_1e5|--n $N8 --m $(mm 100000) --mode back --far $far")
done
ROW_LIST+=("backfull_1e7_1e5|--n $N7 --m $(mm 100000) --mode back_full")

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag args <<< "$row"
  [[ "$tag" =~ $ROWS ]] || continue
  [ -f "$OUT/rows/$tag.json" ] && { echo "skip $tag"; continue; }
  echo "=== $tag $(date -Is)"
  # shellcheck disable=SC2086
  (cd "$OUT" && $PY "$F" $args ${SMOKE_ARGS:-} --out "$OUT/rows/$tag.json") > "$OUT/logs/$tag.log" 2>&1 \
    || echo "FAILED $tag (see $OUT/logs/$tag.log)"
done

# summary: ms (min / median), the target cost over the step, the device peak, the walk's stats
$PY - "$OUT/rows" <<'PYEOF'
import json, sys, pathlib
rows = {p.stem: json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))}
print(f"{'row':34s} {'min ms':>9s} {'med ms':>9s} {'- step':>9s} {'peak GiB':>9s}  stats (queue peak, MAC tests, points)")
for tag, d in rows.items():
    base = rows.get(f"step_{d.get('mac')}_1e8") if d.get("mode") == "targets" else None
    extra = f"{d['ms_min'] - base['ms_min']:9.1f}" if base else f"{'':9s}"
    peak = d.get("peak_bytes")
    peak = f"{peak / 2**30:9.2f}" if isinstance(peak, (int, float)) else f"{'-':>9s}"
    print(f"{tag:34s} {d['ms_min']:9.1f} {d['ms_median']:9.1f} {extra} {peak}  {d.get('stats', '')} ok={d.get('ok', '-')}")
PYEOF
