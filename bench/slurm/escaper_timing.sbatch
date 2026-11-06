#!/bin/bash
# ESC (wave 3): the escaper path's cost at 1e8 -- step time and peak memory with and without escapers, ONE card of
# a dedicated node. Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and
# the paths below. One process at a time on the allocation's first card. Resumable: a row whose JSON exists is
# skipped.
#
#   sbatch bench/slurm/escaper_timing.sbatch                  # every row (~1.5-2 h, rough)
#   ROWS='dehnen' sbatch bench/slurm/escaper_timing.sbatch    # the rows whose tag matches a regex
#
# The IC: the 1e8 light-Sgr merger, UNCLIPPED (100,000,001 rows, a 45.7 Mpc box; A2), copy it to $DATA first (4.4 GB):
#   /export/scratch/tbuck/mn-scratch/a2/sgr/sgr_light_1e8_noclip.npz -> $DATA/sgr_light_1e8_noclip.npz
#
# Rows (bench/escaper_timing.py; 6 segments of 10 steps, dt 1e-3, softening 1.5e-3, energy at every edge):
#   esc_<mac>        escapers ON, r_escape 80 (800 kpc), the unclipped IC (~1.8e5 escapers)
#   off_<mac>        escapers OFF, every particle beyond 80 pulled onto r = 80 (same N): the baseline
#   escclip_<mac>    escapers ON with r_escape 160 on that clipped IC: nobody escapes (the default programs)
#   esc_<mac>_e10    escapers ON, the energy every 10th edge only (the ledger's own jumps still evaluated)
# for <mac> = dehnen (geometric, theta 0.7: the Sgr production run's) and derr (preset galaxy's dehnen_error).
# Compare step_s (segments 2-6) and peak_bytes_per_particle; edge_s is the per-edge host + energy cost.
#
#SBATCH --job-name=jaccpot-esc-escaper-timing
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), yggdrax at the A2 commit or later, jaccpot at the ESC branch
PY=${PY:-python}
JACCPOT=${JACCPOT:-$PWD}
DATA=${DATA:-${WORKSPACE:-$HOME}/esc-data}
OUT=${OUT:-${WORKSPACE:-$HOME}/esc-escaper-timing-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
IC=$DATA/sgr_light_1e8_noclip.npz

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$JACCPOT" rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true
S="$JACCPOT/bench/escaper_timing.py"

ROW_LIST=()
for mac in dehnen derr; do
  M=$([ $mac = derr ] && echo dehnen_error || echo dehnen)
  ROW_LIST+=(
    "off_${mac}|--arm off --clip 80 --mac $M"
    "esc_${mac}|--arm esc --r-escape 80 --mac $M"
    "escclip_${mac}|--arm esc_clip --clip 80 --mac $M"
    "esc_${mac}_e10|--arm esc --r-escape 80 --mac $M --energy-every 10"
  )
done

for entry in "${ROW_LIST[@]}"; do
  tag=${entry%%|*}; args=${entry#*|}
  [[ $tag =~ $ROWS ]] || continue
  out="$OUT/rows/$tag.json"
  if [ -s "$out" ]; then echo "skip $tag (done)"; continue; fi
  echo "== $tag $(date -Is)"
  # shellcheck disable=SC2086
  "$PY" "$S" --ic "$IC" --steps 10 --segments 6 $args --out "$out" > "$OUT/logs/$tag.log" 2>&1 \
    || echo "row $tag failed rc=$? (see $OUT/logs/$tag.log)"
  tail -1 "$OUT/logs/$tag.log" || true
done
echo "done $(date -Is)"
