#!/bin/bash
# HH3a (hybrid Hermite session 3): the K-cluster slow step against the galaxy step at 1e8, timed on ONE card of a
# dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths below.
# One process per row, on the allocation's first card only. Resumable: a row whose JSON exists is skipped.
#
#   sbatch bench/slurm/hh/hybrid_clusters_timing.sbatch                    # every row (~2-3 h, rough)
#   ROWS='^slow_' sbatch bench/slurm/hh/hybrid_clusters_timing.sbatch      # the rows whose tag matches a regex
#
# Every row is bench/hybrid_clusters_timing.py (nornax's FusedClustersForce; one compiled program, warm 1,
# min/median of 3 calls; a Hernquist galaxy of N and K Plummer clusters of M stars at log-uniform radii 0.3-10;
# galaxy dehnen_error 1e-4, leaf 64, p6; the star tree geometric theta 0.5; no reserve):
#   galaxy_1e8                    the galaxy's own step through the force object (the base)
#   slow_1e8_K<K>_<M>_<coupling>  the whole slow evaluation: galaxy step + galaxy -> all K x M stars (one target
#                                 plan) + the stars' field at all 1e8 galaxy rows (the grouped back-reaction) + the
#                                 cluster-cluster coupling (fmm: the star tree minus each cluster's own; none)
#   back_1e8_K<K>_<M>             the grouped back-reaction alone (the star tree's rebuild + all 1e8 rows)
#   coupling_1e6_K150_<M>         the coupling alone (coupling="fmm") at K = 150, M = 1e3 / 1e4 (its galaxy is 1e6:
#                                 the coupling does not read it)
# The clusters' cost is slow_* minus galaxy_1e8; the coupling's is slow_*_fmm minus slow_*_none.
#
# Needs nornax with FusedClustersForce's grouped back-reaction (HH3a): the bundle's H0_NORNAX_HH3 checkout
# (pins.env H0_NORNAX_HH3_COMMIT), put first on PYTHONPATH by job "hh3"; run alone, set PYTHONPATH yourself.
#
# On HoreKa this runs as job "hh3" of the bundle (bench/slurm/h0/README.md: one checkout, one setup, every job).
#
#SBATCH --job-name=jaccpot-hh3-hybrid-clusters
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), yggdrax, nornax (HH3a) and THIS jaccpot checkout (HH3a merged)
PY=${PY:-python}
JACCPOT_HH=${JACCPOT_HH:-$PWD}
OUT=${OUT:-${WORKSPACE:-$HOME}/hh3-hybrid-clusters-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
# SMOKE_N / SMOKE_K / SMOKE_M: every row at this size (bench/slurm/h0/submit_all.sh SMOKE=1, a CPU test)
nn() { echo "${SMOKE_N:-$1}"; }
kk() { echo "${SMOKE_K:-$1}"; }
mm() { echo "${SMOKE_M:-$1}"; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"

echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$JACCPOT_HH" rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true
"$PY" -c "import nornax, os; print(os.path.dirname(nornax.__file__))" > "$OUT/nornax_path.txt" 2>&1 || true
F="$JACCPOT_HH/bench/hybrid_clusters_timing.py"
N8=$(nn 100000000)
N6=$(nn 1000000)

ROW_LIST=("galaxy_1e8|--n $N8 --k $(kk 10) --m $(mm 10000) --mode galaxy --coupling none")
for km in "10 1e4 10000" "150 1e3 1000"; do
  read -r K MT MV <<< "$km"
  for c in fmm none; do
    ROW_LIST+=("slow_1e8_K${K}_${MT}_${c}|--n $N8 --k $(kk "$K") --m $(mm "$MV") --mode slow --coupling $c")
  done
  ROW_LIST+=("back_1e8_K${K}_${MT}|--n $N8 --k $(kk "$K") --m $(mm "$MV") --mode back --coupling none")
done
ROW_LIST+=("coupling_1e6_K150_1e3|--n $N6 --k $(kk 150) --m $(mm 1000) --mode coupling")
ROW_LIST+=("coupling_1e6_K150_1e4|--n $N6 --k $(kk 150) --m $(mm 10000) --mode coupling")

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag args <<< "$row"
  [[ "$tag" =~ $ROWS ]] || continue
  [ -f "$OUT/rows/$tag.json" ] && { echo "skip $tag"; continue; }
  echo "=== $tag $(date -Is)"
  # shellcheck disable=SC2086
  (cd "$OUT" && $PY "$F" $args ${SMOKE_ARGS:-} --out "$OUT/rows/$tag.json") > "$OUT/logs/$tag.log" 2>&1 \
    || echo "FAILED $tag (see $OUT/logs/$tag.log)"
done

# summary: ms (min / median), the clusters' cost over the galaxy step, the device peak and temporaries
$PY - "$OUT/rows" <<'PYEOF'
import json, sys, pathlib
rows = {p.stem: json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))}
base = rows.get("galaxy_1e8")
print(f"{'row':28s} {'min ms':>9s} {'med ms':>9s} {'- galaxy':>9s} {'peak GiB':>9s} {'temp GiB':>9s}  ok / stats")
for tag, d in rows.items():
    extra = f"{d['ms_min'] - base['ms_min']:9.1f}" if base and tag.startswith("slow_") else f"{'':9s}"
    gib = lambda v: f"{v / 2**30:9.2f}" if isinstance(v, (int, float)) else f"{'-':>9s}"
    print(f"{tag:28s} {d['ms_min']:9.1f} {d['ms_median']:9.1f} {extra} {gib(d.get('peak_bytes'))} "
          f"{gib(d.get('temp_bytes'))}  {d.get('ok', '-')} {d.get('stats', '')}")
PYEOF
