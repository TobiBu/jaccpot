#!/bin/bash
# Hybrid Hermite session 4 on a dedicated node (bench/hybrid_clusters_runs.py; HoreKa jobs "hh4a" and "hh4c"):
#   WHAT=share  HH4a item 4: the O(K M^2) internal direct sum's share of a slow step on ONE card, fixed n_sub 16 vs
#               adaptive (Aarseth) internal steps; the galaxy 1e6 (HH4a's local rows) and 1e8 (production)
#   WHAT=mesh   HH4c: the clusters' runner on 1 / 2 / 4 cards of one node (one process, a 1-D mesh): the single-card
#               ClustersRunnerScheme vs MeshClustersRunnerScheme, strong (1e7) and weak (1e7 per card) with
#               10 x 1e4 stars, and the production 1e8 + 150 x 1e3 over 4 and 2 cards (5e7 per card: may pass the
#               40 GB card; the row records the failure)
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths below.
# One process per row. Resumable: a row whose JSON exists is skipped.
#
#   WHAT=share sbatch bench/slurm/hh/hybrid_clusters_runs.sbatch                 # every share row (~2-3 h)
#   WHAT=mesh sbatch bench/slurm/hh/hybrid_clusters_runs.sbatch                  # every mesh row (~3-4 h)
#   WHAT=mesh ROWS='^mesh4_' sbatch bench/slurm/hh/hybrid_clusters_runs.sbatch   # the rows whose tag matches a regex
#
# share rows: share_<N>_K<K>_<M>[_<coupling>] (coupling fmm unless named; "none" attributes the slow force's
#   cluster-cluster part). Per row: the slow force, the internal derivatives, the fixed and adaptive flows and
#   one step of each scheme (median of 3 after a 1.5 s warm-up), the internal pairs, share_fixed16 / share_adaptive.
# mesh rows: card1_* (one card, the single-card scheme), mesh<C>_* (C cards, the mesh scheme); 3 segments x 5 steps,
#   stripping on, cross + coupling fmm, reserve 4096 per device. Per row: every edge (wall, stripped, re-plans,
#   owners, rows per device, drifts), the start's wall (the force's prepare + the start force), the first segment's
#   wall (its compiles), the later segments' wall per step (the steady step) and every card's peak.
#
# On HoreKa this runs as jobs "hh4a" and "hh4c" of the bundle (bench/slurm/h0/README.md: one checkout, one setup,
# every job).
#
#SBATCH --job-name=jaccpot-hh4-hybrid-clusters
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), yggdrax, nornax (main e9097126 or later) and THIS jaccpot checkout
PY=${PY:-python}
JACCPOT_HH=${JACCPOT_HH:-$PWD}
WHAT=${WHAT:?WHAT=share or WHAT=mesh}
OUT=${OUT:-${WORKSPACE:-$HOME}/hh4-$WHAT-$(date +%Y%m%d)}
ROWS=${ROWS:-.}
ROW_TIMEOUT=${ROW_TIMEOUT:-5400}
# SMOKE_N / SMOKE_K / SMOKE_M: every row at this size, SMOKE_ARGS appended (bench/slurm/h0/submit_all.sh SMOKE=1)
nn() { echo "${SMOKE_N:-$1}"; }
kk() { echo "${SMOKE_K:-$1}"; }
mm() { echo "${SMOKE_M:-$1}"; }

CARDS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# the stream-ordered allocator: HH3a's 1e8 slow program and HH4c's mesh runs passed with it (BFC pools fragmented)
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_experimental_enable_fusion_autotuner=false"

echo "node $(hostname) cards $CARDS"
nvidia-smi || true
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$JACCPOT_HH" rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true
"$PY" -c "import nornax, os; print(os.path.dirname(nornax.__file__))" > "$OUT/nornax_path.txt" 2>&1 || true
F="$JACCPOT_HH/bench/hybrid_clusters_runs.py"

ROW_LIST=()  # tag|cards|args
if [ "$WHAT" = share ]; then
  for ng in "1e6 1000000" "1e8 100000000"; do
    read -r NT NV <<< "$ng"
    if [ "$NT" = 1e6 ]; then
      KMS=("1 1e3 1000 fmm" "10 1e3 1000 fmm" "1 1e4 10000 fmm" "1 1e5 100000 fmm" "1 1e5 100000 none")
    else
      KMS=("1 1e5 100000 fmm" "1 1e5 100000 none" "150 1e3 1000 fmm")
    fi
    for kmc in "${KMS[@]}"; do
      read -r K MT MV C <<< "$kmc"
      tag="share_${NT}_K${K}_${MT}"
      [ "$MT" = 1e5 ] && tag="${tag}_$C"
      ROW_LIST+=("$tag|1|share --n $(nn "$NV") --k $(kk "$K") --m $(mm "$MV") --coupling $C")
    done
  done
elif [ "$WHAT" = mesh ]; then
  K10="--k $(kk 10) --m $(mm 10000)"
  K150="--k $(kk 150) --m $(mm 1000)"
  ROW_LIST+=("card1_1e7_K10_1e4|1|mesh --arm card --cards 1 --n $(nn 10000000) $K10")
  ROW_LIST+=("mesh1_1e7_K10_1e4|1|mesh --arm mesh --cards 1 --n $(nn 10000000) $K10")
  ROW_LIST+=("mesh2_1e7_K10_1e4|2|mesh --arm mesh --cards 2 --n $(nn 10000000) $K10")
  ROW_LIST+=("mesh4_1e7_K10_1e4|4|mesh --arm mesh --cards 4 --n $(nn 10000000) $K10")
  ROW_LIST+=("mesh2_2e7_K10_1e4|2|mesh --arm mesh --cards 2 --n $(nn 20000000) $K10")
  ROW_LIST+=("mesh4_4e7_K10_1e4|4|mesh --arm mesh --cards 4 --n $(nn 40000000) $K10")
  # production: 1e8 + 150 x 1e3; slots per device ~2 K / ndev (HH4c open issue 2; overflow grows them)
  ROW_LIST+=("mesh4_1e8_K150_1e3|4|mesh --arm mesh --cards 4 --n $(nn 100000000) $K150 --slots ${SMOKE_K:-75}")
  ROW_LIST+=("mesh2_1e8_K150_1e3|2|mesh --arm mesh --cards 2 --n $(nn 100000000) $K150 --slots ${SMOKE_K:-150}")
  ROW_LIST+=("mesh4_2e8_K150_1e3|4|mesh --arm mesh --cards 4 --n $(nn 200000000) $K150 --slots ${SMOKE_K:-75}")
else
  echo "WHAT=$WHAT: share or mesh" >&2
  exit 2
fi

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag ncards args <<< "$row"
  [[ "$tag" =~ $ROWS ]] || continue
  [ -f "$OUT/rows/$tag.json" ] && { echo "skip $tag"; continue; }
  # the row's cards: the first ncards of the node's (the others stay idle and unallocated)
  cvd=$(tr ',' '\n' <<< "$CARDS" | head -n "$ncards" | paste -sd,)
  xf=$XLA_FLAGS
  # the CPU smoke: forced host devices stand in for the cards
  [ "${H0_SMOKE_CPU:-}" = 1 ] && xf="$xf --xla_force_host_platform_device_count=$ncards"
  echo "=== $tag (cards $cvd) $(date -Is)"
  # shellcheck disable=SC2086
  (cd "$OUT" && CUDA_VISIBLE_DEVICES=$cvd XLA_FLAGS=$xf timeout "$ROW_TIMEOUT" $PY "$F" $args ${SMOKE_ARGS:-} \
    --out "$OUT/rows/$tag.json") > "$OUT/logs/$tag.log" 2>&1 || echo "FAILED $tag rc=$? (see $OUT/logs/$tag.log)"
done

# summary
$PY - "$OUT/rows" "$WHAT" <<'PYEOF'
import json, sys, pathlib
rows = {p.stem: json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))}
gib = lambda v: f"{v / 2**30:6.2f}" if isinstance(v, (int, float)) else "  -   "
if sys.argv[2] == "share":
    print(f"{'row':26s} {'slow s':>8s} {'step16 s':>9s} {'stepad s':>9s} {'share16':>8s} {'shareAd':>8s} "
          f"{'pairs16':>9s} {'pairsAd':>9s} {'peak GiB':>8s}")
    for tag, d in rows.items():
        print(f"{tag:26s} {d['t_slow_s']:8.3f} {d['t_step_fixed16_s']:9.3f} {d['t_step_adaptive_s']:9.3f} "
              f"{d['share_fixed16']:8.3f} {d['share_adaptive']:8.3f} {d['pairs_fixed16_per_step']:9.2e} "
              f"{d['pairs_adaptive_per_step']:9.2e} {gib(d['peak_bytes'][0])}")
else:
    card = rows.get("card1_1e7_K10_1e4")
    print(f"{'row':22s} {'start s':>8s} {'seg1 s':>8s} {'step s':>8s} {'vs card1':>8s} {'stripped':>8s} {'replans':>8s}  peak GiB per card")
    for tag, d in rows.items():
        st = d.get("steady_step_s")
        rel = f"{st / card['steady_step_s']:8.2f}" if card and st and card.get("steady_step_s") and "_1e7_" in tag else f"{'':8s}"
        rp = d["edges"][-1].get("replans") if d.get("edges") else None
        print(f"{tag:22s} {d.get('start_wall_s') or 0:8.1f} {d.get('first_segment_wall_s') or 0:8.1f} {st or 0:8.3f} {rel} {d.get('n_stripped_end')!s:>8s} "
              f"{rp!s:>8s}  {' '.join(gib(v) for v in d.get('peak_bytes', []))}")
PYEOF
