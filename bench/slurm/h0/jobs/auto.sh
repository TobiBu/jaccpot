#!/bin/bash
# H0 job "auto" (~1.5 h): jaccpot's opt-in cell_min_level="auto" against the preset's fixed level 8, on
# mimirax's S10 fused bench (bench/s10_fused_shooting.py --task timing: a self-gravitating disc in nornax's
# MN + NFW potential, preset "galaxy", float32, dehnen_error, dt 0.01, 2 segments of 50 steps), one card:
#   pre_<N>_fc1 / pre_<N>_fc1_auto        a warm pre-pass (gradient_plan with freeze_cache) at level 8 / auto
#   seg_<N>_adj / seg_<N>_auto_adj        a segment's value-and-gradient program (start force in the adjoint)
# at N = 1e5, 1e6, 1e7 (S10 follow-ups: the rows that decide "auto" as the galaxy preset's default; mimirax's
# own sbatch has 1e5 / 1e6, 1e7 added here). Each pair runs A B B A (tags ..._r1, ..._r2) so a drift of the
# node cancels. The auto level the bench picks is in each row's JSON.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight auto
export CUDA_VISIBLE_DEVICES=$FIRST_CARD
S=$H0_MIMIRAX/bench/s10_fused_shooting.py
BFC="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
SIZES="1e5 1e6 1e7"
REPS=${REPS:-5}
L=50
if [ "${SMOKE:-}" = 1 ]; then SIZES="smoke" REPS=1; export H0_SMOKE_INTERPRET=all; fi
mkdir -p "$OUT/rows" "$OUT/logs"
git -C "$H0_MIMIRAX" rev-parse HEAD > "$OUT/mimirax_commit.txt" 2>/dev/null || true

row() {  # tag args...
  local tag=$1; shift
  local json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; return; fi
  # the smoke runs each argument pattern once: an r2 row repeats its r1's arguments (an XLA:CPU gradient
  # compile of 1-2 h and ~240 GB each on a loaded host)
  if [ "${SMOKE:-}" = 1 ] && [[ $tag == *_r2 ]]; then echo "skip $tag (smoke: same arguments as r1)"; return; fi
  echo "=== $tag  $(date -Is)"
  local start=$SECONDS
  # shellcheck disable=SC2086
  if env $BFC "$H0_PY" "$S" "$@" --out "$json" > "$OUT/logs/$tag.log" 2>&1; then status=ok; else status="FAILED rc=$?"; fi
  echo "    $status in $((SECONDS - start)) s (log $OUT/logs/$tag.log)"
}
for n in $SIZES; do
  case $n in 1e5) N=100000 ;; 1e6) N=1000000 ;; 1e7) N=10000000 ;; smoke) N=2048 ;; esac
  base="--task timing --n $N --steps $L --reps $REPS --segments 2"
  pre="$base --only prepass"
  seg="$base --only seggrad --start-force adjoint"
  auto="--cell-min-level auto"
  # A B B A
  row "pre_${n}_fc1_r1" $pre
  row "pre_${n}_fc1_auto_r1" $pre $auto
  row "pre_${n}_fc1_auto_r2" $pre $auto
  row "pre_${n}_fc1_r2" $pre
  row "seg_${n}_adj_r1" $seg
  row "seg_${n}_auto_adj_r1" $seg $auto
  row "seg_${n}_auto_adj_r2" $seg $auto
  row "seg_${n}_adj_r2" $seg
done
"$H0_PY" "$H0_MIMIRAX/bench/slurm/s10_shooting_summary.py" "$OUT/rows" | tee "$OUT/summary.txt" || true
h0_done
