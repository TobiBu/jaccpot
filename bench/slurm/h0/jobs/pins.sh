#!/bin/bash
# H0 job "pins" (~1.5 h): the single-card speed pins on the first card of an exclusive node.
#   L1 8e6 clipped Plummer step, L2 1e8 step, L3 25M disc + bulge step + force rel-L2 (bench/dce_pins.py,
#   production flags: no deterministic ops, fusion autotuner off), with
#   - an A-vs-A control (the same code recorded twice: the envelope), and
#   - the cost of yggdrax #104's extra cummax at 1e8: L2 on yggdrax main (A) against yggdrax main with
#     yggdrax/_cell_partition.py from just before #104 (B), interleaved A B A B A.
# Tags (under $OUT/pins, as /export/scratch/tbuck/dce_pins locally, so `dce_pins.py compare` reads both):
#   h0-L-main (A: L1 L2 L3)  h0-L2-pre104 (B)  h0-L-main-b (A: L1 L2 L3)  h0-L2-pre104-b (B)  h0-L2-main-c (A)
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight pins
export CUDA_VISIBLE_DEVICES=$FIRST_CARD
[ "${SMOKE:-}" = 1 ] && export DCE_PINS_SMOKE_N=2048
ROOT=$OUT/pins
PINS="$H0_PY bench/dce_pins.py"
MAIN_YGG=$H0_YGGDRAX
PRE_YGG=${H0_YGGDRAX_PRE104:-$H0_ROOT/src/yggdrax-pre104}

arm() {  # tag pins yggdrax
  local tag=$1 pins=$2
  H0_YGGDRAX=$3
  h0_paths
  # which _cell_partition this arm imports, and its hash (the B arm must show the pre-#104 file)
  "$H0_PY" -c "import hashlib, yggdrax._cell_partition as c; print(c.__file__, hashlib.sha256(open(c.__file__, 'rb').read()).hexdigest())" \
    | tee "$OUT/arm_$tag.txt"
  if [ -s "$ROOT/$tag/DONE" ]; then echo "skip $tag (done)"; return; fi
  echo "--- $tag [$pins] yggdrax $3 $(date -Is)"
  if $PINS record --pins "$pins" --tag "$tag" --root "$ROOT"; then date -Is > "$ROOT/$tag/DONE"; fi
}
arm h0-L-main L1,L2,L3 "$MAIN_YGG"
arm h0-L2-pre104 L2 "$PRE_YGG"
arm h0-L-main-b L1,L2,L3 "$MAIN_YGG"
arm h0-L2-pre104-b L2 "$PRE_YGG"
arm h0-L2-main-c L2 "$MAIN_YGG"
H0_YGGDRAX=$MAIN_YGG
h0_paths

# the comparisons (dce_pins' own envelope rule: max(3 %, 2x the A-vs-A spread) in time, max(2 %, 2x) in peak)
cmp() {  # name args...
  local name=$1; shift
  echo "--- compare $name: $*" | tee "$OUT/compare_$name.txt"
  $PINS compare "$@" --root "$ROOT" --json "$OUT/compare_$name.json" >> "$OUT/compare_$name.txt" 2>&1 || echo "(compare rc $?)" >> "$OUT/compare_$name.txt"
  cat "$OUT/compare_$name.txt"
}
cmp a_vs_a h0-L-main h0-L-main-b --pins L1,L2,L3
cmp pre104_vs_main h0-L2-pre104 h0-L-main --pins L2 --envelope h0-L2-pre104-b
cmp pre104_vs_main_b h0-L2-pre104-b h0-L-main-b --pins L2 --envelope h0-L2-pre104
cmp main_c h0-L-main h0-L2-main-c --pins L2 --envelope h0-L-main-b
h0_done
