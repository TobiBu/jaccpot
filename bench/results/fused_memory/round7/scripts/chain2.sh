#!/bin/bash
# after step0: atomics probe, list tuning on the 8e6 capture, bitwise arms at 2e6, A/B at 8e6
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^step0 done" $SP/r7/step0.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
A1=/export/home/tbuck/jaccpot-r7-a1
Y0=/export/home/tbuck/yggdrax-r6-main
Y1=/export/home/tbuck/yggdrax-r7-y1
run_py() {  # tree ygg script args...
  local jt=$1 yg=$2; shift 2
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$yg CUDA_VISIBLE_DEVICES=$CARD timeout 1800 $PY "$@" 2>&1 | filt)
  release
}
echo "== atomics probe"; run_py $A1 $Y0 $SP/r7/atomic_probe.py
for k in 0 1; do
  echo "== lists tune c8_lists$k"
  run_py $A1 $Y0 bench/lists_tune.py $SP/r7/cap/c8_lists$k.npz --variants base relaxed=1 rank_rows=4 rank_rows=8 rank_rows=16,lanes=16 rank_rows=8,rank_warps=2 relaxed=1,rank_rows=8 slices=2,relaxed=1
done
# bitwise at 2e6: base twice (control), barrier (A1 default), tree (Y1), fold (A1 + env)
O=$SP/r7/bit; mkdir -p $O
for arm in B1 G B2 Y F; do
  case $arm in
    B*) T=$B; YG=$Y0; E="";;
    G) T=$A1; YG=$Y0; E="";;
    Y) T=$B; YG=$Y1; E="";;
    F) T=$A1; YG=$Y0; E="JACCPOT_COM_RADII_VARIANT=fold";;
  esac
  YGG=$YG ours $T bit_$arm 2000000 stepsave --env "$E" --save-forces $O/$arm.npz --steps 4 --reps 1 --eval-repeats 1
done
python3 - $O <<'PY'
import numpy as np, sys
o = sys.argv[1]
s = {k: np.load(f"{o}/{k}.npz")["state"] for k in ("B1", "B2", "G", "Y", "F")}
for k in ("B2", "G", "Y", "F"):
    a, b = s["B1"], s[k]
    print(f"bitwise B1 vs {k}: {np.array_equal(a, b)} max {float(np.max(np.abs(a - b))):.3e}", flush=True)
PY
# A/B at 8e6, interleaved, 2 rounds
for r in 1 2; do
  for arm in B G Y F; do
    case $arm in
      B) T=$B; YG=$Y0; E="";;
      G) T=$A1; YG=$Y0; E="";;
      Y) T=$B; YG=$Y1; E="";;
      F) T=$A1; YG=$Y0; E="JACCPOT_COM_RADII_VARIANT=fold";;
    esac
    YGG=$YG ours $T ab_${arm}_8e6_$r 8000000 step --env "$E"
  done
done
echo "chain2 done $(date +%H:%M:%S)"
