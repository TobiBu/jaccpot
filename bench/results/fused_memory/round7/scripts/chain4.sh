#!/bin/bash
# after chain3: interleaved list tune over slices; step A/B with JACCPOT_LIST_CSR_SLICES at 8e6
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain3 done" $SP/r7/chain3.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
A3=/export/home/tbuck/jaccpot-r7-a3
Y0=/export/home/tbuck/yggdrax-r6-main
run_py() {
  local jt=$1 yg=$2; shift 2
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$yg CUDA_VISIBLE_DEVICES=$CARD timeout 1800 $PY "$@" 2>&1 | filt)
  release
}
for k in 0 1; do
  echo "== lists tune (interleaved) c8_lists$k"
  run_py $A3 $Y0 bench/lists_tune.py $SP/r7/cap/c8_lists$k.npz --rounds 3 --reps 5 --variants base slices=2 slices=3 slices=4 slices=6 relaxed=1 slices=2,relaxed=1 slices=4,relaxed=1
done
for r in 1 2; do
  ours $B ab_B3_8e6_$r 8000000 step
  ours $B ab_S2_8e6_$r 8000000 step --env "JACCPOT_LIST_CSR_SLICES=2"
  ours $B ab_S4_8e6_$r 8000000 step --env "JACCPOT_LIST_CSR_SLICES=4"
done
echo "chain4 done $(date +%H:%M:%S)"
