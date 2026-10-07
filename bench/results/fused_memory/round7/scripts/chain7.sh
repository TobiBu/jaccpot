#!/bin/bash
# after chain6: COM radii capture (8e6) + tune of table vs fold block sizes
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain6 done" $SP/r7/chain6.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
A6=/export/home/tbuck/jaccpot-r7-a7
Y0=/export/home/tbuck/yggdrax-r6-main
capture $A6 d8 8000000
own
(cd $A6 && PYTHONPATH=$SITE JACCPOT_WORKTREE=$A6 YGGDRAX_WORKTREE=$Y0 CUDA_VISIBLE_DEVICES=$CARD timeout 1800 $PY bench/com_radii_tune.py $SP/r7/cap/d8_comr.npz --rounds 3 --variants table fold fold:block=32 fold:block=64 table:block=32 chain 2>&1 | filt)
release
echo "chain7 done $(date +%H:%M:%S)"
