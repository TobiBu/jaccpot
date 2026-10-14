#!/bin/bash
# after chainF: the 8e6 gradient row with the reference block fixed (S0 branch's grad_step_profile.py)
source /tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime/common.sh
until grep -q "^chainF done" $SP/chainF.out 2>/dev/null; do sleep 30; done
S0=/export/home/tbuck/jaccpot-s0-wt
echo "chainF2 start $(date)"
waitcard
(cd $S0 && PYTHONPATH=$SITE JACCPOT_WORKTREE=$S0 YGGDRAX_WORKTREE=$YMAIN JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$CARD \
  timeout 3600 $PY $SP/grad_wrap.py $O/G_8e6_peak.json --n 8000000 --no-trace --skip-eager-grad --allow-busy --reps 5 \
  --tag G_8e6 --out $O/G_8e6.json 2>&1 | filt > $O/G_8e6.log)
grep -i "jit:\|grad_wrap\|error" $O/G_8e6.log | tail -8
echo "chainF2 done $(date +%H:%M:%S)"
