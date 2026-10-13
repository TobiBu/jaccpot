#!/bin/bash
# follow-ups after chainS0: the Odisseo history-path row (script fixed for SnapshotData)
source /tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime/common.sh
until grep -q "^chainS0 done" $SP/chainS0.out 2>/dev/null; do sleep 30; done
OPYPATH=$SITE:/export/home/tbuck/Odisseo-review-adca4f6:/export/home/tbuck/nornax-dce-wt:/export/scratch/tbuck/odisseo-overlay
echo "chainF start $(date)"
waitcard
(cd $SP && PYTHONPATH=$OPYPATH JACCPOT_WORKTREE=$MAIN YGGDRAX_WORKTREE=$YMAIN CUDA_VISIBLE_DEVICES=$CARD \
  timeout 3600 $PY odisseo_time.py --out $O/O_tun_hist.json --n 8000000 --leaf 64 --order 6 --theta 0.8 --history 2>&1 \
  | filt > $O/O_tun_hist.log)
grep "odisseo_time\]\|Error" $O/O_tun_hist.log | tail -6
echo "chainF done $(date +%H:%M:%S)"
