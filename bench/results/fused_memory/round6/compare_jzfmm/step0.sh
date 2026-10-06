#!/bin/bash
# Step 0: jz-fmm vs our main at EQUAL accuracy (1e8 then 8e6), interleaved.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
B=/export/home/tbuck/jaccpot-r6-base
echo "card $CARD start $(date +%H:%M)"
for N in 100000000 8000000; do
  echo "== N=$N $(date +%H:%M)"
  jz   jzF_$N $N 32 6 0.8 64
  ours $B ouP_${N}_f $N force
  ours $B ouP_${N}_s $N step
  if [ $N = 100000000 ]; then jz jzG_$N $N 32 5 0.7 64; else jz jzB_$N $N 32 5 0.8 64; fi
  ours $B ouQ_${N}_f $N force --order 5
  ours $B ouQ_${N}_s $N step  --order 5
  if [ $N = 100000000 ]; then jz jzH_$N $N 32 6 0.7 64; fi
done
echo "step0 done $(date +%H:%M)"
