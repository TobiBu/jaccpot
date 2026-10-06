#!/bin/bash
# Extension: each side's fast/accurate ends (jz-fmm theta 0.9; jaccpot cml6 x {p6, theta 0.7}, cml5/cml4 at 8e6)
source /tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad/r5/common.sh
for N in 8000000 32000000 100000000; do
  echo "== N=$N $(date +%H:%M)"
  jz   jzD_$N  $N 32 5 0.9 64
  ours ouE_${N}_f $N force --cell-min-level 6 --order 6
  ours ouE_${N}_s $N step  --cell-min-level 6 --order 6
  jz   jzE_$N  $N 32 6 0.9 64
  ours ouF_${N}_f $N force --cell-min-level 6 --theta 0.7
  ours ouF_${N}_s $N step  --cell-min-level 6 --theta 0.7
  if [ $N = 8000000 ]; then
    ours ouG_${N}_f $N force --cell-min-level 5
    ours ouG_${N}_s $N step  --cell-min-level 5
    ours ouH_${N}_f $N force --cell-min-level 4
    ours ouH_${N}_s $N step  --cell-min-level 4
  fi
done
echo "extension done $(date +%H:%M)"
