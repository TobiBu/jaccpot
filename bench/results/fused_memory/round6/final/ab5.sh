#!/bin/bash
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/final; mkdir -p $O; K=/export/home/tbuck/jaccpot-r6-k6; B=/export/home/tbuck/jaccpot-r6-base
echo "ab5 start $(date +%H:%M)"
for r in 1 2; do
  ours $K D_8000000_s$r 8000000 step
  ours $K DT_8000000_s$r 8000000 step --env "JACCPOT_STRICT_CARRY_ORDER=tree"
done
for arm in B D DT; do
  case $arm in B) T=$B; E="";; D) T=$K; E="";; DT) T=$K; E="JACCPOT_STRICT_CARRY_ORDER=tree";; esac
  ours $T ${arm}_2000000_s 2000000 step --env "$E"
  ours $T ${arm}_200000_s 200000 step --env "$E"
done
ours $K D_100000000_f 100000000 force
echo "ab5 done $(date +%H:%M)"
