#!/bin/bash
# tree-order carry with the barrier (k4) against without (k3) at 1e8: time + peak
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/ab; mkdir -p $O
WE="JACCPOT_WALK_NODE_LAYOUT=record JACCPOT_WALK_FUSED_EMIT=1"
TE="JACCPOT_STRICT_CARRY_ORDER=tree"
NE="JACCPOT_NEARFIELD_SOURCE_TILE=8 JACCPOT_NEARFIELD_SOURCE_FLAGS=alr JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE=16"
echo "ab2 start $(date +%H:%M)"
ours /export/home/tbuck/jaccpot-r6-k4 WTb_100000000_s1 100000000 step --env "$WE $TE"
ours /export/home/tbuck/jaccpot-r6-k4 WTNb_100000000_s1 100000000 step --env "$WE $TE $NE"
ours /export/home/tbuck/jaccpot-r6-k4 WTb_8000000_s1 8000000 step --env "$WE $TE"
echo "ab2 done $(date +%H:%M)"
