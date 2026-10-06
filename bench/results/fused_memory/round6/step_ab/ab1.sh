#!/bin/bash
# Full-step A/B on frozen k3: B = defaults (main's paths), W = walk record + fused emit,
# WT = W + tree-order carry. Interleaved; 8e6 two rounds, 1e8 two rounds.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/ab; mkdir -p $O
K=/export/home/tbuck/jaccpot-r6-k3
WE="JACCPOT_WALK_NODE_LAYOUT=record JACCPOT_WALK_FUSED_EMIT=1"
TE="JACCPOT_STRICT_CARRY_ORDER=tree"
NE="JACCPOT_NEARFIELD_SOURCE_TILE=8 JACCPOT_NEARFIELD_SOURCE_FLAGS=alr JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE=16"
echo "ab1 start $(date +%H:%M)"
ours $K B_8000000_f 8000000 force
ours $K W_8000000_f 8000000 force --env "$WE"
ours $K WN_8000000_f 8000000 force --env "$WE $NE"
for r in 1 2; do
  ours $K B_8000000_s$r 8000000 step
  ours $K W_8000000_s$r 8000000 step --env "$WE"
  ours $K WT_8000000_s$r 8000000 step --env "$WE $TE"
  ours $K WTN_8000000_s$r 8000000 step --env "$WE $TE $NE"
done
ours $K W_100000000_f 100000000 force --env "$WE"
ours $K WN_100000000_f 100000000 force --env "$WE $NE"
for r in 1 2; do
  ours $K B_100000000_s$r 100000000 step
  ours $K W_100000000_s$r 100000000 step --env "$WE"
  ours $K WT_100000000_s$r 100000000 step --env "$WE $TE"
  ours $K WTN_100000000_s$r 100000000 step --env "$WE $TE $NE"
done
echo "ab1 done $(date +%H:%M)"
