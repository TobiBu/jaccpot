#!/bin/bash
# after chain5: particle-major L2P -- accuracy and step A/B at 8e6
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain5 done" $SP/r7/chain5.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
A5=/export/home/tbuck/jaccpot-r7-a5
ours $A5 acc_P_8e6 8000000 force --env "JACCPOT_L2P_KERNEL=pallas_particle"
for r in 1 2; do
  ours $B ab_B5_8e6_$r 8000000 step
  ours $A5 ab_P_8e6_$r 8000000 step --env "JACCPOT_L2P_KERNEL=pallas_particle"
done
echo "chain6 done $(date +%H:%M:%S)"
