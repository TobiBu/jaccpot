#!/bin/bash
# after chain2: L2P Pallas arm -- accuracy at 8e6 (force mode) and step A/B at 8e6
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain2 done" $SP/r7/chain2.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
A2=/export/home/tbuck/jaccpot-r7-a2
Y0=/export/home/tbuck/yggdrax-r6-main
ours $B acc_B_8e6 8000000 force
ours $A2 acc_L_8e6 8000000 force --env "JACCPOT_L2P_KERNEL=pallas"
for r in 1 2; do
  ours $B ab_B2_8e6_$r 8000000 step
  ours $A2 ab_L_8e6_$r 8000000 step --env "JACCPOT_L2P_KERNEL=pallas"
done
echo "chain3 done $(date +%H:%M:%S)"
