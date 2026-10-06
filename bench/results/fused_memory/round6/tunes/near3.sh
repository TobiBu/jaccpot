#!/bin/bash
# Step 1c: 2D-indexed tile operands (flag g), frozen k3.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
W=/export/home/tbuck/jaccpot-r6-k3
echo "near3 start $(date +%H:%M)"
V="32:0:1 16:8:1:al 16:8:1:alg 16:16:1:alg 32:8:1:alg 16:4:1:alg 8:8:1:alg 32:16:1:alg 32:8:1:alg:4-8-16-32-64 32:8:1:alg:8-16-32-64 32:4:1:alg:4-8-16-32-64 32:16:1:alg:8-16-32-64 32:8:2:alg:16-32-64 16:8:1:algr 32:0:1"
tune $W c8e6 t3_8e6 $V
tune $W c1e8 t3_1e8 $V
echo "near3 done $(date +%H:%M)"
