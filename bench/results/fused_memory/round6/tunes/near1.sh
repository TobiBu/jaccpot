#!/bin/bash
# Steps 1/2b: capture near-field + walk inputs (8e6, 1e8) on the round-6 tree, then time variants alone.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
W=/export/home/tbuck/jaccpot-r6-k1
echo "near1 start $(date +%H:%M)"
V="32:0:1 16:0:1 16:16:1 16:16:1:p 16:16:1:a 16:16:1:ap 16:16:1:r 16:16:1:ar 32:16:1 32:16:1:a 32:32:1:a 32:32:2:a 32:32:2:ar 16:32:1:ar 16:32:2:ar 32:16:2:ar 16:8:1:a 8:16:1:a 8:32:1:ar 8:64:2:ar 16:64:2:ar 32:32:4 64:16:2:a 32:0:1"
WV="soa:64:2:2048 record:64:2:2048 soa:64:2:2048:1 record:64:2:2048:1 record:128:4:2048:1 record:32:1:4096:1 record:64:2:4096:1 record:64:4:2048:1 soa:64:2:2048"
capture $W c8e6 8000000
tune $W c8e6 t8e6 $V
wtune $W c8e6 w8e6 $WV
capture $W c1e8 100000000
tune $W c1e8 t1e8 $V
wtune $W c1e8 w1e8 $WV
echo "near1 done $(date +%H:%M)"
