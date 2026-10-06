#!/bin/bash
# Step 1b: occupancy classes + lean body (frozen k2), interleaved walk A/B, tree-order carry GPU test.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
W=/export/home/tbuck/jaccpot-r6-k2
echo "near2 start $(date +%H:%M)"
C5=4-8-16-32-64
V="32:0:1 16:8:1:a 16:8:1:al 32:8:1:al:$C5 32:8:1:al:8-16-32-64 32:8:1:al:4-8-16-64 32:16:1:al:$C5 32:4:1:al:$C5 32:8:1:alr:$C5 32:8:1:alp:$C5 32:8:1:a:$C5 32:0:1"
tune $W c8e6 t2_8e6 $V
tune $W c1e8 t2_1e8 $V
WV="soa:64:2:2048 record:64:2:2048:1 record:128:4:2048:1 soa:64:2:2048 record:64:2:2048:1 record:128:4:2048:1 soa:64:2:2048 record:64:2:2048:1 record:128:4:2048:1"
wtune $W c8e6 w2_8e6 $WV
wtune $W c1e8 w2_1e8 $WV
own
(cd $W && PYTHONPATH=$SITE JACCPOT_WORKTREE=$W YGGDRAX_WORKTREE=$YGG CUDA_VISIBLE_DEVICES=$CARD timeout 3000 \
  $PY -m pytest tests/integration/test_strict_run_v2_particle_carry.py -q -p no:cacheprovider -x 2>&1 | tail -15 > $SP/r6/cap/itest_carry.txt)
release
cat $SP/r6/cap/itest_carry.txt
echo "near2 done $(date +%H:%M)"
