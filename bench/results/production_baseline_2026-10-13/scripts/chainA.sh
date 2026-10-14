#!/bin/bash
# Chain A: round-7 code (a10 + y1) vs main at the round-7 configuration; main also at the default kernel.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime/common.sh
echo "chainA start $(date) card $CARD"
for r in 1 2; do
  ours $R7 $YR7 R7_8e6_s$r 8000000 step --ic plummer_clipped
  ours $MAIN $YMAIN Mp_8e6_s$r 8000000 step --ic plummer_clipped --softening-kernel plummer
  ours $MAIN $YMAIN Mf_8e6_s$r 8000000 step --ic plummer_clipped
done
ours $R7 $YR7 R7_8e6_f 8000000 force --ic plummer_clipped
ours $MAIN $YMAIN Mp_8e6_f 8000000 force --ic plummer_clipped --softening-kernel plummer
ours $MAIN $YMAIN Mf_8e6_f 8000000 force --ic plummer_clipped
ours $R7 $YR7 R7_1e8_s 100000000 step --ic plummer_clipped
ours $MAIN $YMAIN Mp_1e8_s 100000000 step --ic plummer_clipped --softening-kernel plummer
ours $MAIN $YMAIN Mf_1e8_s 100000000 step --ic plummer_clipped
echo "chainA done $(date +%H:%M:%S)"
