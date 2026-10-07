#!/bin/bash
# the L2P's in-kernel leaf lookup (a9) against a8: bitwise at 2e6, then (after the CPU unit suite) 8e6 + 1e8 timing
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
source $SP/r7/common.sh
C=/export/home/tbuck/jaccpot-r7-a8
C2=/export/home/tbuck/jaccpot-r7-a9
Y1=/export/home/tbuck/yggdrax-r7-y1
OB=$SP/r7/bit
YGG=$Y1 O=$OB ours $C bit_C 2000000 stepsave --save-forces $OB/C.npz --steps 4 --reps 1 --eval-repeats 1
YGG=$Y1 O=$OB ours $C2 bit_C2 2000000 stepsave --save-forces $OB/C2.npz --steps 4 --reps 1 --eval-repeats 1
python3 - $OB <<'PY'
import numpy as np, sys
o = sys.argv[1]
a, b = np.load(f"{o}/C.npz")["state"], np.load(f"{o}/C2.npz")["state"]
print(f"bitwise C vs C2: {np.array_equal(a, b)} max {float(np.max(np.abs(a - b))):.3e}", flush=True)
PY
# wait for the CPU unit suite (host load poisons timings)
until grep -qE "passed|failed|error" <(tail -1 $SP/r7/unit.out) 2>/dev/null; do sleep 20; done
O=$SP/r7/final
for r in 1 2; do
  YGG=$Y1 ours $C C_8e6_t$r 8000000 step
  YGG=$Y1 ours $C2 C2_8e6_t$r 8000000 step
done
YGG=$Y1 ours $C2 C2_1e8_s 100000000 step
echo "chain9 done $(date +%H:%M:%S)"
