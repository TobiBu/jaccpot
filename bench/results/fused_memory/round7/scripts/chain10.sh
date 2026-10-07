#!/bin/bash
# after chain9: the bisection lookup (a10) against a8 (repeat) and a9 (compare) at 8e6, and at 1e8
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain9 done" $SP/r7/chain9.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
C=/export/home/tbuck/jaccpot-r7-a8
C2=/export/home/tbuck/jaccpot-r7-a9
C3=/export/home/tbuck/jaccpot-r7-a10
Y1=/export/home/tbuck/yggdrax-r7-y1
OB=$SP/r7/bit
YGG=$Y1 O=$OB ours $C3 bit_C3 2000000 stepsave --save-forces $OB/C3.npz --steps 4 --reps 1 --eval-repeats 1
python3 - $OB <<'PY'
import numpy as np, sys
o = sys.argv[1]
a, b = np.load(f"{o}/C.npz")["state"], np.load(f"{o}/C3.npz")["state"]
print(f"bitwise C vs C3: {np.array_equal(a, b)} max {float(np.max(np.abs(a - b))):.3e}", flush=True)
PY
O=$SP/r7/final
for r in 3 4; do
  YGG=$Y1 ours $C C_8e6_t$r 8000000 step
  YGG=$Y1 ours $C3 C3_8e6_t$r 8000000 step
  YGG=$Y1 ours $C2 C2_8e6_t$r 8000000 step
done
YGG=$Y1 ours $C3 C3_1e8_s 100000000 step
echo "chain10 done $(date +%H:%M:%S)"
