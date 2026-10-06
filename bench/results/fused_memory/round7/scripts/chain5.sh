#!/bin/bash
# after chain4: scatter un-permute -- bitwise at 2e6 vs B1, A/B at 8e6 (B, G barrier, S scatter)
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
until grep -q "^chain4 done" $SP/r7/chain4.out 2>/dev/null; do sleep 15; done
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
A1=/export/home/tbuck/jaccpot-r7-a1
A4=/export/home/tbuck/jaccpot-r7-a4
OB=$SP/r7/bit
O=$OB ours $A4 bit_S 2000000 stepsave --save-forces $OB/S.npz --steps 4 --reps 1 --eval-repeats 1
python3 - $OB <<'PY'
import numpy as np, sys
o = sys.argv[1]
a, b = np.load(f"{o}/B1.npz")["state"], np.load(f"{o}/S.npz")["state"]
print(f"bitwise B1 vs S: {np.array_equal(a, b)} max {float(np.max(np.abs(a - b))):.3e}", flush=True)
PY
for r in 1 2; do
  ours $B ab_B4_8e6_$r 8000000 step
  ours $A1 ab_G2_8e6_$r 8000000 step
  ours $A4 ab_S_8e6_$r 8000000 step
done
echo "chain5 done $(date +%H:%M:%S)"
