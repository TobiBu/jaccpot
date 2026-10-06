#!/bin/bash
# tree-order carry vs input-order carry: final states of the same sequence, bitwise
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/bit; mkdir -p $O
K=/export/home/tbuck/jaccpot-r6-k3
WE="JACCPOT_WALK_NODE_LAYOUT=record JACCPOT_WALK_FUSED_EMIT=1"
for N in 2000000 8000000; do
  for arm in I1 T1 I2; do
    case $arm in T*) E="$WE JACCPOT_STRICT_CARRY_ORDER=tree";; *) E="$WE";; esac
    ours $K ${arm}_$N $N stepsave --env "$E" --save-forces $O/${arm}_$N.npz --steps 4 --reps 1 --eval-repeats 1
  done
  python3 - $O $N <<'PY'
import numpy as np, sys
o, n = sys.argv[1], sys.argv[2]
a, b, c = (np.load(f"{o}/{k}_{n}.npz")["state"] for k in ("I1", "T1", "I2"))
d = lambda x, y: float(np.max(np.abs(x - y)))
print(f"N={n}: input vs input bitwise {np.array_equal(a, c)} (max {d(a, c):.3e}); "
      f"tree vs input bitwise {np.array_equal(a, b)} (max {d(a, b):.3e}, rows differing {int(np.sum(np.any(a != b, axis=(1, 2))))})", flush=True)
PY
done
