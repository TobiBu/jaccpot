#!/bin/bash
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/bit; K=/export/home/tbuck/jaccpot-r6-k6
echo "ab4 start $(date +%H:%M)"
for arm in I6 T6; do
  case $arm in T*) E="JACCPOT_STRICT_CARRY_ORDER=tree";; *) E="";; esac
  ours $K ${arm}_2000000 2000000 stepsave --env "$E" --save-forces $O/${arm}_2000000.npz --steps 4 --reps 1 --eval-repeats 1
done
python3 - $O <<'PY'
import numpy as np, sys
o = sys.argv[1]
a, b = (np.load(f"{o}/{k}_2000000.npz")["state"] for k in ("I6", "T6"))
print(f"k6 N=2e6: tree vs input bitwise {np.array_equal(a, b)} rows differing {int(np.sum(np.any(a != b, axis=(1, 2))))}", flush=True)
PY
ours $K T6_100000000_s 100000000 step --env "JACCPOT_STRICT_CARRY_ORDER=tree"
ours $K I6_100000000_s 100000000 step
echo "ab4 done $(date +%H:%M)"
