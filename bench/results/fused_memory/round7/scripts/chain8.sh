#!/bin/bash
# combined round-7 defaults (a8 + yggdrax y1) against main: bitwise subset, 8e6 and 1e8 A/B with
# jz-fmm, accuracy, tree-order carry, profiles
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
source $SP/r7/common.sh
B=/export/home/tbuck/jaccpot-r7-base
C=/export/home/tbuck/jaccpot-r7-a8
Y0=/export/home/tbuck/yggdrax-r6-main
Y1=/export/home/tbuck/yggdrax-r7-y1
O=$SP/r7/final; mkdir -p $O
OB=$SP/r7/bit
# the bitwise part of the round (everything but the L2P) against main, 4 steps at 2e6
YGG=$Y1 O=$OB ours $C bit_X 2000000 stepsave --env "JACCPOT_L2P_KERNEL=xla" --save-forces $OB/X.npz --steps 4 --reps 1 --eval-repeats 1
python3 - $OB <<'PY'
import numpy as np, sys
o = sys.argv[1]
a, b = np.load(f"{o}/B1.npz")["state"], np.load(f"{o}/X.npz")["state"]
print(f"bitwise B1 vs X (round 7 without the L2P): {np.array_equal(a, b)} max {float(np.max(np.abs(a - b))):.3e}", flush=True)
PY
# 8e6: accuracy, then interleaved steps
YGG=$Y1 ours $C C_8e6_f 8000000 force
for r in 1 2; do
  YGG=$Y0 ours $B B_8e6_s$r 8000000 step
  YGG=$Y1 ours $C C_8e6_s$r 8000000 step
  YGG=$Y1 ours $C T_8e6_s$r 8000000 step --env "JACCPOT_STRICT_CARRY_ORDER=tree"
  jz jz_8e6_$r 8000000 32 5 0.8 64
done
# profile of the combined defaults at 8e6
YGG=$Y1 ours $C prof_C_8e6 8000000 step --steps 2 --reps 2 --no-command-buffers \
  --dump-dir $O/dump_8e6 --dump-re '.*_compiled_runner.*' --trace-dir $O/trace_8e6
(cd $C && $PY bench/analyse_trace_by_stage.py $O/trace_8e6 $O/dump_8e6 2 --module-re '*_compiled_runner*' --kernels 6 > $O/stages_8e6.txt 2>&1)
python3 $SP/r7/stage_kernels.py $O/trace_8e6 $O/dump_8e6 2 --min-ms 0.1 --wide 90 --stack 2 > $O/kernels_8e6.txt 2>&1
echo "8e6 done $(date +%H:%M:%S)"
# 1e8
YGG=$Y0 ours $B B_1e8_s 100000000 step
YGG=$Y1 ours $C C_1e8_s 100000000 step
YGG=$Y1 ours $C T_1e8_s 100000000 step --env "JACCPOT_STRICT_CARRY_ORDER=tree"
YGG=$Y1 ours $C C_1e8_f 100000000 force
jz jz_1e8 100000000 32 5 0.7 64
YGG=$Y1 ours $C prof_C_1e8 100000000 step --steps 2 --reps 2 --no-command-buffers \
  --dump-dir $O/dump_1e8 --dump-re '.*_compiled_runner.*' --trace-dir $O/trace_1e8
(cd $C && $PY bench/analyse_trace_by_stage.py $O/trace_1e8 $O/dump_1e8 2 --module-re '*_compiled_runner*' --kernels 6 > $O/stages_1e8.txt 2>&1)
python3 $SP/r7/stage_kernels.py $O/trace_1e8 $O/dump_1e8 2 --min-ms 1 --wide 90 --stack 2 > $O/kernels_1e8.txt 2>&1
echo "chain8 done $(date +%H:%M:%S)"
