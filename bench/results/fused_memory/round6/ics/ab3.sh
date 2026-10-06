#!/bin/bash
# Other ICs and the long-row case: main (r6-base) vs round-6 defaults (k5). Then traces.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
O=$SP/r6/ics; mkdir -p $O
B=/export/home/tbuck/jaccpot-r6-base
K=/export/home/tbuck/jaccpot-r6-k5
# ours() hardcodes --ic plummer_clipped; a later --ic wins (argparse keeps the last)
echo "ab3 start $(date +%H:%M)"
for arm in B K; do T=$([ $arm = B ] && echo $B || echo $K)
  ours $T ${arm}_unc6_2e6_s 2000000 step --ic plummer --cell-min-level 6
  ours $T ${arm}_unc8_2e6_s 2000000 step --ic plummer
  ours $T ${arm}_unc8_8e6_s 8000000 step --ic plummer
  ours $T ${arm}_disc_8e6_s 8000000 step --ic disc
done
for arm in B K; do T=$([ $arm = B ] && echo $B || echo $K)
  ours $T ${arm}_unc8_2e6_f 2000000 force --ic plummer
  ours $T ${arm}_disc_8e6_f 8000000 force --ic disc
done
# traces of the round-6 defaults + tree carry (kernels attributable: no command buffers)
for N in 8000000 100000000; do
  own
  (cd $K && PYTHONPATH=$SITE JACCPOT_WORKTREE=$K YGGDRAX_WORKTREE=$YGG CUDA_VISIBLE_DEVICES=$CARD timeout 2400 \
    $PY bench/fused_memory_budget.py --n $N --ic plummer_clipped --caps unnamed --prealloc 0.88 --skip-eval \
    --steps 2 --reps 2 --no-analysis --no-command-buffers --dump-dir $O/dump_$N --dump-re '.*_compiled_runner.*' \
    --trace-dir $O/trace_$N --env JACCPOT_STRICT_CARRY=particles JACCPOT_STRICT_CARRY_ORDER=tree \
    --out $O/trace_$N.json 2>&1 | filt > $O/trace_$N.log)
  release
  echo "trace $N: $(grep 'scan min' $O/trace_$N.log)"
  (cd $K && python3 bench/analyse_trace_by_stage.py $O/trace_$N $O/dump_$N 2 --module-re '*_compiled_runner*' --kernels 4 > $O/stages_$N.txt 2>&1)
done
echo "ab3 done $(date +%H:%M)"
