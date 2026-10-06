#!/bin/bash
# Round 5 in the full step: old kernels (P2M per leaf, COM radii chain) vs new, at the new default
# (p6, cell_min_level 8), jax 0.11.2, frozen jaccpot-r5f (596b987) + yggdrax-r2-head. Then CSR slices at 1e8,
# then stage profiles of the new default.
while kill -0 130740 2>/dev/null; do sleep 15; done
source /tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad/r6/common.sh
NEW=/export/scratch/tbuck/jax0112-venv/bin/python
JT=/export/home/tbuck/jaccpot-r5f; YT=/export/home/tbuck/yggdrax-r2-head
O=/tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad/r7/rows; mkdir -p $O
OLDK="--env JACCPOT_STRICT_CARRY=particles JACCPOT_P2M_BLOCK=0 JACCPOT_COM_RADII_VARIANT=chain"  # argparse keeps the LAST --env
for N in 8000000 100000000; do
  echo "== N=$N $(date +%H:%M)"
  for rep in 1 2; do
    run_arm $NEW $JT $YT $O/old_${N}_s$rep.json --n $N --ic plummer_clipped --skip-eval --steps 4 --reps 3 --no-analysis $OLDK
    run_arm $NEW $JT $YT $O/new_${N}_s$rep.json --n $N --ic plummer_clipped --skip-eval --steps 4 --reps 3 --no-analysis
  done
  run_arm $NEW $JT $YT $O/old_${N}_f.json --n $N --ic plummer_clipped --no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5 $OLDK
  run_arm $NEW $JT $YT $O/new_${N}_f.json --n $N --ic plummer_clipped --no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5
done
for S in 2 4 8; do
  run_arm $NEW $JT $YT $O/slices${S}_100000000_s.json --n 100000000 --ic plummer_clipped --skip-eval --steps 4 --reps 3 --no-analysis --env JACCPOT_STRICT_CARRY=particles JACCPOT_LIST_CSR_SLICES=$S
done
echo "A/B done $(date +%H:%M)"
for N in 8000000 100000000; do
  own
  (cd $JT && PYTHONPATH=$SITE JACCPOT_WORKTREE=$JT YGGDRAX_WORKTREE=$YT CUDA_VISIBLE_DEVICES=6 timeout 2400 $NEW bench/fused_memory_budget.py \
    --n $N --ic plummer_clipped --caps unnamed --prealloc 0.88 --skip-eval --steps 2 --reps 2 --no-analysis --no-command-buffers \
    --dump-dir $O/dump_$N --dump-re '.*_compiled_runner.*' --trace-dir $O/trace_$N --env JACCPOT_STRICT_CARRY=particles \
    --out $O/trace_$N.json 2>&1 | filt > $O/trace_$N.log)
  release
  echo "trace $N: $(grep 'scan min' $O/trace_$N.log)"
done
echo "all done $(date +%H:%M)"
