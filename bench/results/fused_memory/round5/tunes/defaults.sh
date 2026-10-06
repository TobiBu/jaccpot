#!/bin/bash
# (1) cell_min_level / order defaults across ICs and N on jax 0.11.2; (2) jax 0.10.2 vs 0.11.2 A/B.
# Code: frozen jaccpot-r5 (041956c = round 4 + bench cache) and yggdrax-r2-head (2e8f380): the jz-fmm comparison pair.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad/r6/common.sh
NEW=/export/scratch/tbuck/jax0112-venv/bin/python
OLD=/export/home/tbuck/jaccpot/.venv/bin/python
JT=/export/home/tbuck/jaccpot-r5; YT=/export/home/tbuck/yggdrax-r2-head  # both int64 index defaults (pre #361/#84)
echo "== P2M tune $(date +%H:%M)"
own
for N in 8000000 100000000; do for P in 5 6; do
  (cd /export/home/tbuck/jaccpot-r5-wt && PYTHONPATH=$SITE JACCPOT_WORKTREE=/export/home/tbuck/jaccpot-r5-wt \
   YGGDRAX_WORKTREE=$YT CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 1800 $NEW $SP/r6/p2m_tune.py \
   --n $N --cml 6 --order $P --variants 0:0:0 4:16:4 8:16:4 16:16:4 8:8:4 16:8:4 32:8:4 8:16:2 16:16:8 2>&1 | filt)
done; done
release
O=$SP/r6/defaults; mkdir -p $O
for spec in plummer_clipped:200000 plummer_clipped:2000000 plummer:2000000 disc:2000000 plummer:8000000 disc:8000000; do
  ic=${spec%%:*}; n=${spec##*:}
  echo "== $ic $n $(date +%H:%M)"
  for cfg in "A:5:8" "B:5:6" "E:6:6"; do
    IFS=: read tag p cml <<< "$cfg"
    run_arm $NEW $JT $YT $O/${tag}_${ic}_${n}_f.json --n $n --ic $ic --order $p --cell-min-level $cml \
      --no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5
    run_arm $NEW $JT $YT $O/${tag}_${ic}_${n}_s.json --n $n --ic $ic --order $p --cell-min-level $cml \
      --skip-eval --steps 4 --reps 3 --no-analysis
  done
done
echo "defaults done $(date +%H:%M)"
O=$SP/r6/jaxab; mkdir -p $O
for rep in 1 2; do
  for spec in "A:5:8:8000000" "E:6:6:8000000" "E:6:6:100000000"; do
    IFS=: read tag p cml n <<< "$spec"
    for arm in old new; do
      py=$NEW; [ $arm = old ] && py=$OLD
      run_arm $py $JT $YT $O/${arm}_${tag}_${n}_r$rep.json --n $n --ic plummer_clipped --order $p \
        --cell-min-level $cml --skip-eval --steps 4 --reps 3 --no-analysis
    done
  done
done
echo "jax A/B done $(date +%H:%M)"
