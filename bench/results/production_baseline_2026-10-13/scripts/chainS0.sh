#!/bin/bash
# S0 GPU work after chains B-E: pins (baseline + control), D1 probes, refresh-overhead traces.
source /tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime/common.sh
until grep -q "^chainB done" $SP/chainB.out 2>/dev/null; do sleep 30; done
S0=/export/home/tbuck/jaccpot-s0-wt
OS0=$SP/s0; mkdir -p $OS0
export PYTHONPATH=$SITE JACCPOT_WORKTREE=$S0 YGGDRAX_WORKTREE=$YMAIN JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$CARD
echo "chainS0 start $(date) card $CARD"
# 1. pins: baseline and A-vs-A control
for tag in main-5bc4959 main-5bc4959-b; do
  waitcard
  (cd $S0 && timeout 7200 $PY bench/dce_pins.py record --tag $tag --pins L1,L2,L3,DET1,DET2 2>&1 | filt > $OS0/pins_$tag.log)
  echo "[$(date +%H:%M:%S)] pins $tag: $(tail -1 $OS0/pins_$tag.log)"
done
(cd $S0 && $PY bench/dce_pins.py compare main-5bc4959 main-5bc4959-b --pins L1,L2,L3 --envelope main-5bc4959-b --json $OS0/cmp_L.json > /dev/null 2>&1; echo "L A-vs-A rc=$?")
(cd $S0 && $PY bench/dce_pins.py compare main-5bc4959 main-5bc4959-b --pins DET1,DET2 --json $OS0/cmp_DET.json > /dev/null 2>&1; echo "DET bitwise rc=$?")
# 2. D1 probes
waitcard
(cd $S0 && timeout 3600 $PY bench/blockstep_activity_probe.py $OS0/activity_25m.json --kmax 3,4,5,6 -- \
  --n 25000000 --ic disc --softening 1.5e-3 --softening-kernel ferrers3 --caps unnamed --prealloc 0.88 \
  --no-scan --eval-repeats 1 --no-analysis --out $OS0/activity_25m_bench.json --env JACCPOT_STRICT_CARRY=particles 2>&1 | filt > $OS0/activity_25m.log)
echo "[$(date +%H:%M:%S)] activity probe"; grep "\[probe\]" $OS0/activity_25m.log | tail -6
waitcard
(cd $S0 && timeout 5400 $PY bench/blockstep_frozen_list_probe.py $OS0/frozen_1e6.json --n 1000000 --kmax 3,4,5,6 2>&1 | filt > $OS0/frozen_1e6.log)
echo "[$(date +%H:%M:%S)] frozen-list probe"; grep "\[frozen\]" $OS0/frozen_1e6.log
waitcard
(cd $S0 && timeout 5400 $PY bench/blockstep_hold_vs_max_ab.py $OS0/ab_2e4.json --n 20000 --kmax 3,4 --n-base 32 2>&1 | filt > $OS0/ab_2e4.log)
(cd $S0 && timeout 5400 $PY bench/blockstep_hold_vs_max_ab.py $OS0/ab_2e4_k5.json --n 20000 --kmax 5 --n-base 16 2>&1 | filt > $OS0/ab_2e4_k5.log)
echo "[$(date +%H:%M:%S)] HOLD vs max A/B"; grep "\[ab\]" $OS0/ab_2e4.log $OS0/ab_2e4_k5.log | cut -c1-400
# 3. traces of the refresh overheads (no command buffers, per-kernel attribution)
tr() {  # tag n extra...
  local tag=$1 n=$2; shift 2; waitcard
  (cd $S0 && timeout 3000 $PY bench/fused_memory_budget.py --n $n --caps unnamed --prealloc 0.88 --skip-eval \
    --steps 2 --reps 2 --no-analysis --no-command-buffers --dump-dir $OS0/dump_$tag --dump-re '.*_compiled_runner.*' \
    --trace-dir $OS0/trace_$tag "$@" --env JACCPOT_STRICT_CARRY=particles --out $OS0/$tag.json 2>&1 | filt > $OS0/$tag.log)
  (cd $S0 && $PY bench/analyse_trace_by_stage.py $OS0/trace_$tag $OS0/dump_$tag 2 --module-re '*_compiled_runner*' --kernels 6 > $OS0/stages_$tag.txt 2>&1)
  python3 $SP/stage_kernels.py $OS0/trace_$tag $OS0/dump_$tag 2 --min-ms 0.1 --wide 90 --stack 2 > $OS0/kernels_$tag.txt 2>&1
  echo "[$(date +%H:%M:%S)] trace $tag"; head -25 $OS0/stages_$tag.txt
}
tr plummer_8e6 8000000 --ic plummer_clipped --softening-kernel plummer
tr ferrers3_8e6 8000000 --ic plummer_clipped
D="--ic disc --softening 1.5e-3 --softening-kernel ferrers3"
tr geo_25m 25000000 $D
tr de4_25m 25000000 $D --mac-type dehnen_error --adaptive-eps 1e-4
echo "chainS0 done $(date +%H:%M:%S)"
