#!/bin/bash
# Chains B-E after chain A (waits on its done line; never pgrep).
source /tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime/common.sh
until grep -q "^chainA done" $SP/chainA.out 2>/dev/null; do sleep 30; done
echo "chainB start $(date) card $CARD"
OPYPATH=$SITE:/export/home/tbuck/Odisseo-review-adca4f6:/export/home/tbuck/nornax-dce-wt:/export/scratch/tbuck/odisseo-overlay
# --- B: production 25M disc+bulge (main), ferrers3 at eps 1.5e-3, p6 theta 0.8 leaf 64 cells
D="--ic disc --n 25000000 --softening 1.5e-3 --softening-kernel ferrers3"
ours $MAIN $YMAIN D_geo_s 25000000 step $D
ours $MAIN $YMAIN D_geo_f 25000000 force $D
ours $MAIN $YMAIN D_de4_s 25000000 step $D --mac-type dehnen_error --adaptive-eps 1e-4
ours $MAIN $YMAIN D_de4_f 25000000 force $D --mac-type dehnen_error --adaptive-eps 1e-4
ours $MAIN $YMAIN D_de5_s 25000000 step $D --mac-type dehnen_error --adaptive-eps 1e-5
# --- C: Odisseo integrate() at 8e6 clipped Plummer (eps 1e-7, default ferrers3)
odis() {  # tag args...
  local tag=$1; shift; waitcard; local t0=$(date +%s)
  (cd $SP && PYTHONPATH=$OPYPATH JACCPOT_WORKTREE=$MAIN YGGDRAX_WORKTREE=$YMAIN CUDA_VISIBLE_DEVICES=$CARD \
    timeout 3600 $PY odisseo_time.py --out $O/$tag.json "$@" 2>&1 | filt > $O/$tag.log)
  echo "[$(date +%H:%M:%S)] $tag wall $(( $(date +%s) - t0 )) s"; grep "odisseo_time\]" $O/$tag.log | tail -6
  grep -i "error\|Traceback" $O/$tag.log | tail -3
}
odis O_def_f64 --n 8000000
odis O_tun_f64 --n 8000000 --leaf 64 --order 6 --theta 0.8
odis O_tun_f32 --n 8000000 --leaf 64 --order 6 --theta 0.8 --state-dtype float32
odis O_tun_hist --n 8000000 --leaf 64 --order 6 --theta 0.8 --history --skip-c
# --- D: gradients (main), grad_step_profile operating point (cells64 theta 0.8 p5 fp32)
grad() {  # tag n
  local tag=$1 n=$2; waitcard; local t0=$(date +%s)
  (cd $MAIN && PYTHONPATH=$SITE JACCPOT_WORKTREE=$MAIN YGGDRAX_WORKTREE=$YMAIN JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$CARD \
    timeout 3600 $PY $SP/grad_wrap.py $O/${tag}_peak.json --n $n --no-trace --skip-eager-grad --allow-busy --reps 5 \
    --tag $tag --out $O/$tag.json 2>&1 | filt > $O/$tag.log)
  echo "[$(date +%H:%M:%S)] $tag wall $(( $(date +%s) - t0 )) s"; grep -i "ms\|grad_wrap\|error" $O/$tag.log | tail -14
}
grad G_2e5 200000
grad G_1e6 1000000
grad G_8e6 8000000
# --- E: block-step (Odisseo jitted device lane), fp64, k_max 3
bs() {  # tag args...
  local tag=$1; shift; waitcard; local t0=$(date +%s)
  (cd $SP && PYTHONPATH=$OPYPATH JACCPOT_WORKTREE=$MAIN YGGDRAX_WORKTREE=$YMAIN CUDA_VISIBLE_DEVICES=$CARD \
    timeout 5400 $PY blockstep_time.py --out $O/$tag.json "$@" 2>&1 | filt > $O/$tag.log)
  echo "[$(date +%H:%M:%S)] $tag wall $(( $(date +%s) - t0 )) s"; grep "\[bs\]" $O/$tag.log | tail -4
  grep -i "error\|Traceback" $O/$tag.log | tail -3
}
bs BS_1e5 --n 100000
bs BS_1e6 --n 1000000 --extra 2
echo "chainB done $(date +%H:%M:%S)"
