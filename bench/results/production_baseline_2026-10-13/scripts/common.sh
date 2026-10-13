# Re-timing helpers (2026-10-13). One configuration per process on CARD, booked via autocvd.
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/d2b6dfa5-c9ef-4494-93c4-f91e320f46c5/scratchpad/retime
CARD=${CARD:-7}
export BENCH_DIR=/export/home/tbuck/Odisseo-bench-jint-wt/benchmark_multigpu
SITE=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt
PY=/export/scratch/tbuck/jax0112-venv/bin/python
MAIN=/export/home/tbuck/jaccpot-review-5bc4959
YMAIN=/export/home/tbuck/yggdrax-review-2758fa9
R7=/export/home/tbuck/jaccpot-r7-a10
YR7=/export/home/tbuck/yggdrax-r7-y1
O=${O:-$SP/rows}; mkdir -p $O
filt() { grep -v "cuda_executor\|^E10\|Deprecation\|return FastMulti\|sitecustom\|Source Location\|external/xla"; }

# wait (max 30 min) until no foreign compute process sits on CARD; log what was seen
waitcard() {
  local uuid; uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c=$CARD '$1==c{print $2}')
  for i in $(seq 1 180); do
    local n; n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid")
    [ "$n" = 0 ] && return 0
    [ $i = 1 ] && echo "[waitcard] $(date +%H:%M:%S) foreign process on card $CARD: $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep $uuid)"
    sleep 10
  done
  echo "[waitcard] still busy after 30 min; running anyway"
}

summ() {  # json tag
  python3 - "$1" "$2" <<'PY' || echo "$2: no row"
import json, sys
d = json.load(open(sys.argv[1])); G = 2**30; n = d["n"]
acc = d.get("accuracy", {}); ev = d.get("eval_ms", {}); st = d.get("step_ms", {})
pk = max([e["peak"] for e in d.get("prepare_events", [])] + [0]) / G
fin = d.get("final_peak_gib") or d.get("peak_after_scan_gib") or d.get("peak_after_eval_gib") or pk
c = d.get("scan_contention") or d.get("eval_contention") or {}
s = f"{sys.argv[2]}: "
if st: s += f"step {st['min']:.1f} ms (median {st['median']:.1f}) "
if ev: s += f"eval-only {ev['min']:.1f} ms (median {ev['median']:.1f}) "
if acc: s += f"rel-L2 {acc['rel_l2']:.3e} p90 {acc['p90']:.3e} "
s += f"peak {fin:.2f} GiB ({fin*G/n:.0f} B/p) far {d.get('counts', {}).get('far_pair_count')} "
s += f"load {c.get('loadavg1_max', float('nan')):.0f} foreign {c.get('foreign_pids_on_devices')} contaminated {c.get('contaminated')}"
print(s, flush=True)
PY
}

# ours JT YGG tag n mode(step|force) [--env "A=1 B=2"] extra...
ours() {
  local jt=$1 yg=$2 tag=$3 n=$4 mode=$5; shift 5
  local env="JACCPOT_STRICT_CARRY=particles"
  if [ "$1" = "--env" ]; then env="$env $2"; shift 2; fi
  local args
  if [ $mode = force ]; then args="--no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5"
  else args="--skip-eval --steps 4 --reps 3 --no-analysis"; fi
  waitcard
  local t0=$(date +%s)
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$yg JAX_ENABLE_X64=1 \
    CUDA_VISIBLE_DEVICES=$CARD timeout 3000 $PY bench/fused_memory_budget.py --n $n \
    --caps unnamed --prealloc 0.88 $args "$@" --env $env \
    --out $O/$tag.json 2>&1 | filt > $O/$tag.log)
  echo "[$(date +%H:%M:%S)] $tag wall $(( $(date +%s) - t0 )) s" 
  summ $O/$tag.json $tag
}
