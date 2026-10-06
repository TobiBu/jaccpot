#!/bin/bash
# jz-fmm vs jaccpot on ONE card (6), interleaved per N, card held by guard.py between runs.
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad
source $SP/env.sh
O=$SP/r5/rows; mkdir -p $O
export BENCH_DIR=/export/home/tbuck/Odisseo-bench-jint-wt/benchmark_multigpu
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
filt() { grep -v "cuda_executor\|^E10\|Deprecation\|return FastMulti\|sitecustom"; }

own() {  # the guard releases the card, then the run owns it
  echo yield > $SP/r5/gctl
  until grep -q "^yield" $SP/r5/gack 2>/dev/null; do sleep 0.5; done
}
release() { echo hold > $SP/r5/gctl; }

jz() {  # tag n leaf p theta alloc_fac
  local tag=$1 n=$2 leaf=$3 p=$4 th=$5 af=$6
  own
  (cd $BENCH_DIR && CUDA_VISIBLE_DEVICES=6 timeout 2400 $JZPY codes/jzfmm_force_eval.py --n $n \
    --ic plummer_clipped --p $p --theta $th --leaf $leaf --memory --no-trace --no-lists --allow-busy \
    --ref-targets 4096 --alloc-fac-ilist $af --repeats 5 --warmup 2 --prealloc 0.88 \
    --out $O/$tag.json > $O/$tag.log 2>&1)
  release
  python3 - "$O/$tag.json" "$tag" <<'PY' || echo "$tag: no row"
import json, sys
d = json.load(open(sys.argv[1])); r = d["rows"][0]
t, e, m = r.get("timing", {}), r.get("error", {}), r.get("memory", {})
print(f"{sys.argv[2]}: force {1e3*t.get('min', float('nan')):.1f} ms (median {1e3*t.get('median', float('nan')):.1f}) "
      f"aggL2 {e.get('aggL2', float('nan')):.3e} p90 {e.get('p90', float('nan')):.3e} n_ref {e.get('n_ref')} "
      f"peak {m.get('peak_gib', float('nan')):.2f} GiB ({m.get('peak_bytes_per_particle', float('nan')):.0f} B/p)", flush=True)
PY
}

ours() {  # tag n mode(force|step) extra...
  local tag=$1 n=$2 mode=$3; shift 3
  local args
  if [ $mode = force ]; then args="--no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5"
  else args="--skip-eval --steps 4 --reps 3 --no-analysis"; fi
  own
  (cd /export/home/tbuck/jaccpot-r5 && JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-r2-head \
    CUDA_VISIBLE_DEVICES=6 timeout 2400 $PY bench/fused_memory_budget.py --n $n --ic plummer_clipped \
    --caps unnamed --prealloc 0.88 $args "$@" --env JACCPOT_STRICT_CARRY=particles \
    --out $O/$tag.json 2>&1 | filt > $O/$tag.log)
  release
  python3 - "$O/$tag.json" "$tag" <<'PY' || echo "$tag: no row"
import json, sys
d = json.load(open(sys.argv[1])); G = 2**30; n = d["n"]
acc = d.get("accuracy", {}); ev = d.get("eval_ms", {}); st = d.get("step_ms", {})
pk = max([e["peak"] for e in d.get("prepare_events", [])] + [0]) / G
fin = d.get("final_peak_gib") or d.get("peak_after_scan_gib") or d.get("peak_after_eval_gib") or pk
s = f"{sys.argv[2]}: "
if st: s += f"step {st['min']:.1f} ms (median {st['median']:.1f}) "
if ev: s += f"eval-only {ev['min']:.1f} ms "
if acc: s += f"rel-L2 {acc['rel_l2']:.3e} p90 {acc['p90']:.3e} n_ref {acc['targets']} "
s += f"peak {fin:.2f} GiB ({fin*G/n:.0f} B/p) far {d.get('counts', {}).get('far_pair_count')}"
print(s, flush=True)
PY
}

for N in 8000000 32000000 100000000; do
  echo "== N=$N $(date +%H:%M)"
  jz   jzA_$N  $N 32 4 0.6 128
  ours ouA_${N}_f $N force
  ours ouA_${N}_s $N step
  jz   jzB_$N  $N 32 5 0.8 64
  ours ouB_${N}_f $N force --cell-min-level 6
  ours ouB_${N}_s $N step  --cell-min-level 6
  jz   jzC_$N  $N 32 4 0.7 64
  ours ouC_${N}_f $N force --theta 0.7
  ours ouC_${N}_s $N step  --theta 0.7
  ours ouD_${N}_f $N force --order 4
  ours ouD_${N}_s $N step  --order 4
done
echo "comparison done $(date +%H:%M)"
# stage profiles of our default: trace + HLO dump in ONE run, no command buffers (kernels attributable)
for N in 8000000 100000000; do
  own
  (cd /export/home/tbuck/jaccpot-r5 && JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=/export/home/tbuck/yggdrax-r2-head \
    CUDA_VISIBLE_DEVICES=6 timeout 2400 $PY bench/fused_memory_budget.py --n $N --ic plummer_clipped \
    --caps unnamed --prealloc 0.88 --skip-eval --steps 2 --reps 2 --no-analysis --no-command-buffers \
    --dump-dir $O/dump_$N --dump-re '.*_compiled_runner.*' \
    --trace-dir $O/trace_$N --env JACCPOT_STRICT_CARRY=particles --out $O/trace_$N.json 2>&1 | filt > $O/trace_$N.log)
  release
  echo "trace $N: $(grep 'scan min' $O/trace_$N.log)"
done
echo "all done $(date +%H:%M)"
