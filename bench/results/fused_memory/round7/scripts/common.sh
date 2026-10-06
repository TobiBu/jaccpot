# Round 7 helpers: card $CARD held by guard.py between runs (handshake via $SP/r7/gctl, gack).
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/0f544149-3b57-4277-8c67-aa8871b28ca5/scratchpad
CARD=$(cat $SP/r7/card)
export BENCH_DIR=/export/home/tbuck/Odisseo-bench-jint-wt/benchmark_multigpu
SITE=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt
PY=/export/scratch/tbuck/jax0112-venv/bin/python
JZPY=/export/scratch/tbuck/jzfmm-venv/bin/python
YGG=/export/home/tbuck/yggdrax-r6-main
O=${O:-$SP/r7/rows}; mkdir -p $O
filt() { grep -v "cuda_executor\|^E10\|Deprecation\|return FastMulti\|sitecustom\|Source Location\|external/xla"; }
own() { echo yield > $SP/r7/gctl; until grep -q "^yield" $SP/r7/gack 2>/dev/null; do sleep 0.5; done; }
release() { echo hold > $SP/r7/gctl; }

jz() {  # tag n leaf p theta alloc_fac
  local tag=$1 n=$2 leaf=$3 p=$4 th=$5 af=$6
  own
  (cd $BENCH_DIR && CUDA_VISIBLE_DEVICES=$CARD timeout 2400 $JZPY codes/jzfmm_force_eval.py --n $n \
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

# ours TREE tag n mode(force|step) [--env "A=1 B=2"] extra...
ours() {
  local jt=$1 tag=$2 n=$3 mode=$4; shift 4
  local env="JACCPOT_STRICT_CARRY=particles"
  if [ "$1" = "--env" ]; then env="$env $2"; shift 2; fi
  local args
  if [ $mode = force ]; then args="--no-scan --no-analysis --accuracy-targets 4096 --eval-repeats 5"
  else args="--skip-eval --steps 4 --reps 3 --no-analysis"; fi
  if [ "$mode" = stepsave ]; then args="--no-analysis"; fi
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$YGG \
    CUDA_VISIBLE_DEVICES=$CARD timeout 2400 $PY bench/fused_memory_budget.py --n $n --ic plummer_clipped \
    --caps unnamed --prealloc 0.88 $args "$@" --env $env \
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

# capture TREE tag n extra... : near-field inputs of one eval, to $SP/r7/cap/tag.npz
capture() {
  local jt=$1 tag=$2 n=$3; shift 3
  mkdir -p $SP/r7/cap
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$YGG \
    CUDA_VISIBLE_DEVICES=$CARD timeout 2400 $PY bench/nearfield_capture.py $SP/r7/cap/$tag.npz -- \
    --n $n --ic plummer_clipped --caps unnamed --prealloc 0.88 --no-scan --no-analysis --eval-repeats 1 \
    "$@" --env JACCPOT_STRICT_CARRY=particles --out $SP/r7/cap/${tag}_row.json 2>&1 | filt > $SP/r7/cap/$tag.log)
  release
  grep "\[capture\]" $SP/r7/cap/$tag.log || echo "$tag: capture FAILED"
}

# tune TREE capture_tag out_tag variants... : time near-field kernel variants alone
tune() {
  local jt=$1 tag=$2 out=$3; shift 3
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$YGG \
    CUDA_VISIBLE_DEVICES=$CARD timeout 3000 $PY bench/nearfield_kernel_tune.py $SP/r7/cap/$tag.npz \
    --stats --variants "$@" 2>&1 | filt > $SP/r7/cap/$out.txt)
  release
  cat $SP/r7/cap/$out.txt
}

# wtune TREE capture_tag out_tag variants... : time walk variants alone
wtune() {
  local jt=$1 tag=$2 out=$3; shift 3
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$YGG \
    CUDA_VISIBLE_DEVICES=$CARD timeout 3000 $PY bench/walk_tune.py $SP/r7/cap/${tag}_walk.npz \
    --row $SP/r7/cap/${tag}_row.json --variants "$@" 2>&1 | filt > $SP/r7/cap/$out.txt)
  release
  cat $SP/r7/cap/$out.txt
}
