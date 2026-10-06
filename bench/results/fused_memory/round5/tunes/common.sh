# Shared helpers for round 5 runs on card 6 (guard handshake). Arms pick a venv + frozen trees.
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/6c36057b-f12d-4913-ac8f-1827ce65faa4/scratchpad
export BENCH_DIR=/export/home/tbuck/Odisseo-bench-jint-wt/benchmark_multigpu
SITE=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt
filt() { grep -v "cuda_executor\|^E10\|Deprecation\|return FastMulti\|sitecustom\|Source Location\|external/xla"; }
own() { echo yield > $SP/r5/gctl; until grep -q "^yield" $SP/r5/gack 2>/dev/null; do sleep 0.5; done; }
release() { echo hold > $SP/r5/gctl; }
# run_arm VENV_PY JACCPOT_TREE YGG_TREE OUT_JSON args...
run_arm() {
  local py=$1 jt=$2 yt=$3 out=$4; shift 4
  own
  (cd $jt && PYTHONPATH=$SITE JACCPOT_WORKTREE=$jt YGGDRAX_WORKTREE=$yt CUDA_VISIBLE_DEVICES=6 \
    timeout 2400 $py bench/fused_memory_budget.py --caps unnamed --prealloc 0.88 \
    --env JACCPOT_STRICT_CARRY=particles "$@" --out $out 2>&1 | filt > ${out%.json}.log)
  release
  python3 - "$out" <<'PY' 2>/dev/null || echo "$(basename $out): no row"
import json, os, sys
d = json.load(open(sys.argv[1])); G = 2**30; n = d["n"]
acc = d.get("accuracy", {}); ev = d.get("eval_ms", {}); st = d.get("step_ms", {})
s = f"{os.path.basename(sys.argv[1])[:-5]}: "
if st: s += f"step {st['min']:.1f} ms (median {st['median']:.1f}) peak {d['peak_after_scan_gib']:.2f} GiB ({d['peak_after_scan_gib']*G/n:.0f} B/p) "
if ev: s += f"force {ev['min']:.1f} ms "
if acc: s += f"rel-L2 {acc['rel_l2']:.3e} p90 {acc['p90']:.3e} "
c = d.get("counts", {}); s += f"far {c.get('far_pair_count')} near {c.get('total_neighbors')} leaves {d.get('live_leaves')}"
print(s, flush=True)
PY
}
