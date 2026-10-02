"""Split a per-device GPU trace by named scope, through the optimized HLO's op_name metadata.

Usage: python bench/analyse_trace_by_stage.py <trace dir> <hlo dump dir> <calls>

Take the trace and the dump in ONE probe run, e.g.:

    XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_dump_to=D/hlo --xla_dump_hlo_as_text \
        --xla_dump_hlo_module_re=.*body.*" PROBE_TIME_ARMS=cross PROBE_PROFILE_DIR=D/trace \
        python bench/multigpu_c4_cross_force_probe.py
    python bench/analyse_trace_by_stage.py D/trace/cross D/hlo 5

The stage names are the `jax.named_call` labels in `jaccpot/distributed/cross.py`. Note
the dump spells instructions `input_scatter_fusion.225` where kernel names read `_225`.

Kernel events carry `args.hlo_op` (the HLO instruction that launched them) but not the
op path. The XLA dump's *after_optimizations.txt holds every instruction with
`metadata={op_name="jit(body)/shard_map/cross_hook/cross_export_walk/..."}`; this maps
hlo_op -> op_name and credits each kernel to the innermost `cross_*` scope on it.
Take the trace with `--xla_gpu_enable_command_buffer=` (explicitly empty): inside a CUDA
graph every kernel reports the graph's name instead of its own.
"""

import collections
import glob
import gzip
import json
import re
import sys

tdir, hdir = sys.argv[1], sys.argv[2]
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 1

op_name = {}
inst_re = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*?metadata=\{([^}]*)\}")
name_re = re.compile(r'op_name="([^"]*)"')
files = sorted(glob.glob(f"{hdir}/*body*after_optimizations.txt"))
assert files, f"no after_optimizations dump for the body module in {hdir}"
for path in files:
    for line in open(path):
        m = inst_re.match(line)
        if not m:
            continue
        n = name_re.search(m.group(2))
        if n:
            op_name.setdefault(m.group(1), n.group(1))
print(f"{len(op_name)} instructions with op_name from {len(files)} dump file(s)")

f = sorted(glob.glob(f"{tdir}/**/*.trace.json.gz", recursive=True))[-1]
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
pid_name = {
    e["pid"]: e["args"]["name"]
    for e in ev
    if e.get("ph") == "M" and e.get("name") == "process_name"
}
gpus = sorted(p for p, n in pid_name.items() if re.search(r"/device:GPU:\d+$", n))
scope_re = re.compile(r"(cross_[a-z0-9_]+)")
for p in gpus:
    xs = [e for e in ev if e.get("ph") == "X" and e.get("pid") == p and "dur" in e]
    tot, cnt = collections.Counter(), collections.Counter()
    top = collections.defaultdict(collections.Counter)
    for e in xs:
        hop = e.get("args", {}).get("hlo_op", "")
        path = op_name.get(hop, "")
        if hop.startswith("command_buffer"):
            key = "IN_CUDA_GRAPH"
        elif not hop:
            key = "no_hlo_op (memcpy/memset)"
        elif not path:
            key = "unmapped"
        else:
            hits = scope_re.findall(path)
            key = hits[-1] if hits else "local"
        tot[key] += e["dur"]
        cnt[key] += 1
        top[key][e["name"][:60]] += e["dur"]
    total = sum(tot.values())
    print(
        f"== {pid_name[p]}: kernel time {total / 1e3 / calls:8.2f} ms/call, "
        f"launches {sum(cnt.values()) // calls}"
    )
    for k, v in tot.most_common():
        heavy = ", ".join(
            f"{n} {t / 1e3 / calls:.2f}" for n, t in top[k].most_common(2)
        )
        print(
            f"   {k:<26} {v / 1e3 / calls:8.2f} ms  launches {cnt[k] // calls:>5}   [{heavy}]"
        )
