"""Split a GPU trace (or a buffer assignment) by named scope, through the optimized HLO's op_name metadata.

Usage::

    python bench/analyse_trace_by_stage.py <trace dir> <hlo dump dir> <calls> \\
        [--module-re '*body*'] [--scope-re '(cross_[a-z0-9_]+|fmm_[a-z0-9_]+)']
    python bench/analyse_trace_by_stage.py --buffers <hlo dump dir> [--module-re ...]

Take the trace and the dump in ONE run, e.g. the multi-GPU probe::

    XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_dump_to=D/hlo --xla_dump_hlo_as_text \\
        --xla_dump_hlo_module_re=.*body.*" PROBE_TIME_ARMS=cross PROBE_PROFILE_DIR=D/trace \\
        python bench/multigpu_c4_cross_force_probe.py
    python bench/analyse_trace_by_stage.py D/trace/cross D/hlo 5

or the single-card fused step (``bench/fused_memory_budget.py --trace-dir T --dump-dir D``,
module ``*_compiled_runner*``, scopes ``fmm_*`` -- the ``jax.named_scope`` stages of the
refresh and the force evaluation).

Kernel events carry ``args.hlo_op`` (the HLO instruction that launched them) but not the
op path. The XLA dump's ``*after_optimizations.txt`` holds every instruction with
``metadata={op_name="jit(...)/.../fmm_walk/..."}``; this maps hlo_op -> op_name and credits
each kernel to the INNERMOST matching scope. Take the trace with
``--xla_gpu_enable_command_buffer=`` (explicitly empty): inside a CUDA graph every kernel
reports the graph's name instead of its own. The dump spells instructions
``input_scatter_fusion.225`` where kernel names read ``_225``.

``--buffers`` reads ``*buffer-assignment.txt`` instead: the values at the top of the
preallocated temporary block -- those that set its size -- labelled with their scope, and
the bytes of every scope's values (an upper bound on its share, since values at disjoint
times share addresses).
"""

import argparse
import collections
import glob
import gzip
import json
import re


def _op_names(hdir: str, module_glob: str) -> dict:
    """``{module name: {instruction: op_name}}``; ``""`` holds every module's names.

    Instruction names repeat across modules (``_compiled_runner_start`` and
    ``_compiled_runner`` both have an ``input_scatter_fusion.8``), so a kernel is
    looked up in its OWN module, which the trace names in ``args.hlo_module``; a
    module dumped twice keeps its last compile. The merged ``""`` map (first module
    wins) is only the fallback for events without a module.
    """
    per: dict = {}
    inst_re = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*?metadata=\{([^}]*)\}")
    name_re = re.compile(r'op_name="([^"]*)"')
    files = sorted(glob.glob(f"{hdir}/{module_glob}after_optimizations.txt"))
    assert files, f"no after_optimizations dump matching {module_glob} in {hdir}"
    merged: dict = {}
    head_re = re.compile(r"^%?([\w.\-]+) \(.*\) -> .*\{\s*$")
    any_re = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=(.*)$")
    calls_re = re.compile(r"calls=%?([\w.\-]+)")
    for path in files:
        module = path.split("/")[-1].split(".", 1)[1].split(".sm_")[0]
        names: dict = {}
        body_names: dict = collections.defaultdict(collections.Counter)
        calls: dict = {}
        comp = None
        for line in open(path):
            h = head_re.match(line)
            if h:
                comp = h.group(1)
                continue
            if line.startswith("}"):
                comp = None
                continue
            m = any_re.match(line)
            if not m:
                continue
            n = name_re.search(m.group(2))
            if n:
                names.setdefault(m.group(1), n.group(1))
                if comp is not None:
                    body_names[comp][n.group(1)] += 1
            c = calls_re.search(m.group(2))
            if c:
                calls.setdefault(m.group(1), c.group(1))
        # a fusion without metadata of its own (XLA drops it on some multi-output
        # fusions): the op_name most of its body carries -- the L2P's jvp fusion
        # read as "unmapped" this way, and was taken for the integrator's kick
        for inst, comp_name in calls.items():
            if inst not in names and body_names.get(comp_name):
                names[inst] = body_names[comp_name].most_common(1)[0][0]
        for k, v in names.items():
            merged.setdefault(k, v)
        per[module] = names  # sorted: the last compile of a module wins
    per[""] = merged
    print(f"{len(merged)} instructions with op_name from {len(files)} dump file(s)")
    return per


def _scope(path: str, scope_re: re.Pattern, default: str) -> str:
    hits = scope_re.findall(path)
    return hits[-1] if hits else default


def trace(args: argparse.Namespace) -> None:
    op_name = _op_names(args.hlo_dir, args.module_re)
    scope_re = re.compile(args.scope_re)
    f = sorted(glob.glob(f"{args.trace_dir}/**/*.trace.json.gz", recursive=True))[-1]
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    pid_name = {
        e["pid"]: e["args"]["name"]
        for e in ev
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    gpus = sorted(p for p, n in pid_name.items() if re.search(r"/device:GPU:\d+$", n))
    calls = max(1, int(args.calls))
    for p in gpus:
        xs = [e for e in ev if e.get("ph") == "X" and e.get("pid") == p and "dur" in e]
        tot, cnt = collections.Counter(), collections.Counter()
        top = collections.defaultdict(collections.Counter)
        for e in xs:
            hop = e.get("args", {}).get("hlo_op", "")
            names = op_name.get(e.get("args", {}).get("hlo_module", ""), op_name[""])
            path = names.get(hop, "")
            if hop.startswith("command_buffer"):
                key = "IN_CUDA_GRAPH"
            elif not hop:
                key = "no_hlo_op (memcpy/memset)"
            elif not path:
                key = "unmapped"
            else:
                key = _scope(path, scope_re, "unscoped")
            tot[key] += e["dur"]
            cnt[key] += 1
            top[key][e["name"][:60]] += e["dur"]
        total = sum(tot.values())
        # busy = the union of kernel intervals; the rest of the window is idle (host
        # syncs, launch gaps)
        spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in xs)
        busy, cur_s, cur_e = 0.0, None, None
        for s, t in spans:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, t
            else:
                cur_e = max(cur_e, t)
        if cur_e is not None:
            busy += cur_e - cur_s
        window = (spans[-1][1] - spans[0][0]) if spans else 0.0
        print(
            f"== {pid_name[p]}: kernel time {total / 1e3 / calls:8.2f} ms/call, busy "
            f"{busy / 1e3 / calls:8.2f}, window {window / 1e3 / calls:8.2f}, "
            f"launches {sum(cnt.values()) // calls}"
        )
        for k, v in tot.most_common():
            heavy = ", ".join(
                f"{n} {t / 1e3 / calls:.2f}"
                for n, t in top[k].most_common(int(args.kernels))
            )
            print(
                f"   {k:<26} {v / 1e3 / calls:8.2f} ms  {100 * v / max(total, 1):5.1f} %"
                f"  launches {cnt[k] // calls:>6}   [{heavy}]"
            )


def buffers(args: argparse.Namespace) -> None:
    op_name = _op_names(args.hlo_dir, args.module_re)
    scope_re = re.compile(args.scope_re)
    files = sorted(glob.glob(f"{args.hlo_dir}/{args.module_re}buffer-assignment.txt"))
    assert files, f"no buffer-assignment dump matching {args.module_re}"
    module = files[-1].split("/")[-1].split(".", 1)[1].split(".sm_")[0]
    module_names = op_name.get(module, op_name[""])
    val_re = re.compile(
        r"value: <\d+ ([\w.\-]+)(?:\{[^}]*\})? @\d+> \(size=(\d+),offset=(\d+)\): (\S+)"
    )
    values, in_temp = [], False
    for line in open(files[-1]):
        if line.startswith("allocation "):
            in_temp = "preallocated-temp" in line
        elif in_temp and "value:" in line:
            m = val_re.search(line)
            if m and int(m.group(2)) > 0:
                inst = m.group(1)
                values.append(
                    (
                        int(m.group(3)),
                        int(m.group(2)),
                        inst,
                        m.group(4),
                        _scope(module_names.get(inst, ""), scope_re, "unscoped"),
                    )
                )
    if not values:
        print("no preallocated-temp values")
        return
    peak = max(off + size for off, size, *_ in values)
    print(f"{files[-1].split('/')[-1]}: temporary block {peak / 2**30:.3f} GiB")
    print("values reaching the top 5 % of the block (they set its size):")
    tops = sorted(
        (v for v in values if v[0] + v[1] >= 0.95 * peak),
        key=lambda v: -(v[0] + v[1]),
    )
    for off, size, inst, shape, scope in tops[: args.top]:
        print(
            f"   {scope:<22} {size / 2**20:9.1f} MiB @ {off / 2**30:6.3f} GiB  {inst} {shape}"
        )
    by = collections.Counter()
    for off, size, inst, shape, scope in values:
        by[scope] += size
    print(
        "bytes of values per scope (upper bound -- disjoint lifetimes share addresses):"
    )
    for k, v in by.most_common(15):
        print(f"   {k:<22} {v / 2**30:8.3f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace_dir", nargs="?")
    ap.add_argument("hlo_dir", nargs="?")
    ap.add_argument("calls", nargs="?", default=1)
    ap.add_argument("--buffers", metavar="HLO_DIR", default=None)
    ap.add_argument("--module-re", default="*body*")
    ap.add_argument("--scope-re", default=r"(cross_[a-z0-9_]+|fmm_[a-z0-9_]+)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--kernels", type=int, default=2, help="heaviest kernels listed per stage"
    )
    args = ap.parse_args()
    if args.buffers:
        args.hlo_dir = args.buffers
        buffers(args)
    else:
        trace(args)


if __name__ == "__main__":
    main()
