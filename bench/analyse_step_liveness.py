"""What is live at a compiled program's memory peak: liveness over the scheduled HLO.

``bench/analyse_trace_by_stage.py --buffers`` lists the values at the TOP of the
preallocated temporary block, but a buffer assignment carries offsets, not live
ranges, so it cannot say which values coexist at the peak. This reads the
scheduled module next to it and computes them:

* the module text after optimisation is scheduled (instruction order = schedule);
  while bodies and conditional branches are inlined once at their call, fusions
  are not entered;
* each preallocated-temp value lives from its defining instruction to its last
  use, tuple elements through their ``get-tuple-element`` users (so a sort's
  scratch dies with the sort);
* values sharing an (offset, size) with overlapping lives are one buffer (the
  aliases a while carry or a tuple makes).

It prints the peak instant and what is live there per ``fmm_*`` scope, the live
maximum while each stage runs (the next windows), and on request what is live at
given schedule steps. On the fused step at 8e6 its peak matched the assignment's
block to 1 % (1.560 against 1.570 GiB); the block can exceed the live peak by the
assignment's fragmentation.

Usage::

    XLA_FLAGS="--xla_dump_to=D --xla_dump_hlo_as_text --xla_dump_hlo_module_re=_compiled_runner" ...
    python bench/analyse_step_liveness.py D/module_*.jit__compiled_runner.*after_optimizations.txt
    python bench/analyse_step_liveness.py M.txt --at 963,1019

(``bench/fused_memory_budget.py --dump-dir D --dump-re _compiled_runner`` writes
the dump.) The buffer assignment is read from the same path with
``-buffer-assignment.txt``.
"""

from __future__ import annotations

import argparse
import collections
import re

_COMP = re.compile(r"^(ENTRY )?%([\w.\-]+) .*\{$")
_INST = re.compile(r"^\s+(ROOT )?%([\w.\-]+) = (.*)$")
_REF = re.compile(r"%([\w.\-]+)")
_CALLED = re.compile(
    r"(?:body|condition|true_computation|false_computation)=%([\w.\-]+)"
    r"|branch_computations=\{([^}]*)\}"
)
_OPNAME = re.compile(r'op_name="([^"]*)"')
_GTE = re.compile(r"get-tuple-element\(%([\w.\-]+)\), index=(\d+)")
_VALUE = re.compile(
    r"value: <\d+ ([\w.\-]+)(\{[^}]*\})? @\d+> \(size=(\d+),offset=(\d+)\)"
)


def _schedule(hlo_path: str) -> tuple[list, dict]:
    """The flattened schedule ``[(name, operands)]`` and every instruction's op_name.

    Parameters
    ----------
    hlo_path : str
        ``*after_optimizations.txt`` of the module.

    Returns
    -------
    tuple[list, dict]
        The schedule (while bodies inlined) and ``{instruction: op_name}``.
    """
    comps: dict = {}
    entry, cur = None, None
    for line in open(hlo_path):
        m = _COMP.match(line)
        if m:
            cur = m.group(2)
            comps[cur] = []
            if m.group(1):
                entry = cur
            continue
        m = _INST.match(line)
        if m is None or cur is None:
            continue
        name, rest = m.group(2), m.group(3)
        head = rest.split(" metadata=")[0]
        ops = [r for r in _REF.findall(head) if r != name]
        gm = _GTE.search(rest)
        if gm:
            ops = [f"{gm.group(1)}{{{gm.group(2)}}}"]
        called = []
        for cm in _CALLED.finditer(rest):
            grp = cm.group(1) or cm.group(2)
            called += [c.strip().lstrip("%") for c in grp.split(",") if c.strip()]
        om = _OPNAME.search(rest)
        comps[cur].append((name, ops, called, om.group(1) if om else ""))
    opname: dict = {}
    seq: list = []

    def flatten(cname: str, depth: int = 0) -> None:
        for name, ops, called, on in comps.get(cname, []):
            opname[name] = on
            for c in called:
                if c in comps and depth < 8:
                    flatten(c, depth + 1)
            seq.append((name, ops))

    flatten(entry)
    return seq, opname


def _buffers(ba_path: str, seq: list) -> tuple[list, int]:
    """Temp buffers ``[offset, size, first, last, names]`` and the block size.

    Parameters
    ----------
    ba_path : str
        ``*-buffer-assignment.txt`` of the module.
    seq : list
        The flattened schedule.

    Returns
    -------
    tuple[list, int]
        The buffers (aliases merged) and the preallocated-temp block's size.
    """
    first, last = {}, {}
    gte_users = collections.defaultdict(list)
    for i, (name, ops) in enumerate(seq):
        first.setdefault(name, i)
        for o in ops:
            last[o] = i
        if len(ops) == 1 and ops[0].endswith("}"):
            gte_users[ops[0]].append(name)
    vals, in_temp, block = [], False, 0
    for line in open(ba_path):
        if line.startswith("allocation "):
            in_temp = "preallocated-temp" in line
            if in_temp:
                block = int(re.search(r"size (\d+)", line).group(1))
            continue
        m = _VALUE.search(line) if in_temp else None
        if m is None or int(m.group(3)) == 0:
            continue
        inst, key = m.group(1), m.group(1) + (m.group(2) or "")
        d = first.get(inst)
        if d is None:
            continue
        u = max(last.get(key, d), last.get(inst, d), d)
        for user in gte_users.get(key, ()):
            u = max(u, last.get(user, d))
        vals.append([int(m.group(4)), int(m.group(3)), d, u, key])
    vals.sort(key=lambda v: (v[0], v[1], v[2]))
    bufs: list = []
    for v in vals:
        b = bufs[-1] if bufs else None
        if b and b[0] == v[0] and b[1] == v[1] and v[2] <= b[3] + 1:
            b[3] = max(b[3], v[3])
            b[4].append(v[4])
        else:
            bufs.append([v[0], v[1], v[2], v[3], [v[4]]])
    return bufs, block


def main() -> None:
    """Print the peak, the per-stage maxima and the requested steps."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hlo", help="*after_optimizations.txt of the module")
    ap.add_argument("--top", type=int, default=16, help="values listed per instant")
    ap.add_argument("--at", default="", help="comma-separated schedule steps")
    args = ap.parse_args()
    seq, opname = _schedule(args.hlo)
    bufs, block = _buffers(args.hlo.replace(".txt", "-buffer-assignment.txt"), seq)
    curve = [0] * (len(seq) + 2)
    for _, size, d, u, _ in bufs:
        curve[d] += size
        curve[u + 1] -= size
    for t in range(1, len(curve)):
        curve[t] += curve[t - 1]

    def scope(name: str) -> str:
        on = opname.get(name.split("{")[0], "")
        s = re.findall(r"(fmm_[a-z0-9_]+)", on)
        return s[-1] if s else ("(carry)" if not on else on.split("/")[-1][:28])

    def show(t: int, label: str) -> None:
        at = [b for b in bufs if b[2] <= t <= b[3]]
        print(
            f"{label} step {t}/{len(seq)} ({seq[t][0]} [{scope(seq[t][0])}]): "
            f"{sum(b[1] for b in at) / 2**30:.3f} GiB live"
        )
        per = collections.Counter()
        for b in at:
            per[scope(b[4][0])] += b[1]
        print(
            "  per scope: "
            + ", ".join(f"{k} {v / 2**30:.3f}" for k, v in per.most_common(8))
        )
        for _, size, d, u, names in sorted(at, key=lambda b: -b[1])[: args.top]:
            print(
                f"  {size / 2**20:9.1f} MiB  [{d:>6},{u:>6}] {scope(names[0]):<22} "
                f"{names[0]}"
            )

    peak_t = max(range(len(seq)), key=lambda t: curve[t])
    print(f"temporary block {block / 2**30:.3f} GiB")
    show(peak_t, "peak at")
    stage: dict = {}
    for t in range(len(seq)):
        s = scope(seq[t][0])
        if curve[t] > stage.get(s, (0, 0))[0]:
            stage[s] = (curve[t], t)
    print("live maximum while each stage runs:")
    for s, (v, t) in sorted(stage.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {s:<24} {v / 2**30:8.3f} GiB at step {t}")
    for t in (int(x) for x in args.at.split(",") if x):
        show(t, "at")


if __name__ == "__main__":
    main()
