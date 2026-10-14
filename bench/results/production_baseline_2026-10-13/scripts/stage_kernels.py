"""Per-kernel attribution of a fused-step trace: scope, ms/call, launches, bytes, floor.

usage: python stage_kernels.py TRACE_DIR DUMP_DIR CALLS [--min-ms 0.3] [--scopes a,b] [--bw 1.3e12]

Kernels are matched to the dump of THEIR module (the trace's ``hlo_module``; the
last-numbered dump when a module was compiled twice): instruction names repeat
across modules (``_compiled_runner_start`` vs ``_compiled_runner``).

Bytes: the output shape plus every operand's shape (resolved through the module's
instructions; ``%x#k`` is element k of a tuple). An upper bound for gathers and
scatters that touch part of an operand. Floor = bytes / bw.
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re

BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "s64": 8,
    "u64": 8,
    "f64": 8,
}
SHAPE = re.compile(r"\b(pred|[suf]\d+|bf16)\[([\d,]*)\]")
INST = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$")
OPND = re.compile(r"%([\w.\-]+)(?:#(\d+))?")
NAME = re.compile(r'op_name="([^"]*)"')


def shape_bytes(dt: str, dims: str) -> int:
    n = 1
    for d in filter(None, dims.split(",")):
        n *= int(d)
    return n * BYTES.get(dt, 4)


SF = re.compile(r"stack_frame_id=(\d+)")


def load_frames(path: str):
    """stack_frame_id -> [(file, line, function), ...] innermost first."""
    files, funcs, locs, frames = {}, {}, {}, {}
    sec = None
    for line in open(path):
        if line.startswith(
            ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
        ):
            sec = line.strip()
            continue
        if sec is None:
            continue
        if not line.strip() or line.startswith("HloModule"):
            sec = None
            continue
        k, _, rest = line.strip().partition(" ")
        if not k.isdigit():
            sec = None
            continue
        if sec == "FileNames":
            files[int(k)] = rest.strip('"')
        elif sec == "FunctionNames":
            funcs[int(k)] = rest.strip('"')
        elif sec == "FileLocations":
            d = dict(x.split("=") for x in rest.strip("{}").split())
            locs[int(k)] = (
                files.get(int(d["file_name_id"]), "?"),
                int(d["line"]),
                funcs.get(int(d["function_name_id"]), "?"),
            )
        elif sec == "StackFrames":
            d = dict(x.split("=") for x in rest.strip("{}").split())
            frames[int(k)] = (int(d["file_location_id"]), int(d["parent_frame_id"]))

    def chain(fid: int):
        outl, seen = [], set()
        while fid in frames and fid not in seen:
            seen.add(fid)
            loc, parent = frames[fid]
            outl.append(locs.get(loc, ("?", 0, "?")))
            if parent == fid:
                break
            fid = parent
        return outl

    return chain


def load_module(path: str):
    """name -> (op_name, out element byte sizes, operand refs, text, frame id)."""
    out = {}
    in_entry = False
    for line in open(path):
        if line.startswith("ENTRY "):
            in_entry = True
        elif line.startswith("}"):
            in_entry = False
        m = INST.match(line)
        if not m:
            continue
        nm, rest = m.group(1), m.group(2)
        n = NAME.search(rest)
        head = rest.split(", metadata=")[0]
        # output shapes: everything before the opcode's '('
        k = re.search(r"\s([a-z][\w\-]*)\(", head)
        out_txt = head[: k.start()] if k else head
        args_txt = head[k.end() :] if k else ""
        args_txt = args_txt.split("), ")[0]
        sizes = [shape_bytes(dt, d) for dt, d in SHAPE.findall(out_txt)]
        refs = [(o, int(i) if i else None) for o, i in OPND.findall(args_txt)]
        sf = SF.search(rest)
        prev = out.get(nm)
        if prev is None or in_entry:
            out[nm] = (
                n.group(1) if n else (prev[0] if prev else ""),
                sizes,
                refs,
                head,
                int(sf.group(1)) if sf else (prev[4] if prev else 0),
            )
    return out, load_frames(path)


def inst_bytes(mod: dict, nm: str) -> int:
    rec = mod.get(nm)
    if rec is None:
        return 0
    tot = sum(rec[1])
    for o, i in rec[2]:
        r = mod.get(o)
        if r is None:
            continue
        if i is None:
            tot += sum(r[1])
        elif i < len(r[1]):
            tot += r[1][i]
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("dump_dir")
    ap.add_argument("calls", type=int)
    ap.add_argument("--min-ms", type=float, default=0.3)
    ap.add_argument("--scopes", default="")
    ap.add_argument("--bw", type=float, default=1.3e12)
    ap.add_argument("--scope-re", default=r"(cross_[a-z0-9_]+|fmm_[a-z0-9_]+)")
    ap.add_argument("--wide", type=int, default=150)
    ap.add_argument("--stack", type=int, default=0, help="source frames per kernel")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dump_dir}/module_*after_optimizations.txt"))
    by_mod = {}
    for p in files:
        b = os.path.basename(p)
        mname = b.split(".", 1)[1].split(".sm_")[0]
        by_mod[mname] = p  # sorted: the last compile wins
    mods = {}
    scope_re = re.compile(a.scope_re)
    f = sorted(glob.glob(f"{a.trace_dir}/**/*.trace.json.gz", recursive=True))[-1]
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    pid_name = {
        e["pid"]: e["args"]["name"]
        for e in ev
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    gpus = sorted(p for p, n in pid_name.items() if re.search(r"/device:GPU:\d+$", n))
    want = set(filter(None, a.scopes.split(",")))
    for p in gpus:
        t, c = collections.Counter(), collections.Counter()
        total = 0.0
        for e in ev:
            if e.get("ph") != "X" or e.get("pid") != p or "dur" not in e:
                continue
            args = e.get("args", {})
            hop, hm = args.get("hlo_op", ""), args.get("hlo_module", "")
            if hm and hm not in mods:
                mods[hm] = (
                    load_module(by_mod[hm]) if hm in by_mod else ({}, lambda i: [])
                )
            rec = mods.get(hm, ({},))[0].get(hop)
            path = rec[0] if rec else ""
            hits = scope_re.findall(path)
            scope = hits[-1] if hits else ("unmapped" if hop else "no_hlo_op")
            key = (scope, hm, hop or e["name"][:40])
            t[key] += e["dur"]
            c[key] += 1
            total += e["dur"]
        print(f"== {pid_name[p]}: kernel {total / 1e3 / a.calls:.2f} ms/call")
        per = collections.defaultdict(list)
        for k, v in t.items():
            per[k[0]].append((v / 1e3 / a.calls, c[k] / a.calls, k[1], k[2]))
        for scope in sorted(per, key=lambda s: -sum(x[0] for x in per[s])):
            if want and scope not in want:
                continue
            rows = sorted(per[scope], reverse=True)
            tot = sum(x[0] for x in rows)
            print(f"== {scope}: {tot:.2f} ms, {sum(x[1] for x in rows):.0f} launches")
            rest_ms, rest_n, rest_floor = 0.0, 0.0, 0.0
            for ms, n, hm, hop in rows:
                mod, chain = mods.get(hm, ({}, lambda i: []))
                b = inst_bytes(mod, hop)
                fl = 1e3 * b / a.bw
                if ms < a.min_ms:
                    rest_ms += ms
                    rest_n += n
                    rest_floor += fl * n
                    continue
                rec = mod.get(hop, ("", [], [], "", 0))
                d = rec[3]
                print(
                    f"   {ms:7.2f} ms x{n:<5.1f} {hop[:36]:<36} {b/2**20:8.1f} MiB floor {fl*n:6.2f} ms"
                    f"  {d[:a.wide]}"
                )
                if a.stack:
                    fr = [
                        x
                        for x in chain(rec[4])
                        if "/jax/" not in x[0] and "/jax/_src" not in x[0]
                    ]
                    for fpath, line, fn in fr[: a.stack]:
                        short = "/".join(fpath.split("/")[-3:])
                        print(f"              {short}:{line} {fn}")
            print(
                f"   {rest_ms:7.2f} ms in {rest_n:.0f} launches below {a.min_ms} ms (floor {rest_floor:.2f})"
            )


if __name__ == "__main__":
    main()
