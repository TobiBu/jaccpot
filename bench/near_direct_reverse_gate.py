"""The direct near field's reverse on a captured real step: parity, time, memory (S8).

usage: python bench/near_direct_reverse_gate.py CAPTURE.npz [--parity] [--memory]
       [--table] [--variants Bt:Bs:warps ...] [--reps 7]

``CAPTURE.npz`` (+ ``CAPTURE.json``) comes from ``bench/nearfield_capture.py``: the
sorted particles, leaf ranges and neighbour CSR of the first near-field call of a
real fused step, with its static options. On it, for the loss ``L = sum(acc . w)``
of the near field alone (``w`` a fixed random field):

``--parity``
    the float32 gradients (positions, masses, ``softening_sq``, ``G``) of the direct
    layout's rule (:func:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp`) and of the table layout's
    (:func:`~jaccpot.pallas.nearfield_leafpair_csr.nearfield_leafpair_csr_pallas_cvjp`,
    the differentiable lane before S8), each against the direct rule in float64 on
    the same inputs (the reference: tests/unit/operators/test_nearfield_direct_vjp.py
    pins it to a pure-JAX pair sum), and against each other, with an A-vs-A control
    (the direct float32 gradient twice: elements that differ, max difference).
``--table``
    also time the table layout's gradient (and with ``--memory`` compile it). It
    gathers the ``(L, W)`` leaf tables with a plain ``p[ids]``, whose transpose is a
    scatter-add; the differentiable lane gathers through a custom inverse
    (``_leaf_layout_gather``), so there the table rule is somewhat faster.
default
    time the production forward, the reverse alone and ``jax.grad`` of the direct
    rule (min and median of ``--reps``); ``--variants`` adds reverse tilings
    ``Bt:Bs:warps``. The loss is linear in the accelerations, so under ``jax.grad``
    the forward's value is dead and XLA drops it: the gradient's time is the
    reverse's.
``--memory``
    compile ``jax.grad`` of ``L`` through the direct rule alone and print its
    ``memory_analysis()`` (arguments, outputs, temporaries) per particle, the largest
    array in the compiled program's text in units of ``N`` floats (the S8 gate: no
    buffer of ``N x 16`` floats or more), and the device's peak bytes in use after
    one run (``memory_stats``; the process's inputs included).

Timings are indicative unless the card was free before and after (check
``nvidia-smi``; the script prints the other processes it sees on the card).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

# jaccpot needs 64-bit types (README); neither it nor yggdrax turns them on.
jax.config.update("jax_enable_x64", True)

from jaccpot.pallas.nearfield_leafpair_csr import (  # noqa: E402
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_pallas_cvjp,
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (  # noqa: E402
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp,
    nearfield_leafpair_csr_sorted_direct_reverse_pallas,
)

_NAMES = ("positions", "masses", "leaf_start", "leaf_count", "neighbors", "offsets")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument(
        "--kernel",
        default=None,
        help="softening kernel instead of the captured one (same lists and softening)",
    )
    ap.add_argument("--json", default=None, help="write the measured rows here")
    return ap.parse_args()


def _card_processes() -> str:
    """The compute processes nvidia-smi lists on this process's card."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                cvd.split(",")[0],
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi failed: {exc}"
    mine = str(os.getpid())
    others = [line for line in out.splitlines() if not line.startswith(mine + ",")]
    return "; ".join(others) if others else "none"


def _time(fn, *args, reps):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts), 1e3 * float(np.median(ts))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    max_rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
    l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    return max_rel, l2


def main() -> None:
    args = _args()
    assert jax.devices()[0].platform == "gpu", jax.devices()
    data = dict(np.load(args.capture))
    meta = json.load(open(os.path.splitext(args.capture)[0] + ".json"))
    n = int(data["positions"].shape[0])
    W = int(meta["leaf_width"])
    kernel = args.kernel or meta.get("softening_kernel")
    chunk = int(meta.get("chunk") or 64)
    lc = data["leaf_count"]
    print(
        f"N={n} leaves {lc.size} (live {int((lc > 0).sum())}, mean "
        f"{lc[lc > 0].mean():.1f}) W={W} entries {int(data['counts'].sum())} "
        f"(capacity {data['neighbors'].size}, longest row {int(data['counts'].max())}) "
        f"kernel {kernel} eps^2 {meta['softening_sq']:.3e} G {meta['G']}",
        flush=True,
    )
    print(f"other processes on the card before: {_card_processes()}", flush=True)
    rows: dict = {"capture": args.capture, "n": n, "kernel": kernel}

    def operands(dtype):
        dev = {k: jnp.asarray(data[k]) for k in (*_NAMES, "counts")}
        dev["positions"] = dev["positions"].astype(dtype)
        dev["masses"] = dev["masses"].astype(dtype)
        soft = jnp.asarray(meta["softening_sq"], dtype)
        G = jnp.asarray(meta["G"], dtype)
        topo = tuple(dev[k] for k in _NAMES[2:]) + (dev["counts"],)
        return dev["positions"], dev["masses"], topo, soft, G

    w32 = jnp.asarray(np.random.default_rng(1).standard_normal((n, 3)), jnp.float32)
    common = dict(leaf_width=W, chunk=chunk, softening_kernel=kernel)

    def direct_loss(p, m, topo, s, g, w):
        acc, _ = nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
            p, m, *topo, softening_sq=s, G=g, **common
        )
        return jnp.sum(acc * w)

    direct_grad = jax.jit(jax.grad(direct_loss, argnums=(0, 1, 3, 4)))

    # the table layout: (L, W) leaf tables gathered from the ranges, its chunk table
    starts, counts_l = data["leaf_start"], data["leaf_count"]
    slot = np.arange(W)[None, :]
    mask_np = slot < counts_l[:, None]
    ids_np = np.where(mask_np, starts[:, None] + slot, 0).astype(np.int32)

    def table_parts():
        ids, mask = jnp.asarray(ids_np), jnp.asarray(mask_np)
        offsets = jnp.asarray(data["offsets"])
        counts = jnp.asarray(data["counts"])
        cap = leafpair_chunk_capacity(int(data["neighbors"].size), lc.size, chunk)
        table = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
        return ids, mask, table

    def table_loss(p, m, nbr, ids, mask, table, s, g, w):
        out = nearfield_leafpair_csr_pallas_cvjp(
            p[ids],
            m[ids],
            mask,
            nbr,
            table,
            s,
            g,
            chunk,
            None,
            1,
            None,
            False,
            "input",
            True,
            None,
            kernel,
        )
        return jnp.sum(jnp.where(mask[..., None], out[..., :3] * w[ids], 0.0))

    table_grad = jax.jit(jax.grad(table_loss, argnums=(0, 1, 6, 7)))

    names = ("positions", "masses", "softening_sq", "G")
    if args.parity:
        p64, m64, topo64, s64, g64 = operands(jnp.float64)
        truth = jax.block_until_ready(
            jax.jit(jax.grad(direct_loss, argnums=(0, 1, 3, 4)))(
                p64, m64, topo64, s64, g64, w32.astype(jnp.float64)
            )
        )
        p32, m32, topo32, s32, g32 = operands(jnp.float32)
        d1 = jax.block_until_ready(direct_grad(p32, m32, topo32, s32, g32, w32))
        d2 = jax.block_until_ready(direct_grad(p32, m32, topo32, s32, g32, w32))
        ids, mask, table = table_parts()
        tg = jax.block_until_ready(
            table_grad(p32, m32, topo32[2], ids, mask, table, s32, g32, w32)
        )
        rows["parity"] = {}
        for k, name in enumerate(names):
            a, b = np.asarray(d1[k]), np.asarray(d2[k])
            ndiff = int(np.sum(a != b))
            ctrl = float(np.max(np.abs(a - b))) if ndiff else 0.0
            row = {
                "direct_vs_fp64": _rel(d1[k], truth[k]),
                "table_vs_fp64": _rel(tg[k], truth[k]),
                "direct_vs_table": _rel(d1[k], tg[k]),
                "a_vs_a_differ": ndiff,
                "a_vs_a_max": ctrl,
            }
            rows["parity"][name] = row
            print(
                f"  {name:13s} max-rel/rel-L2 vs fp64: direct "
                f"{row['direct_vs_fp64'][0]:.2e}/{row['direct_vs_fp64'][1]:.2e}  "
                f"table {row['table_vs_fp64'][0]:.2e}/{row['table_vs_fp64'][1]:.2e}  "
                f"direct vs table {row['direct_vs_table'][0]:.2e}/"
                f"{row['direct_vs_table'][1]:.2e}  A-vs-A {ndiff} differ "
                f"(max {ctrl:.2e})",
                flush=True,
            )
        del truth, p64, m64, topo64

    p32, m32, topo32, s32, g32 = operands(jnp.float32)
    if not args.no_timing:
        fwd = jax.jit(
            lambda p, m, topo, s, g: nearfield_leafpair_csr_sorted_direct_pallas(
                p, m, *topo, softening_sq=s, G=g, **common
            )[0]
        )
        t_fwd = _time(fwd, p32, m32, topo32, s32, g32, reps=args.reps)
        t_grad = _time(direct_grad, p32, m32, topo32, s32, g32, w32, reps=args.reps)
        rows["forward_ms"], rows["grad_ms"] = t_fwd, t_grad
        print(
            f"  forward (production)       min {t_fwd[0]:8.2f} ms median {t_fwd[1]:8.2f}"
        )
        print(
            f"  jax.grad direct (fwd+rev)  min {t_grad[0]:8.2f} ms median "
            f"{t_grad[1]:8.2f}  ratio {t_grad[0] / t_fwd[0]:.2f}x",
            flush=True,
        )
        variants = [None] + [tuple(int(t) for t in v.split(":")) for v in args.variants]
        rows["reverse_ms"] = {}
        for v in variants:
            kw = {} if v is None else dict(target_subtile=v[0], source_tile=v[1])
            if v is not None:
                kw["num_warps"] = v[2]

            def rev(p, m, topo, s, g, w, kw=kw):
                return nearfield_leafpair_csr_sorted_direct_reverse_pallas(
                    p, m, *topo, w, softening_sq=s, G=g, **common, **kw
                )

            label = "default" if v is None else ":".join(map(str, v))
            try:
                t_rev = _time(
                    jax.jit(rev), p32, m32, topo32, s32, g32, w32, reps=args.reps
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  reverse {label}: FAILED {str(exc)[:300]}", flush=True)
                continue
            rows["reverse_ms"][label] = t_rev
            print(
                f"  reverse alone {label:12s} min {t_rev[0]:8.2f} ms median "
                f"{t_rev[1]:8.2f}  ratio {t_rev[0] / t_fwd[0]:.2f}x",
                flush=True,
            )
        if args.table:
            ids, mask, table = table_parts()
            t_tab = _time(
                table_grad,
                p32,
                m32,
                topo32[2],
                ids,
                mask,
                table,
                s32,
                g32,
                w32,
                reps=args.reps,
            )
            rows["table_grad_ms"] = t_tab
            print(
                f"  jax.grad table (fwd+rev)   min {t_tab[0]:8.2f} ms median "
                f"{t_tab[1]:8.2f}",
                flush=True,
            )

    if args.memory:
        compiled = direct_grad.lower(p32, m32, topo32, s32, g32, w32).compile()
        ma = compiled.memory_analysis()
        text = compiled.as_text()
        # the largest array an instruction of the program defines (not a parameter,
        # nor a view of one): the S8 gate's "no buffer of N x 16 floats"
        biggest, which = 0, ""
        views = ("parameter", "bitcast", "get-tuple-element", "tuple")
        for dt, dims, op in re.findall(
            r"= (f32|f64|s32|pred|u32|s64)\[([0-9,]+)\][^ ]* ([a-z-]+)\(", text
        ):
            if op in views:
                continue
            size = int(np.prod([int(x) for x in dims.split(",")]))
            per = 8 if dt in ("f64", "s64") else (1 if dt == "pred" else 4)
            if size * per > biggest:
                biggest, which = size * per, f"{op} {dt}[{dims}]"
        mem = {
            "argument_B_per_p": ma.argument_size_in_bytes / n,
            "output_B_per_p": ma.output_size_in_bytes / n,
            "temp_B_per_p": ma.temp_size_in_bytes / n,
            "temp_GiB": ma.temp_size_in_bytes / 2**30,
            "largest_array_in_N_floats": biggest / (4 * n),
            "largest_array": which,
        }
        jax.block_until_ready(compiled(p32, m32, topo32, s32, g32, w32))
        stats = jax.devices()[0].memory_stats() or {}
        mem["device_peak_GiB"] = stats.get("peak_bytes_in_use", 0) / 2**30
        rows["memory"] = mem
        print(
            "  grad memory: args {argument_B_per_p:.1f} B/p, outputs "
            "{output_B_per_p:.1f} B/p, temporaries {temp_B_per_p:.1f} B/p "
            "({temp_GiB:.2f} GiB); largest array {largest_array_in_N_floats:.2f} N "
            "floats ({largest_array}); device peak {device_peak_GiB:.2f} "
            "GiB".format(**mem),
            flush=True,
        )
        if args.table:
            ids, mask, table = table_parts()
            tma = (
                table_grad.lower(p32, m32, topo32[2], ids, mask, table, s32, g32, w32)
                .compile()
                .memory_analysis()
            )
            rows["table_memory"] = {
                "temp_B_per_p": tma.temp_size_in_bytes / n,
                "temp_GiB": tma.temp_size_in_bytes / 2**30,
            }
            print(
                f"  table grad memory: temporaries "
                f"{tma.temp_size_in_bytes / n:.1f} B/p "
                f"({tma.temp_size_in_bytes / 2**30:.2f} GiB)",
                flush=True,
            )

    print(f"other processes on the card after: {_card_processes()}", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
