"""Time forward mode (``ad_mode="forward"``) per step on one card (the HoreKa job).

usage: python bench/forward_mode_timing.py --n N --program PROG [--k K]
       [--directions-per-pass k] [--steps T] [--reps R] --out OUT.json
       python bench/forward_mode_timing.py --n N --tpp-sweep [--tpp-k K] --out OUT.json
       python bench/forward_mode_timing.py --summarize DIR

One program per process, on F8's operating point (``bench/forward_mode_memory.py``,
itself S9's: ``preset="galaxy"``, a clipped Plummer sphere, float32,
``dehnen_error``, the galaxy leaf size and order, ``dt = 1e-3``, the clean-allocator
prepare) and F8's function: the Sgr-like map ``p = (dx, dy, dz, dvx, dvy, dvz, mass
scale, dt)`` of a satellite subset to the two weighted sums of the ``T``-step
rollout's final state. Programs:

* ``forward``: the production forward rollout over ``FusedForceProvider.step``
  (``checkpoint=None``): the reference every ratio is over;
* ``jvp``: ``jax.jvp`` along the first parameter (``K = 1``);
* ``jacfwd``: ``jax.jacfwd`` over the first ``--k`` parameters, in one pass, or with
  ``--directions-per-pass k`` in chunks of ``k`` directions
  (:func:`~jaccpot.runtime.forward_batching.jacfwd_batched`);
* ``vgrad``: ``jax.value_and_grad`` of the two sums' total in the start positions,
  velocities and masses, ``checkpoint="step"`` (S9's reverse).

Per program: ``compile_s`` (lower + compile from shapes), ``memory_analysis()``,
the allocator's ``bytes_in_use`` before and ``peak_bytes_in_use`` after, one
untimed call, then ``--reps`` calls under ``block_until_ready``: min / median / IQR
per call and per step (a call is the start force and ``T`` steps), A-vs-A (the
first and last call's tangents' sha256), the verdict bits, finiteness, the static
sizes (``topology``: total nodes, levels, leaf capacity, order) and the card's
other processes (``contended``: do not quote the row).

``--tpp-sweep``: the tangent op of every seam ALONE at production shapes. One
forward-mode step's seams are recorded (their operands, as ``differentiable_step``
calls them, returned from one jitted jvp of the step), then each seam's tangent op
is jitted and timed with ``--tpp-k`` tangents (the recorded one tiled) for each
tangents-per-program in {1, 2, 4, 8} (``JACCPOT_TANGENT_TPP_<SEAM>``, read at
trace time; track K's ``jaccpot.pallas._tangent_grid``). Without that knob the
sweep is skipped (``"skipped"`` in the JSON) and ``tpp`` is recorded as the
installed fused body. The M2M / L2L cells time the seams' 7-primal tangent ops,
which rerun the primal cascade before the tangent one (the production rule calls
the ``_out`` siblings with the forward's table instead): the same offset in every
tpp cell, so the ranking holds; the absolute time is not the rollout's.

``JACCPOT_COMPILATION_CACHE_DIR`` (opt-in) is applied before anything compiles:
with a warm cache ``compile_s`` measures a cache load, not a compile -- the row
records the directory, so leave it unset for compile timings.

``--cpu-interpret`` runs the small CPU lane (``forward_mode_memory``'s smoke):
the script's smoke test, no timing claims.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROGRAMS = ("forward", "jvp", "jacfwd", "vgrad")
TPP_CHOICES = (1, 2, 4, 8)
_ENV_KEYS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
    "JACCPOT_GRAD_FORWARD_TANGENT_KERNELS",
    "JACCPOT_COMPILATION_CACHE_DIR",
    "XLA_FLAGS",
) + tuple(
    f"JACCPOT_TANGENT_TPP_{s}" for s in ("P2M", "M2M", "L2L", "M2L", "L2P", "NEAR")
)


def _args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--program", default="jvp", choices=PROGRAMS)
    ap.add_argument("--k", type=int, default=1, help="jacfwd: columns (1..8)")
    ap.add_argument(
        "--directions-per-pass",
        type=int,
        default=None,
        help="jacfwd: columns per pass (default: all in one pass)",
    )
    ap.add_argument("--steps", type=int, default=16, help="T, the rollout's steps")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--subset", type=float, default=0.01, help="satellite fraction")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--tpp-sweep", action="store_true", help="time each tangent op")
    ap.add_argument("--tpp-k", type=int, default=8, help="tangents in the sweep")
    ap.add_argument("--cpu-interpret", action="store_true", help="CPU smoke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", default=None)
    return ap.parse_args(argv)


def _digest(*arrays: Any) -> str:
    import jax

    h = hashlib.sha256()
    for a in jax.tree_util.tree_leaves(arrays):
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _card_state() -> dict:
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        return {"other_processes": "none", "utilization": "cpu"}
    import grad_rollout_timing as grt

    return grt._card_state()


def _setup(args: argparse.Namespace) -> None:
    """The CPU smoke's environment, x64 and the opt-in compilation cache."""
    import forward_mode_memory as fmm

    if args.cpu_interpret:
        os.environ.update(fmm._INTERPRET_ENV)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    if args.cpu_interpret:
        jax.default_backend = lambda: "gpu"  # the large-N profile's backend gate
    jax.config.update("jax_enable_x64", True)
    try:  # an installed jaccpot without it: no cache (it is opt-in anyway)
        from jaccpot.runtime.compile_cache import maybe_enable_compilation_cache
    except ImportError:
        return
    maybe_enable_compilation_cache()


# --------------------------------------------------------------------------
# The programs.
# --------------------------------------------------------------------------


def _program(prov: Any, args: argparse.Namespace) -> Any:
    """The jitted program of the row: ``f(theta, p_rest, x, v, m, s, fb, w)``."""
    import forward_mode_memory as fmm
    import jax
    import jax.numpy as jnp

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    if args.program == "forward":

        def forward(theta, p_rest, x, v, m, s, fb, w):
            del theta, p_rest, s
            out = differentiable_rollout(
                prov.step,
                x,
                v,
                m,
                args.dt,
                args.steps,
                aux=(fb, None, None),
                checkpoint=None,
            )
            sums = jnp.stack([jnp.sum(w * out.positions), jnp.sum(w * out.velocities)])
            return sums, sums, out.aux[1]

        return jax.jit(forward)
    if args.program == "vgrad":
        margs = argparse.Namespace(**{**vars(args), "program": "reverse"})
        reverse = fmm._program(prov, margs)

        def vgrad(theta, p_rest, x, v, m, s, fb, w):
            del theta, p_rest
            (value, bits), grads = reverse(x, v, m, s, fb, w)
            return value, grads, bits

        return jax.jit(vgrad)
    margs = argparse.Namespace(
        **{**vars(args), "program": "jvp" if args.program == "jvp" else "jacfwd"}
    )
    if args.program == "jvp" or args.directions_per_pass is None:
        return fmm._program(prov, margs)
    from jaccpot.runtime.forward_batching import jacfwd_batched

    g = fmm._g(prov, margs)

    def jac(theta, p_rest, *rest):
        def f(t):
            sums, bits = g(t, p_rest, *rest)
            return sums, bits

        jacobian, bits = jacfwd_batched(
            f, directions_per_pass=args.directions_per_pass, has_aux=True
        )(theta)
        return jnp.zeros((2,), jacobian.dtype), jacobian, bits

    return jax.jit(jac)


def _stats_ms(ts: list[float], steps: int) -> dict:
    import grad_rollout_timing as grt

    return grt._stats_ms(ts, steps)


def run(args: argparse.Namespace) -> dict:
    """Measure one program; the JSON record."""
    import forward_mode_memory as fmm
    import jax
    import jax.numpy as jnp

    if args.program == "jvp" and args.k != 1:
        raise ValueError("--program jvp is K = 1")
    if not 1 <= args.k <= len(fmm.PARAMETERS):
        raise ValueError(f"--k must be 1..{len(fmm.PARAMETERS)}")
    device = jax.devices()[0]
    rec: dict[str, Any] = {
        "config": {k: v for k, v in vars(args).items() if k != "summarize"},
        "parameters": fmm.PARAMETERS[: args.k],
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "device": str(device),
        "backend": jax.default_backend(),
        "jax": jax.__version__,
        "card_before": _card_state(),
    }
    t0 = time.perf_counter()
    margs = argparse.Namespace(**{**vars(args), "program": "jvp"})
    prov, host = fmm._prepare(margs)
    rec["prepare_s"] = time.perf_counter() - t0
    rec["after_prepare"] = fmm._alloc(device)
    rec["topology"] = fmm._topology(prov)
    pos, vel, mass, fb, w = host
    s = fmm._subset_mask(pos, args.subset)
    p0 = np.asarray([0, 0, 0, 0, 0, 0, 1.0, args.dt], np.float32)
    lead = (p0[: args.k], p0[args.k :])
    arrays = lead + (pos, vel, mass, s, fb, w)
    shapes = tuple(jax.ShapeDtypeStruct(a.shape, jnp.float32) for a in arrays)
    t0 = time.perf_counter()
    compiled = _program(prov, args).lower(*shapes).compile()
    rec["compile_s"] = time.perf_counter() - t0
    fmm._memory(rec, compiled)
    print(
        f"{args.program}: compiled in {rec['compile_s']:.0f} s, temp "
        f"{fmm._gib(rec['temp_bytes'])} GiB",
        flush=True,
    )
    call = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
    rec["before"] = fmm._alloc(device)
    t0 = time.perf_counter()
    out = compiled(*call)
    jax.block_until_ready(out)
    rec["first_call_s"] = time.perf_counter() - t0
    _, tangent, bits = out
    first = _digest(tangent)
    rec["finite"] = bool(
        all(bool(jnp.all(jnp.isfinite(t))) for t in jax.tree_util.tree_leaves(tangent))
    )
    if args.program in ("jvp", "jacfwd"):
        rec["tangent"] = np.asarray(tangent).tolist()
    rec["bits"] = int(np.asarray(bits))
    del out, tangent
    ts = []
    last = None
    for i in range(args.reps):
        t0 = time.perf_counter()
        res = compiled(*call)
        jax.block_until_ready(res)
        ts.append(time.perf_counter() - t0)
        if i == args.reps - 1:
            last = _digest(res[1])
        del res
    rec["a_vs_a_bitwise"] = last == first if last is not None else None
    rec["digest"] = first
    rec["timing"] = _stats_ms(ts, args.steps) if ts else None
    rec["after"] = fmm._alloc(device)
    from jaccpot.runtime.fused_topology import overflow_ok

    rec["verdict_ok"] = bool(overflow_ok(jnp.asarray(rec["bits"], jnp.uint32)))
    n = float(args.n)
    peak = (rec.get("after") or {}).get("peak_bytes_in_use")
    rec["per_particle"] = {
        "temp": rec["temp_bytes"] / n,
        "peak": None if peak is None else peak / n,
    }
    rec["card_after"] = _card_state()
    rec["contended"] = (
        rec["card_before"]["other_processes"] != "none"
        or rec["card_after"]["other_processes"] != "none"
    )
    t = rec["timing"]
    if t:
        print(
            f"{args.program} K {args.k} dpp {args.directions_per_pass}: "
            f"{t['median_ms_per_step']:.1f} ms/step (median of {t['calls']}, IQR "
            f"{t['iqr_ms']:.1f} ms/call); peak {fmm._gib(peak)} GiB; bits "
            f"{rec['bits']:#x}; A-vs-A {rec['a_vs_a_bitwise']}",
            flush=True,
        )
    return rec


# --------------------------------------------------------------------------
# The tangents-per-program sweep.
# --------------------------------------------------------------------------


def _tpp_knob() -> Optional[tuple]:
    """Track K's tangents-per-program seams, or ``None`` if not installed."""
    try:
        from jaccpot.pallas import _tangent_grid  # type: ignore[attr-defined]
    except ImportError:
        return None
    return tuple(getattr(_tangent_grid, "TANGENT_SEAMS", ()))


#: the seam keys of ``_forward_seams.SEAM_TANGENT_OPS`` -> the env spelling
_SEAM_ENV = {
    "p2m": "P2M",
    "m2m": "M2M",
    "l2l": "L2L",
    "m2l": "M2L",
    "l2p": "L2P",
    "near": "NEAR",
}


def _record_seams(prov: Any, x: Any, m: Any, fb: Any) -> dict:
    """Every tangent op one forward-mode step calls: its operands and params.

    ``resolve_tangent`` is wrapped so each op call records its static params at
    trace time and its operands' VALUES at run time (``jax.debug.callback`` to
    host copies: some seams sit inside the step's loops, whose tracers cannot be
    returned), from one run of the jitted jvp of ``differentiable_step``. The
    first call per seam is kept; its operands come back as device arrays.
    """
    import jax
    import jax.numpy as jnp

    import jaccpot.runtime._forward_seams as fs
    from jaccpot.runtime.grad_options import ad_mode_override

    statics: dict = {}
    values: dict = {}
    original = fs.resolve_tangent

    def resolve(seam: str, choice: Optional[str] = None) -> tuple:
        op, source = original(seam, choice)

        def recording(*operands: Any, **params: Any) -> Any:
            if seam not in statics:
                statics[seam] = (source, params, len(operands))

                def store(*host: Any) -> None:
                    values.setdefault(seam, [np.array(h) for h in host])

                jax.debug.callback(store, *operands)
            return op(*operands, **params)

        return recording, source

    fs.resolve_tangent = resolve
    try:

        def program(y: Any, t: Any) -> Any:
            with ad_mode_override("forward"):
                _, t_out = jax.jvp(
                    lambda z: prov.differentiable_step(z, m, aux=(fb, None, None))[0],
                    (y,),
                    (t,),
                )
            return t_out

        t = jnp.ones_like(x) * 1e-3
        jax.block_until_ready(jax.jit(program)(x, t))
        jax.effects_barrier()
    finally:
        fs.resolve_tangent = original
    out: dict = {}
    for seam, (source, params, count) in statics.items():
        host = values.get(seam)
        if host is None or len(host) != count:
            continue
        out[seam] = (source, [jnp.asarray(h) for h in host], params)
    return out


def tpp_sweep(args: argparse.Namespace) -> dict:
    """The tangent op of every seam alone, per tangents-per-program."""
    import forward_mode_memory as fmm
    import jax
    import jax.numpy as jnp

    device = jax.devices()[0]
    rec: dict[str, Any] = {
        "config": {k: v for k, v in vars(args).items() if k != "summarize"},
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "device": str(device),
        "jax": jax.__version__,
        "card_before": _card_state(),
        "seams": {},
    }
    knob = _tpp_knob()
    rec["tpp_knob"] = knob
    if knob is None:
        rec["skipped"] = (
            "no tangents_per_program knob (jaccpot.pallas._tangent_grid) installed"
        )
        print(rec["skipped"], flush=True)
        return rec
    margs = argparse.Namespace(**{**vars(args), "program": "jvp"})
    prov, host = fmm._prepare(margs)
    rec["topology"] = fmm._topology(prov)
    pos, _, mass, fb, _ = host
    x, m = jnp.asarray(pos), jnp.asarray(mass)
    recorded = _record_seams(prov, x, m, jnp.asarray(fb))
    del x, m
    gc.collect()
    k = int(args.tpp_k)
    for seam, (source, ops, params) in recorded.items():
        env = f"JACCPOT_TANGENT_TPP_{_SEAM_ENV[seam]}"
        row: dict[str, Any] = {"source": source, "k": k, "tpp": {}}
        rec["seams"][seam] = row
        if source != "pallas" or _SEAM_ENV[seam] not in knob:
            row["skipped"] = f"tangent source {source!r} / no {env}"
            continue
        op, _ = __import__(
            "jaccpot.runtime._forward_seams", fromlist=["resolve_tangent"]
        ).resolve_tangent(seam, "pallas")
        num_primals = int(op.num_primals)
        primals = list(ops[:num_primals])
        tangents = [jnp.tile(t, (k,) + (1,) * (t.ndim - 1)) for t in ops[num_primals:]]
        saved = os.environ.get(env)
        for tpp in TPP_CHOICES:
            if tpp > k:
                continue
            os.environ[env] = str(tpp)
            jax.clear_caches()
            fn = jax.jit(lambda *a, _p=params: op(*a, **_p))
            try:
                t0 = time.perf_counter()
                compiled = fn.lower(*primals, *tangents).compile()
                compile_s = time.perf_counter() - t0
                jax.block_until_ready(compiled(*primals, *tangents))
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    jax.block_until_ready(compiled(*primals, *tangents))
                    ts.append(time.perf_counter() - t0)
                ma = compiled.memory_analysis()
                row["tpp"][str(tpp)] = {
                    "compile_s": compile_s,
                    "timing": _stats_ms(ts, 1),
                    "temp_bytes": (None if ma is None else int(ma.temp_size_in_bytes)),
                }
                print(
                    f"{seam} K {k} tpp {tpp}: compile {compile_s:.1f} s, median "
                    f"{1e3 * float(np.median(ts)):.2f} ms",
                    flush=True,
                )
                del compiled
            except Exception as exc:  # noqa: BLE001 -- one bad tpp must not end it
                row["tpp"][str(tpp)] = {"error": repr(exc)[:2000]}
                print(f"{seam} tpp {tpp}: FAILED {exc!r}"[:400], flush=True)
        if saved is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = saved
        del primals, tangents
        gc.collect()
    rec["card_after"] = _card_state()
    rec["contended"] = (
        rec["card_before"]["other_processes"] != "none"
        or rec["card_after"]["other_processes"] != "none"
    )
    return rec


# --------------------------------------------------------------------------
# The summary.
# --------------------------------------------------------------------------


def summarize(directory: str) -> list[dict]:
    """One line per row of a directory, with the per-step ratios over ``forward``.

    Parameters
    ----------
    directory : str
        Where the ``--out`` JSONs are.

    Returns
    -------
    list[dict]
        The table rows (also printed; the tpp sweeps after them).
    """
    recs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as fh:
            recs.append((os.path.basename(path), json.load(fh)))
    forward = {
        (r["config"]["n"], r["config"]["steps"]): r
        for _, r in recs
        if r.get("timing") and r["config"]["program"] == "forward"
    }
    rows = []
    for name, r in recs:
        if not r.get("timing"):
            continue
        c = r["config"]
        ref = forward.get((c["n"], c["steps"]))
        rows.append(
            {
                "file": name,
                "n": c["n"],
                "steps": c["steps"],
                "program": c["program"],
                "k": c["k"],
                "dpp": c.get("directions_per_pass"),
                "ms_per_step": r["timing"]["median_ms_per_step"],
                "over_forward": (
                    r["timing"]["median_ms"] / ref["timing"]["median_ms"]
                    if ref
                    else None
                ),
                "compile_s": r["compile_s"],
                "temp_bpp": r["per_particle"]["temp"],
                "peak_bpp": r["per_particle"]["peak"],
                "a_vs_a": r.get("a_vs_a_bitwise"),
                "bits": r.get("bits"),
                "contended": r.get("contended"),
            }
        )
    rows.sort(
        key=lambda q: (q["n"], PROGRAMS.index(q["program"]), q["k"], q["dpp"] or 0)
    )
    for q in rows:
        ratio = "-" if q["over_forward"] is None else f"{q['over_forward']:.2f}x"
        peak = "-" if q["peak_bpp"] is None else f"{q['peak_bpp']:.0f}"
        print(
            f"N {q['n']:>11,} T {q['steps']:>3} {q['program']:>7} K {q['k']} dpp "
            f"{q['dpp'] or '-':>2}  {q['ms_per_step']:9.1f} ms/step {ratio:>7}  "
            f"compile {q['compile_s']:7.0f} s  temp {q['temp_bpp']:.0f} peak {peak} "
            f"B/p  A-vs-A {q['a_vs_a']} bits {q['bits']}"
            + ("  CONTENDED" if q["contended"] else "")
        )
    for name, r in recs:
        if "seams" not in r:
            continue
        if r.get("skipped"):
            print(f"{name}: tpp sweep skipped ({r['skipped']})")
            continue
        for seam, row in r["seams"].items():
            cells = []
            for tpp, cell in row.get("tpp", {}).items():
                if "timing" in cell:
                    cells.append(
                        f"tpp {tpp}: {cell['timing']['median_ms']:.2f} ms "
                        f"(compile {cell['compile_s']:.0f} s)"
                    )
                else:
                    cells.append(f"tpp {tpp}: error")
            print(f"{name} {seam} K {row.get('k')}: " + "; ".join(cells or ["-"]))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    """Run one row, a tpp sweep, or ``--summarize``.

    Parameters
    ----------
    argv : Optional[list[str]]
        The arguments.

    Returns
    -------
    int
        The exit status.
    """
    args = _args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0
    _setup(args)
    rec = tpp_sweep(args) if args.tpp_sweep else run(args)
    rec["done"] = True
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        os.replace(args.out + ".tmp", args.out)
        print(f"wrote {args.out}", flush=True)
    if rec.get("contended"):
        print("CONTENDED: another process was on the card; do not quote the timings")
    if args.tpp_sweep:
        return 0
    return 0 if rec["finite"] and rec["verdict_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
