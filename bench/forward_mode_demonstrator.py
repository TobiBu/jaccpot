"""Forward-mode demonstrator: a satellite's 7-parameter Jacobian through a long rollout.

usage: python bench/forward_mode_demonstrator.py --n N [--steps T]
       [--directions-per-pass k] --out OUT.json
       python bench/forward_mode_demonstrator.py --n N --check [--steps 4] --out OUT.json

``jax.jacfwd`` over seven parameters -- a satellite's 6D phase-space offset and its
mass scale, ``p = (dx, dy, dz, dvx, dvy, dvz, mass scale)`` applied to a satellite
subset of the particles (the ``--subset`` fraction nearest to ``(5, 0, 0)``, as in
``bench/forward_mode_memory.py``) -- of the satellite's final centre of mass
(position and velocity, mass-weighted over the subset) and the two weighted sums of
the whole system's final state: an ``8 x 7`` Jacobian through a ``T``-step rollout
of the production gradient lane (``FusedForceProvider.differentiable_step``,
``ad_mode="forward"``: a plain scan, memory flat in ``T``) at S9's operating point
(``preset="galaxy"``, clipped Plummer, float32, ``dehnen_error``, ``dt = 1e-3``).

The point: the reverse path stores per-step residuals and ran out of memory near
45 steps at 2.5e7 on an A100-40 (S9 / F8); forward mode runs ``T = 200`` (or any
``T``) in the memory of ``directions_per_pass`` tangents. At 1e8 pass
``--directions-per-pass 1`` (one direction at a time: ``K = 1`` peaks at 384 B/p,
35.8 GiB): seven passes, each recomputing the primal, plus one primal for the value.

``--check`` (short ``T``, default 4): the same Jacobian against the reverse path's
``value_and_grad`` of ``<w, f(p)>`` (``checkpoint="step"``) for a random output
weight ``w``: ``w^T J`` (forward) against ``J^T w`` (reverse) per parameter, and the
F2 identity ``<w, J d> == <J^T w, d>`` for a random ``d``. Float32 lane: agreement
to ~1e-4 relative is round-off (the exit status uses 1e-3).

Records ``compile_s``, the single call's wall time (``run_s``; a demonstrator, not a
timing row -- ``bench/forward_mode_timing.py`` has those), ``memory_analysis()``,
the allocator peak, the Jacobian, the verdict bits and the card's other processes.
``--cpu-interpret``: the small CPU lane (the script's smoke test).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PARAMETERS = ("dx", "dy", "dz", "dvx", "dvy", "dvz", "mass_scale")
OUTPUTS = ("sat_x", "sat_y", "sat_z", "sat_vx", "sat_vy", "sat_vz", "sum_wx", "sum_wv")


def _args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--n", type=int, default=25_000_000)
    ap.add_argument("--steps", type=int, default=None, help="T (200; --check 4)")
    ap.add_argument("--directions-per-pass", type=int, default=None)
    ap.add_argument("--subset", type=float, default=0.01, help="satellite fraction")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compact", action="store_true")
    ap.add_argument("--check", action="store_true", help="against value_and_grad")
    ap.add_argument("--cpu-interpret", action="store_true", help="CPU smoke")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.steps is None:
        args.steps = 4 if args.check else 200
    return args


def _objective(prov: Any, args: argparse.Namespace, ad_mode: str) -> Any:
    """``f(p, x, v, m, s, fb, w) -> (outputs[8], bits)``."""
    import jax.numpy as jnp

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def f(p, x, v, m, s, fb, w):
        m_sat = m * (1.0 + s * (p[6] - 1.0))
        out = differentiable_rollout(
            prov.differentiable_step,
            x + s[:, None] * p[0:3],
            v + s[:, None] * p[3:6],
            m_sat,
            args.dt,
            args.steps,
            aux=(fb, None, None),
            checkpoint=None if ad_mode == "forward" else "step",
            ad_mode=ad_mode,
        )
        ms = s * m_sat
        total = jnp.sum(ms)
        com_x = jnp.sum(ms[:, None] * out.positions, axis=0) / total
        com_v = jnp.sum(ms[:, None] * out.velocities, axis=0) / total
        sums = jnp.stack([jnp.sum(w * out.positions), jnp.sum(w * out.velocities)])
        return jnp.concatenate([com_x, com_v, sums]), out.aux[1]

    return f


def run(args: argparse.Namespace) -> dict:
    """The demonstrator (and with ``--check`` the reverse comparison); the record."""
    import forward_mode_memory as fmm
    import forward_mode_timing as fmt
    import jax
    import jax.numpy as jnp

    from jaccpot.runtime.forward_batching import jacfwd_batched
    from jaccpot.runtime.fused_topology import overflow_ok

    device = jax.devices()[0]
    rec: dict[str, Any] = {
        "config": vars(args),
        "parameters": PARAMETERS,
        "outputs": OUTPUTS,
        "env": {k: os.environ.get(k) for k in fmt._ENV_KEYS},
        "device": str(device),
        "jax": jax.__version__,
        "card_before": fmt._card_state(),
    }
    t0 = time.perf_counter()
    margs = argparse.Namespace(**{**vars(args), "program": "jvp", "k": 1})
    prov, host = fmm._prepare(margs)
    rec["prepare_s"] = time.perf_counter() - t0
    rec["topology"] = fmm._topology(prov)
    pos, vel, mass, fb, w = host
    s = fmm._subset_mask(pos, args.subset)
    p0 = np.asarray([0, 0, 0, 0, 0, 0, 1.0], np.float32)
    arrays = (p0, pos, vel, mass, s, fb, w)
    shapes = tuple(jax.ShapeDtypeStruct(a.shape, jnp.float32) for a in arrays)
    f = _objective(prov, args, "forward")

    def jac(p, *rest):
        return jacfwd_batched(
            lambda q: f(q, *rest),
            directions_per_pass=args.directions_per_pass,
            has_aux=True,
        )(p)

    t0 = time.perf_counter()
    compiled = jax.jit(jac).lower(*shapes).compile()
    rec["compile_s"] = time.perf_counter() - t0
    fmm._memory(rec, compiled)
    call = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
    rec["before"] = fmm._alloc(device)
    t0 = time.perf_counter()
    jacobian, bits = compiled(*call)
    jax.block_until_ready(jacobian)
    rec["run_s"] = time.perf_counter() - t0
    rec["after"] = fmm._alloc(device)
    J = np.asarray(jacobian, np.float64)
    rec["jacobian"] = J.tolist()
    rec["bits"] = int(np.asarray(bits))
    rec["verdict_ok"] = bool(overflow_ok(jnp.asarray(rec["bits"], jnp.uint32)))
    rec["finite"] = bool(np.all(np.isfinite(J)))
    peak = rec["after"].get("peak_bytes_in_use")
    rec["peak_bytes_per_particle"] = None if peak is None else peak / float(args.n)
    print(
        f"N {args.n:,} T {args.steps} dpp {args.directions_per_pass}: compile "
        f"{rec['compile_s']:.0f} s, run {rec['run_s']:.1f} s, peak "
        f"{fmm._gib(peak)} GiB; bits {rec['bits']:#x}, finite {rec['finite']}",
        flush=True,
    )
    with np.printoptions(precision=4, suppress=False, linewidth=160):
        print("J (rows " + ", ".join(OUTPUTS) + "; columns " + ", ".join(PARAMETERS))
        print(J)
    del compiled, jacobian
    if args.check:
        rng = np.random.default_rng(args.seed + 7)
        wo = rng.normal(size=len(OUTPUTS))
        d = rng.normal(size=len(PARAMETERS))
        fr = _objective(prov, args, "reverse")

        def scalar(p, *rest):
            out, b = fr(p, *rest)
            return jnp.dot(jnp.asarray(wo, out.dtype), out), b

        t0 = time.perf_counter()
        (value, rbits), grad = jax.jit(jax.value_and_grad(scalar, has_aux=True))(*call)
        jax.block_until_ready(grad)
        rec["reverse_s"] = time.perf_counter() - t0
        jt_w = np.asarray(grad, np.float64)
        wt_j = wo @ J
        rel = np.abs(wt_j - jt_w) / np.maximum(np.abs(jt_w), 1e-30)
        lhs, rhs = float(wo @ (J @ d)), float(jt_w @ d)
        rec["check"] = {
            "w": wo.tolist(),
            "d": d.tolist(),
            "wT_J_forward": wt_j.tolist(),
            "JT_w_reverse": jt_w.tolist(),
            "per_parameter_rel": rel.tolist(),
            "f2_lhs": lhs,
            "f2_rhs": rhs,
            "f2_rel": abs(lhs - rhs) / max(abs(rhs), 1e-30),
            "reverse_bits": int(np.asarray(rbits)),
        }
        rec["check_ok"] = bool(rec["check"]["f2_rel"] <= 1e-3 and np.all(rel <= 1e-3))
        print(
            f"check: <w, J d> {lhs:.8e} vs <J^T w, d> {rhs:.8e} (rel "
            f"{rec['check']['f2_rel']:.2e}); per-parameter max rel {rel.max():.2e}",
            flush=True,
        )
    rec["card_after"] = fmt._card_state()
    rec["contended"] = (
        rec["card_before"]["other_processes"] != "none"
        or rec["card_after"]["other_processes"] != "none"
    )
    return rec


def main(argv: Optional[list[str]] = None) -> int:
    """Run the demonstrator.

    Parameters
    ----------
    argv : Optional[list[str]]
        The arguments.

    Returns
    -------
    int
        The exit status (0: finite, verdict ok, and with ``--check`` the identity
        held).
    """
    import json

    import forward_mode_timing as fmt

    args = _args(argv)
    fmt._setup(args)
    rec = run(args)
    rec["done"] = True
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out + ".tmp", "w") as fh:
            json.dump(rec, fh, indent=1, default=str)
        os.replace(args.out + ".tmp", args.out)
        print(f"wrote {args.out}", flush=True)
    ok = rec["finite"] and rec["verdict_ok"] and rec.get("check_ok", True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
