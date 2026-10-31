"""Compiled memory of a gradient through ``differentiable_rollout``, against steps.

Compiles ``jax.grad`` of a loss over ``jaccpot.runtime.grad_rollout.
differentiable_rollout`` for each horizon ``T`` (``--steps``) and checkpoint mode
(``--modes``), and records XLA's ``compiled.memory_analysis()``: the temporaries,
the arguments and the outputs. Nothing is executed: the numbers are the compiled
program's buffer assignment, so they are card-independent and the same on every
run of one configuration.

The memory model (plan S9, G6) says the temporaries grow by one stored carry per
step under ``"step"`` and by ``n_outer + n_inner`` carries under ``"nested"``, the
carry being ``(x, v, a)`` plus the step aux's per-particle leaves: 36 B/particle in
float32, 72 in float64 (plus f_b's 4 under the fused lane's ``dehnen_error``). The
script fits ``temp = c + r * S(T)`` over the rows with ``T >= 2`` (XLA inlines a
one-step scan, which changes the buffer assignment) and prints ``r / N`` beside the
carry's bytes per particle. ``S(T)`` is ``T`` for ``None`` and ``"step"``, and
``n_outer + max(n_inner, tail + 1)`` for ``"nested"`` (a remainder runs as a
shorter last segment whose input is stored once more).

Forces:

* ``--force direct`` -- the softened direct sum (``direct_sum_step_fn``), on CPU
  at N ~ 4096. Its plain scan stores the ``[N, N]`` pair arrays per step: the row
  that shows what checkpointing removes.
* ``--force provider`` -- the fused lane's ``FusedForceProvider`` at production
  settings (``preset="galaxy"``, or ``--mac dehnen`` for the geometric MAC; a
  clipped Plummer sphere), on a GPU: ``--provider-step step`` with
  ``--forward-only`` (the forward rollout's memory: no reverse pass, so nothing is
  stored per step), or a differentiable step method's name (track A's
  ``differentiable_step``) for the gradient rows.

Run on one reserved card, one configuration per process::

    on-reserved-card python bench/grad_rollout_memory.py --force provider \\
        --n 2000000 --forward-only --out row.json
    python bench/grad_rollout_memory.py --force direct --n 4096 --dtype float32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Optional


def _args(argv: Optional[list[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", default="direct", choices=("direct", "provider"))
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="the state's dtype (the provider's working dtype too)",
    )
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["none", "step", "nested"],
        help="none|step|nested",
    )
    ap.add_argument(
        "--nested-steps",
        type=int,
        nargs="*",
        default=[4, 9, 16, 25, 36],
        help="nested: extra horizons for the fit (perfect squares: no remainder, "
        "inner >= 2)",
    )
    ap.add_argument(
        "--inner",
        type=int,
        default=None,
        help="nested: steps per segment (default round(sqrt(T)))",
    )
    ap.add_argument(
        "--forward-only",
        action="store_true",
        help="compile the jitted rollout, not its gradient",
    )
    ap.add_argument(
        "--provider-step",
        default="step",
        help="the provider method used as step_fn (step, or a differentiable one)",
    )
    ap.add_argument(
        "--mac",
        default="dehnen_error",
        choices=("dehnen_error", "dehnen"),
        help="provider: the galaxy preset's MAC, or the geometric one",
    )
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the rows as JSON here")
    return ap.parse_args(argv)


def _plummer(n: int, seed: int, rmax: float = 20.0):
    """A Plummer sphere clipped at ``rmax`` scale radii, small random velocities."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), rmax)
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = 0.1 * rng.normal(size=(n, 3))
    return pos, vel, np.full((n,), 1.0 / n)


def _direct_force(args: argparse.Namespace) -> tuple[Callable, Any]:
    from jaccpot.runtime.grad_rollout import direct_sum_step_fn

    return direct_sum_step_fn(G=1.0, softening=args.softening), None


def _provider_force(args: argparse.Namespace, x0, m) -> tuple[Callable, Any]:
    """The fused lane's step at production settings, and its first step's aux."""
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod, FusedForceProvider
    from jaccpot.production import GALAXY_SETTINGS

    kw: dict = {}
    if args.mac == "dehnen":
        kw = dict(mac_type="dehnen", adaptive_eps=None)
    solver = FastMultipoleMethod(
        preset="galaxy", softening=args.softening, working_dtype=x0.dtype, **kw
    )
    prov, topo = FusedForceProvider.prepare(
        solver,
        x0,
        m,
        leaf_size=GALAXY_SETTINGS.leaf_size,
        max_order=GALAXY_SETTINGS.max_order,
    )
    aux = None
    if prov.scale_carry:
        # strict_run_v2's first f_b: the eager prepare's eq (16b) prepass, back in
        # input order
        fb = jnp.asarray(solver._impl._last_force_scale_particles, x0.dtype)
        perm = jnp.asarray(topo.prepared.tree.particle_indices)
        aux = (jnp.zeros_like(fb).at[perm].set(fb), None, None)
    del topo
    return getattr(prov, args.provider_step), aux


def _segments(steps: int, inner: Optional[int]) -> tuple[int, int, int]:
    """``(inner, outer, tail)`` of a nested rollout, as the rollout splits it."""
    k = max(1, round(math.sqrt(steps))) if inner is None else int(inner)
    k = min(k, steps)
    outer, tail = divmod(steps, k)
    return k, outer, tail


def _stored_carries(mode: str, steps: int, inner: Optional[int]) -> int:
    """``S(T)``: the carries the model says the reverse pass stores."""
    if mode in ("none", "step"):
        return steps
    k, outer, tail = _segments(steps, inner)
    return outer + (max(k, tail + 1) if tail else k)


def _fit(rows: list[dict]) -> Optional[dict]:
    import numpy as np

    # T >= 2 (a one-step scan is inlined); nested: whole segments of >= 2 steps (a
    # one-step inner scan is inlined too, and a remainder adds its own buffers)
    use = [r for r in rows if r["steps"] >= 2 and r.get("fit", True)]
    if len(use) < 2 or len({r["stored_carries"] for r in use}) < 2:
        return None
    s = np.array([r["stored_carries"] for r in use], np.float64)
    t = np.array([r["temp_bytes"] for r in use], np.float64)
    slope, intercept = np.polyfit(s, t, 1)
    resid = t - (slope * s + intercept)
    return {
        "slope_bytes": float(slope),
        "intercept_bytes": float(intercept),
        "max_residual_bytes": float(np.max(np.abs(resid))),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = _args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    dtype = jnp.dtype(args.dtype)
    pos, vel, mass = _plummer(args.n, args.seed)
    x0, v0 = jnp.asarray(pos, dtype), jnp.asarray(vel, dtype)
    m = jnp.asarray(mass, dtype)
    w = jnp.asarray(np.random.default_rng(args.seed + 1).normal(size=pos.shape), dtype)
    if args.force == "direct":
        step_fn, aux0 = _direct_force(args)
    else:
        step_fn, aux0 = _provider_force(args, x0, m)
    # the carry's bytes per particle: x, v, a, plus the aux's [N, ...] leaves
    carry_bpp = 9 * dtype.itemsize
    aux_shape = jax.eval_shape(lambda x, mm, a: step_fn(x, mm, None, a)[1], x0, m, aux0)
    for leaf in jax.tree_util.tree_leaves(aux_shape):
        if leaf.ndim >= 1 and leaf.shape[0] == args.n:
            carry_bpp += leaf.dtype.itemsize * int(np.prod(leaf.shape[1:]))

    result: dict = {
        "config": {
            **vars(args),
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "carry_bytes_per_particle": carry_bpp,
        },
        "rows": [],
        "fits": {},
    }
    for mode in args.modes:
        checkpoint = None if mode == "none" else mode
        rows = []
        horizons = list(args.steps)
        if mode == "nested":
            horizons += [t for t in args.nested_steps if t not in horizons]
        for steps in horizons:
            inner = args.inner if mode == "nested" else None

            def rollout(x, v, mm, a, _steps=steps, _inner=inner):
                return differentiable_rollout(
                    step_fn,
                    x,
                    v,
                    mm,
                    args.dt,
                    _steps,
                    aux=a,
                    checkpoint=checkpoint,
                    inner_steps=None if _inner is None else min(_inner, _steps),
                )

            if args.forward_only:
                fn = jax.jit(rollout)
            else:

                def loss(x, v, mm, a, w_in, _rollout=rollout):
                    out = _rollout(x, v, mm, a)
                    return jnp.sum(w_in * out.positions) + jnp.sum(
                        w_in * out.velocities
                    )

                fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
            call = (x0, v0, m, aux0) + (() if args.forward_only else (w,))
            t0 = time.perf_counter()
            stats = fn.lower(*call).compile().memory_analysis()
            row = {
                "mode": mode,
                "steps": int(steps),
                "inner": inner,
                "stored_carries": _stored_carries(mode, steps, inner),
                "fit": mode != "nested"
                or (
                    _segments(steps, inner)[0] >= 2 and _segments(steps, inner)[2] == 0
                ),
                "temp_bytes": int(stats.temp_size_in_bytes),
                "argument_bytes": int(stats.argument_size_in_bytes),
                "output_bytes": int(stats.output_size_in_bytes),
                "alias_bytes": int(stats.alias_size_in_bytes),
                "compile_s": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(
                f"{mode:>6} T={steps:<3} S={row['stored_carries']:<3} "
                f"temp {row['temp_bytes'] / 2**20:10.2f} MiB  "
                f"temp/N {row['temp_bytes'] / args.n:9.1f} B  "
                f"args {row['argument_bytes'] / 2**20:8.2f} MiB  "
                f"compile {row['compile_s']} s",
                flush=True,
            )
        fit = _fit(rows)
        if fit is not None:
            fit["slope_bytes_per_particle"] = fit["slope_bytes"] / args.n
            fit["ratio_to_carry"] = fit["slope_bytes_per_particle"] / carry_bpp
            print(
                f"{mode:>6} slope {fit['slope_bytes_per_particle']:.2f} B/particle per "
                f"stored carry (carry {carry_bpp} B/particle, ratio "
                f"{fit['ratio_to_carry']:.4f}; max residual "
                f"{fit['max_residual_bytes']:.0f} B)",
                flush=True,
            )
        result["rows"] += rows
        result["fits"][mode] = fit
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
