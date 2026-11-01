"""Compiled memory of a gradient through ``differentiable_rollout``, against steps.

Compiles ``jax.grad`` of a loss over ``jaccpot.runtime.grad_rollout.
differentiable_rollout`` for each horizon ``T`` (``--steps``) and checkpoint mode
(``--modes``), and records XLA's ``compiled.memory_analysis()``: the temporaries,
the arguments and the outputs. Nothing is executed: the numbers are the compiled
program's buffer assignment, so they are card-independent and the same on every
run of one configuration.

The memory model (plan S9, G6): ``peak ~ W_step + S r``. Under ``"step"`` the
reverse pass stores one STEP RESIDUAL per step -- the drifted positions plus the
aux's per-particle leaves (12 B/particle in float32, 24 in float64, +4 for the
fused lane's f_b under ``dehnen_error``), since the reverse recomputes the force
and the kick from the drifted positions (``_adjoint_rollout``). Under ``"nested"`` it stores
one CARRY ``(x, v, a, aux)`` per segment (36 B/particle float32, + f_b) and one
step residual per step of a segment. The script fits, over the rows with
``T >= 2`` (XLA inlines a one-step scan):

* ``"step"`` and ``None``: ``temp = W + r T``, and prints ``r / N`` (B/particle
  per step) and ``W``, the temporaries a horizon of zero steps would need;
* ``"nested"``: ``temp = W + a n_outer + b n_inner`` over whole segments of at
  least 2 steps, and prints ``a / N`` (the carry) and ``b / N`` (the step
  residual). Horizons ``T:inner`` pin the segment length, so the two counts vary
  independently;
* ``"offload"`` (``"nested"`` with its carries and step residuals in pinned host
  memory): the same model fitted twice, to the DEVICE temporaries (both slopes
  ~0: flat in ``T`` and ``inner``) and to the HOST temporaries
  (``host_temp_size_in_bytes``: ``a`` the carry, ``b`` the step residual). Each
  row also lists the program's host transfers, read off the compiled HLO: every
  ``dynamic-update-slice`` into a host buffer (``S(5)``, device -> host) and every
  ``dynamic-slice`` out of one (host -> device), with the bytes per particle it
  moves per execution and its loop (the forward's outer scan, or the reverse
  pass's segment recompute).

``--execute`` also RUNS the gradient at the largest horizon of ``--execute-mode``
(default ``"step"``) and records the allocator's peak (``memory_stats()``), to
check the model against a measured peak. The peak is the process's: one executed
mode per process. ``--check`` runs the GPU functional gate instead: the
forward-only rollout over ``differentiable_step`` against the one over ``step``
(bitwise), and the gradient twice (finite, and A-vs-A bitwise).

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
import re
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
        help="none|step|nested|offload",
    )
    ap.add_argument(
        "--nested-steps",
        nargs="*",
        default=["4", "9", "16", "8:2", "16:2", "12:3", "24:3"],
        help="nested / offload: extra horizons for the fit, T or T:inner",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="also run the gradient at the largest horizon of --execute-mode "
        "(measured peak)",
    )
    ap.add_argument(
        "--execute-mode",
        default="step",
        help="the mode --execute runs (one per process: the peak is the process's)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="the functional gate (forward bitwise to step, gradient A-vs-A) "
        "instead of the memory rows",
    )
    ap.add_argument(
        "--inner",
        type=int,
        default=None,
        help="nested / offload: steps per segment (default round(sqrt(T)))",
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
    differentiable = args.check or args.provider_step != "step"
    prov, topo = FusedForceProvider.prepare(
        solver,
        x0,
        m,
        leaf_size=GALAXY_SETTINGS.leaf_size,
        max_order=GALAXY_SETTINGS.max_order,
        **({"differentiable": True} if differentiable else {}),
    )
    aux = None
    if prov.scale_carry:
        # strict_run_v2's first f_b: the eager prepare's eq (16b) prepass, back in
        # input order
        fb = jnp.asarray(solver._impl._last_force_scale_particles, x0.dtype)
        perm = jnp.asarray(topo.prepared.tree.particle_indices)
        aux = (jnp.zeros_like(fb).at[perm].set(fb), None, None)
    del topo
    if args.check:
        return prov, aux
    return getattr(prov, args.provider_step), aux


def _segments(steps: int, inner: Optional[int]) -> tuple[int, int, int]:
    """``(inner, outer, tail)`` of a nested rollout, as the rollout splits it."""
    k = max(1, round(math.sqrt(steps))) if inner is None else int(inner)
    k = min(k, steps)
    outer, tail = divmod(steps, k)
    return k, outer, tail


def _counts(mode: str, steps: int, inner: Optional[int]) -> tuple[int, int]:
    """``(carries, step residuals)`` the model says the reverse pass stores."""
    if mode in ("none", "step"):
        return 0, steps
    k, outer, tail = _segments(steps, inner)
    # a remainder segment's input is one more carry; its recompute stores its own
    # step residuals, in buffers the inner segments' may share
    return outer + (1 if tail else 0), max(k, tail)


def _fit(rows: list[dict], mode: str, key: str = "temp_bytes") -> Optional[dict]:
    import numpy as np

    # T >= 2 (a one-step scan is inlined); nested: whole segments of >= 2 steps (a
    # one-step inner scan is inlined too, and a remainder adds its own buffers)
    use = [r for r in rows if r["steps"] >= 2 and r["fit"]]
    t = np.array([r[key] for r in use], np.float64)
    if mode in ("nested", "offload"):
        a = np.array([[1.0, r["carries"], r["step_residuals"]] for r in use])
        if len(use) < 3 or np.linalg.matrix_rank(a) < 3:
            return None
        coef = np.linalg.lstsq(a, t, rcond=None)[0]
        resid = t - a @ coef
        return {
            "W_bytes": float(coef[0]),
            "carry_bytes": float(coef[1]),
            "step_bytes": float(coef[2]),
            "max_residual_bytes": float(np.max(np.abs(resid))),
        }
    s = np.array([r["step_residuals"] for r in use], np.float64)
    if len(set(s)) < 2:
        return None
    slope, intercept = np.polyfit(s, t, 1)
    resid = t - (slope * s + intercept)
    return {
        "W_bytes": float(intercept),
        "step_bytes": float(slope),
        "max_residual_bytes": float(np.max(np.abs(resid))),
    }


_SHAPE = re.compile(r"^\s*(?:ROOT )?%(\S+) = (\w+)\[([\d,]*)\]\{([^}]*)\}")
_ITEMSIZE = {"f64": 8, "f32": 4, "s32": 4, "u32": 4, "s64": 8, "u64": 8, "f16": 2}


def _host_transfers(hlo: str, n: int) -> list[dict]:
    """The compiled program's transfers to and from host memory, per site.

    A device -> host write is a ``dynamic-update-slice`` whose result is a host
    buffer (layout memory space ``S(5)``); a host -> device read a
    ``dynamic-slice`` of one. Each site moves its slice once per execution of the
    loop body it sits in; ``loop`` names that loop from the op's metadata.

    Parameters
    ----------
    hlo : str
        ``compiled.as_text()``.
    n : int
        Particles, for the bytes per particle.

    Returns
    -------
    list[dict]
        ``{"direction", "bytes_per_particle", "loop"}`` per site.
    """
    shapes: dict[str, tuple[str, int, bool]] = {}
    lines = hlo.splitlines()
    for line in lines:
        m = _SHAPE.match(line)
        if m:
            name, dt, dims, layout = m.groups()
            size = _ITEMSIZE.get(dt, 4) * math.prod(
                int(d) for d in dims.split(",") if d
            )
            shapes[name] = (dt, size, "S(5)" in layout)
    sites = []
    for line in lines:
        m = _SHAPE.match(line)
        if not m:
            continue
        name, _, _, layout = m.groups()
        op = re.search(r"\} (dynamic-update-slice|dynamic-slice)\(([^)]*)\)", line)
        if not op:
            continue
        operands = [o.strip().lstrip("%") for o in op.group(2).split(",")]
        meta = re.search(r'op_name="([^"]*)"', line)
        where = meta.group(1) if meta else ""
        loop = (
            "reverse: segment recompute"
            if "rematted_computation/while" in where
            else ("reverse" if "transpose(" in where else "forward")
        )
        if op.group(1) == "dynamic-update-slice" and "S(5)" in layout:
            size = shapes.get(operands[1], ("", 0, False))[1]
            sites.append({"direction": "to_host", "bytes": size, "loop": loop})
        elif op.group(1) == "dynamic-slice" and shapes.get(operands[0], (0, 0, 0))[2]:
            sites.append(
                {"direction": "to_device", "bytes": shapes[name][1], "loop": loop}
            )
    for site in sites:
        site["bytes_per_particle"] = site.pop("bytes") / n
    return sites


def _bitwise(a, b) -> int:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return int(np.sum(a.view(np.uint8) != b.view(np.uint8)))


def _check(args, prov, aux0, x0, v0, m, w) -> dict:
    """The GPU functional gate: forward bitwise to ``step``, gradient A-vs-A."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.runtime.grad_rollout import differentiable_rollout, rollout_ok

    steps = max(args.steps)
    out: dict = {"steps": steps}

    def rollout(step_fn, checkpoint):
        def run(x, v, mm, a):
            return differentiable_rollout(
                step_fn, x, v, mm, args.dt, steps, aux=a, checkpoint=checkpoint
            )

        return run

    ref = jax.jit(rollout(prov.step, "step"))(x0, v0, m, aux0)
    out["forward_ok"] = rollout_ok(ref)
    for mode in args.modes:
        checkpoint = None if mode == "none" else mode
        got = jax.jit(rollout(prov.differentiable_step, checkpoint))(x0, v0, m, aux0)
        out[f"forward_{mode}_elements_differ"] = sum(
            _bitwise(g, r) for g, r in zip(got[:3], ref[:3])
        )
        print(f"forward {mode}: {out[f'forward_{mode}_elements_differ']} differ")

        def loss(x, v, mm, a, w_in, _run=rollout(prov.differentiable_step, checkpoint)):
            res = _run(x, v, mm, a)
            value = jnp.sum(w_in * res.positions) + jnp.sum(w_in * res.velocities)
            return value, res

        grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))
        (_, res_a), g_a = grad(x0, v0, m, aux0, w)
        (_, res_b), g_b = grad(x0, v0, m, aux0, w)
        out[f"grad_{mode}_finite"] = all(bool(np.all(np.isfinite(g))) for g in g_a)
        out[f"grad_{mode}_ok"] = rollout_ok(res_a, gradient=True)
        out[f"grad_{mode}_a_vs_a_differ"] = sum(
            _bitwise(a, b) for a, b in zip(g_a, g_b)
        )
        out[f"grad_{mode}"] = [g for g in g_a]
        print(
            f"grad {mode}: finite {out[f'grad_{mode}_finite']}, verdict "
            f"{out[f'grad_{mode}_ok']}, A-vs-A {out[f'grad_{mode}_a_vs_a_differ']} "
            "differ",
            flush=True,
        )
    modes = [k for k in out if k.startswith("grad_") and isinstance(out[k], list)]
    for i, ref_key in enumerate(modes):
        for key in modes[i + 1 :]:
            rel = [
                float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                / max(float(np.max(np.abs(np.asarray(b)))), 1e-300)
                for a, b in zip(out[key], out[ref_key])
            ]
            differ = sum(_bitwise(a, b) for a, b in zip(out[key], out[ref_key]))
            out[f"{key}_vs_{ref_key}_maxrel"] = rel
            out[f"{key}_vs_{ref_key}_differ"] = differ
            print(f"{key} vs {ref_key}: max rel {max(rel):.2e}, {differ} differ")
    for key in modes:
        del out[key]
    return out


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
    if args.check:
        result = {"config": {**vars(args), "device": str(jax.devices()[0])}}
        result["check"] = _check(args, step_fn, aux0, x0, v0, m, w)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(result, fh, indent=1, default=str)
        return 0
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
        horizons = [(int(t), args.inner) for t in args.steps]
        if mode in ("nested", "offload"):
            for item in args.nested_steps:
                t, _, k = str(item).partition(":")
                pair = (int(t), int(k) if k else args.inner)
                if pair not in horizons:
                    horizons.append(pair)
        for steps, inner in horizons:
            inner = inner if mode in ("nested", "offload") else None

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
            compiled = fn.lower(*call).compile()
            stats = compiled.memory_analysis()
            row = {
                "mode": mode,
                "steps": int(steps),
                "inner": inner,
                "carries": _counts(mode, steps, inner)[0],
                "step_residuals": _counts(mode, steps, inner)[1],
                "fit": mode not in ("nested", "offload")
                or (
                    _segments(steps, inner)[0] >= 2 and _segments(steps, inner)[2] == 0
                ),
                "temp_bytes": int(stats.temp_size_in_bytes),
                "argument_bytes": int(stats.argument_size_in_bytes),
                "output_bytes": int(stats.output_size_in_bytes),
                "alias_bytes": int(stats.alias_size_in_bytes),
                "host_temp_bytes": int(stats.host_temp_size_in_bytes),
                "compile_s": round(time.perf_counter() - t0, 1),
            }
            if mode == "offload" and not args.forward_only:
                row["host_transfers"] = _host_transfers(compiled.as_text(), args.n)
            rows.append(row)
            print(
                f"{mode:>6} T={steps:<3} inner={inner} "
                f"temp {row['temp_bytes'] / 2**20:10.2f} MiB  "
                f"temp/N {row['temp_bytes'] / args.n:9.1f} B  "
                f"args {row['argument_bytes'] / 2**20:8.2f} MiB  "
                f"host temp {row['host_temp_bytes'] / 2**20:10.2f} MiB  "
                f"compile {row['compile_s']} s",
                flush=True,
            )
            for site in row.get("host_transfers", ()):
                print(
                    f"        {site['direction']:>9} {site['bytes_per_particle']:6.2f} "
                    f"B/particle per execution  ({site['loop']})"
                )
        fit = _fit(rows, mode)
        if fit is not None:
            per = {k: v / args.n for k, v in fit.items() if k.endswith("_bytes")}
            fit.update(
                {k.replace("_bytes", "_per_particle"): v for k, v in per.items()}
            )
            print(
                f"{mode:>6} fit: W {fit['W_bytes'] / 2**20:.2f} MiB "
                f"({per['W_bytes']:.1f} B/particle), step residual "
                f"{per['step_bytes']:.2f} B/particle"
                + (
                    f", carry {per['carry_bytes']:.2f} B/particle"
                    if "carry_bytes" in per
                    else ""
                )
                + f" (state carry {carry_bpp} B/particle; max residual "
                f"{fit['max_residual_bytes']:.0f} B)",
                flush=True,
            )
        result["rows"] += rows
        result["fits"][mode] = fit
        if mode == "offload":
            host_fit = _fit(rows, mode, key="host_temp_bytes")
            if host_fit is not None:
                host_fit.update(
                    {
                        k.replace("_bytes", "_per_particle"): v / args.n
                        for k, v in host_fit.items()
                        if k.endswith("_bytes")
                    }
                )
                print(
                    f"{mode:>6} HOST fit: W {host_fit['W_bytes'] / 2**20:.2f} MiB, "
                    f"carry {host_fit['carry_per_particle']:.2f} B/particle, step "
                    f"residual {host_fit['step_per_particle']:.2f} B/particle (max "
                    f"residual {host_fit['max_residual_bytes']:.0f} B)",
                    flush=True,
                )
            result["fits"]["offload_host"] = host_fit
        if args.execute and mode == args.execute_mode and not args.forward_only:
            # the measured peak at the largest horizon: the last compiled program
            device = jax.devices()[0]
            before = (device.memory_stats() or {}).get("peak_bytes_in_use")
            jax.block_until_ready(fn(*call))
            stats_after = device.memory_stats() or {}
            result["executed"] = {
                "steps": int(steps),
                "peak_bytes_before": before,
                "peak_bytes": stats_after.get("peak_bytes_in_use"),
                "bytes_in_use_after": stats_after.get("bytes_in_use"),
                "temp_bytes": row["temp_bytes"],
                "argument_bytes": row["argument_bytes"],
                "output_bytes": row["output_bytes"],
            }
            print(f"executed T={steps}: {result['executed']}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
