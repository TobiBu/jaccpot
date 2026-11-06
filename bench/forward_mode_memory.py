"""Device memory of forward mode (``ad_mode="forward"``) against tangents and steps (F8).

usage: python bench/forward_mode_memory.py --n N --program PROG --k K --steps T
       [--out-dir DIR] [--cpu-interpret]
       python bench/forward_mode_memory.py --summarize DIR

Functional and memory only: no timing is recorded or claimed. One row per process
(``--program``, ``--k``, ``--steps`` at ``--n``), on S9's operating point
(``bench/grad_rollout_timing.py``'s ``_prepare``: ``preset="galaxy"``, a clipped
Plummer sphere, float32, ``dehnen_error``, the galaxy leaf size and order,
``dt = 1e-3``) and its clean-allocator recipe: prepare, drop the solver's ``_last_*``
side channels, re-place the provider's template so it lands first, compile from
shapes, then place the inputs.

THE FUNCTION. A Sgr-like few-parameter map: a satellite subset (``--subset``
particles nearest to ``(5, 0, 0)``) shifted in phase space and rescaled in mass,
and the timestep::

    p = (dx, dy, dz, dvx, dvy, dvz, mass scale, dt)
    g(p) = (sum(w . x_T), sum(w . v_T)) of the T-step rollout from
           x0 + s dx, v0 + s dv, m (1 + s (scale - 1)), dt        (s the subset mask)

over ``FusedForceProvider.differentiable_step``, ``differentiable_rollout(...,
ad_mode="forward")`` (the plain scan GradientPlan's forward twins run per segment).
The first ``K`` entries of ``p`` are differentiated, the rest are held at their
values. Programs:

* ``jvp`` (``K = 1``): ``jax.jvp`` of ``g`` along the first parameter;
* ``jacfwd`` (any ``K``): ``jax.jacfwd`` of ``g`` over the first ``K`` -- the tangent
  kernels carry the ``K`` tangents on their tangent axis (up to 8 per launch);
* ``reverse``: S9's reference, ``jax.value_and_grad`` of ``sum(w . x_T) + sum(w . v_T)``
  in the start positions, velocities and masses with ``checkpoint="step"``
  (``grad_rollout_timing``'s ``grad`` program);
* ``plan_jvp`` (``K = 1``): :meth:`GradientPlan.jvp <jaccpot.production_grad.
  GradientPlan.jvp>` of the same loss along the start positions, after a
  :class:`~jaccpot.production.ProductionRunner` pre-pass of one segment of ``T``
  steps -- the user API's program (with its start force and chain glue).

Per row: the static sizes the program was traced at (``topology``: ``total_nodes``,
``num_levels`` -- the cascades' static level count --, ``leaf_capacity`` -- the near
field's padded leaf width ``W`` --, the expansion ``order`` and its coefficient count,
the far list's capacity); the compiled program's ``memory_analysis()`` (temporaries, arguments,
outputs) and the allocator's ``bytes_in_use`` before / ``peak_bytes_in_use`` after,
each also per particle; the verdict bits; finiteness; A-vs-A (two calls of the same
executable, the tangents / gradients' sha256 equal); the allocator environment
(``XLA_PYTHON_CLIENT_*``, set by the caller: prealloc at 0.95 for 1e8, S9-D).

Rows are JSON files ``n{N}_{program}_k{K}_t{T}.json`` in ``--out-dir`` (``..._g{g}.json``
with the tangents in groups of ``g``, ``JACCPOT_FORWARD_TANGENT_GROUP``); a row whose
file exists (with ``"done": true``) is skipped, so a sweep resumes.

``--summarize DIR`` prints the rows and fits, per ``(N, T, g)``, ``W_fwd(K) = a + b K``
to the forward rows' temporaries (``jvp`` as ``K = 1``), checks flatness in ``T``
(each forward row against its ``T = 1`` twin) and sets the reverse row beside it;
it writes ``DIR/summary.json``.

``--cpu-interpret`` runs on CPU with every Pallas kernel in interpret mode (the walk
the CPU's) and ``jax.default_backend`` reporting ``"gpu"``, on the strict tests' small
lane (``preset="large_n_gpu"``, leaf 16, order 2, theta 0.6, ``dehnen_error``,
float32): the smoke test of the script, not a measurement.
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

PROGRAMS = ("jvp", "jacfwd", "reverse", "plan_jvp")
#: the static sizes every row records (``_topology``) and the summary carries
TOPOLOGY_KEYS = ("total_nodes", "num_levels", "leaf_capacity", "order")
PARAMETERS = ("dx", "dy", "dz", "dvx", "dvy", "dvz", "mass_scale", "dt")
DEFAULT_OUT = "/export/scratch/tbuck/fwd-scratch/gpu/f8"
_ENV_KEYS = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "CUDA_VISIBLE_DEVICES",
    "JACCPOT_GRAD_FORWARD_TANGENT_KERNELS",
    "JACCPOT_FORWARD_TANGENT_GROUP",
    "JACCPOT_COMPILATION_CACHE_DIR",
    "XLA_FLAGS",
)
_INTERPRET_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}
# (the GPU walk in interpret mode only with --walk-interpret: at N = 2e4 its eager
# run did not finish in 30 min on CPU; the CPU's flat walk otherwise, whose compile
# took over 20 min at N = 4096 -- the strict tests' N = 512 runs either in seconds)


def _args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--program", default="jvp", choices=PROGRAMS)
    ap.add_argument("--k", type=int, default=1, help="tangents (1..8)")
    ap.add_argument("--steps", type=int, default=1, help="T, the rollout's steps")
    ap.add_argument("--subset", type=float, default=0.01, help="satellite fraction")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="redo an existing row")
    ap.add_argument(
        "--cpu-interpret",
        action="store_true",
        help="CPU smoke: Pallas interpret mode, the large-N gate open",
    )
    ap.add_argument(
        "--walk-interpret",
        action="store_true",
        help="with --cpu-interpret: the GPU walk in interpret mode too",
    )
    ap.add_argument(
        "--no-compact",
        action="store_true",
        help="keep the provider's template where the prepare left it",
    )
    ap.add_argument("--summarize", default=None, help="fit the rows of this directory")
    return ap.parse_args(argv)


def _row_path(args: argparse.Namespace) -> str:
    # tangents in groups (JACCPOT_FORWARD_TANGENT_GROUP): a row of its own
    group = os.environ.get("JACCPOT_FORWARD_TANGENT_GROUP", "").strip()
    tail = f"_g{group}" if group not in ("", "0") else ""
    return os.path.join(
        args.out_dir, f"n{args.n}_{args.program}_k{args.k}_t{args.steps}{tail}.json"
    )


def _alloc(device: Any) -> dict:
    stats = device.memory_stats() or {}
    return {
        k: stats.get(k)
        for k in (
            "bytes_in_use",
            "peak_bytes_in_use",
            "largest_free_block_bytes",
            "bytes_limit",
        )
    }


def _digest(*arrays: Any) -> str:
    import jax

    h = hashlib.sha256()
    for a in jax.tree_util.tree_leaves(arrays):
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _gib(v: Any) -> Any:
    return None if v is None else round(v / 2**30, 3)


# --------------------------------------------------------------------------
# The programs.
# --------------------------------------------------------------------------


def _subset_mask(pos: np.ndarray, fraction: float) -> np.ndarray:
    """The ``fraction`` of particles nearest to ``(5, 0, 0)``: a satellite."""
    n = int(pos.shape[0])
    count = max(1, int(round(fraction * n)))
    d2 = np.sum((pos - np.array([5.0, 0.0, 0.0])) ** 2, axis=1)
    mask = np.zeros(n, np.float32)
    mask[np.argpartition(d2, count - 1)[:count]] = 1.0
    return mask


def _g(prov: Any, args: argparse.Namespace) -> Any:
    """``g(theta, p_rest, x, v, m, s, fb, w) -> ((xsum, vsum), bits)`` in forward mode."""
    import jax.numpy as jnp

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def g(theta, p_rest, x, v, m, s, fb, w):
        p = jnp.concatenate([theta, p_rest])
        out = differentiable_rollout(
            prov.differentiable_step,
            x + s[:, None] * p[0:3],
            v + s[:, None] * p[3:6],
            m * (1.0 + s * (p[6] - 1.0)),
            p[7],
            args.steps,
            aux=(fb, None, None),
            checkpoint=None,
            ad_mode="forward",
        )
        sums = jnp.stack([jnp.sum(w * out.positions), jnp.sum(w * out.velocities)])
        return sums, out.aux[1]

    return g


def _program(prov: Any, args: argparse.Namespace) -> Any:
    """The jitted program of the row (``plan_jvp`` excepted)."""
    import jax
    import jax.numpy as jnp

    from jaccpot.runtime.grad_rollout import differentiable_rollout

    if args.program == "reverse":

        def loss(x, v, m, s, fb, w):
            del s
            out = differentiable_rollout(
                prov.differentiable_step,
                x,
                v,
                m,
                args.dt,
                args.steps,
                aux=(fb, None, None),
                checkpoint="step",
            )
            value = jnp.sum(w * out.positions) + jnp.sum(w * out.velocities)
            return value, out.aux[1]

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))
    g = _g(prov, args)
    if args.program == "jvp":

        def jvp(theta, p_rest, *rest):
            def f(t):
                return g(t, p_rest, *rest)

            (sums, bits), (t_sums, _) = jax.jvp(f, (theta,), (jnp.ones_like(theta),))
            return sums, t_sums, bits

        return jax.jit(jvp)

    def jac(theta, p_rest, *rest):
        def f(t):
            sums, bits = g(t, p_rest, *rest)
            return sums, (sums, bits)

        jacobian, (sums, bits) = jax.jacfwd(f, has_aux=True)(theta)
        return sums, jacobian, bits

    return jax.jit(jac)


#: the CPU smoke's lane: the strict tests' (``test_grad_rollout_differentiable.py``)
#: small configuration in float32 -- on CPU the galaxy preset's walk compiled for
#: over 45 min at N = 4096 (the slow-compile alarm on ``_flat_walk_lists``)
SMOKE_LEAF, SMOKE_ORDER, SMOKE_THETA = 16, 2, 0.6


def _smoke_solver(args: argparse.Namespace) -> Any:
    """The CPU smoke's solver (``--cpu-interpret``)."""
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    criterion: dict[str, Any] = dict(
        adaptive_eps=1e-5,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=SMOKE_THETA,
        G=1.0,
        softening=args.softening,
        working_dtype=jnp.float32,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=SMOKE_LEAF,
                leaf_partition="cells",
                leaf_capacity="auto",
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=SMOKE_ORDER,
        **criterion,
    )


def _prepare(args: argparse.Namespace) -> tuple:
    """The provider and the host inputs ``(pos, vel, mass, fb, w)``.

    On a GPU S9's (``grad_rollout_timing._prepare``: the galaxy preset on a clean
    allocator); with ``--cpu-interpret`` the smoke's small lane.
    """
    import grad_rollout_memory as grm
    import grad_rollout_timing as grt

    if not args.cpu_interpret:
        targs = argparse.Namespace(
            n=args.n,
            softening=args.softening,
            seed=args.seed,
            no_compact=args.no_compact,
        )
        return grt._prepare(targs)
    import jax.numpy as jnp

    from jaccpot import FusedForceProvider

    solver = _smoke_solver(args)
    pos, vel, mass = grm._plummer(args.n, args.seed)
    x0, m = jnp.asarray(pos, jnp.float32), jnp.asarray(mass, jnp.float32)
    prov, topo = FusedForceProvider.prepare(
        solver,
        x0,
        m,
        leaf_size=SMOKE_LEAF,
        max_order=SMOKE_ORDER,
        theta=SMOKE_THETA,
        differentiable=True,
    )
    fb = jnp.asarray(solver._impl._last_force_scale_particles, jnp.float32)
    perm = jnp.asarray(topo.prepared.tree.particle_indices)
    fb = np.asarray(jnp.zeros_like(fb).at[perm].set(fb))
    w = np.random.default_rng(args.seed + 1).normal(size=pos.shape)
    f32 = np.float32
    return prov, (
        pos.astype(f32),
        vel.astype(f32),
        mass.astype(f32),
        fb.astype(f32),
        w.astype(f32),
    )


def _plan_row(args: argparse.Namespace, rec: dict, device: Any) -> None:
    """``plan_jvp``: the production runner's pre-pass, then ``GradientPlan.jvp``."""
    import grad_rollout_memory as grm
    import jax
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod
    from jaccpot.production import (
        GALAXY_SETTINGS,
        GalaxySettings,
        ProductionRunner,
        SegmentConfig,
    )

    pos, vel, mass = grm._plummer(args.n, args.seed)
    w = np.random.default_rng(args.seed + 1).normal(size=pos.shape).astype(np.float32)
    if args.cpu_interpret:
        solver = _smoke_solver(args)
        settings = GalaxySettings(
            leaf_size=SMOKE_LEAF,
            max_order=SMOKE_ORDER,
            theta=SMOKE_THETA,
            mac_type="dehnen_error",
            adaptive_eps=1e-5,
            precision="fp32",
        )
    else:
        solver = FastMultipoleMethod(
            preset="galaxy", softening=args.softening, working_dtype=jnp.float32
        )
        settings = GALAXY_SETTINGS
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=args.steps, diagnostics_every=0),
    )
    t0 = time.perf_counter()
    state = jnp.stack([jnp.asarray(pos, jnp.float32), jnp.asarray(vel, jnp.float32)], 1)
    plan = runner.gradient_plan(
        state, jnp.asarray(mass, jnp.float32), dt=args.dt, num_steps=args.steps
    )
    del state
    jax.clear_caches()
    gc.collect()
    rec["prepare_s"] = time.perf_counter() - t0
    rec["after_prepare"] = _alloc(device)
    rec["topology"] = _topology(plan.segments[0].provider)
    upward = plan.segments[0].plan.get("upward_num_levels")
    if upward is not None:
        rec["topology"]["num_levels"] = int(upward)
    w_dev = jnp.asarray(w)

    def loss(result: Any, inputs: dict) -> Any:
        del inputs
        return jnp.sum(w_dev * result.positions) + jnp.sum(w_dev * result.velocities)

    wrt = ("positions",)
    diff, static, dynamic = plan._forward_split(None, wrt)
    program = plan._forward_program("jvp", loss, wrt, static, "outside")
    tangents = {
        "positions": jnp.asarray(_subset_mask(pos, args.subset))[:, None]
        * jnp.ones((1, 3), jnp.float32)
    }
    t0 = time.perf_counter()
    compiled = program.lower(diff, dynamic, tangents).compile()
    rec["compile_s"] = time.perf_counter() - t0
    _memory(rec, compiled)
    rec["before"] = _alloc(device)
    outs = []
    for _ in range(2):
        res = plan.jvp(None, tangents, wrt=wrt, fn=loss)
        outs.append((float(res.value), _digest(res.tangent), int(res.result.aux[1])))
        del res
    rec["after"] = _alloc(device)
    rec["bits"] = outs[0][2]
    rec["finite"] = bool(np.isfinite(outs[0][0]))
    rec["a_vs_a_bitwise"] = outs[0][1] == outs[1][1]


def _topology(prov: Any) -> dict:
    """The static sizes the row's program was traced at (for attributing W(K)).

    ``total_nodes`` / ``num_internal`` / ``num_leaves`` from the provider template's
    tree, ``leaf_capacity`` the near field's padded leaf width ``W`` (the template's
    ``nearfield_leaf_particle_indices``; else the grad statics' slot capacity),
    ``num_levels`` the cascades' static level count (the engine's planned upward
    depth bound, ``_cells_upward_num_levels``; else the tree's ``level_offsets``
    bound), ``order`` the expansion order and ``num_far_pairs`` the far list's
    capacity.
    """
    import jax

    out: dict[str, Any] = {
        "order": getattr(prov, "max_order", None),
        "leaf_size": getattr(prov, "leaf_size", None),
    }
    statics = getattr(prov, "grad_statics", None)
    if statics is not None:
        out["order"] = int(statics.order)
        out["max_leaf_size"] = int(statics.max_leaf_size)
        out["num_far_pairs"] = int(statics.num_far_pairs)
    tpl = getattr(prov, "template", None)
    if tpl is not None:
        state = jax.tree_util.tree_unflatten(tpl.treedef, list(tpl.leaves))
        tree = state.tree
        out["total_nodes"] = int(tree.parent.shape[0])
        out["num_internal"] = int(tree.left_child.shape[0])
        out["num_leaves"] = out["total_nodes"] - out["num_internal"]
        out["level_offsets_bound"] = int(tree.level_offsets.shape[0]) - 1
        idx = getattr(state, "nearfield_leaf_particle_indices", None)
        if idx is not None:
            out["leaf_capacity"] = int(idx.shape[-1])
    if "leaf_capacity" not in out and "max_leaf_size" in out:
        out["leaf_capacity"] = out["max_leaf_size"]
    engine = getattr(prov, "engine", None)
    levels = None
    for holder in (engine, getattr(engine, "_impl", None)):
        levels = getattr(holder, "_cells_upward_num_levels", None) or levels
    if levels is None:
        levels = out.get("level_offsets_bound")
    out["num_levels"] = None if levels is None else int(levels)
    if out.get("order") is not None:
        out["coefficients"] = (int(out["order"]) + 1) ** 2
    return out


def _memory(rec: dict, compiled: Any) -> None:
    ma = compiled.memory_analysis()
    rec["temp_bytes"] = int(ma.temp_size_in_bytes)
    rec["argument_bytes"] = int(ma.argument_size_in_bytes)
    rec["output_bytes"] = int(ma.output_size_in_bytes)
    rec["host_temp_bytes"] = int(getattr(ma, "host_temp_size_in_bytes", 0) or 0)


def run(args: argparse.Namespace) -> dict:
    """Measure one row; the JSON record."""
    import jax
    import jax.numpy as jnp

    if args.program in ("jvp", "plan_jvp") and args.k != 1:
        raise ValueError(f"--program {args.program} is K = 1")
    if not 1 <= args.k <= len(PARAMETERS):
        raise ValueError(f"--k must be 1..{len(PARAMETERS)}")
    device = jax.devices()[0]
    rec: dict[str, Any] = {
        "config": {k: v for k, v in vars(args).items() if k != "summarize"},
        "parameters": PARAMETERS[: args.k],
        "env": {k: os.environ.get(k) for k in _ENV_KEYS},
        "device": str(device),
        "backend": jax.default_backend(),
        "jax": jax.__version__,
    }
    if args.program == "plan_jvp":
        _plan_row(args, rec, device)
    else:
        t0 = time.perf_counter()
        prov, host = _prepare(args)
        rec["prepare_s"] = time.perf_counter() - t0
        rec["after_prepare"] = _alloc(device)
        rec["topology"] = _topology(prov)
        pos, vel, mass, fb, w = host
        s = _subset_mask(pos, args.subset)
        p0 = np.asarray([0, 0, 0, 0, 0, 0, 1.0, args.dt], np.float32)
        f32 = jnp.float32

        def spec(a: np.ndarray) -> Any:
            return jax.ShapeDtypeStruct(a.shape, f32)

        arrays = (pos, vel, mass, s, fb, w)
        if args.program == "reverse":
            shapes: tuple = (
                spec(pos),
                spec(vel),
                spec(mass),
                spec(s),
                spec(fb),
                spec(w),
            )
            lead: tuple = ()
        else:
            lead = (p0[: args.k], p0[args.k :])
            shapes = tuple(spec(a) for a in lead + arrays)
        t0 = time.perf_counter()
        compiled = _program(prov, args).lower(*shapes).compile()
        rec["compile_s"] = time.perf_counter() - t0
        _memory(rec, compiled)
        # the inputs after the compile, on the template's heels
        call = tuple(jnp.asarray(a, f32) for a in lead + arrays)
        rec["before"] = _alloc(device)
        digests: list[str] = []
        bits, finite = 0, True
        for _ in range(2):  # A-vs-A: two calls of the same executable
            out = compiled(*call)
            jax.block_until_ready(out)
            if args.program == "reverse":
                (value, b), grads = out
                digests.append(_digest(grads))
                finite = finite and bool(
                    all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
                    and bool(jnp.isfinite(value))
                )
            else:
                sums, tangent, b = out
                digests.append(_digest(tangent))
                finite = finite and bool(jnp.all(jnp.isfinite(tangent)))
                rec["value"] = [float(v) for v in np.asarray(sums)]
                rec["tangent"] = np.asarray(tangent).tolist()
            bits = int(np.asarray(b))
            del out
        rec["after"] = _alloc(device)
        rec["bits"] = bits
        rec["finite"] = finite
        rec["a_vs_a_bitwise"] = digests[0] == digests[1]
        rec["digest"] = digests[0]
    n = float(args.n)
    peak = (rec.get("after") or {}).get("peak_bytes_in_use")
    rec["per_particle"] = {
        "temp": rec["temp_bytes"] / n,
        "argument": rec["argument_bytes"] / n,
        "output": rec["output_bytes"] / n,
        "in_use_before": ((rec.get("before") or {}).get("bytes_in_use") or 0) / n,
        "peak": None if peak is None else peak / n,
    }
    from jaccpot.runtime.fused_topology import overflow_ok

    rec["verdict_ok"] = bool(overflow_ok(jnp.asarray(int(rec["bits"]), jnp.uint32)))
    rec["done"] = True
    return rec


# --------------------------------------------------------------------------
# The summary.
# --------------------------------------------------------------------------


def _group_of(rec: dict) -> int:
    """The row's tangent group (``JACCPOT_FORWARD_TANGENT_GROUP``; 0: off).

    Parameters
    ----------
    rec : dict
        A row.

    Returns
    -------
    int
        ``g``, or 0.
    """
    raw = str((rec.get("env") or {}).get("JACCPOT_FORWARD_TANGENT_GROUP") or "")
    return int(raw) if raw.strip().isdigit() else 0


def summarize(directory: str) -> dict:
    """Print the rows, fit ``W_fwd(K) = a + b K`` per ``(N, T)``, check T-flatness.

    Parameters
    ----------
    directory : str
        Where the rows are.

    Returns
    -------
    dict
        The summary (also written to ``DIR/summary.json``).
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "n*_*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if not rec.get("done"):
            continue
        c = rec["config"]
        rows.append(
            {
                "n": c["n"],
                "program": c["program"],
                "k": c["k"],
                "steps": c["steps"],
                "group": _group_of(rec),
                "temp_bpp": rec["per_particle"]["temp"],
                "peak_bpp": rec["per_particle"]["peak"],
                "temp_GiB": _gib(rec["temp_bytes"]),
                "peak_GiB": _gib((rec.get("after") or {}).get("peak_bytes_in_use")),
                "bits": rec["bits"],
                "finite": rec["finite"],
                "a_vs_a": rec["a_vs_a_bitwise"],
                **{key: (rec.get("topology") or {}).get(key) for key in TOPOLOGY_KEYS},
            }
        )
    rows.sort(
        key=lambda r: (
            r["n"],
            r["steps"],
            r["group"],
            PROGRAMS.index(r["program"]),
            r["k"],
        )
    )
    for r in rows:
        peak = "-" if r["peak_bpp"] is None else f"{r['peak_bpp']:.1f}"
        print(
            f"N {r['n']:>11,} T {r['steps']:>3} {r['program']:>8} K {r['k']} "
            f"g {r['group']}  "
            f"temp {r['temp_bpp']:8.1f} B/p ({r['temp_GiB']} GiB)  peak {peak} B/p "
            f"({r['peak_GiB']} GiB)  bits {r['bits']:#x} finite {r['finite']} "
            f"A-vs-A {r['a_vs_a']}  nodes {r['total_nodes']} levels "
            f"{r['num_levels']} W {r['leaf_capacity']} p {r['order']}"
        )
    fits: dict = {}
    forward = [r for r in rows if r["program"] in ("jvp", "jacfwd")]
    for n, t, g in sorted({(r["n"], r["steps"], r["group"]) for r in forward}):
        pts = sorted(
            {
                (r["k"], r["temp_bpp"])
                for r in forward
                if (r["n"], r["steps"], r["group"]) == (n, t, g)
            }
        )
        key = f"n{n}_t{t}" + (f"_g{g}" if g else "")
        if len(pts) >= 2:
            k = np.asarray([p[0] for p in pts], float)
            y = np.asarray([p[1] for p in pts], float)
            b, a = np.polyfit(k, y, 1)
            fits[key] = {"a_bpp": float(a), "b_bpp_per_k": float(b), "points": pts}
            print(f"N {n:,} T {t} g {g}: W_fwd(K) = {a:.1f} + {b:.1f} K  B/particle")
    flat = []
    for r in forward:
        if r["steps"] == 1:
            continue
        base = [
            q
            for q in forward
            if (q["n"], q["program"], q["k"], q["steps"], q["group"])
            == (r["n"], r["program"], r["k"], 1, r["group"])
        ]
        if base:
            ratio = r["temp_bpp"] / base[0]["temp_bpp"]
            flat.append(
                {
                    "n": r["n"],
                    "program": r["program"],
                    "k": r["k"],
                    "steps": r["steps"],
                    "temp_over_t1": ratio,
                }
            )
            print(
                f"N {r['n']:,} {r['program']} K {r['k']}: T {r['steps']} / T 1 "
                f"temporaries {ratio:.3f}"
            )
    summary = {"rows": rows, "fits": fits, "flat_in_t": flat}
    with open(os.path.join(directory, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    """Run one row (or ``--summarize``).

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
    path = _row_path(args)
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            if json.load(fh).get("done"):
                print(f"skip: {path} exists", flush=True)
                return 0
    if args.cpu_interpret:
        os.environ.update(_INTERPRET_ENV)
        if args.walk_interpret:
            os.environ["JACCPOT_WALK_PALLAS_INTERPRET"] = "1"
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    if args.cpu_interpret:
        jax.default_backend = lambda: "gpu"  # the large-N profile's backend gate
    jax.config.update("jax_enable_x64", True)
    from jaccpot.runtime.compile_cache import maybe_enable_compilation_cache

    maybe_enable_compilation_cache()  # opt-in: JACCPOT_COMPILATION_CACHE_DIR
    rec = run(args)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    os.replace(path + ".tmp", path)
    pp = rec["per_particle"]
    peak = "-" if pp["peak"] is None else f"{pp['peak']:.1f}"
    print(
        f"N {args.n:,} {args.program} K {args.k} T {args.steps}: temp "
        f"{pp['temp']:.1f} B/p ({_gib(rec['temp_bytes'])} GiB), peak {peak} B/p; "
        f"bits {rec['bits']:#x}, finite {rec['finite']}, A-vs-A "
        f"{rec['a_vs_a_bitwise']}; wrote {path}",
        flush=True,
    )
    return 0 if rec["finite"] and rec["a_vs_a_bitwise"] and rec["verdict_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
