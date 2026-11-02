"""S1b: the particle-major L2P forward, Triton-Pallas vs Mosaic GPU, at the galaxy preset.

``jaccpot.pallas.l2p_real_mosaic`` ports the Triton L2P forward
(``jaccpot.pallas.l2p_real``) to Mosaic GPU. This bench compares them on the inputs of a
real force evaluation at ``preset="galaxy"`` (leaf 64, p6, ``dehnen_error``, fp32):

* ``--mode check`` (a local card, functional): the L2P alone per kernel -- bitwise
  against Triton, and the max relative difference of each against an fp64 reference
  (the ``"xla"`` autodiff form on float64 inputs); then one full force step
  (``FusedForceProvider.step``) per kernel, bitwise against Triton's; and the compiled
  temp bytes of both programs per kernel. No timings.
* ``--mode time`` (HoreKa, one card of an exclusive node): the same programs, compiled
  once each (AOT), timed in interleaved repetitions with an A-vs-A control (Triton
  compiled twice: the spread of identical programs is the noise floor).

The kernel is selected at trace time by ``JACCPOT_L2P_KERNEL`` (``pallas``,
``mosaic``, ``mosaic_select``); each program is lowered and compiled with its value
set, so the compiled executables are fixed afterwards.

    python bench/s1b_mosaic_l2p.py --mode check --n 200000 --out row.json
    python bench/s1b_mosaic_l2p.py --mode time --n 25000000 --reps 7 --out row.json
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("check", "time"), default="check")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument(
        "--ic",
        choices=("disc", "plummer"),
        default="disc",
        help="disc: a prefix of the shuffled disc+bulge IC (n <= its size)",
    )
    ap.add_argument(
        "--disc-ic",
        default=os.environ.get(
            "DISC_IC", "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
        ),
    )
    ap.add_argument(
        "--kernels",
        default="pallas,mosaic,mosaic_select",
        help="JACCPOT_L2P_KERNEL values; the first is the reference",
    )
    ap.add_argument("--reps", type=int, default=5, help="time: interleaved rounds")
    ap.add_argument("--inner", type=int, default=5, help="time: calls per round")
    ap.add_argument("--no-step", action="store_true", help="the L2P alone only")
    ap.add_argument("--no-fp64", action="store_true", help="check: skip fp64 ref")
    ap.add_argument("--out", default=None, help="JSON row (appended)")
    return ap.parse_args()


def _ic(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, float]:
    if args.ic == "plummer":
        sys.path.insert(0, str(HERE))
        from dce_pins import _plummer

        pos, _, mass = _plummer(args.n)
        return pos, mass, 1.0, 1e-3
    ic = np.load(args.disc_ic)
    n_all = int(ic["state0"].shape[0])
    if args.n > n_all:
        raise SystemExit(f"--n {args.n} > the disc IC's {n_all}: use --ic plummer")
    # the IC is shuffled: a prefix is a random subsample, rescaled to the galaxy
    # (bench/multigpu_force_repeat.py's convention)
    pos = np.asarray(ic["state0"][: args.n, 0], np.float32)
    mass = np.asarray(
        np.asarray(ic["mass"][: args.n], np.float64) * (n_all / args.n), np.float32
    )
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(args.n / 1e5))
    return pos, mass, g, soft


def _temp(compiled) -> dict:
    ma = compiled.memory_analysis()
    return {
        "temp_bytes": int(ma.temp_size_in_bytes),
        "argument_bytes": int(ma.argument_size_in_bytes),
        "output_bytes": int(ma.output_size_in_bytes),
    }


def _compile_with(kernel: str, fn, *args):
    os.environ["JACCPOT_L2P_KERNEL"] = kernel
    try:
        return fn.lower(*args).compile()
    finally:
        os.environ.pop("JACCPOT_L2P_KERNEL", None)


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return {
        "differing_elements": int(np.sum(a32.view(np.uint32) != b32.view(np.uint32))),
        "elements": int(a32.size),
        "max_abs": float(np.max(np.abs(a32.astype(np.float64) - b32))),
    }


def _rel_to(ref: np.ndarray, x: np.ndarray) -> dict:
    ref = np.asarray(ref, np.float64)
    d = np.linalg.norm(np.asarray(x, np.float64) - ref, axis=1)
    r = np.linalg.norm(ref, axis=1)
    ok = r > 0
    rel = d[ok] / r[ok]
    return {
        "max_rel": float(np.max(rel)),
        "p999_rel": float(np.quantile(rel, 0.999)),
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(r)),
    }


def main() -> int:
    args = _args()
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # yggdrax's Morton codes need it

    from jaccpot import FastMultipoleMethod, FusedForceProvider
    from jaccpot.downward.local_expansions import LocalExpansionData
    from jaccpot.production import GALAXY_SETTINGS
    from jaccpot.runtime.dtypes import INDEX_DTYPE
    from jaccpot.runtime.kernels._evaluate import (
        _evaluate_local_expansions_particle_major,
    )

    dev = jax.devices()[0]
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    row: dict = {
        "bench": "s1b_mosaic_l2p",
        "mode": args.mode,
        "n": args.n,
        "ic": args.ic,
        "kernels": kernels,
        "device": str(dev.device_kind),
        "jax": jax.__version__,
        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": os.uname().nodename,
    }
    pos, mass, g, soft = _ic(args)
    row.update(G=g, softening=soft)
    fmm = FastMultipoleMethod(
        preset="galaxy", G=g, softening=soft, working_dtype=jnp.float32
    )
    leaf, order = GALAXY_SETTINGS.leaf_size, GALAXY_SETTINGS.max_order
    row.update(leaf_size=leaf, order=order)
    p, m = jnp.asarray(pos), jnp.asarray(mass)
    t0 = time.perf_counter()
    state = fmm.prepare_state(p, m, leaf_size=leaf, max_order=order)
    row["prepare_s"] = time.perf_counter() - t0

    # ---- the L2P alone, on the prepared state's own inputs ---------------------
    local = state.local_data
    leaf_nodes = jnp.asarray(state.neighbor_list.leaf_indices, INDEX_DTYPE)
    ranges = jnp.asarray(state.tree.node_ranges, INDEX_DTYPE)
    xs = jnp.asarray(state.positions_sorted)
    counts = np.asarray(ranges[leaf_nodes][:, 1] - ranges[leaf_nodes][:, 0] + 1)
    row["leaves"] = {
        "num": int(leaf_nodes.shape[0]),
        "live": int(np.sum(counts > 0)),
        "mean_live_occupancy": float(np.mean(counts[counts > 0])),
        "max": int(counts.max()),
    }

    def l2p(coef, cent, x, ln, nr, *, kernel):
        data = LocalExpansionData(order=order, centers=cent, coefficients=coef)
        return _evaluate_local_expansions_particle_major(
            data, x, leaf_nodes=ln, node_ranges=nr, order=order, chunk=1 << 21,
            kernel=kernel,
        )  # fmt: skip

    l2p_args = (local.coefficients, local.centers, xs, leaf_nodes, ranges)
    compiled_l2p = {}
    names = list(kernels)
    if args.mode == "time":
        names.insert(1, kernels[0] + "#ctrl")  # A-vs-A: the reference twice
    for name in names:
        kernel = name.split("#")[0]
        t0 = time.perf_counter()
        try:
            fn = jax.jit(functools.partial(l2p, kernel=kernel))
            compiled_l2p[name] = _compile_with(kernel, fn, *l2p_args)
        except Exception as exc:  # noqa: BLE001
            row.setdefault("errors", {})[f"l2p/{name}"] = repr(exc)[:600]
            print(f"l2p {name}: FAILED {exc!r}"[:800], flush=True)
            continue
        row.setdefault("l2p_compile_s", {})[name] = time.perf_counter() - t0
        row.setdefault("l2p_memory", {})[name] = _temp(compiled_l2p[name])
    ref_name = names[0]

    def timed(fn, a) -> float:
        jax.block_until_ready(fn(*a))
        ts = []
        for _ in range(args.inner):
            t = time.perf_counter()
            jax.block_until_ready(fn(*a))
            ts.append(time.perf_counter() - t)
        return float(np.median(ts))

    def time_table(label: str, table: dict, a: tuple) -> None:
        # interleaved rounds: every program once per round, the A-vs-A control too
        per: dict = {k: [] for k in table}
        for _ in range(args.reps):
            for k, c in table.items():
                per[k].append(1e3 * timed(c, a))
        row[label] = {
            k: {"median": float(np.median(v)), "min": float(np.min(v)), "all": v}
            for k, v in per.items()
        }
        print(json.dumps({label: row[label]}), flush=True)

    f64 = None
    if args.mode == "check" and not args.no_fp64:
        f64_fn = jax.jit(
            lambda c, ce, x, ln, nr: _evaluate_local_expansions_particle_major(
                LocalExpansionData(order=order, centers=ce, coefficients=c),
                x, leaf_nodes=ln, node_ranges=nr, order=order, chunk=1 << 20,
                kernel="xla",
            )  # fmt: skip
        )
        f64 = np.asarray(
            f64_fn(
                jnp.asarray(local.coefficients, jnp.float64),
                jnp.asarray(local.centers, jnp.float64),
                jnp.asarray(xs, jnp.float64),
                leaf_nodes,
                ranges,
            )
        )
    # one output on the host at a time (1.2 GB each at 1e8)
    ref_out = None
    if ref_name in compiled_l2p:
        ref_out = np.asarray(compiled_l2p[ref_name](*l2p_args))
    for k, c in compiled_l2p.items():
        out = ref_out if k == ref_name else np.asarray(c(*l2p_args))
        if ref_out is not None and k != ref_name:
            row.setdefault("l2p_vs_ref", {})[k] = _diff(ref_out, out)
        if f64 is not None:
            row.setdefault("l2p_vs_fp64", {})[k] = _rel_to(f64, out)
        # equal digests: bitwise-equal outputs (e.g. the two Mosaic gathers)
        row.setdefault("l2p_sha256", {})[k] = hashlib.sha256(out.tobytes()).hexdigest()
        del out
    del ref_out, f64
    for key in ("leaves", "l2p_memory", "l2p_vs_ref", "l2p_vs_fp64", "l2p_sha256"):
        print(json.dumps({key: row.get(key)}), flush=True)
    if args.mode == "time" and compiled_l2p:
        time_table("l2p_ms", compiled_l2p, l2p_args)
    del state, local, xs, l2p_args, compiled_l2p

    # ---- one full force step (rebuild + accelerations) --------------------------
    if not args.no_step:
        prov, topo = FusedForceProvider.prepare(
            fmm, p, m, leaf_size=leaf, max_order=order
        )
        # under dehnen_error the rebuild reads the previous force's f_b: seed it
        # from the prepared topology (input order, as step returns it)
        _, f_b = jax.jit(lambda t: prov.accelerations(t, force_scale=True))(topo)
        del topo
        aux0 = (f_b, None, None)
        step_args = (p, m, aux0)
        compiled_step = {}
        for name in names:
            kernel = name.split("#")[0]
            t0 = time.perf_counter()
            try:
                # a fresh jit per kernel: the kernel is read at trace time
                step_fn = jax.jit(lambda x, mm, aux: prov.step(x, mm, None, aux))
                compiled_step[name] = _compile_with(kernel, step_fn, *step_args)
            except Exception as exc:  # noqa: BLE001
                row.setdefault("errors", {})[f"step/{name}"] = repr(exc)[:600]
                print(f"step {name}: FAILED {exc!r}"[:800], flush=True)
                continue
            row.setdefault("step_compile_s", {})[name] = time.perf_counter() - t0
            row.setdefault("step_memory", {})[name] = _temp(compiled_step[name])
        ref_acc = None
        if ref_name in compiled_step:
            ref_acc = np.asarray(compiled_step[ref_name](*step_args)[0])
            row.setdefault("step_sha256", {})[ref_name] = hashlib.sha256(
                ref_acc.tobytes()
            ).hexdigest()
        for k, c in compiled_step.items():
            if ref_acc is None or k == ref_name:
                continue
            acc = np.asarray(c(*step_args)[0])
            row.setdefault("step_sha256", {})[k] = hashlib.sha256(
                acc.tobytes()
            ).hexdigest()
            row.setdefault("step_vs_ref", {})[k] = dict(
                _diff(ref_acc, acc), **_rel_to(ref_acc, acc)
            )
            del acc
        del ref_acc
        for key in ("step_memory", "step_vs_ref", "step_sha256"):
            print(json.dumps({key: row.get(key)}), flush=True)
        if args.mode == "time" and compiled_step:
            time_table("step_ms", compiled_step, step_args)

    stats = dev.memory_stats() or {}
    row["peak_bytes_in_use"] = int(stats.get("peak_bytes_in_use", 0))
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
