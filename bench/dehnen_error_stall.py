"""The dehnen_error stall (#487): per-step wall time and list volume from a restart.

Restores a ``ProductionRunner`` restart and runs it on, one step per segment, under
one of three MAC arms, recording each segment's wall time and each traced list
build's near-list volume (the sum and the longest row) and far-pair count. The
build statistics come from a wrapper around the flat-walk builder
(``_interaction_cache._build_flat_walk_artifacts_strict_streamed``, a private seam
that main and the A1 branch share), reported through ``jax.debug.callback``.

Arms:

``error``  ``preset="galaxy"`` (eq 16a at ``adaptive_eps`` 1e-4): the stall on main; on the
           A1 branch, without it (long CSR rows merge-sorted, no NaN walk-table entry);
``floor``  the same with ``mac_force_scale_floor`` (opt-in; the A1 branch only);
``geoTT``  the geometric Dehnen MAC at theta ``TT / 10`` (``geo07``: the fallback).

Usage (one card; the restart files of the 1e8 bar disc and the 1.06e8 merger are
listed in ``bench/slurm/a1_dehnen_error_stall.sbatch``)::

    python bench/dehnen_error_stall.py --restart restart_3.npz --steps 10 \\
        --arm floor --out row.json

Timings on a shared card are contended: ratios of a 20x stall survive it, a few
per cent do not (``docs/production_2026-11.md``, section A1).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--restart", required=True, help="a ProductionRunner restart file")
    ap.add_argument("--steps", type=int, default=10, help="steps to run past it")
    ap.add_argument("--arm", default="error", help="error, floor or geoTT")
    ap.add_argument("--floor", type=float, default=0.5, help="the floor arm's fraction")
    ap.add_argument("--softening", type=float, default=1.5e-3)
    ap.add_argument("--out", required=True, help="the row's JSON")
    return ap.parse_args()


ARGS = _args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

BUILDS: list = []


def _install_build_probe() -> None:
    import jaccpot.runtime._interaction_cache as ic

    orig = ic._build_flat_walk_artifacts_strict_streamed

    def wrapped(**kw):
        out = orig(**kw)
        if kw.get("caps_only"):
            return out
        counts = jnp.asarray(out.neighbor_list.counts)
        if not isinstance(counts, jax.core.Tracer):
            return out
        info = dict(
            max_row=jnp.max(counts),
            near=jnp.sum(counts),
            rows_over_2048=jnp.sum(counts > 2048),
            far=jnp.asarray(out.compact_far_pairs.far_pair_count),
        )
        jax.debug.callback(
            lambda d: BUILDS.append(
                {k: int(v) for k, v in d.items()} | {"t": time.time()}
            ),
            info,
        )
        return out

    ic._build_flat_walk_artifacts_strict_streamed = wrapped


def _arm():
    from jaccpot import FastMultipoleMethod
    from jaccpot.production import GALAXY_SETTINGS, GalaxySettings

    if ARGS.arm == "error":
        return (
            FastMultipoleMethod(preset="galaxy", softening=ARGS.softening),
            GALAXY_SETTINGS,
        )
    if ARGS.arm == "floor":
        return (
            FastMultipoleMethod(
                preset="galaxy",
                softening=ARGS.softening,
                mac_force_scale_floor=ARGS.floor,
            ),
            GALAXY_SETTINGS,
        )
    if ARGS.arm.startswith("geo"):
        theta = int(ARGS.arm[3:]) / 10.0
        base = FastMultipoleMethod(preset="galaxy").advanced
        solver = FastMultipoleMethod(
            preset="galaxy",
            softening=ARGS.softening,
            theta=theta,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
        return solver, GalaxySettings(mac_type="dehnen", adaptive_eps=None, theta=theta)
    raise SystemExit(f"unknown arm {ARGS.arm!r}")


def main() -> None:
    from jaccpot.production import ProductionRunner, SegmentConfig

    with np.load(ARGS.restart) as z:
        step0 = int(json.loads(bytes(z["meta"]).decode())["counters"]["step"])
    _install_build_probe()
    solver, settings = _arm()
    walls: list = []

    def edge(e):
        walls.append({"segment": e.segment, "step": e.step, "wall_s": e.wall_s})
        print(f"step {e.step} wall {e.wall_s:.2f} s", flush=True)

    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=1, diagnostics_every=0, restart_every=0),
        callbacks=(edge,),
    )
    t0 = time.time()
    res = runner.run(
        dt=float(
            json.loads(bytes(np.load(ARGS.restart)["meta"]).decode())["counters"]["dt"]
        ),
        num_steps=step0 + ARGS.steps,
        restart_from=ARGS.restart,
    )
    peak = [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    ]
    row = dict(
        arm=ARGS.arm,
        restart=os.path.abspath(ARGS.restart),
        step0=step0,
        steps=ARGS.steps,
        total_s=time.time() - t0,
        segments=walls,
        builds=BUILDS,
        peak_bytes=peak,
        device=str(jax.devices()[0].device_kind),
        restored=res.restored,
    )
    with open(ARGS.out, "w") as fh:
        json.dump(row, fh, indent=1, default=str)
    print(json.dumps({k: row[k] for k in ("arm", "total_s", "peak_bytes")}), flush=True)


if __name__ == "__main__":
    main()
