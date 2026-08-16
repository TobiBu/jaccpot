"""Warm-call end-to-end wall-clock: the mass MAC against the geometric MAC.

WHY THIS EXISTS. Step 4's accept condition in
``docs/dehnen_mass_mac_status_and_plan.md`` is *"warm-call wall-clock within 1.3x of
the geometric MAC"*, and it is the last one still open. Everything measured so far is
a proxy for it and not a substitute:

- **interaction work** (``pair_work``) counts pairs, not seconds, and the two arms do
  not spend the same time per pair -- the criterion's accept mask is less regular, so
  it can cost more per pair than a fixed angle even at equal counts;
- **prepare overhead** (``force_scale_prepare_cost.py``) times ``prepare_state`` only,
  on the generic lane in fp64, at N=16384. Production is the large-N GPU lane in fp32
  at N=1e6, and an N-body step pays *evaluation* as well as prepare.

There was also a structural worry that Step 3' removed but nobody re-timed: if the
criterion ran the generic lane while fixed theta ran the fast lane, end-to-end it
would lose regardless of accuracy. The criterion reaches the large-N lane now, so
this harness times both arms **on the same lane** and settles it.

WHAT IT COMPARES. Not the same knob -- the same *accuracy*. Timing two arms at
knobs that produce different errors measures nothing, so each arm is run at a knob
whose delta-a/f median was measured by the leaf sweep, and the pairing is stated in
the output. Defaults are the production configuration the leaf sweep settled on
(N=1e6, leaf 1024, large-N lane, fp32) with fixed theta=0.65 (median 8.7e-6) against
eq (16a) eps=3e-4 (median 7.7e-6) -- within 12 % of each other, the criterion being
the slightly *more* accurate of the two.

TWO WAYS TO GET THIS WRONG, both already paid for elsewhere in this project:

- **Never quote a cold call.** Cold ``prepare_state`` is dominated by JAX compilation
  (~155 s vs ~34 s warm at N=16384/p=8; a cold-vs-cold comparison once reported 1.29x
  where the steady state was 3.5x). Every timed call here follows at least one
  discarded warm-up, and the medians are over ``--repeats`` calls.
- **Never time a dispatch.** ``prepare_state`` and the evaluation both return device
  arrays, so a bare timer measures queueing. Every call is followed by
  ``block_until_ready`` on the whole returned pytree.

Run it on a card you have to yourself: the leaf-sweep runs shared cards four at a
time, which is exactly why their ``prepare_s``/``evaluate_s`` are not quotable and
this harness exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from bench.validation.mac_error_distribution import make_distribution
from jaccpot import FastMultipoleMethod, FMMAdvancedConfig

__all__ = ["time_arm", "timed_median"]


def timed_median(call: Callable[[], Any], *, repeats: int, warmups: int) -> dict:
    """Median wall-clock of a materialised call, after discarding warm-ups.

    Parameters
    ----------
    call : Callable[[], Any]
        Zero-argument callable returning a pytree of device arrays.
    repeats : int
        Timed calls contributing to the median.
    warmups : int
        Untimed calls run first; at least one is required or the first timed call
        carries the JAX compilation of the whole pipeline.

    Returns
    -------
    dict
        ``median``, ``min``, ``max`` and the raw ``samples``, all in seconds.
    """

    for _ in range(warmups):
        jax.block_until_ready(call())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - start)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def time_arm(
    *,
    arm: str,
    knob: float,
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    order: int,
    leaf_size: int,
    softening: float,
    repeats: int,
    warmups: int,
    perturb: float = 0.0,
) -> dict:
    """Time one arm's prepare and evaluation at one knob value.

    Parameters
    ----------
    arm : str
        ``"fixed"`` for the geometric MAC swept over theta, ``"mass"`` for Dehnen
        eq (16a), ``"mass_16b_est"`` for eq (16b) via the O(N) estimator.
    knob : float
        ``theta`` for the fixed arm, ``eps`` for the criterion arms.
    positions : jnp.ndarray
        Particle positions ``[N, 3]``, already at the solver dtype.
    masses : jnp.ndarray
        Particle masses ``[N]``, already at the solver dtype.
    order : int
        Expansion order ``p``.
    leaf_size : int
        Particles per leaf. Load-bearing: the criterion's advantage and its cost
        both depend on it.
    softening : float
        Plummer softening; 0.0 matches the leaf sweep.
    repeats : int
        Timed calls per stage.
    warmups : int
        Discarded calls per stage. No traversal capacities are pre-sized here on
        purpose: the ``large_n_gpu`` contract pins the interaction cache on, so the
        retry-recompile cycle is paid once inside the warm-up and the timed calls
        reuse the converged capacity. That is what a warm-call median is for, and
        it avoids a second knob whose wrong value would quietly change the timing.
    perturb : float
        Per-call random displacement as a fraction of ``r_rms``. ``0.0`` re-prepares
        identical positions, which the interaction cache answers without rebuilding;
        a small non-zero value (1e-3) makes each prepare do the work an N-body step
        would. Report which one a number came from -- they differ by ~100x.

    Returns
    -------
    dict
        Per-stage timings plus the lane diagnostics that prove the large-N lane
        actually ran, since a silent fallback would reproduce generic-lane numbers.

    Raises
    ------
    RuntimeError
        If the large-N lane declined, or ``prepare_state`` returned anything other
        than a ``LargeNPreparedState``. Timing a generic-lane fallback against a
        large-N arm would compare two different pipelines and read as a result.
    """

    cfg = FMMAdvancedConfig()
    cfg = replace(cfg, tree=replace(cfg.tree, tree_type="radix"))
    runtime = replace(
        cfg.runtime,
        memory_objective="minimum_memory",
        prepare_stage_memory_split_enabled=True,
    )
    cfg = replace(
        cfg,
        mac_type="dehnen" if arm == "fixed" else "dehnen_error",
        runtime=runtime,
    )

    kwargs: dict[str, Any] = dict(
        advanced=cfg,
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        softening=softening,
        streamed_far_pairs=True,
    )
    if arm == "fixed":
        kwargs["theta"] = float(knob)
    else:
        # theta does not gate acceptance in paper mode; eps is the knob.
        kwargs["theta"] = 1.0
        kwargs["adaptive_eps"] = float(knob)
        kwargs["dehnen_geometry_mode"] = "com"
        kwargs["mac_force_scale_mode"] = (
            "paper_fb_cached" if arm == "mass_16b_est" else "paper_cached"
        )

    fmm = FastMultipoleMethod(**kwargs)

    # `enable_interaction_cache` is pinned True by the large_n_gpu contract, and the
    # cache is keyed on the acceptance criterion as well as the geometry. So calling
    # `prepare_state` repeatedly on *identical* positions measures a cache hit, not a
    # rebuild -- 0.22 s at N=1e6 against ~21 s for the real thing. That is a
    # legitimate steady-state reading only for a static configuration; an N-body step
    # moves particles and rebuilds. `--perturb` displaces the positions between calls
    # by a fraction of r_rms so each prepare does the work a real step would.
    # Displacements stay far below the 8.5 %-of-r_rms cliff at which a cached force
    # scale starts to over-accept (trap 6), so `paper_cached` remains valid here.
    rng = np.random.default_rng(seed=90210)
    radius_rms = float(np.sqrt(np.mean(np.sum(np.asarray(positions) ** 2, axis=1))))
    step = perturb * radius_rms

    def prepare() -> Any:
        if step > 0.0:
            offsets = rng.normal(size=np.asarray(positions).shape) * step
            moved = positions + jnp.asarray(offsets, dtype=positions.dtype)
        else:
            moved = positions
        return fmm.prepare_state(moved, masses, leaf_size=leaf_size, max_order=order)

    prepare_timing = timed_median(prepare, repeats=repeats, warmups=warmups)
    state = jax.block_until_ready(prepare())

    def evaluate() -> Any:
        return fmm.evaluate_prepared_state(state, return_potential=False)

    evaluate_timing = timed_median(evaluate, repeats=repeats, warmups=warmups)

    # A silent lane decline would reproduce generic-lane timings and read as a
    # result, so assert the lane rather than trusting the preset (trap 15: on
    # `large_n_gpu` a decline raises, but say so here anyway).
    declined = fmm.get_runtime_diagnostics().get("large_n_path_declined_reason")
    if declined is not None or type(state).__name__ != "LargeNPreparedState":
        raise RuntimeError(
            f"large-N lane did not engage for arm={arm} knob={knob:g}: "
            f"declined={declined!r}, state={type(state).__name__}"
        )

    return {
        "arm": arm,
        "knob": knob,
        "prepare_s": prepare_timing,
        "evaluate_s": evaluate_timing,
        "total_s": prepare_timing["median"] + evaluate_timing["median"],
        "prepared_state_type": type(state).__name__,
        "large_n_path_declined_reason": declined,
    }


def main() -> int:
    """Time both arms at matched accuracy and report the ratio."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--leaf-size", type=int, default=1024)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--distribution", default="plummer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softening", type=float, default=0.0)
    parser.add_argument(
        "--theta",
        type=float,
        default=0.65,
        help="fixed arm knob. Default 0.65 measured delta-a/f median 8.7e-6 at "
        "N=1e6/leaf 1024 in the leaf sweep.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3.0e-4,
        help="criterion arm knob. Default 3e-4 measured median 7.7e-6 on the same "
        "run -- within 12 %% of the fixed arm, and on the more accurate side.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="discarded calls per stage. Never 0: the first call compiles.",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0.0,
        help="per-call displacement as a fraction of r_rms. 0 measures a cached "
        "re-prepare; 1e-3 measures the rebuild an N-body step actually pays.",
    )
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--arms", default="fixed,mass,mass_16b_est")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    dtype = jnp.float32 if args.precision == "fp32" else jnp.float64
    pos_np, mass_np = make_distribution(args.distribution, args.n, args.seed)
    positions = jnp.asarray(pos_np, dtype=dtype)
    masses = jnp.asarray(mass_np, dtype=dtype)

    print(
        f"backend={jax.default_backend()} n={args.n} leaf={args.leaf_size} "
        f"order={args.order} dist={args.distribution} precision={args.precision} "
        f"repeats={args.repeats} warmups={args.warmups}",
        flush=True,
    )
    print(
        f"{'arm':14s} {'knob':>9s} {'prepare s':>11s} {'evaluate s':>11s} "
        f"{'total s':>10s}  lane",
        flush=True,
    )

    records = []
    for arm in args.arms.split(","):
        knob = args.theta if arm == "fixed" else args.eps
        record = time_arm(
            arm=arm,
            knob=knob,
            positions=positions,
            masses=masses,
            order=args.order,
            leaf_size=args.leaf_size,
            softening=args.softening,
            repeats=args.repeats,
            warmups=args.warmups,
            perturb=args.perturb,
        )
        records.append(record)
        print(
            f"{record['arm']:14s} {record['knob']:9.2e} "
            f"{record['prepare_s']['median']:11.3f} "
            f"{record['evaluate_s']['median']:11.3f} "
            f"{record['total_s']:10.3f}  {record['prepared_state_type']}"
            f" declined={record['large_n_path_declined_reason']}",
            flush=True,
        )

    baseline = next((r for r in records if r["arm"] == "fixed"), None)
    if baseline is not None:
        print(
            "\nratio to the geometric MAC (>1 means the criterion is SLOWER; "
            "Step 4 accepts <= 1.3):",
            flush=True,
        )
        for record in records:
            if record["arm"] == "fixed":
                continue
            print(
                f"  {record['arm']:14s} prepare "
                f"{record['prepare_s']['median'] / baseline['prepare_s']['median']:5.2f}x"
                f"  evaluate "
                f"{record['evaluate_s']['median'] / baseline['evaluate_s']['median']:5.2f}x"
                f"  total "
                f"{record['total_s'] / baseline['total_s']:5.2f}x",
                flush=True,
            )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({"args": vars(args), "records": records}, handle, indent=1)
        print(f"\nwrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
