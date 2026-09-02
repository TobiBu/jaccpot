"""Cost and accuracy against expansion order and leaf size, on the large-N fast lane.

Serves tracks B2 and B3 of ``docs/plan_2026-08_B_nearfield.md``, which are the same
measurement on two axes:

* **B2** claims order 4 -> 8 costs nothing detectable and buys 17-26x in rms, because
  the far field is under 1 % of the evaluation.
* **B3** claims leaf 128 is 2.5x faster than leaf 1024 at equal or better accuracy,
  with a floor somewhere below 128 where the neighbour lists OOM.

Both are configuration questions with no code risk, and both need the same three
columns: a warm wall clock, an error against an exact reference, and the knowledge of
whether that error is real or round-off.

WHAT IT REUSES, AND WHY THAT MATTERS
------------------------------------
The error metric, the reference and the summary come from
``bench/validation/mac_error_distribution``: ``chunked_direct_accelerations`` (exact
fp64, chunked and target-subsampled so N=1e6 is tractable),
``chunked_force_scale`` / ``per_particle_dehnen_scaled_error`` (Dehnen's ``|da|/f_b``)
and ``error_summary``. Inventing a second error definition here would make these
numbers incomparable with the MAC study's and, worse, with the documented fp32 floor
that decides whether a row means anything.

WHAT IT DOES NOT REUSE. ``mac_error_distribution.measure`` times a single
``evaluate_prepared_state`` immediately after ``prepare_state``, so its ``evaluate_s``
includes compilation. The plan's own trap list says never to quote a cold call -- a
cold-vs-cold comparison in this project once reported 1.29x where the steady state was
3.5x -- so the timing here is a median over ``--repeats`` warm calls, each followed by
``block_until_ready``, after a discarded warm-up.

THE FP32 FLOOR IS A REPORTED COLUMN, NOT A FOOTNOTE
---------------------------------------------------
In fp32 the near-field summation floors the error at roughly ``sqrt(N) * 1.2e-7``:
~1.9e-6 at N=1e6, which is documented in
``bench/validation/leaf_sweep_common_target`` together with what it cost to learn
(a matched target below the floor once published p99.99 = 0.57, an outlier in every
column at once). An order sweep is exactly where this bites, because raising the order
drives truncation error down *towards* the floor -- and once there, the next order
buys nothing and the table looks like diminishing returns from the method rather than
from the arithmetic. Every row therefore carries ``floor_estimate`` and
``rms_over_floor``, and rows within ``--floor-factor`` of the floor are flagged
``floor_limited``.

TWO ASSERTIONS, BOTH BORROWED FROM ``measure``
----------------------------------------------
A silent fall back to the generic lane would reproduce the old numbers exactly and
look like a successful large-N run, so the lane is asserted twice: the runtime must
report no ``large_n_path_declined_reason``, and ``prepare_state`` must return a
``LargeNPreparedState``. And because ``precision="fp32"`` does not mean fp32
everywhere -- yggdrax forces x64 at import -- the resolved working dtype is reported
on every row rather than assumed from the inputs.

WHAT IT FOUND (A100, disc, N=1e6, fp32, order 4 unless stated)
--------------------------------------------------------------
**B2 -- the order axis.** The plan's "order 4 to 8 costs nothing detectable and buys
17-26x in rms" reproduces in the metric its table uses (per-particle relative), at
11.3x for leaf 128, but with two corrections. The cost is **+8 %** at leaf 128 and
+5 % at leaf 256, not nothing; and at the leaf production freezes (256) the gain is
only **6.9x**, saturating between order 6 and 8 (1.4x over that step against 2.3x at
leaf 128). Order 4 is retained as the default. If accuracy is wanted, **order 6 is
the better buy than order 8** on this lane -- see the lane note below for why 8
returns so little.

**B3 -- the leaf axis, theta 0.6.** Leaf 128 is the optimum, and it is an *interior*
one:

    leaf     16    32     64     96    128    256    512   1024
    eval s   OOM   OOM  0.554  0.524  0.494  0.547  0.689  0.999
    rms       --    --  5.15   4.69   4.56   3.84   3.49   3.16   (x 1e-4)

The plan says the optimum lies between 16 and 128 and asks for it to be mapped;
there is nothing there to map. It also says leaf 128 is 2.5x faster than leaf 1024
"at equal or better accuracy": the speed reproduces (2.03x) and **the accuracy does
not** -- error improves monotonically with leaf size, so 128 is 1.44x *worse* than
1024. The same monotone ordering holds at theta 0.5, 0.6 and 0.7 and in both error
metrics. The plan's own table is non-monotone across that axis, which is a noise
signature; monotone-improving is also the expected direction, since a larger leaf
moves more of the interaction into the exactly-computed near field. So the 2x is a
trade-off to state, not a free win.

Leaf 32 OOMs at 16 GiB and is a real floor. Leaf 64 OOMs only at the default
``XLA_PYTHON_CLIENT_MEM_FRACTION`` of 0.75 and runs at 0.95 -- a ceiling on the
fraction, not on the code, and worth checking before any leaf floor is reported.

**A lane accuracy gap, which is not what this module set out to find.** At the same
N, leaf, order, theta and precision, and over the *same* far-pair set, the
``large_n_gpu`` fast lane is far less accurate than the generic lane at high order.
At N=1e6, leaf 256, order 8 the rms ratio has median 8.25x (range 1.20 to 12.28 over
three seeds, 16384 reference targets) and the p99 ratio median 10.12x (range 3.62 to
10.32). It is not capacity (identical ``recent_dual_far_pair_count``, every overflow
flag clear), not precision (both fp32, and the generic lane reaches 7e-6 against a
measured round-off floor of 2.4e-6 while the fast lane sits at 8.7e-5), and not the
expansion basis (solidfmm and real agree to five figures within each lane). It grows
with order -- negligible at 4, 1.4-3.6x at 6, 8-12x at 8 -- which is why order 8 buys
so little on the production lane.

USAGE
-----
    # B2: the order axis at three N, production leaf
    python -u -m bench.validation.order_leaf_accuracy_sweep \
        --orders 4,6,8 --leaves 256 --num-particles 262144,1048576 --theta 0.5

    # B3: the leaf axis at one N
    python -u -m bench.validation.order_leaf_accuracy_sweep \
        --orders 4 --leaves 32,64,128,256,512,1024 --num-particles 1048576 --theta 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bench.validation.mac_error_distribution import (  # noqa: E402
    chunked_direct_accelerations,
    chunked_force_scale,
    error_summary,
    make_distribution,
    per_particle_dehnen_scaled_error,
    per_particle_relative_error,
)
from jaccpot.solver import FastMultipoleMethod  # noqa: E402

_RESULTS = REPO_ROOT / "bench" / "results" / "near_field"

# THE FP32 FLOOR IS MEASURED HERE, NOT PREDICTED.
#
# ``bench/validation/leaf_sweep_common_target`` documents it as "~1.9-2.4e-6 at
# N=1e6 ... ~sqrt(N)*1.2e-7". Those two do not agree: ``sqrt(1e6) * 1.2e-7`` is
# 1.2e-4, sixty-three times the stated 1.9e-6. The ``N`` in that expression cannot be
# the particle count -- most likely it is the near-field count per target, which
# depends on leaf size and theta and so is not something this module can reconstruct.
#
# Rather than pick a reading and label rows with it, the floor is obtained the way it
# is defined: run the same configuration in fp64 and in fp32. fp64 gives truncation
# error alone; the two combine in quadrature, so the round-off contribution is
# ``sqrt(max(0, rms_fp32**2 - rms_fp64**2))``. That needs no coefficient and no
# assumption about what scales with what.


def _inferred_floor(rms_fp32: float, rms_fp64: float) -> float:
    """Round-off contribution implied by an fp32/fp64 pair at one configuration.

    Truncation and round-off are independent, so they add in quadrature; the fp64 arm
    isolates truncation and the residual is the floor.

    Parameters
    ----------
    rms_fp32 : float
        Scaled-error rms measured in fp32.
    rms_fp64 : float
        Scaled-error rms measured in fp64 at the same order, leaf and theta.

    Returns
    -------
    float
        The inferred round-off floor, or 0.0 when fp32 is no worse than fp64 (which
        means round-off is not resolvable at this configuration).
    """
    return float(np.sqrt(max(0.0, float(rms_fp32) ** 2 - float(rms_fp64) ** 2)))


def _one_point(
    *,
    positions: np.ndarray,
    masses: np.ndarray,
    reference: Any,
    force_scale: Any,
    targets: np.ndarray,
    order: int,
    leaf: int,
    theta: float,
    softening: float,
    gravity: float,
    repeats: int,
    lane: str = "large_n",
) -> dict[str, Any]:
    """Prepare, warm-time and score one (order, leaf) point.

    ``lane`` selects ``"large_n"`` (the production fast lane, which PINS fp32) or
    ``"generic"`` (which honours the input dtype, and is therefore the only lane on
    which a precision comparison is possible at all).

    Parameters
    ----------
    positions : np.ndarray
        Particle positions; their dtype selects the precision of the run.
    masses : np.ndarray
        Particle masses.
    reference : Any
        Exact fp64 accelerations for ``targets``.
    force_scale : Any
        Dehnen's per-particle force scale for ``targets``.
    targets : np.ndarray
        Indices the reference covers; errors are taken over these rows only.
    order : int
        Expansion order.
    leaf : int
        Tree leaf size.
    theta : float
        Opening angle.
    softening : float
        Plummer softening length.
    gravity : float
        Gravitational constant.
    repeats : int
        Warm evaluations to time, after one discarded warm-up.
    lane : str
        ``"large_n"`` or ``"generic"``.

    Returns
    -------
    dict[str, Any]
        One record: timings, error summaries, the resolved lane and dtype.

    Raises
    ------
    RuntimeError
        If the requested lane declined, returned the wrong prepared-state type, or
        resolved a working dtype other than the one the inputs asked for.
    """
    kwargs: dict[str, Any] = dict(
        theta=float(theta), G=float(gravity), softening=float(softening)
    )
    if lane == "large_n":
        kwargs.update(preset="large_n_gpu", expansion_basis="solidfmm")
    fmm = FastMultipoleMethod(**kwargs)
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        jnp.asarray(positions), jnp.asarray(masses), leaf_size=leaf, max_order=order
    )
    prepare_s = time.perf_counter() - t0

    # Assert the lane rather than hope for it: a generic-lane fallback would
    # reproduce the pre-fast-lane numbers and read as a successful large-N run.
    if lane == "large_n":
        declined = fmm.get_runtime_diagnostics().get("large_n_path_declined_reason")
        if declined is not None:
            raise RuntimeError(
                f"large-N lane declined at order={order} leaf={leaf}: {declined!r}"
            )
        if type(state).__name__ != "LargeNPreparedState":
            raise RuntimeError(
                f"prepare_state returned {type(state).__name__}, not "
                f"LargeNPreparedState (order={order}, leaf={leaf})"
            )
    # Assert the dtype ACHIEVED, not the one requested. The large-N preset pins
    # working_dtype=float32 -- one of the fast lane's five requirements -- so handing
    # it float64 inputs silently yields a float32 run, and a floor measurement built
    # on that comparison compares fp32 with fp32 and reads zero. This session did
    # exactly that before the assertion existed.
    got = str(getattr(state, "working_dtype", "MISSING"))
    want = "float64" if np.asarray(positions).dtype == np.float64 else "float32"
    if got != want:
        raise RuntimeError(
            f"asked for {want} on the {lane!r} lane and got working_dtype={got}: a "
            "precision comparison on this lane would be vacuous"
        )

    # Warm-up, discarded: the first evaluate carries compilation.
    accel = fmm.evaluate_prepared_state(state, return_potential=False)
    jax.block_until_ready(accel)

    times = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        accel = fmm.evaluate_prepared_state(state, return_potential=False)
        jax.block_until_ready(accel)
        times.append(time.perf_counter() - t0)

    measured = np.asarray(accel)[targets]
    dehnen = per_particle_dehnen_scaled_error(measured, reference, force_scale)
    relative = per_particle_relative_error(measured, reference)
    summary = error_summary(dehnen, prefix="dehnen_")
    summary.update(error_summary(relative, prefix="rel_"))
    return {
        "order": int(order),
        "leaf_size": int(leaf),
        "theta": float(theta),
        "num_particles": int(positions.shape[0]),
        "lane": str(lane),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_kind": str(jax.devices()[0].device_kind),
        "prepare_seconds": round(prepare_s, 4),
        "evaluate_median_seconds": round(float(statistics.median(times)), 5),
        "evaluate_min_seconds": round(float(min(times)), 5),
        "evaluate_max_seconds": round(float(max(times)), 5),
        # Off the prepared state, not off `fmm`: the engine exposes no
        # `working_dtype`, so reading it there reports "?" and the column meant to
        # catch a silent fp64 run catches nothing. This is the plan's own trap
        # ("check what you actually got") applied to the check itself.
        "working_dtype": str(getattr(state, "working_dtype", "MISSING")),
        "state_input_dtype": str(getattr(state, "input_dtype", "MISSING")),
        "input_dtype": str(positions.dtype),
        **summary,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Sweep (order x leaf x N) and write the records with their floor context.

    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status; 0 on completion.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-particles", default="1048576")
    ap.add_argument("--orders", default="4,6,8")
    ap.add_argument("--leaves", default="256")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--softening", type=float, default=0.001)
    ap.add_argument("--gravity", type=float, default=1.0)
    ap.add_argument("--distribution", default="disc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--reference-targets",
        type=int,
        default=4096,
        help="targets for the exact fp64 reference; sources are always all N",
    )
    ap.add_argument(
        "--measure-floor",
        action="store_true",
        help="repeat each point in fp64 to measure the round-off floor by quadrature",
    )
    ap.add_argument(
        "--floor-factor",
        type=float,
        default=3.0,
        help="flag a row floor_limited when its rms is within this factor of the floor",
    )
    ap.add_argument(
        "--reference-block-bytes",
        type=int,
        default=2_000_000_000,
        help="budget for the reference's [block, N, 3] fp64 temporary",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in str(args.num_particles).split(",") if x]
    orders = [int(x) for x in str(args.orders).split(",") if x]
    leaves = [int(x) for x in str(args.leaves).split(",") if x]
    print(
        f"# dist={args.distribution} theta={args.theta} soft={args.softening} "
        f"repeats={args.repeats} ref_targets={args.reference_targets} "
        f"x64={bool(jax.config.read('jax_enable_x64'))} "
        f"devices={[d.device_kind for d in jax.devices()]}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for num_particles in sizes:
        positions, masses = make_distribution(
            str(args.distribution), int(num_particles), int(args.seed)
        )
        # fp32 by INPUT DTYPE. yggdrax forces x64 at import, so asking for fp32 any
        # other way silently compares fp64 to fp64 -- the plan lists this trap and
        # `working_dtype` is reported per row so the claim can be checked.
        positions32 = positions.astype(np.float32)
        masses32 = masses.astype(np.float32)

        rng = np.random.default_rng(int(args.seed) + 1)
        n_targets = min(int(args.reference_targets), int(num_particles))
        targets = np.sort(rng.choice(int(num_particles), size=n_targets, replace=False))
        # `chunked_direct_accelerations` chunks over TARGETS only and materialises a
        # [block, N, 3] fp64 temporary per block, so its cost is memory-bound in N
        # rather than pair-bound. Its docstring's "1e4 targets against 1e6 sources is
        # tractable" is true of the pair count and false of the footprint: at the
        # default block=512 that temporary is 12 GB at N=1e6. Size the block to a
        # budget instead.
        block = max(1, int(args.reference_block_bytes // (24 * max(1, num_particles))))
        block = min(block, n_targets)
        t0 = time.perf_counter()
        ref_args = dict(
            softening=float(args.softening),
            G=float(args.gravity),
            targets=targets,
            block=block,
        )
        pos64 = jnp.asarray(positions, dtype=jnp.float64)
        mass64 = jnp.asarray(masses, dtype=jnp.float64)
        reference = chunked_direct_accelerations(pos64, mass64, **ref_args)
        force_scale = chunked_force_scale(pos64, mass64, **ref_args)
        print(
            f"# N={num_particles} reference built in {time.perf_counter()-t0:.1f}s "
            f"over {n_targets} targets (block={block})",
            flush=True,
        )

        for leaf in leaves:
            for order in orders:
                try:
                    row = _one_point(
                        positions=positions32,
                        masses=masses32,
                        reference=reference,
                        force_scale=force_scale,
                        targets=targets,
                        order=order,
                        leaf=leaf,
                        theta=float(args.theta),
                        softening=float(args.softening),
                        gravity=float(args.gravity),
                        repeats=int(args.repeats),
                    )
                except Exception as exc:  # noqa: BLE001 -- a wall is a result
                    rows.append(
                        {
                            "num_particles": int(num_particles),
                            "leaf_size": int(leaf),
                            "order": int(order),
                            "failed": f"{type(exc).__name__}: {str(exc)[:400]}",
                        }
                    )
                    print(
                        f"  N={num_particles:8d} leaf={leaf:5d} order={order:2d}  "
                        f"FAILED {type(exc).__name__}: {str(exc)[:120]}",
                        flush=True,
                    )
                    continue
                # The floor is measured, not predicted: the same configuration in
                # fp64 isolates truncation, and round-off is the quadrature residual.
                # Only worth paying for where it decides the reading, so it is opt-in.
                if args.measure_floor:
                    try:
                        # SAME LANE, only the dtype differs. The large-N lane cannot
                        # do this -- it pins fp32 -- so the pair runs on the generic
                        # lane, where fp32 against fp64 isolates round-off from
                        # truncation with nothing else changing. The floor is a property
                        # of the arithmetic and transfers; timings are not read here.
                        common = dict(
                            reference=reference,
                            force_scale=force_scale,
                            targets=targets,
                            order=order,
                            leaf=leaf,
                            theta=float(args.theta),
                            softening=float(args.softening),
                            gravity=float(args.gravity),
                            repeats=1,
                            lane="generic",
                        )
                        g32 = _one_point(
                            positions=positions.astype(np.float32),
                            masses=masses.astype(np.float32),
                            **common,
                        )
                        g64 = _one_point(
                            positions=positions.astype(np.float64),
                            masses=masses.astype(np.float64),
                            **common,
                        )
                        floor = _inferred_floor(g32["dehnen_rms"], g64["dehnen_rms"])
                        row["generic_fp32_dehnen_rms"] = g32["dehnen_rms"]
                        row["generic_fp64_dehnen_rms"] = g64["dehnen_rms"]
                        row["generic_fp32_working_dtype"] = g32["working_dtype"]
                        row["generic_fp64_working_dtype"] = g64["working_dtype"]
                        row["measured_floor"] = float(f"{floor:.4g}")
                        row["rms_over_floor"] = (
                            round(row["dehnen_rms"] / floor, 2) if floor > 0 else None
                        )
                        row["floor_limited"] = bool(
                            row["rms_over_floor"] is not None
                            and row["rms_over_floor"] <= float(args.floor_factor)
                        )
                    except Exception as exc:  # noqa: BLE001
                        row["floor_probe_failed"] = (
                            f"{type(exc).__name__}: {str(exc)[:200]}"
                        )
                rows.append(row)
                floor_text = (
                    ""
                    if (not args.measure_floor) or "floor_probe_failed" in row
                    else (
                        f"  g32={row['generic_fp32_dehnen_rms']:.3e}"
                        f"  g64={row['generic_fp64_dehnen_rms']:.3e}"
                        f"  floor={row['measured_floor']:.3g}"
                        f"  rms/floor={row['rms_over_floor']}"
                        f"{'  FLOOR-LIMITED' if row['floor_limited'] else ''}"
                    )
                )
                print(
                    f"  N={num_particles:8d} leaf={leaf:5d} order={order:2d}  "
                    f"eval={row['evaluate_median_seconds']:.4f}s  "
                    f"rms={row['dehnen_rms']:.3e}  p99={row['dehnen_p99']:.3e}  "
                    f"max={row['dehnen_max']:.3e}  "
                    f"dtype={row['working_dtype']}{floor_text}",
                    flush=True,
                )

    _RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"order_leaf_sweep_{args.distribution}_theta{args.theta}"
    path = _RESULTS / f"{stem}{('_' + args.tag) if args.tag else ''}.json"
    path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2) + "\n")
    print(f"# wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
