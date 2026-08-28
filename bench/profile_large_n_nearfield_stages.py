"""Where the large-N evaluation's time goes, and whether a kernel change could help.

Track B4 of ``docs/plan_2026-08_B_nearfield.md``. The near field is essentially the
whole single-card evaluation, so this is the last remaining performance question
there; the plan's prior is that the cost is gather/scatter rather than arithmetic,
and it asks for that to be confirmed or refuted before anyone proposes a kernel.

HOW THE STAGES ARE SEPARATED
----------------------------
Not with host-side timers, which are unreliable against a jitted device-resident
lane. The production fast lane already carries its own cumulative ablation modes
(``JACCPOT_LARGE_N_EVAL_DIAG_MODE``), and each compiles its own executable, so the
attribution is a difference of separately measured whole-program timings:

    zero              returns before the body runs -- dispatch and return only
    permutation_only  body runs with both fields off: the inverse-permutation
                      gather and the output cast, and nothing else
    near_only         near field + that gather
    far_only          far field + that gather
    full              both + that gather

    permutation = permutation_only - zero
    near        = near_only        - permutation_only
    far         = far_only         - permutation_only

``permutation_only`` is what makes this better than a bare near/far split: without
it, the gather's cost is silently attributed to whichever field is measured.

AN ATTRIBUTED STAGE IS A DIFFERENCE OF TWO NOISY NUMBERS. ``profile_fused_stage_ablation``
records that ``eval+nearfield`` had a 35 % within-mode spread on an A100 -- four times
the cross-tree difference then being attributed to a refactor. So every mode is timed
``--repeats`` times, the spread is reported beside the median, and any attributed
stage smaller than the summed spread of its two modes is printed as NOT READABLE
rather than as a number.

THE ROOFLINE IS THE POINT, NOT THE BREAKDOWN
--------------------------------------------
Commit ``6b7cc1b`` is the model the plan names: it established that M2L time tracked
*kernel count* rather than FLOPs (~121 HLO ops per coefficient, 0.24 % of fp64 peak)
and thereby retired a kernel proposal aimed at the wrong term. The same discipline
applied here means asking what the near field is actually limited by before proposing
anything, so this reports the achieved arithmetic rate as a fraction of the device's
peak, computed from the pair count the tree really produces:

    near particle pairs = sum over target leaves of (neighbour leaves x |target| x |source|)

A rate near peak means arithmetic and a kernel change buys little. A rate far below it
means the plan's gather/scatter prior stands, and the ceiling a kernel change could
reach is quotable rather than hoped for.

WHAT IT FOUND (A100-PCIE-40GB, disc, N=1e6, fp32, on a verified-idle card)
-------------------------------------------------------------------------
==================  =====  ======  ==========  =======  ==============
config              near   far     pairs       % peak   kernel ceiling
==================  =====  ======  ==========  =======  ==============
leaf 256, p4, t0.6  97.5%    1.2%   1.442e11    27.8%           3.6x
leaf 128, p8, t0.5  92.8%   10.8%   1.241e11    21.9%           4.6x
leaf 1024, p4, t0.6 99.5%    0.7%   3.071e11    31.6%           3.2x
==================  =====  ======  ==========  =======  ==============

**B4's answer is no-go on a near-field kernel, and the plan's reason for expecting
otherwise does not hold.** It predicts the cost is gather/scatter rather than
arithmetic. A gather-bound kernel sits at single-digit percent of peak; this one runs
at 22-32 %, and ``permutation+cast`` -- the obvious gather suspect -- is below noise
in every configuration. The ceiling on a rewrite is 3.2-4.6x, while the pair count
moves further: leaf 1024 evaluates 3.07e11 pairs against leaf 256's 1.44e11 (2.1x)
for 1.83x the time, because efficiency *improves* with leaf size. Configuration moves
more than a kernel could, which is the plan's "fewer pairs" conclusion by a different
route.

**One correction to the track's founding premise.** It asserts the far field is under
1 % and builds on that. True at order 4 (0.7-1.2 %), false at order 8, where it is
**10.8 %**. The plan's own ablation table reports 0.8 % there, and also reports
``near only`` (0.938 s) exceeding ``full`` (0.664 s), which cannot happen -- these
stages compose to within 3.7 %. "The far field is negligible" is the stated reason
three M2L projects were judged misaimed, and it stops being true at high order.

**B5 -- ``JACCPOT_M2L_DEGREE_BATCHED``: leave it off, for a better reason than the
plan gives.** Measured at leaf 128 / order 8, where the far field is largest, with
the arms INTERLEAVED over three passes and the card's load recorded around every
measurement:

    flag off   far_only 48.52 ms   full 637.08 ms
    flag on    far_only 45.32 ms   full 645.77 ms

No end-to-end effect (1.014x *slower*, inside a 2.4-5.8 % within-arm spread) and none
on the stage either: far_only's 1.07x sits under a 26-73 % spread. The plan expects
the 1.64x stage win to be real but invisible because the stage is under 1 %; it is not
visible **in the stage**, which matches the code -- the flag lives in
``operators/m2l_real_rot_scale`` (the REAL basis) and this lane runs solidfmm. So the
flag does not reach the production lane, and anyone wanting that 1.64x would have to
put the lane on the real basis first. Note it is read at module import, so it cannot
be A/B'd inside one process.

A NOTE ON MEASURING ANYTHING ABSOLUTE ON THIS BOX
-------------------------------------------------
The first pass at both B4 and B5 measured contention rather than code. The same
configuration read 1307 ms and 1441 ms on busy cards against 542 ms on an idle one --
a 2.6x artefact that had already reached a reported roofline (11.4 % of peak instead
of 27.8 %), and the first B5 attempt put both flag-off arms on a busy card and both
flag-on arms on a freeing one, manufacturing a win. The box runs at load ~120 with
~290 users and card occupancy changes on a minute timescale, so: sample
``utilization.gpu`` several times before trusting a card, record it *around* each
measurement rather than once at the start, and interleave the arms of any A/B. Ratios
measured back-to-back in one process (the stage shares here) survive contention;
absolute seconds do not.

USAGE
-----
    export CUDA_VISIBLE_DEVICES=<one card, all session>
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
      python -u -m bench.profile_large_n_nearfield_stages \
        --num-particles 1048576 --leaf-size 256 --max-order 4 --theta 0.6 --repeats 5
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bench.validation.mac_error_distribution import make_distribution  # noqa: E402
from jaccpot.solver import FastMultipoleMethod  # noqa: E402

_RESULTS = REPO_ROOT / "bench" / "results" / "near_field"

#: Cumulative ablation modes, cheapest first. Order matters only for readability.
_MODES = ("zero", "permutation_only", "near_only", "far_only", "full")

#: Flops per softened particle pair: 3 subs, 3 mul + 2 add for r^2, 1 add for eps^2,
#: an rsqrt-and-cube (counted as 3), then 3 fma to accumulate. Deliberately a round
#: number -- the conclusion below turns on an order of magnitude, not on whether this
#: is 18 or 22.
_FLOPS_PER_PAIR = 20.0

#: A100-PCIE-40GB, fp32 non-tensor, and HBM2e. Overridable because the conclusion is
#: a fraction of peak and quoting the wrong peak moves it.
_DEFAULT_PEAK_FLOPS = 19.5e12
_DEFAULT_PEAK_BW = 1555e9


def _near_particle_pairs(state: Any) -> int:
    """Count target-source PARTICLE pairs the near field evaluates.

    Leaf-pair counts understate the work by the square of the leaf size, which is the
    factor that decides whether this kernel is anywhere near peak.

    Parameters
    ----------
    state : Any
        A ``LargeNPreparedState``.

    Returns
    -------
    int
        Number of particle pairs, counting padding as the kernel does.
    """
    neighbors = state.neighbor_list
    node_ranges = np.asarray(state.tree.node_ranges)
    leaf_indices = np.asarray(neighbors.leaf_indices)
    sizes = (node_ranges[leaf_indices, 1] - node_ranges[leaf_indices, 0]).astype(
        np.int64
    )
    counts = np.asarray(neighbors.counts).astype(np.int64)
    source_ids = np.asarray(neighbors.neighbors)
    offsets = np.asarray(getattr(neighbors, "offsets", None))
    if offsets is None or offsets.shape[0] != counts.shape[0] + 1:
        # Uniform fallback: mean source size per neighbour, which is exact when the
        # tree's leaves are equal-sized (they are, on this lane).
        return int((counts * sizes * int(sizes.mean())).sum())
    total = 0
    for i, leaf in enumerate(leaf_indices):
        lo, hi = int(offsets[i]), int(offsets[i]) + int(counts[i])
        src = source_ids[lo:hi]
        src = src[src >= 0]
        src_sizes = (node_ranges[src, 1] - node_ranges[src, 0]).astype(np.int64)
        total += int(sizes[i]) * int(src_sizes.sum())
    return int(total)


def _time_mode(
    fmm: Any, state: Any, mode: str, repeats: int
) -> tuple[float, float, list[float]]:
    """Compile, warm up and time one ablation mode.

    Parameters
    ----------
    fmm : Any
        Engine holding the prepared state.
    state : Any
        The prepared state.
    mode : str
        Value for ``JACCPOT_LARGE_N_EVAL_DIAG_MODE``.
    repeats : int
        Timed calls after the discarded warm-up.

    Returns
    -------
    tuple[float, float, list[float]]
        Median seconds, spread (max - min) as a fraction of the median, and the
        raw samples.
    """
    os.environ["JACCPOT_LARGE_N_EVAL_DIAG_MODE"] = mode
    out = fmm.evaluate_prepared_state(state, return_potential=False)
    jax.block_until_ready(out)
    samples = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fmm.evaluate_prepared_state(state, return_potential=False)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    median = float(statistics.median(samples))
    spread = float((max(samples) - min(samples)) / median) if median > 0 else 0.0
    return median, spread, samples


def main(argv: Optional[list[str]] = None) -> int:
    """Profile the stages and report a go/no-go on a near-field kernel change.

    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit status; 0 on completion.

    Raises
    ------
    RuntimeError
        If ``prepare_state`` did not land on the large-N lane, which would make
        every ablation mode below inapplicable.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-particles", type=int, default=1_048_576)
    ap.add_argument("--leaf-size", type=int, default=256)
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--softening", type=float, default=0.001)
    ap.add_argument("--distribution", default="disc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--peak-flops", type=float, default=_DEFAULT_PEAK_FLOPS)
    ap.add_argument("--peak-bandwidth", type=float, default=_DEFAULT_PEAK_BW)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    positions, masses = make_distribution(
        str(args.distribution), int(args.num_particles), int(args.seed)
    )
    positions = positions.astype(np.float32)
    masses = masses.astype(np.float32)
    fmm = FastMultipoleMethod(
        theta=float(args.theta),
        G=1.0,
        softening=float(args.softening),
        preset="large_n_gpu",
        expansion_basis="solidfmm",
    )
    state = fmm.prepare_state(
        jnp.asarray(positions),
        jnp.asarray(masses),
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
    )
    if type(state).__name__ != "LargeNPreparedState":
        raise RuntimeError(f"not on the large-N lane: got {type(state).__name__}")

    pairs = _near_particle_pairs(state)
    device = jax.devices()[0]
    print(
        f"# N={args.num_particles} leaf={args.leaf_size} order={args.max_order} "
        f"theta={args.theta} card={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device={device.device_kind} repeats={args.repeats}",
        flush=True,
    )
    print(
        f"# near particle pairs = {pairs:,} "
        f"({pairs / max(1, args.num_particles):,.0f} sources per target)",
        flush=True,
    )

    timings: dict[str, dict[str, Any]] = {}
    for mode in _MODES:
        median, spread, samples = _time_mode(fmm, state, mode, int(args.repeats))
        timings[mode] = {
            "median_s": median,
            "spread_frac": spread,
            "samples": [round(v, 5) for v in samples],
        }
        print(
            f"  {mode:17s} {median * 1e3:9.2f} ms   spread {spread * 100:5.1f}%",
            flush=True,
        )
    os.environ.pop("JACCPOT_LARGE_N_EVAL_DIAG_MODE", None)

    def attribute(name: str, high: str, low: str) -> dict[str, Any]:
        delta = timings[high]["median_s"] - timings[low]["median_s"]
        noise = (
            timings[high]["spread_frac"] * timings[high]["median_s"]
            + timings[low]["spread_frac"] * timings[low]["median_s"]
        )
        readable = abs(delta) > noise
        return {
            "stage": name,
            "seconds": delta,
            "noise_seconds": noise,
            "readable": bool(readable),
            "share_of_full": delta / timings["full"]["median_s"],
        }

    stages = [
        attribute("dispatch", "zero", "zero"),
        attribute("permutation+cast", "permutation_only", "zero"),
        attribute("near", "near_only", "permutation_only"),
        attribute("far", "far_only", "permutation_only"),
    ]
    stages[0]["seconds"] = timings["zero"]["median_s"]
    stages[0]["share_of_full"] = (
        timings["zero"]["median_s"] / timings["full"]["median_s"]
    )
    stages[0]["readable"] = True

    print("\n# stage attribution (difference of separately measured modes)")
    for st in stages:
        if st["readable"]:
            print(
                f"  {st['stage']:17s} {st['seconds'] * 1e3:9.2f} ms  "
                f"{st['share_of_full'] * 100:5.1f}% of full",
                flush=True,
            )
        else:
            print(
                f"  {st['stage']:17s}    NOT READABLE  (|{st['seconds'] * 1e3:.2f}| ms "
                f"< {st['noise_seconds'] * 1e3:.2f} ms of noise)",
                flush=True,
            )
    residual = (
        timings["full"]["median_s"]
        - timings["near_only"]["median_s"]
        - timings["far_only"]["median_s"]
        + timings["permutation_only"]["median_s"]
    )
    print(
        f"  {'residual':17s} {residual * 1e3:9.2f} ms  "
        f"(full - near_only - far_only + permutation_only; large means the stages "
        f"do not compose)",
        flush=True,
    )

    near_s = next(s for s in stages if s["stage"] == "near")["seconds"]
    achieved = (pairs * _FLOPS_PER_PAIR) / near_s if near_s > 0 else 0.0
    frac_peak = achieved / float(args.peak_flops)
    print("\n# roofline for the near field")
    print(f"  pairs            {pairs:,}")
    print(f"  flops (at {_FLOPS_PER_PAIR:.0f}/pair) {pairs * _FLOPS_PER_PAIR:.3e}")
    print(f"  achieved         {achieved / 1e12:.2f} TFLOP/s")
    print(f"  device peak      {float(args.peak_flops) / 1e12:.2f} TFLOP/s (fp32)")
    print(f"  fraction of peak {frac_peak * 100:.1f}%")
    if frac_peak >= 0.4:
        verdict = (
            "ARITHMETIC-BOUND: the near field is already near peak, so a kernel "
            "change has little to win. The lever is fewer pairs."
        )
    elif frac_peak >= 0.1:
        verdict = (
            f"PARTLY BOUND: at {frac_peak:.0%} of peak a perfect kernel could win at "
            f"most {1 / frac_peak:.1f}x, and only if nothing else binds. Weigh that "
            "against fewer pairs before writing one."
        )
    else:
        verdict = (
            f"NOT ARITHMETIC-BOUND: at {frac_peak:.1%} of peak the plan's "
            "gather/scatter prior stands. A kernel change has headroom, but the "
            "ceiling is set by data movement, not by flops."
        )
    print(f"\n# verdict: {verdict}", flush=True)

    _RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"nearfield_stages_n{args.num_particles}_leaf{args.leaf_size}"
    path = _RESULTS / f"{stem}{('_' + args.tag) if args.tag else ''}.json"
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "device_kind": device.device_kind,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "near_particle_pairs": pairs,
                "timings": timings,
                "stages": stages,
                "residual_seconds": residual,
                "achieved_flops": achieved,
                "fraction_of_peak": frac_peak,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"# wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
