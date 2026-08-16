"""Read a leaf sweep at a *common absolute* matched median, not per-leaf midpoints.

WHY THIS EXISTS. ``mac_error_distribution.compare_arms`` matches the two arms at five
log-spaced targets spanning *their own* overlapping error range. That is the right thing
within one configuration and the wrong thing across a leaf-size axis: each leaf size
reaches a different accuracy window, and the criterion's advantage grows as the tolerance
loosens, so comparing each leaf at its own range midpoint confounds leaf size with
accuracy level. Item 2b of ``docs/dehnen_mass_mac_status_and_plan.md`` established the
leaf-16..256 trend at N=1e5 by reading a **common** target at every leaf size; this module
is that read, done in code rather than by hand.

WHAT IT GUARDS. Three filters, each of which has produced a wrong published number at
least once in this project:

- the two ``compare_arms`` guards (``--min-far-pairs``, ``--max-p9999``), applied to the
  records before anything is interpolated (traps 8 and 3);
- the **fp32 median floor**. At N=1e6 in fp32 the delta-a/f median saturates at
  ~1.9-2.4e-6 -- summation round-off over a 1e6-particle near field, ~sqrt(N)*1.2e-7 --
  and the fixed arm's median is then *non-monotone in theta*, which truncation error
  cannot be. A matched target below the floor is matching round-off: one such row already
  reported p99.99 = 0.57 at work 2.02, an outlier in every column at once. Targets below
  ``--median-floor`` are refused rather than reported;
- **extrapolation**. A common target that falls outside either arm's measured range at a
  given leaf size is reported as absent, never as an interpolated number.

COST. Pure post-processing of committed JSON artifacts -- no solver, no device.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from bench.validation.mac_error_distribution import log_interp_at

__all__ = ["non_monotone_knobs", "read_at_common_target", "usable_median_range"]

#: Statistics reported as fixed/mass ratios, in the order Dehnen quotes them.
RATIO_FIELDS = ("rms", "p9999", "p99")


def usable_median_range(
    records: list[dict[str, Any]],
    *,
    metric: str,
    min_far_pairs: int,
    max_p9999: float,
) -> tuple[list[dict[str, Any]], Optional[tuple[float, float]]]:
    """Apply the two ``compare_arms`` guards and report the surviving median range.

    Parameters
    ----------
    records : list[dict[str, Any]]
        One arm's per-configuration records from a sweep artifact.
    metric : str
        Error-family prefix, ``"dehnen_"`` for Dehnen's own delta-a/f measure.
    min_far_pairs : int
        Configurations accepting fewer far pairs than this are dropped: with no far
        field the run is direct summation and its error is round-off (trap 8).
    max_p9999 : float
        Configurations whose 99.99th percentile exceeds this are dropped as diverged
        expansions rather than coarse points on the same curve (trap 3).

    Returns
    -------
    tuple[list[dict[str, Any]], Optional[tuple[float, float]]]
        The surviving records, and their ``(min, max)`` matched-median range -- or
        ``None`` for the range when fewer than two records survive, which is the
        "widen the grid" signal rather than an empty comparison.
    """

    kept = [
        r
        for r in records
        if r.get("far_pairs", 0) >= min_far_pairs
        and 0.0 < r.get(f"{metric}p9999", 0.0) <= max_p9999
    ]
    medians = [r[f"{metric}median"] for r in kept if r[f"{metric}median"] > 0]
    if len(medians) < 2:
        return kept, None
    return kept, (min(medians), max(medians))


def non_monotone_knobs(
    records: list[dict[str, Any]],
    *,
    metric: str,
) -> list[tuple[float, float]]:
    """Find where the matched median stops rising with the knob -- the fp32-floor tell.

    For the fixed arm the knob is ``theta`` and the median must rise with it: a
    *tighter* opening angle cannot produce a larger truncation error. Where it does
    not, the median is not measuring truncation any more, it is sitting on the
    summation-round-off floor. At N=1e6/fp32 that floor is ~1.9-2.4e-6 and it made
    the fixed arm's median read 2.44e-6, 1.91e-6, 1.87e-6, 2.87e-6, 6.44e-6 across
    theta = 0.40..0.60 -- non-monotone over the first three, which is the signature.

    Parameters
    ----------
    records : list[dict[str, Any]]
        One arm's records, in any order; sorted by knob internally.
    metric : str
        Error-family prefix, e.g. ``"dehnen_"``.

    Returns
    -------
    list[tuple[float, float]]
        ``(knob, median)`` for each configuration whose median failed to exceed that
        of the next-smaller knob. Empty means the arm is monotone and every point is
        plausibly above the floor.
    """

    ordered = sorted(records, key=lambda r: r["knob"])
    key = f"{metric}median"
    flagged = []
    for previous, current in zip(ordered, ordered[1:]):
        if current[key] <= previous[key]:
            flagged.append((float(current["knob"]), float(current[key])))
    return flagged


def read_at_common_target(
    artifact: dict[str, Any],
    *,
    target: float,
    mass_arm: str,
    metric: str = "dehnen_",
    min_far_pairs: int = 5000,
    max_p9999: float = 1.0,
) -> Optional[dict[str, Any]]:
    """Interpolate both arms at one absolute matched median.

    Parameters
    ----------
    artifact : dict[str, Any]
        A parsed ``mac_error_distribution`` JSON artifact.
    target : float
        The matched median, the *same* value at every leaf size.
    mass_arm : str
        Which criterion arm to read -- ``"mass"`` for eq (16a), ``"mass_16b_est"`` for
        eq (16b) via the O(N) estimator.
    metric : str
        Error-family prefix; see :func:`usable_median_range`.
    min_far_pairs : int
        Passed through to :func:`usable_median_range`.
    max_p9999 : float
        Passed through to :func:`usable_median_range`.

    Returns
    -------
    Optional[dict[str, Any]]
        Ratios and both arms' raw values at ``target``, or ``None`` when the target
        lies outside either arm's measured range at this leaf size. ``None`` means
        "not measured here", never "no effect".
    """

    records = artifact["records"]
    fixed, _ = usable_median_range(
        [r for r in records if r["arm"] == "fixed"],
        metric=metric,
        min_far_pairs=min_far_pairs,
        max_p9999=max_p9999,
    )
    mass, _ = usable_median_range(
        [r for r in records if r["arm"] == mass_arm],
        metric=metric,
        min_far_pairs=min_far_pairs,
        max_p9999=max_p9999,
    )
    if len(fixed) < 2 or len(mass) < 2:
        return None

    key = f"{metric}median"
    row: dict[str, Any] = {"matched_median": target, "mass_arm": mass_arm}
    for label, arm in (("fixed", fixed), ("mass", mass)):
        for name in RATIO_FIELDS + ("pair_work", "far_pairs"):
            field = f"{metric}{name}" if name in RATIO_FIELDS else name
            value = log_interp_at(arm, target_p90=target, field=field, p90_key=key)
            if value is None:
                return None
            row[f"{label}_{name}"] = value
    for name in RATIO_FIELDS:
        row[f"{name}_ratio"] = row[f"fixed_{name}"] / row[f"mass_{name}"]
    # Work is reported fixed/mass like the error ratios, so > 1 means the criterion
    # does LESS work. That convention is stated at every site because it has been
    # misread once already.
    row["work_ratio"] = row["fixed_pair_work"] / row["mass_pair_work"]
    return row


def main() -> int:
    """Report a leaf sweep at one or more common absolute matched medians."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="sweep JSONs, one per leaf size, in the order to report them",
    )
    parser.add_argument(
        "--target",
        default="",
        help=(
            "comma-separated matched medians. Empty picks targets automatically from "
            "the intersection of every leaf's usable range, which is the only choice "
            "that is a common target for all of them."
        ),
    )
    parser.add_argument(
        "--median-floor",
        type=float,
        default=3e-6,
        help=(
            "refuse targets at or below this. Default 3e-6 is the measured fp32 "
            "delta-a/f floor at N=1e6; below it the comparison matches round-off."
        ),
    )
    parser.add_argument("--metric", default="dehnen_")
    parser.add_argument("--min-far-pairs", type=int, default=5000)
    parser.add_argument("--max-p9999", type=float, default=1.0)
    parser.add_argument("--arm", default="mass,mass_16b_est")
    args = parser.parse_args()

    loaded = []
    for path in args.artifacts:
        with open(path) as handle:
            artifact = json.load(handle)
        leaf = artifact["meta"]["args"]["leaf_size"]
        loaded.append((leaf, path, artifact))

    for arm in args.arm.split(","):
        print(f"\n=== arm {arm} ===", flush=True)

        ranges = []
        for leaf, path, artifact in loaded:
            for label, name in (("fixed", "fixed"), ("mass", arm)):
                kept, span = usable_median_range(
                    [r for r in artifact["records"] if r["arm"] == name],
                    metric=args.metric,
                    min_far_pairs=args.min_far_pairs,
                    max_p9999=args.max_p9999,
                )
                if span is None:
                    print(
                        f"  leaf {leaf:5d} {label:5s}: fewer than 2 configs survive "
                        "the guards -- widen the grid for this leaf size",
                        flush=True,
                    )
                else:
                    print(
                        f"  leaf {leaf:5d} {label:5s}: median range "
                        f"{span[0]:.3e} .. {span[1]:.3e}",
                        flush=True,
                    )
                    ranges.append(span)
                # Both arms' medians must RISE with the knob (theta up, or eps up).
                # Where one does not, that point is on the round-off floor, not on
                # the truncation curve, and matching there compares noise.
                for knob, median in non_monotone_knobs(kept, metric=args.metric):
                    print(
                        f"    !! leaf {leaf} {label} knob={knob:g}: median "
                        f"{median:.3e} did not rise with the knob -- round-off "
                        "floor, not truncation",
                        flush=True,
                    )

        if args.target:
            targets = [float(t) for t in args.target.split(",")]
        elif ranges:
            lo = max(max(s[0] for s in ranges), args.median_floor)
            hi = min(s[1] for s in ranges)
            targets = [lo, (lo * hi) ** 0.5, hi] if hi > lo else []
        else:
            targets = []

        refused = [t for t in targets if t <= args.median_floor]
        targets = [t for t in targets if t > args.median_floor]
        for t in refused:
            print(
                f"  REFUSED target {t:.3e}: at or below the fp32 median floor "
                f"({args.median_floor:.1e}) -- that comparison matches round-off",
                flush=True,
            )
        if not targets:
            print(
                "  no common target above the floor lies inside every leaf's range "
                "-- the grids do not overlap; widen them rather than reading each "
                "leaf at its own midpoint",
                flush=True,
            )
            continue

        for target in targets:
            print(f"\n  matched median = {target:.3e}", flush=True)
            print(
                f"    {'leaf':>6s} {'rms x':>9s} {'p99.99 x':>9s} {'p99 x':>9s} "
                f"{'work x':>8s} {'fixed far':>12s} {'mass far':>12s}",
                flush=True,
            )
            for leaf, path, artifact in loaded:
                row = read_at_common_target(
                    artifact,
                    target=target,
                    mass_arm=arm,
                    metric=args.metric,
                    min_far_pairs=args.min_far_pairs,
                    max_p9999=args.max_p9999,
                )
                if row is None:
                    print(
                        f"    {leaf:6d} {'--':>9s} {'--':>9s} {'--':>9s} {'--':>8s}"
                        "   target outside this leaf's measured range",
                        flush=True,
                    )
                    continue
                print(
                    f"    {leaf:6d} {row['rms_ratio']:9.2f} {row['p9999_ratio']:9.2f} "
                    f"{row['p99_ratio']:9.2f} {row['work_ratio']:8.2f} "
                    f"{row['fixed_far_pairs']:12.0f} {row['mass_far_pairs']:12.0f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
