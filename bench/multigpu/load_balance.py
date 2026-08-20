"""Figure 11 data -- per-device work on a clustered distribution.

The question is whether the work-weighted space-filling-curve partition balances
*cost*, not particle count. A uniform cube cannot answer it: any sensible
partition balances a uniform cube by construction. A centrally concentrated
distribution can, because there the two notions come apart -- equal particle
counts per device leave the device holding the dense centre with far more pair
work than the ones holding the outskirts.

So this reports, per device, the near- and far-pair counts the driver's own
diagnostic vector carries. Those are the load-balance signal: P2P cost tracks
near pairs and far-field cost tracks far pairs, and an imbalance in them is an
imbalance in time regardless of how evenly the particles were divided.

Both distributions are swept by default so the figure can show the contrast
rather than assert it.

Usage
-----
::

    python -m bench.multigpu.load_balance --ndevs 4 --per-device-n 8000
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.multigpu import _sweep_cli  # noqa: E402
from bench.multigpu.harness import sweep  # noqa: E402

DEFAULT_OUT = "multigpu/load_balance.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _sweep_cli.add_sweep_args(p)
    p.add_argument(
        "--per-device-n",
        type=int,
        default=_sweep_cli.HEALTHY_PER_DEVICE_N,
        help="particles per device (default: the healthy 8000)",
    )
    p.add_argument(
        "--distributions",
        default="uniform,plummer",
        help="comma-separated; the contrast is the point (default: uniform,plummer)",
    )
    return p.parse_args()


def main() -> int:
    """Measure per-device work for each distribution and write the artifact.

    Returns
    -------
    int
        0 when at least one point is valid, 1 when none is.
    """

    args = _parse_args()
    ndevs = [int(v) for v in str(args.ndevs).split(",") if v.strip()]
    if any(d < 2 for d in ndevs):
        raise SystemExit(
            "--ndevs must all be >= 2; a 1-device run has nothing to balance"
        )
    dists = [d.strip() for d in str(args.distributions).split(",") if d.strip()]

    records = []
    for dist in dists:
        per_dist = argparse.Namespace(**vars(args))
        per_dist.distribution = dist
        pts = [{"ndev": d, "n": int(args.per_device_n) * d} for d in ndevs]
        got = sweep(pts, extra_argv=_sweep_cli.worker_argv(per_dist))
        for r in got:
            r["distribution"] = dist
        records.extend(got)

    _sweep_cli.write_sweep(
        DEFAULT_OUT,
        args,
        records,
        extra_config={
            "distributions": dists,
            "per_device_n_fixed": int(args.per_device_n),
        },
    )
    return 0 if any(r.get("valid") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
