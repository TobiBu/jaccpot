"""Figure 08 data -- strong scaling: fixed total N, wall clock vs device count.

Each device count runs in its own process (JAX fixes its device count at backend
initialisation, so a sweep cannot happen in one), via
:func:`bench.multigpu.harness.sweep`.

Read the output with the density limit in mind. Holding N fixed while adding
devices *lowers* the per-device load, so the interesting direction here is the
low-device-count end, where per-device N is largest -- and that is exactly where
the fixed-topology traversal buffers overflow and the forces come back
truncated. Such points are recorded with ``valid=false`` and must not be plotted:
their wall clock is padded-buffer overhead over a wrong answer, not a slow
correct answer.

The default total N is chosen so the *largest* device count sits at the measured
healthy per-device load; smaller device counts are then expected to be invalid,
and that expectation is the figure's main finding rather than a failure of the
run.

Usage
-----
::

    python -m bench.multigpu.strong_scaling --n 40000 --ndevs 2,3,4,5
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

DEFAULT_OUT = "multigpu/strong_scaling.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _sweep_cli.add_sweep_args(p)
    p.add_argument(
        "--n",
        type=int,
        default=40_000,
        help=(
            "total particle count, held fixed across device counts "
            "(default: 40000, i.e. the healthy per-device load at 5 devices)"
        ),
    )
    return p.parse_args()


def main() -> int:
    """Run the strong-scaling sweep and write the artifact.

    Returns
    -------
    int
        0 when at least one point is valid, 1 when none is.
    """

    args = _parse_args()
    ndevs = [int(v) for v in str(args.ndevs).split(",") if v.strip()]
    if any(d < 2 for d in ndevs):
        raise SystemExit("--ndevs must all be >= 2; a 1-device run is not distributed")

    points = [{"ndev": d, "n": int(args.n)} for d in ndevs]
    for p in points:
        per = -(-p["n"] // p["ndev"])
        if per > _sweep_cli.HEALTHY_PER_DEVICE_N:
            print(
                f"[strong] ndev={p['ndev']} -> {per} particles/device, above the "
                f"measured healthy {_sweep_cli.HEALTHY_PER_DEVICE_N}; expect overflow"
            )
    records = sweep(points, extra_argv=_sweep_cli.worker_argv(args))
    _sweep_cli.write_sweep(DEFAULT_OUT, args, records, extra_config={"fixed_n": args.n})
    return 0 if any(r.get("valid") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
