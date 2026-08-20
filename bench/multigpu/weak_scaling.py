"""Figure 09 data -- weak scaling: fixed per-device N, throughput vs device count.

This is the load-bearing scaling curve for this implementation. Strong scaling is
capacity-limited above roughly 8000 particles per device (see
``strong_scaling.py``), so the way to reach scale here is to add devices while
holding the per-device load at its healthy value -- which is precisely what weak
scaling measures.

Each device count runs in its own process, via
:func:`bench.multigpu.harness.sweep`.

Usage
-----
::

    python -m bench.multigpu.weak_scaling --per-device-n 8000 --ndevs 2,3,4,5
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

DEFAULT_OUT = "multigpu/weak_scaling.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _sweep_cli.add_sweep_args(p)
    p.add_argument(
        "--per-device-n",
        type=int,
        default=_sweep_cli.HEALTHY_PER_DEVICE_N,
        help="particles per device, held fixed (default: the healthy 8000)",
    )
    return p.parse_args()


def main() -> int:
    """Run the weak-scaling sweep and write the artifact.

    Returns
    -------
    int
        0 when at least one point is valid, 1 when none is.
    """

    args = _parse_args()
    ndevs = [int(v) for v in str(args.ndevs).split(",") if v.strip()]
    if any(d < 2 for d in ndevs):
        raise SystemExit("--ndevs must all be >= 2; a 1-device run is not distributed")

    per = int(args.per_device_n)
    if per > _sweep_cli.HEALTHY_PER_DEVICE_N:
        print(
            f"[weak] --per-device-n {per} is above the measured healthy "
            f"{_sweep_cli.HEALTHY_PER_DEVICE_N}; expect every point to overflow"
        )
    points = [{"ndev": d, "n": per * d} for d in ndevs]
    records = sweep(points, extra_argv=_sweep_cli.worker_argv(args))
    _sweep_cli.write_sweep(
        DEFAULT_OUT, args, records, extra_config={"per_device_n_fixed": per}
    )
    return 0 if any(r.get("valid") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
