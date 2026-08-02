"""Figure 03 -- geometric vs mass-dependent MAC at matched force error.

A **thin wrapper**. It does not implement the comparison: it calls
``bench.validation.mac_error_distribution.main()`` with a constructed argv and
re-wraps that script's output into the paper's ``{config, meta, data}`` envelope.
The engineering benchmark stays the single source of truth for the numerics, so
the figure and the branch's own measurements cannot drift apart.

What the comparison does (see that script's docstring for the full reasoning):
sweeps the geometric criterion over theta and the Dehnen eq (16a) mass-dependent
criterion over eps, records the full per-particle error distribution and
hardware-independent cost proxies for each, then log-interpolates both arms onto
a common 90th-percentile error and reports the cost ratio there. Matching on a
percentile rather than the median is deliberate: with a shallow far field most
particles are pure near-field and the median saturates at machine precision,
which makes median-matching degenerate.

Three arms are available. ``fixed`` is the geometric baseline, ``mass`` is eq
(16a) verbatim, and ``mass_16b`` is eq (16b) supplied with the exact O(N^2) force
scale ``f_b`` -- a measurement of the ceiling, not a production path, since no
real solver would compute an O(N^2) quantity to save far-field work.

**Expected outcome: no net compute advantage for the mass-dependent criterion.**
This is an honest head-to-head, not a demonstration. Nothing here is tuned to
produce a win; if the mass MAC ties or loses, that is the result and the caption
says so. In the reported ratios, ``> 1`` favours the mass MAC.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.validation.mac_comparison \\
        --n 2048 --order 4 --distribution plummer --theta 0.4,0.6 --eps 1e-4,1e-5 \\
        --arm fixed,mass --gpu-select none --json-out /tmp/smoke.json

Paper run::

    python -m bench.validation.mac_comparison --n 16384 --order 4,8 \\
        --distribution plummer,bulge_halo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "validation/mac_comparison.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # These mirror mac_error_distribution's own flags and are forwarded verbatim.
    p.add_argument("--n", default="16384", help="Comma-separated particle counts")
    p.add_argument("--order", default="4,8", help="Comma-separated expansion orders")
    p.add_argument(
        "--distribution",
        default="plummer,bulge_halo",
        help=(
            "Comma-separated distributions. Clustered by default: a uniform cube "
            "is where a mass-dependent criterion has least to offer, so testing "
            "only there would stack the comparison against it."
        ),
    )
    # leaf=16 keeps the clustered far field populated; see error_vs_theta.py.
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--theta", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80")
    p.add_argument("--eps", default="1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6")
    p.add_argument(
        "--arm",
        default="fixed,mass,mass_16b",
        help="Arms to run; must include 'fixed' (the comparison baseline)",
    )
    p.add_argument("--geometry-mode", default="com")
    p.add_argument("--theta-max", type=float, default=None)
    p.add_argument(
        "--metric", choices=("relative", "scaled", "dehnen"), default="dehnen"
    )
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    runmeta.add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    runmeta.enable_x64(args.dtype)

    # Imported after the device is pinned: this module imports jax at module scope.
    from bench.validation import mac_error_distribution as engine  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = pathlib.Path(tmp) / "mac_raw.json"
        forwarded = [
            "--n",
            str(args.n),
            "--order",
            str(args.order),
            "--distribution",
            str(args.distribution),
            "--leaf-size",
            str(args.leaf_size),
            "--theta",
            str(args.theta),
            "--eps",
            str(args.eps),
            "--arm",
            str(args.arm),
            "--geometry-mode",
            str(args.geometry_mode),
            "--metric",
            str(args.metric),
            "--softening",
            str(args.softening),
            "--G",
            str(args.g),
            "--seed",
            str(args.seed),
            "--json-out",
            str(raw_path),
        ]
        if args.theta_max is not None:
            forwarded += ["--theta-max", str(args.theta_max)]

        # Drive the engineering script exactly as its CLI would, so the paper
        # figure and the branch's measurements execute identical code.
        saved_argv = sys.argv
        sys.argv = ["mac_error_distribution", *forwarded]
        try:
            status = engine.main()
        finally:
            sys.argv = saved_argv
        if status != 0:
            print(f"[fig03] engine exited {status}", file=sys.stderr)
            return int(status)

        raw = json.loads(raw_path.read_text())

    records: list[dict[str, Any]] = raw["records"]
    comparisons: list[dict[str, Any]] = raw["comparisons"]

    ns = sorted({int(r["n"]) for r in records})
    orders = sorted({int(r["order"]) for r in records})
    arms = sorted({str(r["arm"]) for r in records})
    dists = sorted({str(r["distribution"]) for r in records})

    config = {
        "n": ns if len(ns) > 1 else (ns[0] if ns else None),
        # theta is one arm's swept knob and eps the other's, so neither is a
        # single held-fixed value: both grids are recorded rather than a scalar.
        "theta": [float(v) for v in str(args.theta).split(",") if v.strip()],
        "eps": [float(v) for v in str(args.eps).split(",") if v.strip()],
        "order": orders,
        # The MAC comparison is a traversal-criterion question, not a basis one;
        # the engine runs the solver's default (real) basis throughout.
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "distribution": dists,
        "arms": arms,
        "mac_types": {
            "fixed": "mac_type='dehnen' (geometric sphere), swept over theta",
            "mass": "mac_type='dehnen_error', eq (16a), swept over eps",
            "mass_16b": "mac_type='dehnen_error' with exact O(N^2) f_b, eq (16b)",
        },
        "matched_metric": args.metric,
        "geometry_mode": args.geometry_mode,
        "softening": float(args.softening),
        "G": float(args.g),
    }

    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta(
            {
                "argv": sys.argv[1:],
                "engine": "bench/validation/mac_error_distribution.py",
                "engine_args": raw.get("meta", {}).get("args", {}),
                "note": (
                    "Wrapper output. The engine script is the source of truth for "
                    "these numerics; this file only re-envelopes its records."
                ),
            }
        ),
        data={"records": records, "comparisons": comparisons},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
