"""Is ``target_owned_block_size`` an axis a regression guard can stand on?

``bench/guard_large_n_radix_fast_lane.py`` compares block size 0 against 8 and calls
them "baseline" and "fast_lane", but the production large-N radix/solidfmm path is
locked to fast-lane execution (``runtime/_large_n_nearfield.resolve_large_n_execution_config``
sets ``radix_fast_lane = True`` unconditionally) and block 0 means "default to 32". So
both arms are the same lane and the guard is a 32-vs-8 A/B held to a 2x bar.

Measured on an uncontended A100, N=1048576, leaf 256, order 4, fp32: 0.3957 s at 32
against 0.4053 s at 8, i.e. **0.976x**. The bar cannot be met because the axis does
not separate the arms.

Before rebuilding the guard around this axis, find out whether *any* block size is
materially different -- a pessimised control arm needs one that is. If none is, the
axis cannot carry a guard in either direction and the guard needs a fixed reference
instead.

Reuses the guard's own ``FROZEN_CONFIG`` and worker invocation so this measures the
same thing the guard does, rather than something nearby.

USAGE
-----
    python -u -m bench.sweep_large_n_target_block_size --cuda-visible-devices 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any, Optional

from bench.guard_large_n_radix_fast_lane import (
    FROZEN_CONFIG,
    _build_worker_env,
    _metric_key,
    _run_case,
)

_RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "near_field"

#: Powers of two around the shipped default of 32, down to 1. If the near field is
#: sensitive to this at all, it shows up at the ends.
_BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128)


def main(argv: Optional[list[str]] = None) -> int:
    """Sweep the target block size and report whether it moves the near field."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-particles", type=int, default=1_048_576)
    ap.add_argument("--leaf-size", type=int, default=256)
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--benchmark-scope", choices=("steady_eval", "full"), default="steady_eval"
    )
    ap.add_argument("--blocks", default=",".join(str(b) for b in _BLOCKS))
    ap.add_argument("--repeat", type=int, default=2, help="passes over the whole sweep")
    ap.add_argument("--cuda-visible-devices", type=str, default=None)
    ap.add_argument("--use-autocvd", action="store_true", default=False)
    ap.add_argument("--autocvd-num-gpus", type=int, default=1)
    ap.add_argument("--autocvd-exclude", nargs="*", default=[])
    ap.add_argument("--allow-missing-autocvd", action="store_true", default=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    worker_env, visible = _build_worker_env(args)
    metric = _metric_key(str(args.benchmark_scope))
    blocks = [int(b) for b in str(args.blocks).split(",") if b]
    print(
        f"# N={args.num_particles} leaf={args.leaf_size} order={args.max_order} "
        f"scope={args.benchmark_scope} metric={metric} card={visible or '(inherited)'} "
        f"repeat={args.repeat}",
        flush=True,
    )

    # Interleaved passes rather than all repeats of one block back to back: if the
    # card's state drifts over the sweep, interleaving spreads the drift across every
    # block instead of loading it onto whichever one ran last.
    samples: dict[int, list[float]] = {b: [] for b in blocks}
    rows: list[dict[str, Any]] = []
    for pass_index in range(int(args.repeat)):
        for block in blocks:
            row = _run_case(
                block_size=block,
                args=args,
                cfg=dict(FROZEN_CONFIG),
                worker_env=worker_env,
            )
            seconds = float(row.get(metric, float("nan")))
            samples[block].append(seconds)
            rows.append(
                {
                    "pass": pass_index,
                    "block_size": block,
                    "seconds": seconds,
                    "resolved_nearfield_mode": row.get("resolved_nearfield_mode"),
                    "resolved_nearfield_edge_chunk_size": row.get(
                        "resolved_nearfield_edge_chunk_size"
                    ),
                }
            )
            print(
                f"  pass={pass_index} block={block:4d}  {seconds:.4f}s  "
                f"mode={row.get('resolved_nearfield_mode')}",
                flush=True,
            )

    medians = {b: statistics.median(v) for b, v in samples.items() if v}
    best = min(medians, key=lambda b: medians[b])
    worst = max(medians, key=lambda b: medians[b])
    spread = medians[worst] / medians[best]
    print("\n# block   median s   ratio to best   samples", flush=True)
    for block in blocks:
        vals = samples[block]
        print(
            f"# {block:5d}   {medians[block]:8.4f}   {medians[block]/medians[best]:13.3f}"
            f"   {[round(v, 4) for v in vals]}",
            flush=True,
        )
    print(
        (
            f"\n# best={best} worst={worst} spread={spread:.3f}x "
            f"(default is 32, at {medians[32]/medians[best]:.3f}x of best)"
            if 32 in medians
            else f"\n# best={best} worst={worst} spread={spread:.3f}x"
        ),
        flush=True,
    )
    verdict = (
        "USABLE as a guard axis"
        if spread >= 1.20
        else "NOT usable as a guard axis: the whole sweep fits inside run-to-run spread"
    )
    print(f"# verdict: {verdict}", flush=True)

    _RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"block_size_sweep_n{args.num_particles}_leaf{args.leaf_size}"
    path = _RESULTS / f"{stem}{('_' + args.tag) if args.tag else ''}.json"
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "cuda_visible_devices": visible,
                "metric_key": metric,
                "frozen_config": FROZEN_CONFIG,
                "medians": {str(k): v for k, v in medians.items()},
                "spread": spread,
                "best_block": best,
                "worst_block": worst,
                "verdict": verdict,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"# wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
