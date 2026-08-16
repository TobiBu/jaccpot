"""Prepare-only far-pair census: is a sweep configuration measuring anything?

Trap 11 in ``docs/dehnen_mass_mac_status_and_plan.md``: leaf_size and N are
coupled, and getting it wrong fails *silently*. At N=1e5 / leaf 256 the fixed-theta
arm accepts 0 / 0 / 4 / 218 far pairs at theta = 0.30 / 0.34 / 0.38 / 0.42, so the
whole arm sits at machine precision with no error range to match against -- and the
run is *slow*, because with no far field accepted it degenerates to all-to-all
direct summation. That reads as "the tight-eps configs are just expensive". It is
not expensive, it is empty. Eight sweep attempts have failed on configuration this
way, two of them multi-hour.

The diagnosis is minutes rather than hours because it needs neither the O(N^2)
reference nor the evaluation: build the tree, run the traversal, count the accepted
far pairs. That is what this does.

The count comes from :mod:`bench.validation._lane_probe`, which hooks the dual-tree
build rather than reading the prepared state -- see that module for why the state
cannot supply it on the large-N lane. The probe also asserts that a ``pair_policy``
was really installed and that the acceptance threshold is not the constant produced
by the unit force-scale fallback.

Usage::

    eval $(autocvd -l -q)
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD python -u \\
      -m bench.validation.far_pair_census \\
        --n 1000000 --leaf-size 256 --order 8 --distribution plummer \\
        --theta 0.4,0.5,0.6,0.7 --eps 1e-3,1e-4,1e-5 \\
        --runtime-lane large_n --json-out bench/results/validation/census_1e6_leaf256.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from yggdrax.interactions import DualTreeTraversalConfig  # noqa: E402

from bench.validation._lane_probe import DualBuildProbe  # noqa: E402
from bench.validation.mac_error_distribution import make_distribution  # noqa: E402
from jaccpot.config import FMMAdvancedConfig  # noqa: E402
from jaccpot.solver import FastMultipoleMethod  # noqa: E402


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in str(text).split(",") if v.strip())


def _solver(
    *,
    arm: str,
    knob: float,
    order: int,
    geometry_mode: str,
    softening: float,
    G: float,
    runtime_lane: str,
    force_scale_mode: str,
    split_build: str,
    max_pair_queue: Optional[int],
    max_interactions_per_node: Optional[int],
) -> FastMultipoleMethod:
    cfg = FMMAdvancedConfig()
    if runtime_lane == "large_n":
        cfg = replace(cfg, tree=replace(cfg.tree, tree_type="radix"))
    # `prepare_stage_memory_split_enabled` has to be set EXPLICITLY for the lane's
    # low-peak split build. Its default,
    # `_streamed_minimum_memory_gpu_default_split_build`, is computed in
    # `__init__` from `memory_objective`/`streamed_far_pairs` *before*
    # `_apply_large_n_gpu_production_contract` coerces them -- so on the
    # `large_n_gpu` preset the predicate reads `memory_objective="balanced"` and
    # comes out False, and the preset silently runs the monolithic build it exists
    # to avoid. Measured consequence: the N=1e7 census OOMed in
    # `_dual_tree_build_raw` trying to allocate 4.77 GiB.
    runtime = replace(
        cfg.runtime,
        retain_traversal_result=True,
        retain_interactions=True,
        prepare_stage_memory_split_enabled=(
            True
            if split_build == "on"
            else (False if split_build == "off" else (runtime_lane == "large_n"))
        ),
    )
    if max_pair_queue is not None and max_interactions_per_node is not None:
        base = runtime.traversal_config
        fields = {
            "max_pair_queue": int(max_pair_queue),
            "max_interactions_per_node": int(max_interactions_per_node),
        }
        runtime = replace(
            runtime,
            traversal_config=(
                replace(base, **fields)
                if base is not None
                else DualTreeTraversalConfig(process_block=512, **fields)
            ),
        )
    mac_type = "dehnen" if arm == "fixed" else "dehnen_error"
    cfg = replace(cfg, mac_type=mac_type, runtime=runtime)

    kwargs: dict[str, Any] = dict(advanced=cfg, G=G, softening=softening)
    if arm == "fixed":
        kwargs["theta"] = float(knob)
    else:
        # theta does not gate acceptance in paper mode; eps is the knob. The
        # prepass angle is resolved separately (mac_force_scale_prepass_theta).
        kwargs["theta"] = 1.0
        kwargs["adaptive_eps"] = float(knob)
        kwargs["dehnen_geometry_mode"] = geometry_mode
        kwargs["mac_force_scale_mode"] = force_scale_mode
    if runtime_lane == "large_n":
        kwargs["preset"] = "large_n_gpu"
        kwargs["expansion_basis"] = "solidfmm"
    return FastMultipoleMethod(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000000)
    ap.add_argument("--leaf-size", type=int, default=256)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--distribution", default="plummer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", default="0.4,0.5,0.6,0.7")
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5")
    ap.add_argument("--arms", default="fixed,mass")
    ap.add_argument("--geometry-mode", default="com")
    ap.add_argument("--softening", type=float, default=0.0)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument(
        "--force-scale-mode",
        default="paper_cached",
        help="paper_cached = eq (16a); paper_fb = eq (16b)'s O(N) estimator.",
    )
    ap.add_argument("--runtime-lane", choices=("generic", "large_n"), default="generic")
    ap.add_argument(
        "--split-build",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Force the low-peak split traversal build on or off. 'auto' means on for "
            "--runtime-lane large_n. Use 'on'/'off' to A/B the two builds' accept "
            "masks: they are bit-identical at small N in fp64, but at N=1e6 in fp32 "
            "they differed by 2 far pairs in 1783416 at eps=3e-5."
        ),
    )
    ap.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    ap.add_argument(
        "--min-far-pairs",
        type=int,
        default=5000,
        help="Warn when an arm falls below this; such a config measures nothing.",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = jnp.float32 if args.precision == "fp32" else jnp.float64
    pos_np, mass_np = make_distribution(args.distribution, args.n, args.seed)
    positions = jnp.asarray(pos_np, dtype=dtype)
    masses = jnp.asarray(mass_np, dtype=dtype)
    print(
        f"backend={jax.default_backend()} n={args.n} leaf={args.leaf_size} "
        f"order={args.order} dist={args.distribution} lane={args.runtime_lane} "
        f"precision={args.precision}",
        flush=True,
    )

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    records: list[dict[str, Any]] = []
    print(
        f"{'arm':6s} {'knob':>10s} {'far pairs':>12s} {'near leaf pairs':>16s} "
        f"{'prepare s':>10s} {'nodes':>8s}  force scale",
        flush=True,
    )
    for arm in arms:
        for knob in _floats(args.theta if arm == "fixed" else args.eps):
            fmm = _solver(
                arm=arm,
                knob=knob,
                order=args.order,
                geometry_mode=args.geometry_mode,
                softening=args.softening,
                G=args.G,
                runtime_lane=args.runtime_lane,
                force_scale_mode=args.force_scale_mode,
                split_build=args.split_build,
                max_pair_queue=None,
                max_interactions_per_node=None,
            )
            engine = getattr(fmm, "_impl", fmm)
            t0 = time.perf_counter()
            with DualBuildProbe() as probe:
                state = fmm.prepare_state(
                    positions,
                    masses,
                    leaf_size=args.leaf_size,
                    max_order=args.order,
                )
            elapsed = time.perf_counter() - t0
            declined = fmm.get_runtime_diagnostics().get("large_n_path_declined_reason")
            if args.runtime_lane == "large_n":
                if declined is not None:
                    raise RuntimeError(
                        f"--runtime-lane large_n requested but the lane declined: "
                        f"{declined!r} (arm={arm}, knob={knob})"
                    )
                if type(state).__name__ != "LargeNPreparedState":
                    raise RuntimeError(
                        "--runtime-lane large_n requested but prepare_state returned "
                        f"{type(state).__name__} (arm={arm}, knob={knob})"
                    )
            final = probe.final
            if arm != "fixed":
                probe.check_criterion_was_applied(context=f"arm={arm} knob={knob:g}")
            threshold = final.get("threshold")
            scale_text = (
                "geometric"
                if threshold is None
                else (
                    f"eps*s in [{threshold['min']:.3e}, {threshold['max']:.3e}] "
                    f"{threshold['dtype']}"
                )
            )
            rec = {
                "arm": arm,
                "knob": float(knob),
                "far_pairs": final.get("far_pairs", -1),
                "near_leaf_pairs": final.get("near_leaf_pairs", -1),
                "prepare_s": round(elapsed, 2),
                "nodes": int(state.tree.parent.shape[0]),
                "dual_calls": len(probe.calls),
                "runtime_lane": args.runtime_lane,
                "prepared_state_type": type(state).__name__,
                "large_n_path_declined_reason": declined,
                "threshold": threshold,
                "pair_policy_installed": final.get("pair_policy_installed"),
                "retry_final_caps": [
                    {
                        "max_pair_queue": int(getattr(e, "max_pair_queue", -1) or -1),
                        "max_interactions_per_node": int(
                            getattr(e, "max_interactions_per_node", -1) or -1
                        ),
                    }
                    for e in getattr(engine, "_recent_retry_events", ()) or ()
                ],
            }
            records.append(rec)
            print(
                f"{arm:6s} {knob:10.3g} {rec['far_pairs']:12d} "
                f"{rec['near_leaf_pairs']:16d} {rec['prepare_s']:10.2f} "
                f"{rec['nodes']:8d}  {scale_text}",
                flush=True,
            )
            del state, fmm

    print("\n=== verdict ===")
    for arm in arms:
        arm_recs = [r for r in records if r["arm"] == arm]
        healthy = [r for r in arm_recs if r["far_pairs"] >= args.min_far_pairs]
        if not healthy:
            print(
                f"{arm}: NO configuration reaches {args.min_far_pairs} far pairs. This "
                "grid measures nothing -- widen the knob range before sweeping."
            )
        else:
            lo = min(r["knob"] for r in healthy)
            hi = max(r["knob"] for r in healthy)
            print(
                f"{arm}: {len(healthy)}/{len(arm_recs)} configs healthy "
                f"(>= {args.min_far_pairs} far pairs), knob range [{lo:g}, {hi:g}]"
            )

    if args.json_out:
        path = pathlib.Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "records": records,
                },
                indent=2,
            )
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
