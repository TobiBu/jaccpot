"""Figure 06 -- per-stage time breakdown vs N (build / P2M+M2M / M2L / L2L / P2P).

Adapted from ``bench/profile_refresh_stage_breakdown.py``, which answers "where
does per-step time actually go" at one N; this sweeps N so the *shares* can be
plotted as a stacked series.

Why this figure is measured on the large-N GPU path
--------------------------------------------------
Per-stage timers exist only on the strict refresh path, and
``refresh_prepared_state`` is implemented only for
``preset="large_n_gpu"``/radix/``solidfmm`` -- asking for it on the default path
raises. That is the production per-step path at the N this figure covers, so it
is the right thing to profile, but the breakdown describes *that* configuration
and the JSON's ``config`` says so. It is not a breakdown of the
``accurate``/``real`` default that figures 01-03 use.

What "per-step" means here
--------------------------
The timers are reset after warmup and accumulate over ``--steps``
refresh-and-evaluate calls, with the particles drifting slightly between steps so
the topology is reused as it would be inside an integrator. Reported values are
per step.

Unattributed time
-----------------
The stage timers do not partition the wall clock exactly: they cover tree build,
the upward pass, the dual/downward pass and the near field, but the
local-to-particle evaluation and assorted host-side bookkeeping are not
separately instrumented. Rather than rescale the stages to sum to the measured
wall clock -- which would inflate whichever stage happens to be largest -- the
remainder is reported explicitly as ``unattributed``, and the notebook plots it.
If it is a large share, the figure should say so rather than hide it.

Usage
-----
This figure needs a GPU; the strict path is GPU-only::

    python -m bench.scaling.stage_breakdown --n 65536,262144,1048576 --steps 10
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/stage_breakdown.json"

# Environment the strict non-fused path needs for its per-stage timers to be
# populated, lifted from bench/profile_refresh_stage_breakdown.py. Fused mode is
# off on purpose: fusing the stages is exactly what makes them unmeasurable.
PROFILE_ENV = {
    "JACCPOT_REFRESH_TIMING_ENABLE": "1",
    "JACCPOT_PROFILE_UPWARD_STAGES": "1",
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "off",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
}

# Raw `_refresh_timing_<name>_seconds` counters grouped into the stages the figure
# shows. Only counters named here are summed, and the rest of the measured wall
# clock becomes `unattributed`. That ordering matters: the counters are a
# *hierarchy*, not a partition -- `nearfield` is itself the sum of the
# `nearfield_*` children, and `tree_upward` the sum of the `upward_*` ones -- so a
# scheme that sums "everything not explicitly excluded" double-counts. Measured at
# N=65536: nearfield 80.33 ms against its children summing to 79.4 ms, which the
# first version of this map charged twice.
#
# Sum-only-what-is-named makes double counting impossible by construction. The
# risk it trades for is a genuinely-measured stage silently joining
# `unattributed` because nobody mapped it, so every run also records
# `unmapped_nonzero_s` and prints it.
STAGE_MAP: dict[str, tuple[str, ...]] = {
    # Zero during a refresh loop, and that is the point: topology is reused, so a
    # rebuilt tree would show up here.
    "tree_build": ("tree_build",),
    # Its own band, not folded into P2M/M2M. Measured at 634 ms of a 1079 ms step
    # at N=65536 -- 59%, the single largest cost in the refresh, and burying it
    # inside an upward-pass band was actively misleading.
    "upward_geometry": ("upward_geometry",),
    "p2m_m2m": ("upward_p2m", "upward_m2m", "upward_mass_moments"),
    "m2l": ("dual_m2l_compute",),
    "l2l": ("dual_l2l_compute",),
    "nearfield_p2p": ("nearfield",),
    "traversal_setup": (
        "dual_setup",
        "dual_artifact_build",
        "dual_far_pair_plan",
        "dual_select_interactions",
    ),
}

# Counters that are known sums of other counters. Not plotted, and reported
# separately so `unmapped_nonzero_s` stays a list of genuine surprises.
KNOWN_AGGREGATES = {
    "total",
    "tree_upward",
    "upward_compute",
    "dual_downward",
    "dual_downward_compute",
    "nearfield_leaf_groups",
    "nearfield_precompute",
    "nearfield_target_blocks",
    "nearfield_block_sort",
    "nearfield_speed_layout",
    "nearfield_overflow_profile",
    "nearfield_radix_payload",
    "nearfield_neighbor_padding",
    "nearfield_state_pack",
    "nearfield_residual",
    "input",
    "profile_accounting",
    "compile_or_sync_suspect",
    "dual_split_combined",
    "dual_raw_combined",
    "dual_finalize",
    "dual_residual",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default="65536,131072,262144,524288,1048576",
        help="Comma-separated particle counts",
    )
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--leaf-size", type=int, default=256)
    p.add_argument("--drift", type=float, default=1e-4)
    p.add_argument(
        "--max-pair-queue",
        type=int,
        default=1 << 22,
        help=(
            "Traversal pair-queue capacity. The default 131072 fails at and above "
            "N=262144 with 'Pair queue capacity exceeded', which cost this figure "
            "its top two ladder points on the first run (default: 4194304)"
        ),
    )
    p.add_argument("--max-neighbors-per-leaf", type=int, default=1 << 16)
    p.add_argument("--max-interactions-per-node", type=int, default=1 << 16)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    for key, value in PROFILE_ENV.items():
        os.environ.setdefault(key, value)

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from yggdrax.interactions import DualTreeTraversalConfig  # noqa: E402

    from jaccpot import (  # noqa: E402
        FastMultipoleMethod,
        FMMAdvancedConfig,
        RuntimePolicyConfig,
    )

    if jax.default_backend() == "cpu":
        print(
            "[fig06] WARNING: the strict refresh path this figure instruments is "
            "GPU-only, so every N will be recorded as an error on CPU."
        )

    ns = [int(v) for v in str(args.n).split(",") if v.strip()]
    records: list[dict[str, Any]] = []

    for n in ns:
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, "uniform_cube", n))
        k_pos, k_mass, k_vel = jax.random.split(key, 3)
        pos = jax.random.uniform(
            k_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
        )
        mass = jax.random.uniform(
            k_mass, (n,), minval=0.1, maxval=1.0, dtype=jnp.float32
        )
        vel = 0.01 * jax.random.normal(k_vel, (n, 3), dtype=jnp.float32)

        advanced = FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=DualTreeTraversalConfig(
                    max_pair_queue=int(args.max_pair_queue),
                    process_block=512,
                    max_interactions_per_node=int(args.max_interactions_per_node),
                    max_neighbors_per_leaf=int(args.max_neighbors_per_leaf),
                )
            )
        )
        solver = FastMultipoleMethod(
            advanced=advanced,
            preset="large_n_gpu",
            runtime_path="large_n",
            expansion_basis="solidfmm",
            complex_rotation="solidfmm",
            theta=float(args.theta),
            nearfield_mode="bucketed",
            nearfield_edge_chunk_size=64,
            grouped_interactions=False,
            working_dtype=jnp.float32,
            tree_build_mode="static_radix",
            fixed_order=int(args.order),
        )

        def step(prepared, p):
            prepared, acc = solver.strict_prepare_refresh_and_evaluate(
                prepared,
                p,
                mass,
                leaf_size=int(args.leaf_size),
                max_order=int(args.order),
                theta=float(args.theta),
            )
            jax.block_until_ready(acc)
            return prepared, acc

        try:
            prepared = None
            p = pos
            for _ in range(max(1, int(args.warmup))):
                prepared, _ = step(prepared, p)
                p = p + float(args.drift) * vel

            impl = solver._impl
            # Reset the accumulators after warmup, so compile time is excluded.
            for attr in list(vars(impl)):
                if attr.startswith("_refresh_timing_") and attr.endswith("_seconds"):
                    setattr(impl, attr, 0.0)
            impl._refresh_timing_enabled = True

            t0 = time.perf_counter()
            for _ in range(int(args.steps)):
                impl._refresh_timing_active = True
                prepared, _ = step(prepared, p)
                impl._refresh_timing_active = False
                p = p + float(args.drift) * vel
            wall = time.perf_counter() - t0
        except Exception as exc:
            print(f"N={n:<9d} FAILED: {str(exc)[:160]}")
            records.append({"n": int(n), "error": str(exc)[:400]})
            continue

        raw = {
            attr[len("_refresh_timing_") : -len("_seconds")]: float(getattr(impl, attr))
            for attr in vars(impl)
            if attr.startswith("_refresh_timing_") and attr.endswith("_seconds")
        }
        steps = max(1, int(args.steps))
        per_step_raw = {k: v / steps for k, v in raw.items()}

        mapped: dict[str, float] = {}
        claimed: set[str] = set()
        for stage, counters in STAGE_MAP.items():
            total = 0.0
            for counter in counters:
                if counter in per_step_raw:
                    total += per_step_raw[counter]
                    claimed.add(counter)
            mapped[stage] = total

        # Any nonzero counter that is neither mapped nor a known aggregate is a
        # stage this taxonomy has not accounted for. It is reported rather than
        # folded in, so it cannot quietly inflate a band.
        unmapped = {
            k: v
            for k, v in per_step_raw.items()
            if v > 0 and k not in claimed and k not in KNOWN_AGGREGATES
        }
        if unmapped:
            print(
                "  [warn] nonzero counters not in STAGE_MAP: "
                + ", ".join(f"{k}={v*1e3:.2f}ms" for k, v in sorted(unmapped.items()))
            )

        per_step_wall = wall / steps
        stage_sum = sum(mapped.values())
        mapped["unattributed"] = max(per_step_wall - stage_sum, 0.0)

        record = {
            "n": int(n),
            "per_step_wall_s": per_step_wall,
            "stages_s": mapped,
            "stage_sum_s": stage_sum,
            "attributed_fraction": (
                (stage_sum / per_step_wall) if per_step_wall else None
            ),
            "raw_per_step_s": per_step_raw,
            "unmapped_nonzero_s": unmapped,
            "steps": steps,
        }
        records.append(record)
        shares = "  ".join(
            f"{k}={v / per_step_wall * 100:4.1f}%"
            for k, v in mapped.items()
            if per_step_wall and v > 0
        )
        print(f"N={n:<9d} per-step={per_step_wall*1e3:8.2f} ms  {shares}", flush=True)

    config = {
        "n": ns,
        "theta": float(args.theta),
        "order": int(args.order),
        "basis": "solidfmm",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "leaf_size": int(args.leaf_size),
        "preset": "large_n_gpu",
        "runtime_path": "large_n",
        "distribution": "uniform_cube",
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "drift": float(args.drift),
        "profile_env": dict(PROFILE_ENV),
        "max_pair_queue": int(args.max_pair_queue),
        "note": (
            "Per-stage timers exist only on the strict non-fused refresh path, "
            "implemented for preset='large_n_gpu'/radix/solidfmm. This breakdown "
            "describes that configuration, not the accurate/real default used by "
            "figures 01-03."
        ),
        "stage_map": {k: list(v) for k, v in STAGE_MAP.items()},
        "known_aggregates": sorted(KNOWN_AGGREGATES),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
