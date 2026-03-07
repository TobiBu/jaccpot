"""Per-N GPU radix benchmark worker for process-isolated runtime measurements."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Optional

import jax
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples import benchmark_utils as bench_utils
from jaccpot import (  # noqa: E402
    FMMAdvancedConfig,
    FMMPreset,
    FarFieldConfig,
    FastMultipoleMethod,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from yggdrax import Tree, compute_tree_geometry  # noqa: E402
from yggdrax.interactions import (  # noqa: E402
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sweep", "prepare"), required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--leaf-size", type=int, required=True)
    parser.add_argument("--max-order", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autotune-cache", default=None)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.float64
    return jnp.float32


def _build_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    preset_norm = str(config.get("preset", "fast")).strip().lower()
    autotune_default = preset_norm == "large_n_gpu"
    traversal_raw = config.get("traversal_config")
    traversal_cfg: Optional[DualTreeTraversalConfig]
    if traversal_raw is None:
        traversal_cfg = None
    else:
        traversal_cfg = DualTreeTraversalConfig(
            process_block=int(traversal_raw["process_block"]),
            max_neighbors_per_leaf=int(traversal_raw["max_neighbors_per_leaf"]),
            max_interactions_per_node=int(traversal_raw["max_interactions_per_node"]),
            max_pair_queue=int(traversal_raw["max_pair_queue"]),
        )

    advanced = FMMAdvancedConfig(
        tree=TreeConfig(
            tree_type=str(config["tree_type"]),
            leaf_target=int(config["leaf_target"]),
        ),
        farfield=FarFieldConfig(
            rotation=str(config.get("farfield_rotation", "solidfmm")),
            mode=str(config.get("farfield_mode", "auto")),
            grouped_interactions=bool(config.get("grouped_interactions", False)),
            streamed_far_pairs=config.get("streamed_far_pairs"),
            mixed_order=bool(config.get("mixed_order", False)),
            mixed_order_min_order=(
                None
                if config.get("mixed_order_min_order") is None
                else int(config["mixed_order_min_order"])
            ),
        ),
        nearfield=NearFieldConfig(
            mode=str(config.get("nearfield_mode", "auto")),
            edge_chunk_size=int(config.get("nearfield_edge_chunk_size", 256)),
            precompute_scatter_schedules=bool(
                config.get("precompute_scatter_schedules", True)
            ),
        ),
        runtime=RuntimePolicyConfig(
            pair_process_block=(
                None
                if config.get("pair_process_block") is None
                else int(config["pair_process_block"])
            ),
            traversal_config=traversal_cfg,
            jit_traversal=bool(config.get("jit_traversal", True)),
            enable_interaction_cache=bool(
                config.get("enable_interaction_cache", True)
            ),
            retain_traversal_result=bool(
                config.get("retain_traversal_result", True)
            ),
            retain_interactions=bool(config.get("retain_interactions", True)),
            autotune_m2l_chunk=bool(
                config.get("autotune_m2l_chunk", autotune_default)
            ),
        ),
        mac_type=str(config.get("mac_type", "dehnen")),
    )
    return dict(
        preset=FMMPreset(str(config["preset"])),
        basis=str(config["basis"]),
        theta=float(config["theta"]),
        softening=float(config["softening"]),
        working_dtype=_dtype_from_name(str(config["working_dtype"])),
        adaptive_order=bool(config.get("adaptive_order", False)),
        p_gears=tuple(int(v) for v in config.get("p_gears", [])),
        adaptive_error_model=str(config.get("adaptive_error_model", "tail_proxy")),
        adaptive_eps=(
            None
            if config.get("adaptive_eps") is None
            else float(config.get("adaptive_eps"))
        ),
        mac_force_scale_mode=str(config.get("mac_force_scale_mode", "prev")),
        advanced=advanced,
    )


def _make_row_error(
    *, mode: str, num_particles: int, message: str
) -> dict[str, Any]:
    if mode == "sweep":
        return {
            "num_particles": int(num_particles),
            "mean_seconds": float("nan"),
            "std_seconds": float("nan"),
            "prepare_mean_seconds": float("nan"),
            "prepare_std_seconds": float("nan"),
            "evaluate_mean_seconds": float("nan"),
            "evaluate_std_seconds": float("nan"),
            "error": str(message),
        }
    return {
        "num_particles": int(num_particles),
        "tree_build_mean_seconds": float("nan"),
        "upward_mean_seconds": float("nan"),
        "interactions_mean_seconds": float("nan"),
        "downward_mean_seconds": float("nan"),
        "prepare_component_sum_seconds": float("nan"),
        "error": str(message),
    }


def _run_sweep_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    fmm = FastMultipoleMethod(**fmm_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    full_timing = bench_utils.time_callable(
        fmm.compute_accelerations,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        reuse_prepared_state=False,
        warmup=int(warmup),
        runs=int(runs),
    )
    prepare_once_timing = bench_utils.time_callable(
        fmm.prepare_state,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    prepared_state = prepare_once_timing.result
    eval_timing = bench_utils.time_callable(
        fmm.evaluate_prepared_state,
        prepared_state,
        warmup=int(warmup),
        runs=int(runs),
    )
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    return {
        "num_particles": int(num_particles),
        "mean_seconds": float(full_timing.mean),
        "std_seconds": float(full_timing.std),
        "prepare_mean_seconds": float(prepare_once_timing.mean),
        "prepare_std_seconds": float(prepare_once_timing.std),
        "evaluate_mean_seconds": float(eval_timing.mean),
        "evaluate_std_seconds": float(eval_timing.std),
        "error": "",
    }


def _run_prepare_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    fmm = FastMultipoleMethod(**fmm_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )

    tree_type = str(getattr(fmm._impl, "tree_type", "radix"))
    tree_mode = str(getattr(fmm._impl, "tree_build_mode", "lbvh"))
    ygg_build_mode = "fixed_depth" if tree_mode == "fixed_depth" else "adaptive"
    theta_val = float(getattr(fmm._impl, "theta", fmm_kwargs.get("theta", 0.6)))
    traversal_cfg = fmm.advanced.runtime.traversal_config
    mac_type = str(getattr(fmm, "mac_type", "dehnen"))
    dehnen_radius_scale = float(getattr(fmm._impl, "dehnen_radius_scale", 1.0))

    tree_timing = bench_utils.time_callable(
        Tree.from_particles,
        positions,
        masses,
        tree_type=tree_type,
        build_mode=ygg_build_mode,
        return_reordered=True,
        leaf_size=int(leaf_size),
        warmup=int(warmup),
        runs=int(runs),
    )
    tree = Tree.from_particles(
        positions,
        masses,
        tree_type=tree_type,
        build_mode=ygg_build_mode,
        return_reordered=True,
        leaf_size=int(leaf_size),
    )
    geometry = compute_tree_geometry(tree, tree.positions_sorted)
    interactions_timing = bench_utils.time_callable(
        build_interactions_and_neighbors,
        tree,
        geometry,
        theta=theta_val,
        traversal_config=traversal_cfg,
        mac_type=mac_type,
        dehnen_radius_scale=dehnen_radius_scale,
        warmup=int(warmup),
        runs=int(runs),
    )
    prepare_timing = bench_utils.time_callable(
        fmm.prepare_state,
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        warmup=int(warmup),
        runs=int(runs),
    )
    residual = max(
        float(prepare_timing.mean)
        - float(tree_timing.mean)
        - float(interactions_timing.mean),
        0.0,
    )
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    return {
        "num_particles": int(num_particles),
        "tree_build_mean_seconds": float(tree_timing.mean),
        "interactions_mean_seconds": float(interactions_timing.mean),
        "upward_mean_seconds": float(residual),
        "downward_mean_seconds": 0.0,
        "prepare_component_sum_seconds": float(prepare_timing.mean),
        "error": "",
    }


def main() -> None:
    args = _parse_args()
    cfg = json.loads(args.config_json)
    fmm_kwargs = _build_runtime_config(cfg)
    dtype = _dtype_from_name(args.dtype)
    autotune_cache_path: Optional[str] = args.autotune_cache
    if autotune_cache_path is None:
        env_cache = os.environ.get("JACCPOT_AUTOTUNE_CACHE_PATH")
        autotune_cache_path = None if env_cache is None else str(env_cache).strip()
    if autotune_cache_path == "":
        autotune_cache_path = None
    try:
        if args.mode == "sweep":
            row = _run_sweep_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
        else:
            row = _run_prepare_case(
                num_particles=args.num_particles,
                leaf_size=args.leaf_size,
                max_order=args.max_order,
                runs=args.runs,
                warmup=args.warmup,
                dtype=dtype,
                seed=args.seed,
                fmm_kwargs=fmm_kwargs,
                autotune_cache_path=autotune_cache_path,
            )
    except Exception as exc:  # pragma: no cover - worker fallback path
        row = _make_row_error(
            mode=args.mode,
            num_particles=args.num_particles,
            message=f"{type(exc).__name__}: {exc}",
        )
    print(json.dumps(row))


if __name__ == "__main__":
    main()
