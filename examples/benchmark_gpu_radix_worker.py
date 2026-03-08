"""Per-N GPU radix benchmark worker for process-isolated runtime measurements."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pathlib
import sys
import time
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yggdrax import Tree, compute_tree_geometry  # noqa: E402
from yggdrax.interactions import (  # noqa: E402
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
)

from examples import benchmark_utils as bench_utils
from jaccpot import (  # noqa: E402
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
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


def _block_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _evaluate_prepared_kwargs(fmm: FastMultipoleMethod) -> dict[str, Any]:
    params = set(inspect.signature(fmm.evaluate_prepared_state).parameters)
    if "jit_traversal" in params:
        return {"jit_traversal": True}
    return {}


def _runtime_overrides(
    fmm_kwargs: dict[str, Any],
    *,
    traversal_cfg_dict: Optional[dict[str, int]] = None,
    nearfield_edge_chunk_size: Optional[int] = None,
) -> dict[str, Any]:
    advanced: FMMAdvancedConfig = fmm_kwargs["advanced"]
    runtime_cfg = advanced.runtime
    nearfield_cfg = advanced.nearfield
    if traversal_cfg_dict is not None:
        traversal_cfg = DualTreeTraversalConfig(
            process_block=int(traversal_cfg_dict["process_block"]),
            max_neighbors_per_leaf=int(traversal_cfg_dict["max_neighbors_per_leaf"]),
            max_interactions_per_node=int(
                traversal_cfg_dict["max_interactions_per_node"]
            ),
            max_pair_queue=int(traversal_cfg_dict["max_pair_queue"]),
        )
        runtime_cfg = replace(runtime_cfg, traversal_config=traversal_cfg)
    if nearfield_edge_chunk_size is not None:
        nearfield_cfg = replace(
            nearfield_cfg,
            edge_chunk_size=int(nearfield_edge_chunk_size),
        )
    out = dict(fmm_kwargs)
    out["advanced"] = replace(advanced, runtime=runtime_cfg, nearfield=nearfield_cfg)
    return out


def _device_autotune_signature(
    *,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    num_particles: int,
    max_order: int,
    dtype: Any,
) -> str:
    """Build a stable key for worker-side runtime autotune reuse."""
    try:
        dev = jax.devices()[0]
        device_name = str(getattr(dev, "device_kind", getattr(dev, "platform", "cpu")))
    except Exception:
        device_name = "unknown"
    payload = {
        "device": device_name,
        "platform": jax.default_backend(),
        "index_precision": os.environ.get("JACCPOT_INDEX_PRECISION", "int64"),
        "dtype": str(jnp.dtype(dtype)),
        "preset": str(cfg.get("preset", "")),
        "basis": str(cfg.get("basis", "")),
        "theta": float(cfg.get("theta", 0.6)),
        "adaptive_order": bool(cfg.get("adaptive_order", False)),
        "max_order": int(max_order),
        "num_particles": int(num_particles),
        "tree_type": str(cfg.get("tree_type", "")),
        "farfield_mode": str(cfg.get("farfield_mode", "")),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _runtime_autotune_cache_path(
    *,
    cfg: dict[str, Any],
    autotune_cache_path: Optional[str],
) -> Optional[pathlib.Path]:
    """Resolve worker runtime autotune cache path."""
    raw = cfg.get("runtime_autotune_cache_path")
    if raw is None and autotune_cache_path:
        base = pathlib.Path(str(autotune_cache_path))
        raw = str(base.with_name("runtime_worker_autotune_cache.json"))
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return pathlib.Path(text)


def _load_runtime_autotune_entry(
    *,
    path: Optional[pathlib.Path],
    signature: str,
) -> Optional[dict[str, Any]]:
    """Load one runtime-autotune cache entry."""
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return None
    entry = entries.get(signature)
    return entry if isinstance(entry, dict) else None


def _save_runtime_autotune_entry(
    *,
    path: Optional[pathlib.Path],
    signature: str,
    traversal_cfg: Optional[dict[str, int]],
    nearfield_edge_chunk_size: Optional[int],
) -> None:
    """Persist one runtime-autotune cache entry."""
    if path is None:
        return
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    entries = payload.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        payload["entries"] = entries
    entries[signature] = {
        "worker_traversal_config": traversal_cfg,
        "worker_nearfield_edge_chunk_size": (
            None
            if nearfield_edge_chunk_size is None
            else int(nearfield_edge_chunk_size)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _measure_prepare_once(
    *,
    fmm_kwargs: dict[str, Any],
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    autotune_cache_path: Optional[str],
) -> float:
    fmm = FastMultipoleMethod(**fmm_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
    )
    _ = _block_ready(state)
    dt = float(time.perf_counter() - t0)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    return dt


def _worker_autotune_runtime_kwargs(
    *,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    positions: Any,
    masses: Any,
    leaf_size: int,
    max_order: int,
    autotune_cache_path: Optional[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tuned_kwargs = dict(fmm_kwargs)
    autotune_default = str(cfg.get("preset", "")).strip().lower() == "large_n_gpu"
    runtime_cache_path = _runtime_autotune_cache_path(
        cfg=cfg,
        autotune_cache_path=autotune_cache_path,
    )
    signature = _device_autotune_signature(
        cfg=cfg,
        fmm_kwargs=tuned_kwargs,
        num_particles=int(positions.shape[0]),
        max_order=int(max_order),
        dtype=positions.dtype,
    )
    info: dict[str, Any] = {
        "worker_traversal_config": None,
        "worker_nearfield_edge_chunk_size": None,
    }
    cached_entry = _load_runtime_autotune_entry(
        path=runtime_cache_path,
        signature=signature,
    )
    if isinstance(cached_entry, dict):
        cached_traversal = cached_entry.get("worker_traversal_config")
        cached_nf = cached_entry.get("worker_nearfield_edge_chunk_size")
        if isinstance(cached_traversal, dict):
            try:
                tuned_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    traversal_cfg_dict={
                        "max_pair_queue": int(cached_traversal["max_pair_queue"]),
                        "process_block": int(cached_traversal["process_block"]),
                        "max_interactions_per_node": int(
                            cached_traversal["max_interactions_per_node"]
                        ),
                        "max_neighbors_per_leaf": int(
                            cached_traversal["max_neighbors_per_leaf"]
                        ),
                    },
                )
                info["worker_traversal_config"] = {
                    "max_pair_queue": int(cached_traversal["max_pair_queue"]),
                    "process_block": int(cached_traversal["process_block"]),
                    "max_interactions_per_node": int(
                        cached_traversal["max_interactions_per_node"]
                    ),
                    "max_neighbors_per_leaf": int(
                        cached_traversal["max_neighbors_per_leaf"]
                    ),
                }
            except Exception:
                pass
        if cached_nf is not None:
            try:
                tuned_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    nearfield_edge_chunk_size=int(cached_nf),
                )
                info["worker_nearfield_edge_chunk_size"] = int(cached_nf)
            except Exception:
                pass
        if (
            info["worker_traversal_config"] is not None
            or info["worker_nearfield_edge_chunk_size"] is not None
        ):
            return tuned_kwargs, info
    traversal_candidates_raw = cfg.get("traversal_candidates", [])
    if bool(cfg.get("worker_autotune_traversal", autotune_default)) and isinstance(
        traversal_candidates_raw, list
    ):
        best_time = float("inf")
        best_cfg: Optional[dict[str, int]] = None
        for candidate in traversal_candidates_raw:
            if not isinstance(candidate, dict):
                continue
            try:
                trial_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    traversal_cfg_dict={
                        "max_pair_queue": int(candidate["max_pair_queue"]),
                        "process_block": int(candidate["process_block"]),
                        "max_interactions_per_node": int(
                            candidate["max_interactions_per_node"]
                        ),
                        "max_neighbors_per_leaf": int(
                            candidate["max_neighbors_per_leaf"]
                        ),
                    },
                )
                t = _measure_prepare_once(
                    fmm_kwargs=trial_kwargs,
                    positions=positions,
                    masses=masses,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    autotune_cache_path=autotune_cache_path,
                )
                if t < best_time:
                    best_time = t
                    best_cfg = {
                        "max_pair_queue": int(candidate["max_pair_queue"]),
                        "process_block": int(candidate["process_block"]),
                        "max_interactions_per_node": int(
                            candidate["max_interactions_per_node"]
                        ),
                        "max_neighbors_per_leaf": int(
                            candidate["max_neighbors_per_leaf"]
                        ),
                    }
            except Exception:
                continue
        if best_cfg is not None:
            tuned_kwargs = _runtime_overrides(tuned_kwargs, traversal_cfg_dict=best_cfg)
            info["worker_traversal_config"] = best_cfg

    nf_candidates_raw = cfg.get("nearfield_chunk_candidates", [])
    if bool(
        cfg.get("worker_autotune_nearfield_chunk", autotune_default)
    ) and isinstance(nf_candidates_raw, list):
        best_time = float("inf")
        best_nf: Optional[int] = None
        for candidate in nf_candidates_raw:
            try:
                candidate_nf = int(candidate)
            except Exception:
                continue
            if candidate_nf <= 0:
                continue
            try:
                trial_kwargs = _runtime_overrides(
                    tuned_kwargs,
                    nearfield_edge_chunk_size=candidate_nf,
                )
                t = _measure_prepare_once(
                    fmm_kwargs=trial_kwargs,
                    positions=positions,
                    masses=masses,
                    leaf_size=int(leaf_size),
                    max_order=int(max_order),
                    autotune_cache_path=autotune_cache_path,
                )
                if t < best_time:
                    best_time = t
                    best_nf = candidate_nf
            except Exception:
                continue
        if best_nf is not None:
            tuned_kwargs = _runtime_overrides(
                tuned_kwargs,
                nearfield_edge_chunk_size=int(best_nf),
            )
            info["worker_nearfield_edge_chunk_size"] = int(best_nf)

    _save_runtime_autotune_entry(
        path=runtime_cache_path,
        signature=signature,
        traversal_cfg=info["worker_traversal_config"],
        nearfield_edge_chunk_size=info["worker_nearfield_edge_chunk_size"],
    )
    return tuned_kwargs, info


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
            enable_interaction_cache=bool(config.get("enable_interaction_cache", True)),
            retain_traversal_result=bool(config.get("retain_traversal_result", True)),
            retain_interactions=bool(config.get("retain_interactions", True)),
            autotune_m2l_chunk=bool(config.get("autotune_m2l_chunk", autotune_default)),
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


def _make_row_error(*, mode: str, num_particles: int, message: str) -> dict[str, Any]:
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
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)
    benchmark_scope = str(cfg.get("benchmark_scope", "steady_eval")).strip().lower()
    if benchmark_scope not in ("full", "steady_eval"):
        benchmark_scope = "steady_eval"

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
        **_evaluate_prepared_kwargs(fmm),
    )

    if benchmark_scope == "steady_eval":
        full_mean = float(eval_timing.mean)
        full_std = float(eval_timing.std)
    else:
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
        full_mean = float(full_timing.mean)
        full_std = float(full_timing.std)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmm.save_m2l_autotune_cache(str(cache_path))
    row = {
        "num_particles": int(num_particles),
        "mean_seconds": full_mean,
        "std_seconds": full_std,
        "prepare_mean_seconds": float(prepare_once_timing.mean),
        "prepare_std_seconds": float(prepare_once_timing.std),
        "evaluate_mean_seconds": float(eval_timing.mean),
        "evaluate_std_seconds": float(eval_timing.std),
        "benchmark_scope": benchmark_scope,
        "error": "",
    }
    row.update(worker_tune_info)
    return row


def _run_prepare_case(
    *,
    num_particles: int,
    leaf_size: int,
    max_order: int,
    runs: int,
    warmup: int,
    dtype: jnp.dtype,
    seed: int,
    cfg: dict[str, Any],
    fmm_kwargs: dict[str, Any],
    autotune_cache_path: Optional[str] = None,
) -> dict[str, Any]:
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(num_particles))
    positions, masses, _ = bench_utils.generate_random_distribution(
        int(num_particles),
        key=key,
        dtype=dtype,
    )
    tuned_kwargs, worker_tune_info = _worker_autotune_runtime_kwargs(
        cfg=cfg,
        fmm_kwargs=fmm_kwargs,
        positions=positions,
        masses=masses,
        leaf_size=int(leaf_size),
        max_order=int(max_order),
        autotune_cache_path=autotune_cache_path,
    )
    fmm = FastMultipoleMethod(**tuned_kwargs)
    if autotune_cache_path:
        cache_path = pathlib.Path(str(autotune_cache_path))
        if cache_path.exists():
            fmm.load_m2l_autotune_cache(str(cache_path), merge=True)

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
    row = {
        "num_particles": int(num_particles),
        "tree_build_mean_seconds": float(tree_timing.mean),
        "interactions_mean_seconds": float(interactions_timing.mean),
        "upward_mean_seconds": float(residual),
        "downward_mean_seconds": 0.0,
        "prepare_component_sum_seconds": float(prepare_timing.mean),
        "error": "",
    }
    row.update(worker_tune_info)
    return row


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
                cfg=cfg,
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
                cfg=cfg,
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
