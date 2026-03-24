"""Scriptable memory benchmark for compile/prepare/evaluate phases."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod, FMMPreset


@dataclass(frozen=True)
class MemorySnapshot:
    label: str
    gpu_used_mb: Optional[float]
    gpu_total_mb: Optional[float]
    wall_time_s: float


def _sample_problem(n: int, *, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=dtype,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=dtype,
    )
    return positions, masses


def _gpu_memory_snapshot(label: str, *, gpu_index: int) -> MemorySnapshot:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            text=True,
        ).strip()
        used_mb_str, total_mb_str = [part.strip() for part in output.split(",", 1)]
        return MemorySnapshot(
            label=label,
            gpu_used_mb=float(used_mb_str),
            gpu_total_mb=float(total_mb_str),
            wall_time_s=time.perf_counter(),
        )
    except Exception:
        return MemorySnapshot(
            label=label,
            gpu_used_mb=None,
            gpu_total_mb=None,
            wall_time_s=time.perf_counter(),
        )


def _block_until_ready(value: Any) -> Any:
    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


def _tree_nbytes(value: Any) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "dtype") and hasattr(leaf, "shape"):
            total += int(jnp.asarray(leaf).size) * int(jnp.asarray(leaf).dtype.itemsize)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-particles", type=int, default=131072)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--preset", type=str, default="large_n_gpu")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--skip-memory-analysis", action="store_true")
    args = parser.parse_args()

    dtype = getattr(jnp, str(args.dtype))
    solver = FastMultipoleMethod(
        preset=FMMPreset(str(args.preset).strip().lower()),
        basis=args.basis,
    )
    positions, masses = _sample_problem(int(args.num_particles), dtype=dtype)
    snapshots = [_gpu_memory_snapshot("start", gpu_index=int(args.gpu_index))]

    eval_fn = jax.jit(lambda state: solver.evaluate_prepared_state(state))

    compile_rows: list[dict[str, Any]] = []
    compile_prepare_s = float("nan")
    snapshots.append(
        _gpu_memory_snapshot("after_prepare_compile", gpu_index=int(args.gpu_index))
    )

    compile_rows.append(
        {
            "phase": "prepare_compile",
            "memory_analysis_error": (
                "prepare_state is intentionally not wrapped in an outer jax.jit "
                "for large-N benchmarks; internal runtime compilation still occurs "
                "during the first prepare_state call"
            ),
        }
    )

    warm_prepare_start = time.perf_counter()
    state = solver.prepare_state(
        positions,
        masses,
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
    )
    _block_until_ready(state)
    warm_prepare_s = time.perf_counter() - warm_prepare_start
    snapshots.append(
        _gpu_memory_snapshot("after_warm_prepare", gpu_index=int(args.gpu_index))
    )

    eval_compile_start = time.perf_counter()
    eval_compiled = eval_fn.lower(state).compile()
    compile_eval_s = time.perf_counter() - eval_compile_start
    snapshots.append(
        _gpu_memory_snapshot("after_eval_compile", gpu_index=int(args.gpu_index))
    )

    if not bool(args.skip_memory_analysis):
        try:
            stats = eval_compiled.memory_analysis()
            compile_rows.append(
                {
                    "phase": "eval_compile",
                    "generated_code_size": getattr(
                        stats, "generated_code_size_in_bytes", None
                    ),
                    "argument_size": getattr(stats, "argument_size_in_bytes", None),
                    "output_size": getattr(stats, "output_size_in_bytes", None),
                    "temp_size": getattr(stats, "temp_size_in_bytes", None),
                    "alias_size": getattr(stats, "alias_size_in_bytes", None),
                }
            )
        except Exception as exc:
            compile_rows.append(
                {"phase": "eval_compile", "memory_analysis_error": str(exc)}
            )

    warm_eval_start = time.perf_counter()
    result = eval_compiled(state)
    _block_until_ready(result)
    warm_eval_s = time.perf_counter() - warm_eval_start
    snapshots.append(
        _gpu_memory_snapshot("after_warm_eval", gpu_index=int(args.gpu_index))
    )

    retained_state_bytes = _tree_nbytes(state)
    del result
    del eval_compiled
    del state
    gc.collect()
    snapshots.append(
        _gpu_memory_snapshot("after_cleanup", gpu_index=int(args.gpu_index))
    )

    output = {
        "backend": jax.default_backend(),
        "num_particles": int(args.num_particles),
        "leaf_size": int(args.leaf_size),
        "max_order": int(args.max_order),
        "preset": str(args.preset),
        "basis": str(args.basis),
        "dtype": str(dtype),
        "compile_prepare_s": compile_prepare_s,
        "warm_prepare_s": warm_prepare_s,
        "compile_eval_s": compile_eval_s,
        "warm_eval_s": warm_eval_s,
        "retained_state_bytes": retained_state_bytes,
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
        "memory_analysis": compile_rows,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
