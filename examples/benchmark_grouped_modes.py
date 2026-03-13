"""Compare far-field execution modes on runtime and GPU memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jaccpot import (
    FMMAdvancedConfig,
    FMMPreset,
    FarFieldConfig,
    FastMultipoleMethod,
    RuntimePolicyConfig,
)


@dataclass(frozen=True)
class PeakSummary:
    phase: str
    wall_seconds: float
    gpu_used_before_mb: float | None
    gpu_used_after_mb: float | None
    gpu_peak_used_mb: float | None
    gpu_peak_delta_mb: float | None
    num_samples: int


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


def _query_gpu_memory_mb(gpu_index: int) -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=%d" % int(gpu_index),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used_mb, total_mb = [float(x.strip()) for x in out.splitlines()[0].split(",")[:2]]
        return used_mb, total_mb
    except Exception:
        return None, None


def _block_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _peak_gpu_memory_trace(fn, *args, phase: str, gpu_index: int, poll_interval_s: float = 0.02, **kwargs):
    samples: list[float] = []
    stop_event = threading.Event()

    def _poll() -> None:
        while not stop_event.is_set():
            used_mb, _ = _query_gpu_memory_mb(gpu_index)
            if used_mb is not None:
                samples.append(float(used_mb))
            stop_event.wait(poll_interval_s)

    before_used, _ = _query_gpu_memory_mb(gpu_index)
    worker = threading.Thread(target=_poll, daemon=True)
    worker.start()
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
        out = _block_ready(out)
    finally:
        stop_event.set()
        worker.join(timeout=max(1.0, 10.0 * poll_interval_s))
    elapsed = time.perf_counter() - t0
    after_used, _ = _query_gpu_memory_mb(gpu_index)
    peak_used = max(samples) if samples else after_used
    return out, PeakSummary(
        phase=phase,
        wall_seconds=float(elapsed),
        gpu_used_before_mb=before_used,
        gpu_used_after_mb=after_used,
        gpu_peak_used_mb=peak_used,
        gpu_peak_delta_mb=(
            None if peak_used is None or before_used is None else float(peak_used - before_used)
        ),
        num_samples=len(samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-particles", type=int, default=131072)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--preset", type=str, default="large_n_gpu")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    dtype = getattr(jnp, str(args.dtype))
    positions, masses = _sample_problem(int(args.num_particles), dtype=dtype)
    rows: list[dict[str, Any]] = []

    mode_specs = (
        {
            "label": "streamed_pair",
            "grouped_interactions": False,
            "farfield_mode": "pair_grouped",
            "streamed_far_pairs": True,
            "memory_objective": "minimum_memory",
        },
        {
            "label": "pair_grouped",
            "grouped_interactions": True,
            "farfield_mode": "pair_grouped",
            "streamed_far_pairs": False,
            "memory_objective": "balanced",
        },
        {
            "label": "class_major",
            "grouped_interactions": True,
            "farfield_mode": "class_major",
            "streamed_far_pairs": False,
            "memory_objective": "balanced",
        },
    )

    for mode_spec in mode_specs:
        fmm = FastMultipoleMethod(
            preset=FMMPreset(str(args.preset).strip().lower()),
            basis=args.basis,
            advanced=FMMAdvancedConfig(
                farfield=FarFieldConfig(
                    grouped_interactions=bool(mode_spec["grouped_interactions"]),
                    mode=str(mode_spec["farfield_mode"]),
                    streamed_far_pairs=bool(mode_spec["streamed_far_pairs"]),
                ),
                runtime=RuntimePolicyConfig(
                    retain_traversal_result=False,
                    memory_objective=str(mode_spec["memory_objective"]),
                ),
            ),
        )
        state, prepare_peak = _peak_gpu_memory_trace(
            fmm.prepare_state,
            positions,
            masses,
            leaf_size=int(args.leaf_size),
            max_order=int(args.max_order),
            phase=f"{mode_spec['label']}:prepare",
            gpu_index=int(args.gpu_index),
        )
        _, eval_peak = _peak_gpu_memory_trace(
            fmm.evaluate_prepared_state,
            state,
            phase=f"{mode_spec['label']}:evaluate",
            gpu_index=int(args.gpu_index),
        )
        rows.append(
            {
                "mode": str(mode_spec["label"]),
                "grouped_interactions": bool(mode_spec["grouped_interactions"]),
                "farfield_mode": str(mode_spec["farfield_mode"]),
                "streamed_far_pairs": bool(mode_spec["streamed_far_pairs"]),
                "memory_objective": str(mode_spec["memory_objective"]),
                "prepare": asdict(prepare_peak),
                "evaluate": asdict(eval_peak),
            }
        )

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
