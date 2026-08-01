"""Profile prepare/evaluate stages with coarse wall-clock timing."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod, FMMPreset


def _sample_problem(n: int, *, dtype):
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


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-particles", type=int, default=131072)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--preset", type=str, default="large_n_gpu")
    parser.add_argument("--basis", type=str, default="solidfmm")
    parser.add_argument(
        "--runtime-path",
        choices=("auto", "large_n"),
        default="auto",
    )
    args = parser.parse_args()

    solver = FastMultipoleMethod(
        preset=FMMPreset(str(args.preset).strip().lower()),
        basis=args.basis,
        runtime_path=str(args.runtime_path).strip().lower(),
    )
    positions, masses = _sample_problem(args.num_particles, dtype=jnp.float32)
    impl = solver._impl

    timings: list[tuple[str, float]] = []

    t0 = time.perf_counter()
    tree_artifacts = impl._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
        refine_local_val=bool(impl.refine_local),
        max_refine_levels_val=int(impl.max_refine_levels),
        aspect_threshold_val=float(impl.aspect_threshold),
        jit_tree_override=solver.advanced.runtime.jit_tree,
        upward_center_mode="aabb" if bool(impl.grouped_interactions) else "com",
        allow_stateful_cache=False,
    )
    jax.block_until_ready(tree_artifacts.upward.multipoles.packed)
    timings.append(("tree+upward", time.perf_counter() - t0))

    runtime_overrides = impl._resolve_runtime_execution_overrides(
        num_particles=int(args.num_particles)
    )
    t0 = time.perf_counter()
    dual_downward = impl._prepare_state_dual_and_downward(
        tree_artifacts=tree_artifacts,
        force_scale_nodes=None,
        upward_center_mode=runtime_overrides.center_mode,
        theta_val=float(impl.theta),
        mac_type_val=impl.mac_type,
        dehnen_radius_scale=float(impl.dehnen_radius_scale),
        runtime_traversal_config=runtime_overrides.traversal_config,
        runtime_m2l_chunk_size=runtime_overrides.m2l_chunk_size,
        runtime_l2l_chunk_size=runtime_overrides.l2l_chunk_size,
        grouped_interactions=runtime_overrides.grouped_interactions,
        farfield_mode=runtime_overrides.farfield_mode,
        record_retry=lambda _event: None,
        refine_local_val=bool(impl.refine_local),
        max_refine_levels_val=int(impl.max_refine_levels),
        aspect_threshold_val=float(impl.aspect_threshold),
        allow_stateful_cache=False,
    )
    jax.block_until_ready(dual_downward.downward.locals.coefficients)
    timings.append(("dual+downward", time.perf_counter() - t0))

    state, elapsed = _time_call(
        solver.prepare_state,
        positions,
        masses,
        leaf_size=int(args.leaf_size),
        max_order=int(args.max_order),
    )
    timings.append(("prepare_state", elapsed))
    timings.append(("prepared_state_type", 0.0))

    _, elapsed = _time_call(solver.evaluate_prepared_state, state)
    timings.append(("evaluate_prepared_state", elapsed))

    for label, seconds in timings:
        if label == "prepared_state_type":
            print(f"{label:24s} {type(state).__name__}")
        else:
            print(f"{label:24s} {seconds:10.4f}s")


if __name__ == "__main__":
    main()
