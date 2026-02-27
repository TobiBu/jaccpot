"""Cross-compare yggdrax tree/traversal vs jaccpot prepare/evaluate timings.

Run with:
    JAX_ENABLE_X64=1 conda run -n expanse python examples/compare_yggdrax_jaccpot_prepare.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from yggdrax import Tree, compute_tree_geometry
from yggdrax.interactions import (
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
)

from jaccpot import FastMultipoleMethod


def _sync(value):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _time_mean(fn, *args, repeats: int = 3, **kwargs) -> float:
    out = fn(*args, **kwargs)
    _sync(out)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _sync(out)
        samples.append(time.perf_counter() - t0)
    return float(sum(samples) / len(samples))


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    num_particles = 16_000
    key = jax.random.PRNGKey(0)
    positions = jax.random.uniform(
        key,
        (num_particles, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jnp.ones((num_particles,), dtype=jnp.float32)

    print(f"N={num_particles}")

    ygg_kdtree_cfg = DualTreeTraversalConfig(
        max_pair_queue=65_536,
        process_block=64,
        max_interactions_per_node=512,
        max_neighbors_per_leaf=2_048,
    )

    for tree_type in ("radix", "kdtree"):
        cfg = ygg_kdtree_cfg if tree_type == "kdtree" else None
        tree_build_s = _time_mean(
            lambda p, m: Tree.from_particles(
                p,
                m,
                tree_type=tree_type,
                build_mode="adaptive",
                return_reordered=True,
                leaf_size=32,
            ),
            positions,
            masses,
        )
        tree = Tree.from_particles(
            positions,
            masses,
            tree_type=tree_type,
            build_mode="adaptive",
            return_reordered=True,
            leaf_size=32,
        )
        geometry = compute_tree_geometry(tree, tree.positions_sorted)
        traversal_s = _time_mean(
            lambda t, g: build_interactions_and_neighbors(
                t,
                g,
                theta=0.6,
                traversal_config=cfg,
                mac_type="dehnen",
                dehnen_radius_scale=1.0,
            ),
            tree,
            geometry,
        )
        print(
            f"[yggdrax {tree_type}] tree_build={tree_build_s:.4f}s "
            f"interactions={traversal_s:.4f}s nodes={int(tree.num_nodes)}"
        )

    for tree_type in ("radix", "kdtree"):
        fmm = FastMultipoleMethod(
            preset="fast",
            basis="solidfmm",
            theta=0.6,
            working_dtype=jnp.float32,
            tree_type=tree_type,
            target_leaf_particles=32,
        )
        prepare_s = _time_mean(
            lambda: fmm.prepare_state(
                positions,
                masses,
                leaf_size=32,
                max_order=4,
            ),
            repeats=2,
        )
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=32,
            max_order=4,
        )
        eval_s = _time_mean(
            lambda st: fmm.evaluate_prepared_state(st), state, repeats=3
        )
        print(
            f"[jaccpot {tree_type}] prepare_state={prepare_s:.4f}s "
            f"evaluate_prepared_state={eval_s:.4f}s"
        )


if __name__ == "__main__":
    main()
