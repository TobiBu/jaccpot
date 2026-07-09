"""De-risk: the local far-field FMM path runs under jax.shard_map (Phase 4c).

Before the full multi-GPU assembly, verify every jaccpot stage the assembly
needs traces + runs inside a shard_map body (per device): Tree.from_particles,
compute_tree_geometry, build_interactions_and_neighbors (the dual-tree walk),
compute_node_multipoles, initialize/accumulate/propagate local expansions, and
the L2P evaluation. Small N so the compile is quick.

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        pytest tests/test_distributed_shardmap_local.py -q
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax import build_interactions_and_neighbors, compute_tree_geometry
from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.tree import Tree
from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.partition import global_bounds

from jaccpot.upward.tree_expansions import compute_node_multipoles
from jaccpot.downward.local_expansions import (
    accumulate_m2l_contributions,
    initialize_local_expansions,
    propagate_local_expansions,
)
from jaccpot.runtime._fmm_impl import _evaluate_local_expansions_for_particles

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="shard_map local-path needs >= 2 devices"
)

_P = 2
_LEAF = 8


def test_local_far_path_under_shard_map():
    ndev = min(2, device_count())
    mesh = make_mesh(ndev)
    per = 32
    n = per * ndev
    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32))
    mass = jnp.asarray(rng.uniform(0.5, 2.0, size=(n,)).astype(np.float32))

    cfg = DualTreeTraversalConfig(
        max_interactions_per_node=256,
        max_neighbors_per_leaf=256,
        max_pair_queue=8192,
        process_block=64,
    )

    def fn(p, m):
        bounds = global_bounds(p)
        tree = Tree.from_particles(
            p, m, tree_type="radix", bounds=bounds, return_reordered=True,
            leaf_size=_LEAF,
        )
        geom = compute_tree_geometry(tree, tree.positions_sorted, max_leaf_size=_LEAF)
        inter, nbr = build_interactions_and_neighbors(
            tree, geom, theta=0.5, traversal_config=cfg, mac_type="bh"
        )
        mp = compute_node_multipoles(
            tree, tree.positions_sorted, tree.masses_sorted, max_order=_P
        )
        local = initialize_local_expansions(tree, mp.centers, max_order=_P)
        local = accumulate_m2l_contributions(inter, mp, local)
        # Under shard_map the NamedTuple's int `order` field is a traced leaf;
        # propagate_local_expansions does int(local_data.order). Force it static.
        local = local._replace(order=_P)
        local = propagate_local_expansions(tree, local)
        far = _evaluate_local_expansions_for_particles(
            local,
            tree.positions_sorted,
            leaf_nodes=nbr.leaf_indices,
            node_ranges=tree.node_ranges,
            max_leaf_size=_LEAF,
            order=_P,
            expansion_basis="cartesian",
            return_potential=False,
        )[0]
        return jnp.sum(far)[None], jnp.sum(jnp.isfinite(far).astype(jnp.int32))[None]

    tot, nfinite = shard_map(
        fn, mesh=mesh, in_specs=(P("gpus"), P("gpus")), out_specs=(P("gpus"), P("gpus")),
        check_vma=False,
    )(pos, mass)
    tot = np.asarray(tot)
    nfinite = np.asarray(nfinite)
    # every per-device far field is finite (path traced + ran)
    assert np.all(np.isfinite(tot))
    assert int(nfinite.sum()) == per * 3 * ndev  # per-device 32 particles * 3 comps
