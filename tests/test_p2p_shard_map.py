"""Near-field P2P correctness under jax.shard_map (padding-mask fix).

Root cause fixed in prepare_leaf_neighbor_pairs: under shard_map the neighbour
list is not compacted (traced branch of _result_to_neighbors keeps the padded
[num_leaves * max_neighbors] buffer). The pair derivation processed the -1
padding tail, and leaf_lookup[-1] wraps to a real leaf row -> garbage pairs ->
~100x-wrong near forces (correct single-device, where the list is compacted).
Masking padding neighbours (neighbors >= 0) fixes BOTH near-field kernels.

This test drives the DEFAULT compute_leaf_p2p_accelerations (no overrides) under
shard_map and checks it equals the explicit-index/override groups path -- which
was already bit-exact -- proving the default path is now correct. Before the
fix this diverged ~300x.

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        pytest tests/test_p2p_shard_map.py -q
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax import build_interactions_and_neighbors, compute_tree_geometry
from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.partition import global_bounds
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.interactions import DualTreeTraversalConfig
from yggdrax.tree import Tree

from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations

_G = 1.0
_SOFT = 0.02
_LEAF = 8
_CFG = DualTreeTraversalConfig(
    max_interactions_per_node=256,
    max_neighbors_per_leaf=64,
    max_pair_queue=8192,
    process_block=64,
)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="shard_map near-field test needs >= 2 devices"
)


def _override_near(tree, nbr, lp, lm):
    """Explicit-index groups path (rebuild CSR from counts) -- the reference."""
    leaf_nodes = jnp.asarray(nbr.leaf_indices, INDEX_DTYPE)
    nl = leaf_nodes.shape[0]
    nr = jnp.asarray(tree.node_ranges, INDEX_DTYPE)
    total_nodes = nr.shape[0]
    node_to_row = (
        jnp.full((total_nodes + 1,), -1, INDEX_DTYPE)
        .at[leaf_nodes]
        .set(jnp.arange(nl, dtype=INDEX_DTYPE))
    )
    cnt = jnp.asarray(nbr.counts, INDEX_DTYPE)
    srcn = jnp.asarray(nbr.neighbors, INDEX_DTYPE)
    Ne = srcn.shape[0]
    cum = jnp.cumsum(cnt)
    e = jnp.arange(Ne, dtype=INDEX_DTYPE)
    tgt = jnp.searchsorted(cum, e, side="right")
    vv = (e < cum[-1]) & (srcn >= 0)
    srow = node_to_row[jnp.clip(srcn, 0, total_nodes)]
    vv = vv & (srow >= 0)
    tgt = jnp.where(vv, tgt, nl)
    srow = jnp.where(vv, srow, nl)
    srt = jnp.argsort(tgt)
    counts_csr = jnp.bincount(tgt, length=nl).astype(INDEX_DTYPE)
    offs = jnp.concatenate([jnp.zeros((1,), INDEX_DTYPE), jnp.cumsum(counts_csr)])
    kk = jnp.arange(_LEAF, dtype=INDEX_DTYPE)
    lr = nr[leaf_nodes]
    lidx = jnp.clip(lr[:, 0][:, None] + kk[None, :], 0, lp.shape[0] - 1)
    lmask = kk[None, :] < (lr[:, 1] - lr[:, 0] + 1)[:, None]
    return compute_leaf_p2p_accelerations(
        tree,
        nbr,
        lp,
        lm,
        G=_G,
        softening=_SOFT,
        nearfield_mode="baseline",
        node_ranges_override=jnp.zeros((nl + 1, 2), INDEX_DTYPE),
        leaf_nodes_override=jnp.arange(nl, dtype=INDEX_DTYPE),
        neighbor_offsets_override=offs,
        neighbor_indices_override=srow[srt],
        neighbor_counts_override=counts_csr,
        leaf_particle_indices_override=lidx,
        leaf_particle_mask_override=lmask,
    )


def test_default_near_field_correct_under_shard_map():
    ndev = min(2, device_count())
    mesh = make_mesh(ndev)
    per = 48
    rng = np.random.default_rng(1)
    pts = jnp.asarray(rng.uniform(-1, 1, size=(per * ndev, 3)).astype(np.float32))
    mass = jnp.asarray(rng.uniform(0.5, 2, size=(per * ndev,)).astype(np.float32))

    def fn(p, m):
        b = global_bounds(p)
        tree = Tree.from_particles(
            p, m, tree_type="radix", bounds=b, return_reordered=True, leaf_size=_LEAF
        )
        lp, lm = tree.positions_sorted, tree.masses_sorted
        geom = compute_tree_geometry(tree, lp, max_leaf_size=_LEAF)
        _inter, nbr = build_interactions_and_neighbors(
            tree, geom, theta=0.5, traversal_config=_CFG, mac_type="bh"
        )
        near_default = compute_leaf_p2p_accelerations(
            tree,
            nbr,
            lp,
            lm,
            G=_G,
            softening=_SOFT,
            max_leaf_size=_LEAF,
            nearfield_mode="baseline",
        )
        near_ref = _override_near(tree, nbr, lp, lm)
        return (
            jnp.linalg.norm(near_default - near_ref)[None],
            jnp.linalg.norm(near_ref)[None],
        )

    dnorm, refnorm = shard_map(
        fn,
        mesh=mesh,
        in_specs=(P("gpus"), P("gpus")),
        out_specs=(P("gpus"), P("gpus")),
        check_vma=False,
    )(pts, mass)
    rel = np.asarray(dnorm) / (np.asarray(refnorm) + 1e-30)
    assert np.all(rel < 1e-4), f"default-vs-override near rel err {rel}"
