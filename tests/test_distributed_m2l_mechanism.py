"""Separate-array M2L mechanism de-risk (distributed FMM, Phase 4a).

Distributed FMM needs M2L where target nodes index the LOCAL tree but source
multipoles come from a SEPARATE (remote) array. This test proves
``accumulate_m2l_contributions`` supports that, via two operator-agnostic
invariants (single device, fast, no shard_map):

- **additivity**: a multi-source interaction list accumulates the same locals as
  summing the per-source single-interaction runs (routing/accumulation correct);
- **source-array permutation invariance**: permuting the source multipole array
  and remapping ``interactions.sources`` accordingly gives identical locals
  (sources are genuinely indexed into the separate array).

Together these mean a remote multipole array can be dropped in as the M2L source.
"""

import jax.numpy as jnp
import numpy as np
from yggdrax.bounds import infer_bounds
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.interactions import NodeInteractionList
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import (
    accumulate_m2l_contributions,
    initialize_local_expansions,
)
from jaccpot.upward.tree_expansions import NodeMultipoleData, compute_node_multipoles

_ORDER = 2
_LEAF = 8


def _tree(points, bounds):
    pts = jnp.asarray(points)
    mass = jnp.ones(pts.shape[0], dtype=pts.dtype)
    tree = Tree.from_particles(
        pts,
        mass,
        tree_type="radix",
        bounds=bounds,
        return_reordered=True,
        leaf_size=_LEAF,
    )
    return tree, tree.positions_sorted, tree.masses_sorted


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    tgt = rng.uniform(-1.0, -0.2, size=(40, 3)).astype(np.float32)
    src = rng.uniform(0.2, 1.0, size=(48, 3)).astype(np.float32)
    bounds = infer_bounds(jnp.asarray(np.concatenate([tgt, src], 0)))
    tgt_tree, tgt_pos, tgt_mass = _tree(tgt, bounds)
    src_tree, src_pos, src_mass = _tree(src, bounds)
    src_mp = compute_node_multipoles(
        src_tree, src_pos, src_mass, max_order=_ORDER, center_mode="com"
    )
    tgt_mp = compute_node_multipoles(
        tgt_tree, tgt_pos, tgt_mass, max_order=_ORDER, center_mode="com"
    )
    local0 = initialize_local_expansions(tgt_tree, tgt_mp.centers, max_order=_ORDER)
    return tgt_tree, src_tree, src_mp, local0


def _ilist(offsets, sources, counts):
    """NodeInteractionList; only offsets/sources/counts are read by M2L."""
    z = jnp.zeros_like(sources)
    return NodeInteractionList(
        offsets=jnp.asarray(offsets, INDEX_DTYPE),
        sources=jnp.asarray(sources, INDEX_DTYPE),
        targets=z,
        counts=jnp.asarray(counts, INDEX_DTYPE),
        level_offsets=jnp.zeros((1,), INDEX_DTYPE),
        target_levels=z,
    )


def _per_target_sources(n_targets, n_sources, k_each=3):
    """Assign each target node k_each source node ids (round-robin)."""
    src_cols = [(np.arange(n_targets) + k) % n_sources for k in range(k_each)]
    sources = np.stack(src_cols, axis=1).reshape(-1)  # [n_targets*k_each]
    counts = np.full(n_targets, k_each)
    offsets = np.arange(n_targets) * k_each
    return offsets, sources, counts, src_cols


def test_m2l_additivity_over_sources():
    _tgt_tree, _src_tree, src_mp, local0 = _setup(0)
    n_t = int(local0.centers.shape[0])
    n_s = int(src_mp.centers.shape[0])
    offsets, sources, counts, src_cols = _per_target_sources(n_t, n_s)

    multi = accumulate_m2l_contributions(
        _ilist(offsets, sources, counts), src_mp, local0
    )

    # Sum of per-source single-interaction runs (each from zero local0).
    acc = np.asarray(local0.coefficients)
    for col in src_cols:
        one = accumulate_m2l_contributions(
            _ilist(np.arange(n_t), col, np.ones(n_t)), src_mp, local0
        )
        acc = acc + np.asarray(one.coefficients)

    np.testing.assert_allclose(
        np.asarray(multi.coefficients), acc, rtol=1e-4, atol=1e-5
    )
    # non-trivial
    assert np.abs(np.asarray(multi.coefficients)).max() > 0


def test_m2l_source_array_permutation_invariance():
    _tgt_tree, _src_tree, src_mp, local0 = _setup(1)
    n_t = int(local0.centers.shape[0])
    n_s = int(src_mp.centers.shape[0])
    offsets, sources, counts, _ = _per_target_sources(n_t, n_s)

    base = accumulate_m2l_contributions(
        _ilist(offsets, sources, counts), src_mp, local0
    )

    rng = np.random.default_rng(2)
    perm = rng.permutation(n_s)
    inv = np.argsort(perm)
    src_perm = NodeMultipoleData(
        order=src_mp.order,
        centers=src_mp.centers[perm],
        moments=None,  # unused on the M2L path
        packed=src_mp.packed[perm],
        component_matrix=None,
        source_motion_packed=None,
    )
    sources_perm = inv[np.asarray(sources)]
    permd = accumulate_m2l_contributions(
        _ilist(offsets, sources_perm, counts), src_perm, local0
    )

    np.testing.assert_allclose(
        np.asarray(permd.coefficients),
        np.asarray(base.coefficients),
        rtol=1e-5,
        atol=1e-6,
    )
