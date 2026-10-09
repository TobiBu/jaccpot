"""The particle-major far-field evaluation equals the padded leaf-major sweep.

``_evaluate_local_expansions_particle_major`` evaluates each particle in its own
leaf's expansion, chunk by chunk, instead of sweeping padded ``[leaves, width]``
blocks (whose ``value_and_grad`` residuals were the fused step's memory peak). Same
arithmetic per particle, so the gradients must match: on leaves of every occupancy,
empty padding leaves at the end, dead rows past the leaves' particles (a mesh shard's
padding), and chunks that do not divide N.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.downward.local_expansions import LocalExpansionData
from jaccpot.runtime.dtypes import INDEX_DTYPE
from jaccpot.runtime.kernels._evaluate import (
    _evaluate_local_expansions_for_particles,
    _evaluate_local_expansions_particle_major,
)

_ORDER = 4
_WIDTH = 8


def _case(seed: int, n_live: int, n_rows: int, num_pad_leaves: int):
    rng = np.random.default_rng(seed)
    # leaves of every occupancy 1.._WIDTH covering the n_live particles in order
    counts = []
    left = n_live
    while left > 0:
        c = int(min(left, rng.integers(1, _WIDTH + 1)))
        counts.append(c)
        left -= c
    num_live_leaves = len(counts)
    num_leaves = num_live_leaves + num_pad_leaves
    num_internal = num_leaves - 1
    ranges = np.zeros((num_internal + num_leaves, 2), np.int64)
    start = 0
    for i, c in enumerate(counts):
        ranges[num_internal + i] = (start, start + c - 1)
        start += c
    for j in range(num_pad_leaves):
        # empty padding leaves: start = n_live, end = start - 1
        ranges[num_internal + num_live_leaves + j] = (n_live, n_live - 1)
    num_nodes = num_internal + num_leaves
    coeff_count = (_ORDER + 1) ** 2
    local = LocalExpansionData(
        order=_ORDER,
        centers=jnp.asarray(rng.normal(size=(num_nodes, 3)), jnp.float32),
        coefficients=jnp.asarray(
            rng.normal(size=(num_nodes, coeff_count)), jnp.float32
        ),
    )
    positions = jnp.asarray(rng.normal(size=(n_rows, 3)), jnp.float32)
    leaf_nodes = jnp.arange(num_internal, num_nodes, dtype=INDEX_DTYPE)
    return local, positions, leaf_nodes, jnp.asarray(ranges, INDEX_DTYPE)


@pytest.mark.parametrize(
    "n_live, n_rows, pad, chunk",
    [(200, 200, 0, 64), (200, 200, 3, 1000), (187, 200, 2, 50), (97, 97, 1, 7)],
)
def test_particle_major_equals_leaf_major(n_live, n_rows, pad, chunk):
    local, positions, leaf_nodes, ranges = _case(11, n_live, n_rows, pad)
    leaf, _, _ = _evaluate_local_expansions_for_particles(
        local,
        positions,
        leaf_nodes=leaf_nodes,
        node_ranges=ranges,
        max_leaf_size=_WIDTH,
        order=_ORDER,
        expansion_basis="solidfmm",
        return_potential=False,
    )
    part = _evaluate_local_expansions_particle_major(
        local,
        positions,
        leaf_nodes=leaf_nodes,
        node_ranges=ranges,
        order=_ORDER,
        chunk=chunk,
    )
    leaf = np.asarray(leaf)
    part = np.asarray(part)
    assert part.shape == (n_rows, 3)
    # same arithmetic per particle; the compiler may contract it differently
    # (fp32 rounding level, ~1e-5 of the largest component on CPU)
    scale = float(np.abs(leaf).max())
    np.testing.assert_allclose(part[:n_live], leaf[:n_live], rtol=0, atol=2e-5 * scale)
    # dead rows (past every leaf's particles) are zero
    assert not np.any(part[n_live:])
    assert np.any(part[:n_live])


@pytest.mark.parametrize("dtype, tol", [(jnp.float64, 1e-13), (jnp.float32, 1e-5)])
@pytest.mark.parametrize("order", [1, 4, 6])
def test_pallas_cartesian_l2p_equals_the_autodiff_l2p(dtype, tol, order):
    """The Pallas particle kernel (interpret mode; gradient by the Cartesian
    recurrence of the solid harmonics) gives the autodiff gradient of the
    spherical-angle form to float rounding: the same function, other operations.
    Empty padding leaves, dead rows past the leaves' particles (zero), a block
    past the last particle."""
    import jaccpot.runtime.kernels._evaluate as ev

    global _ORDER
    saved, _ORDER = _ORDER, order
    try:
        local, positions, leaf_nodes, ranges = _case(11, 187, 200, 2)
    finally:
        _ORDER = saved
    local = LocalExpansionData(
        order=order,
        centers=local.centers.astype(dtype),
        coefficients=local.coefficients.astype(dtype),
    )
    positions = positions.astype(dtype)
    kw = dict(leaf_nodes=leaf_nodes, node_ranges=ranges, order=order, chunk=64)
    ref = np.asarray(
        ev._evaluate_local_expansions_particle_major(local, positions, **kw)
    )
    got = np.asarray(
        ev._evaluate_local_expansions_particle_major(
            local, positions, kernel="interpret", **kw
        )
    )
    assert got.dtype == ref.dtype
    scale = float(np.abs(ref).max())
    np.testing.assert_allclose(got, ref, rtol=0, atol=tol * scale)
    assert not np.any(got[187:])
    assert np.any(got[:187])


@pytest.mark.parametrize("block", [32, 64])
def test_pallas_l2p_finds_each_particles_leaf_in_its_block(block):
    """The kernel finds each particle's leaf from its block's first leaf (no
    per-particle leaf array): blocks of 32 / 64 particles over leaves of 1-8 (up to
    32 leaves a block), empty padding leaves and dead rows at the end."""
    from jaccpot.pallas.l2p_real import l2p_real_particles_pallas

    local, positions, leaf_nodes, ranges = _case(5, 187, 200, 3)
    lr = ranges[leaf_nodes]
    counts = jnp.maximum(lr[:, 1] - lr[:, 0] + 1, 0)
    ref = np.asarray(
        _evaluate_local_expansions_particle_major(
            local,
            positions,
            leaf_nodes=leaf_nodes,
            node_ranges=ranges,
            order=_ORDER,
            chunk=64,
            kernel="interpret",
        )
    )
    got = np.asarray(
        l2p_real_particles_pallas(
            local.coefficients,
            local.centers,
            positions,
            leaf_nodes,
            lr[:, 0],
            counts,
            order=_ORDER,
            block=block,
            interpret=True,
        )
    )
    assert np.array_equal(got, ref)


def test_the_removed_leaf_flatten_switch_raises(monkeypatch):
    """``JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN=1`` reshaped the leaf-major block
    into particle order instead of scattering it -- right only when every leaf is
    full and in order (the case below has neither). It was removed in the 2026-10
    cleanup (X5) and is refused by name.

    The switch is read at trace time and the function is jitted, so the refusal
    is checked on a shape no other test in this module traces."""
    local, positions, leaf_nodes, ranges = _case(13, 113, 120, 1)
    kw = dict(
        leaf_nodes=leaf_nodes,
        node_ranges=ranges,
        max_leaf_size=_WIDTH,
        order=_ORDER,
        expansion_basis="solidfmm",
        return_potential=False,
    )
    monkeypatch.setenv("JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN", "1")
    with pytest.raises(ValueError, match="removed in the 2026-10 cleanup"):
        _evaluate_local_expansions_for_particles(local, positions, **kw)
    monkeypatch.setenv("JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN", "0")
    grad, _, _ = _evaluate_local_expansions_for_particles(local, positions, **kw)
    assert np.any(np.asarray(grad)[:113]) and not np.any(np.asarray(grad)[113:])
