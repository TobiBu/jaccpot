"""Grouped/cached real-basis M2M and L2L == the ungrouped rotate-scale operators.

The grouped/cached kernels precompute the per-degree real rotation blocks once per
interaction class (shared displacement) and apply them with a cached einsum, running only
the z-axis translation per node. They must be bit-identical (up to fp) to the reference
``m2m_real`` / ``l2l_real``, which are the force-validated Dehnen rotate-scale operators
(off-axis field correct upstream). This is the Phase-1 gate for the M2M/L2L launch-count
refactor: correctness is guaranteed at the operator level before any cascade wiring.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaccpot.operators.m2l_real_rot_scale import (
    l2l_rot_scale_real_batch_cached_blocks,
    m2m_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_from_z_multipole_batch,
    real_rotation_blocks_to_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.real_harmonics import (
    l2l_real,
    m2m_real,
    sh_index,
    sh_size,
    translate_along_z_l2l_real,
    translate_along_z_m2m_real,
)

_TOL = 1e-10


def _rand(order, n, seed):
    rng = np.random.default_rng(seed)
    coeffs = jnp.asarray(rng.standard_normal((n, sh_size(order))))
    deltas = jnp.asarray(rng.standard_normal((n, 3)))  # generic off-axis displacements
    return coeffs, deltas


def _z_shift_unrolled(coeffs, dz, order, which):
    """Reference (unrolled) real z-axis M2M/L2L shift: out[n,m]=sum_k dz^k/k! * src."""
    import math

    p = int(order)
    out = np.zeros((sh_size(p),), dtype=np.float64)
    c = np.asarray(coeffs, dtype=np.float64)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            acc = 0.0
            krange = range(n - abs(m) + 1) if which == "m2m" else range(p - n + 1)
            for k in krange:
                src_n = n - k if which == "m2m" else n + k
                if src_n < abs(m) or src_n > p:
                    continue
                acc += (dz**k) / math.factorial(k) * c[sh_index(src_n, m)]
            out[sh_index(n, m)] = acc
    return out


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("which", ["m2m", "l2l"])
def test_z_translate_vectorized_matches_unrolled(order, which):
    """The table-vectorised M2M/L2L z-translate == the per-(n,m) unrolled reference."""
    rng = np.random.default_rng(int(order) * 10 + (which == "l2l"))
    coeffs = rng.standard_normal(sh_size(order))
    dz = float(abs(rng.standard_normal()) + 0.3)
    fn = translate_along_z_m2m_real if which == "m2m" else translate_along_z_l2l_real
    got = np.asarray(fn(jnp.asarray(coeffs), jnp.asarray(dz), order=order))
    ref = _z_shift_unrolled(coeffs, dz, order, which)
    assert float(np.max(np.abs(got - ref))) < 1e-12


@pytest.mark.parametrize("order", [2, 4, 6])
def test_m2m_cached_matches_m2m_real(order):
    coeffs, deltas = _rand(order, 24, 11)
    ref = jax.vmap(lambda c, d: m2m_real(c, d, order=order))(coeffs, deltas)
    bt = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    bf = real_rotation_blocks_from_z_multipole_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    got = m2m_rot_scale_real_batch_cached_blocks(coeffs, deltas, bt, bf, order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL


@pytest.mark.parametrize("order", [2, 4, 6])
def test_l2l_cached_matches_l2l_real(order):
    coeffs, deltas = _rand(order, 24, 22)
    ref = jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)
    bt = real_rotation_blocks_to_z_local_batch(deltas, order=order, dtype=coeffs.dtype)
    bf = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=coeffs.dtype
    )
    got = l2l_rot_scale_real_batch_cached_blocks(coeffs, deltas, bt, bf, order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL


@pytest.mark.parametrize("order", [2, 4, 6])
def test_fastlane_grouped_l2l_cascade_matches_per_node(order):
    """The grouped real L2L branch of the runtime cascade
    ``_propagate_solidfmm_locals_by_level(l2l_grouped=True)`` is bit-identical to the
    per-node `_l2l_real_batch_kernel` path, for any tree/centres (grouping only precomputes
    the rotation blocks once per displacement class). Guards the Phase-4 fast-lane building
    block."""
    from jaccpot.runtime.kernels.core import _propagate_solidfmm_locals_by_level

    rng = np.random.default_rng(order)
    # small binary tree: internal {0,1,2}, leaves {3,4,5,6}
    left_child = jnp.asarray([1, 3, 5])
    right_child = jnp.asarray([2, 4, 6])
    node_levels = jnp.asarray([0, 1, 1, 2, 2, 2, 2])
    total_nodes = 7
    centers = jnp.asarray(rng.standard_normal((total_nodes, 3)))
    coeffs_np = rng.standard_normal((total_nodes, sh_size(order)))
    kw = dict(
        order=order, rotation="solidfmm", total_nodes=total_nodes, basis_mode="real"
    )
    ref = np.asarray(
        _propagate_solidfmm_locals_by_level(
            jnp.asarray(coeffs_np), centers, left_child, right_child, node_levels, **kw
        )
    )
    grp = np.asarray(
        _propagate_solidfmm_locals_by_level(
            jnp.asarray(coeffs_np),
            centers,
            left_child,
            right_child,
            node_levels,
            l2l_grouped=True,
            mm_class_capacity=64,
            **kw,
        )
    )
    assert float(np.max(np.abs(grp - ref))) < 1e-11


@pytest.mark.parametrize("op", ["m2m", "l2l"])
def test_class_grouping_matches_ungrouped(op):
    """Mirror the octree wiring: jnp.unique -> per-class blocks -> blocks[cls_id]."""
    order, n_classes, n = 4, 8, 50
    rng = np.random.default_rng(33)
    class_deltas = rng.standard_normal((n_classes, 3))
    idx = rng.integers(0, n_classes, size=n)
    deltas = jnp.asarray(class_deltas[idx])
    coeffs = jnp.asarray(rng.standard_normal((n, sh_size(order))))

    if op == "m2m":
        ref = jax.vmap(lambda c, d: m2m_real(c, d, order=order))(coeffs, deltas)
        to_fn, from_fn = (
            real_rotation_blocks_to_z_multipole_batch,
            real_rotation_blocks_from_z_multipole_batch,
        )
        cached = m2m_rot_scale_real_batch_cached_blocks
    else:
        ref = jax.vmap(lambda c, d: l2l_real(c, d, order=order))(coeffs, deltas)
        to_fn, from_fn = (
            real_rotation_blocks_to_z_local_batch,
            real_rotation_blocks_from_z_local_batch,
        )
        cached = l2l_rot_scale_real_batch_cached_blocks

    uniq, cls_id = jnp.unique(
        deltas, axis=0, size=n_classes, return_inverse=True, fill_value=0.0
    )
    cls_id = cls_id.reshape(-1)
    bt = to_fn(uniq, order=order, dtype=coeffs.dtype)
    bf = from_fn(uniq, order=order, dtype=coeffs.dtype)
    got = cached(coeffs, deltas, bt[cls_id], bf[cls_id], order=order)
    assert float(np.max(np.abs(np.asarray(got - ref)))) < _TOL
