"""The near-field half of the acceleration-derivative tower.

Until 2026-10, ``max_acc_derivative_order`` returned only the far field's
derivatives. :mod:`jaccpot.nearfield.acc_derivatives` adds the direct sum's,
over the same near pairs. These tests check it (fp64) in three ways:

* The pair rule against ``jax.jacfwd`` of the masked direct sum, for every kernel
  at orders 1-3, with pairs inside and outside the support and with masked slots.
* A coincident pair of distinct particles stays finite: ``-G m g(0) I`` at order 1
  and zero at order 2, for every kernel. The C2 Wendland's ``sqrt(r2)`` has no
  forward derivative at ``r = 0``.
* The leaf route against the full self-excluded direct sum, with a block size that
  does not divide the pair count, so the padded tail block runs.

The system-level oracle, near and far against a direct sum through the solver,
is ``tests/integration/test_real_basis_runtime.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.nearfield.acc_derivatives import (
    compute_leaf_p2p_acc_derivatives,
    pair_acc_derivative_tower,
)
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d
from jaccpot.softening import pair_factors, softening_params

jax.config.update("jax_enable_x64", True)

_KERNELS = ("plummer", "ferrers3", "wendland_c2")
_EPS = 0.35  # compact supports h = 0.86 / 1.05 against the spacings below
_G = 1.3


def _rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def _packed(full, order):
    """``[T, 3, 3, ...]`` derivative tensor -> the tower's packed ``[T, 3, C]``."""
    columns = []
    for nx, ny, nz in symmetric_multi_indices_3d(order):
        axes = (0,) * nx + (1,) * ny + (2,) * nz
        columns.append(full[(slice(None), slice(None)) + axes])
    return jnp.stack(columns, axis=-1)


def _direct_acc(kernel, sources, masses):
    """``acc(x, active_row)``: the masked direct sum at one point."""
    params = softening_params(kernel, _EPS, jnp.float64)

    def acc(x, row):
        d = x[None, :] - sources
        r2 = jnp.sum(d * d, axis=1)
        g = pair_factors(jnp.where(row, r2, 1.0), params, kernel)[0]
        return -_G * jnp.sum(jnp.where(row, masses * g, 0.0)[:, None] * d, axis=0)

    return acc


@pytest.mark.parametrize("kernel", _KERNELS)
def test_pair_tower_matches_autodiff(kernel):
    rng = np.random.default_rng(3)
    t, s = 5, 24
    targets = jnp.asarray(rng.standard_normal((t, 3)) * 0.6)
    sources = jnp.asarray(rng.standard_normal((s, 3)) * 0.6)
    masses = jnp.asarray(np.abs(rng.standard_normal(s)) + 0.1)
    active = jnp.asarray(rng.random((t, s)) > 0.2)
    diff = targets[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(np.asarray(diff), axis=-1)
    assert np.any(r < 0.86) and np.any(r > 1.05)  # inside and outside the support

    levels = pair_acc_derivative_tower(
        diff,
        jnp.broadcast_to(masses, (t, s)),
        active,
        G=_G,
        params=softening_params(kernel, _EPS, jnp.float64),
        kernel=kernel,
        max_order=3,
    )
    assert len(levels) == 3
    derivative = _direct_acc(kernel, sources, masses)
    for k, level in enumerate(levels, start=1):
        derivative = jax.jacfwd(derivative)
        reference = _packed(jax.vmap(derivative)(targets, active), k)
        assert level.shape == reference.shape
        assert _rel(level, reference) < 1e-12, (k, _rel(level, reference))


@pytest.mark.parametrize("kernel", _KERNELS)
def test_coincident_pair_is_finite(kernel):
    params = softening_params(kernel, _EPS, jnp.float64)
    masses = jnp.asarray([[0.7, 1.1]])
    diff = jnp.asarray([[[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]]])

    def tower(active):
        return pair_acc_derivative_tower(
            diff,
            masses,
            jnp.asarray([active]),
            G=_G,
            params=params,
            kernel=kernel,
            max_order=2,
        )

    both, other = tower([True, True]), tower([False, True])
    g0 = pair_factors(jnp.asarray(0.0), params, kernel)[0]
    for level in both:
        assert bool(jnp.all(jnp.isfinite(level)))
    np.testing.assert_allclose(
        both[0] - other[0], -_G * 0.7 * g0 * jnp.eye(3)[None], rtol=1e-13, atol=0
    )
    np.testing.assert_allclose(both[1], other[1], rtol=1e-13, atol=0)


def test_leaf_route_matches_direct_sum():
    from jaccpot import FastMultipoleMethod

    n, kernel = 48, "ferrers3"
    rng = np.random.default_rng(11)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (n, 3)))
    masses = jnp.asarray(np.abs(rng.standard_normal(n)) + 0.2)
    fmm = FastMultipoleMethod(
        preset="accurate", basis="real", theta=1e-4, softening=_EPS
    )
    state = fmm.prepare_state(positions, masses, leaf_size=4, max_order=2)
    pos_sorted, mass_sorted = state.positions_sorted, state.masses_sorted

    derivative = _direct_acc(kernel, pos_sorted, mass_sorted)
    not_self = jnp.asarray(~np.eye(n, dtype=bool))
    references = []
    for k in (1, 2):
        derivative = jax.jacfwd(derivative)
        references.append(_packed(jax.vmap(derivative)(pos_sorted, not_self), k))

    for chunk in (5, 4096):  # 5 leaves a padded tail block
        levels = compute_leaf_p2p_acc_derivatives(
            state.tree,
            state.neighbor_list,
            pos_sorted,
            mass_sorted,
            G=_G,
            softening=_EPS,
            softening_kernel=kernel,
            max_leaf_size=int(state.max_leaf_size),
            max_derivative_order=2,
            edge_chunk_size=chunk,
        )
        for level, reference in zip(levels, references):
            assert _rel(level, reference) < 1e-13, (chunk, _rel(level, reference))
