"""Primary correctness gate for the differentiable FMM force.

The direct O(N^2) sum (:func:`direct_sum_gravitational_acceleration`) is
exactly differentiable, so ``jax.grad`` of the fixed-topology FMM
(:meth:`FastMultipoleMethod.differentiable_accelerations`) must agree with
``jax.grad`` of the direct sum to the FMM's own *force* accuracy -- not to
machine precision. This is the strongest cross-check that the reverse pass is
wired correctly (see ``docs/differentiable_fmm_design.md``).

The configs here use a deliberately *multi-level* tree (small ``leaf_size``) so
the MAC accepts a non-empty M2L interaction list. That is essential: with a
shallow tree the MAC accepts *no* box pairs, the FMM reduces to the near-field
direct sum and equals it to machine precision (~1e-16), and the tolerance below
becomes vacuous -- it would pass regardless of whether the multipole->local
(M2L) reverse pass is correct. Each test asserts the M2L list is non-empty (via
``_num_far_pairs``) and that the gradient error is bounded *below* as well as
above, so it genuinely exercises the far-field reverse path.

Memory: a reverse-mode compile that carries an M2L list has a ~1.15 GB floor,
largely independent of ``n``/``leaf_size`` (a deeper tree at smaller ``n`` costs
about the same) -- expansion order is the main lever. The
``_bound_diff_fmm_compile_cache`` fixture (tests/conftest.py) drops JAX's
compile cache after each of these tests, so the ``-n0`` peak *plateaus* at
~1.5 GB across all the deep configs instead of accumulating; that is safe on the
7 GB / ``-n 2`` CI runner (~3 GB). The ~1.1 GB measured previously was the
*vacuous* far=0 regime, not a floor reachable while genuinely exercising M2L.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod


def _direct_sum_accelerations(positions, masses, *, softening, G):
    diffs = positions[:, None, :] - positions[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1) + softening**2
    inv_dist = jnp.where(dist2 > 0, dist2**-0.5, 0.0)
    inv_dist3 = inv_dist**3
    weights = masses[None, :] * inv_dist3
    weights = weights * (1.0 - jnp.eye(positions.shape[0], dtype=positions.dtype))
    return -G * jnp.einsum("ij,ijk->ik", weights, diffs)


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    probe = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    return positions, masses, probe


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def _num_far_pairs(state):
    """Number of M2L (multipole->local) interactions in the frozen topology.

    Zero means the tree is so shallow the MAC accepts no box pairs: there is no
    far field, so ``grad(FMM)`` collapses onto ``grad(direct-sum)`` at machine
    precision and the tolerance is vacuous. The tests assert this is positive so
    they genuinely exercise the M2L reverse pass.
    """
    inter = state.interactions
    if inter is not None:
        return int(jnp.sum(inter.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


# Both solidfmm submodes at a single deliberately *multi-level* point:
# ``leaf_size=4`` at ``theta=0.6`` builds a tree deep enough that the MAC accepts
# a non-empty M2L interaction list (~16 pairs at n=64), so grad(FMM) genuinely
# differs from grad(direct-sum) at the finite-order tolerance -- unlike the old
# shallow ``leaf_size=16`` setting, where the MAC accepted *no* box pairs and the
# check passed at machine precision regardless of the M2L reverse pass. One
# (theta, order, leaf) point per basis is enough: the reverse pass is
# order-independent to validate, and the deep-config memory plateaus rather than
# accumulating (see the module docstring and tests/conftest.py).
@pytest.mark.parametrize("basis", ["complex", "real"])
@pytest.mark.parametrize("theta,order,leaf,tol", [(0.6, 4, 4, 5e-3)])
def test_grad_fmm_matches_grad_directsum_positions(basis, theta, order, leaf, tol):
    n = 64
    positions, masses, probe = _system(n)
    softening, G = 1e-2, 1.5
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=G, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)
    n_far = _num_far_pairs(state)
    assert n_far > 0, (
        f"config must exercise the far field (got {n_far} M2L pairs); otherwise "
        "this test validates only the near-field reverse path"
    )

    def fmm_loss(pos):
        return jnp.sum(fmm.differentiable_accelerations(state, pos, masses) * probe)

    def dense_loss(pos):
        return jnp.sum(
            _direct_sum_accelerations(pos, masses, softening=softening, G=G) * probe
        )

    g_fmm = jax.grad(fmm_loss)(positions)
    g_dense = jax.grad(dense_loss)(positions)
    assert jnp.all(jnp.isfinite(g_fmm))
    err = _rel_l2(g_fmm, g_dense)
    # Upper bound: grad(FMM) matches grad(direct-sum) to the FMM's force accuracy.
    # Lower bound: guard against silently regressing to the vacuous far=0 regime
    # (err ~ 1e-16), where a broken M2L reverse pass would pass unnoticed.
    assert 1e-5 < err < tol, (
        f"position grad rel-L2 {err:.3e} outside (1e-5, {tol:.1e}) "
        f"(basis={basis}, theta={theta}, far_pairs={n_far})"
    )


@pytest.mark.parametrize("basis", ["complex", "real"])
def test_grad_fmm_matches_grad_directsum_masses(basis):
    n = 64
    positions, masses, probe = _system(n, seed=1)
    # leaf_size=4 at theta=0.6 -> multi-level tree with a non-empty M2L list, so
    # the mass gradient flows through the far field (P2M/M2M/M2L/L2L/L2P), not
    # just the near-field P2P (see module docstring for the far-field rationale
    # and memory characterization).
    softening, G, theta, order, leaf = 1e-2, 1.5, 0.6, 4, 4
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=G, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)
    n_far = _num_far_pairs(state)
    assert n_far > 0, f"config must exercise the far field (got {n_far} M2L pairs)"

    def fmm_loss(m):
        return jnp.sum(fmm.differentiable_accelerations(state, positions, m) * probe)

    def dense_loss(m):
        return jnp.sum(
            _direct_sum_accelerations(positions, m, softening=softening, G=G) * probe
        )

    g_fmm = jax.grad(fmm_loss)(masses)
    g_dense = jax.grad(dense_loss)(masses)
    assert jnp.all(jnp.isfinite(g_fmm))
    err = _rel_l2(g_fmm, g_dense)
    # Bounded below (guard the vacuous far=0 regime) and above (force accuracy).
    assert (
        1e-5 < err < 5e-3
    ), f"mass grad rel-L2 {err:.3e} outside (1e-5, 5e-3) (basis={basis}, far_pairs={n_far})"


def test_differentiable_matches_forward_compute_accelerations():
    """The differentiable path must reproduce the forward FMM force it differentiates."""
    n = 64
    positions, masses, _ = _system(n, seed=2)
    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.4, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=16)
    a_diff = fmm.differentiable_accelerations(state, positions, masses)
    a_fwd = fmm.compute_accelerations(positions, masses, max_order=4, theta=0.4)
    assert a_diff.shape == positions.shape
    assert _rel_l2(a_diff, a_fwd) < 1e-6
