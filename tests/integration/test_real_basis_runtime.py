"""Runtime validation for real-basis rotate+scale M2L path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d


def _direct_sum_accelerations(
    positions: np.ndarray, masses: np.ndarray, *, softening: float
) -> np.ndarray:
    n = int(positions.shape[0])
    out = np.zeros_like(positions)
    eps = np.finfo(positions.dtype).eps
    soft_sq = softening * softening
    for i in range(n):
        delta = positions[i] - positions
        dist_sq = np.sum(delta * delta, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        inv_dist3 = 1.0 / (dist_sq * dist + eps)
        inv_dist3[i] = 0.0
        out[i] = -np.sum((masses[:, None] * inv_dist3[:, None]) * delta, axis=0)
    return out


def _solver_with_capacity(*, basis: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset="accurate",
        basis=basis,
        theta=0.5,
        softening=1.0e-2,
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=DualTreeTraversalConfig(
                    max_pair_queue=131072,
                    process_block=512,
                    max_interactions_per_node=65536,
                    max_neighbors_per_leaf=65536,
                )
            )
        ),
        softening_kernel="plummer",
    )


def test_real_basis_matches_direct_sum_small_n():
    """Real-basis FMM should agree with direct summation at small N."""
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for stable tolerance")

    n = 96
    dtype = jnp.float64
    key = jax.random.PRNGKey(123)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    fmm_real = _solver_with_capacity(basis="real")
    acc_real = np.asarray(
        fmm_real.compute_accelerations(positions, masses, leaf_size=8, max_order=4)
    )
    acc_ref = _direct_sum_accelerations(
        np.asarray(positions), np.asarray(masses), softening=1.0e-2
    )

    rel_l2 = np.linalg.norm(acc_real - acc_ref) / (np.linalg.norm(acc_ref) + 1.0e-12)
    assert rel_l2 < 5.0e-3


def test_real_basis_far_field_converges_with_order():
    """Real-basis far-field must converge with expansion order.

    Regression guard for the runtime real-basis path (Q-based complex->Dehnen
    conversion + real M2L/L2L/L2P routing). A far-field-engaged configuration
    must show the error dropping monotonically as the order grows -- the old
    path (wrong sqrt(2) conversion + conjugate-symmetry applied to real locals)
    plateaued at ~10% regardless of order.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for stable tolerance")

    n = 1500
    dtype = jnp.float64
    key = jax.random.PRNGKey(7)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    soft = 1.0e-3
    acc_ref = _direct_sum_accelerations(
        np.asarray(positions), np.asarray(masses), softening=soft
    )

    def rel_err(order: int) -> float:
        fmm = FastMultipoleMethod(
            preset="accurate",
            basis="real",
            theta=0.6,
            softening=soft,
            softening_kernel="plummer",
        )
        acc = np.asarray(
            fmm.compute_accelerations(positions, masses, leaf_size=16, max_order=order)
        )
        return float(np.linalg.norm(acc - acc_ref) / np.linalg.norm(acc_ref))

    errors = [rel_err(o) for o in (2, 4, 6)]
    # Monotone improvement with order, and a tight final accuracy.
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1.0e-3


def _far_pair_count(state) -> int:
    """M2L interactions in a prepared topology; 0 means the far field never runs."""
    interactions = state.interactions
    if interactions is not None:
        return int(jnp.sum(interactions.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


def _far_source_mask(state, n: int) -> np.ndarray:
    """``mask[i, j]``: source ``j`` reaches target ``i`` through an accepted M2L pair.

    Built from the prepared state's own far-pair list and node ranges, in input
    order. Every source is reached at most once (asserted): a pair list that covered
    a source twice would double-count it, and this oracle would then hide it.
    """
    interactions = state.interactions
    total = int(jnp.sum(interactions.counts))
    sources = np.asarray(interactions.sources)[:total]
    targets = np.asarray(interactions.targets)[:total]
    ranges = np.asarray(state.tree.node_ranges)
    hits = np.zeros((n, n), dtype=np.int64)
    for src, tgt in zip(sources, targets):
        if src < 0 or tgt < 0:
            continue
        (t0, t1), (s0, s1) = ranges[tgt], ranges[src]
        hits[t0 : t1 + 1, s0 : s1 + 1] += 1
    assert hits.max() <= 1, "a source reaches a target through two far pairs"
    to_sorted = np.asarray(state.inverse_permutation)
    return hits[to_sorted][:, to_sorted].astype(bool)


def _packed_symmetric(full: jnp.ndarray, order: int) -> jnp.ndarray:
    """``[N, 3, 3, ...]`` derivative tensor -> the solver's packed layout ``[N, 3, C]``."""
    columns = []
    for nx, ny, nz in symmetric_multi_indices_3d(order):
        axes = (0,) * nx + (1,) * ny + (2,) * nz
        columns.append(full[(slice(None), slice(None)) + axes])
    return jnp.stack(columns, axis=-1)


def test_real_basis_acceleration_derivatives_match_direct_sum():
    """The acceleration's spatial-derivative tower against a direct sum.

    ``max_acc_derivative_order`` returns the derivatives of the full
    acceleration. The near field contributes its direct sum's, exactly
    (:mod:`jaccpot.nearfield.acc_derivatives`, 2026-10); before that the tower was
    far-field only. Three oracles, all ``jax.jacfwd`` at the target of a
    self-excluded direct sum:

    * the full tower against the full direct sum;
    * the far half alone (the engine's ``near_field_derivatives=False``, the
      jerk path's Jacobian) against the field of exactly the sources the FMM's
      accepted far pairs cover for that target, read off the prepared state;
    * the near half (full minus far) against the field of every other source, to
      round-off. That pins that the near pairs are exactly the complement of the
      far ones, with no source missing or counted twice.

    The far half replaces ``test_real_basis_acceleration_derivatives_match_complex``
    (2026-10 cleanup, C). That test compared the tower with the complex basis's to
    1e-6 at ``leaf_size=16``, ``theta=0.4``, a configuration with **no** M2L pair at
    this N, so it compared two zero towers. Here ``leaf_size=4`` engages 52 pairs.

    The local expansion of order ``p`` gives the k-th acceleration derivative to an
    effective order ``p - k - 1``, so the far half converges more slowly than the
    acceleration. The near pairs at softening 1e-3 dominate the full tower's
    norm, so its relative error sits far below the far half's. Measured (fp64,
    N = 128, leaf 4, theta 0.4):

        order   acceleration   far D1     far D2     full D1    full D2
          4        2.31e-4     4.51e-2    2.00e-1    1.52e-4    3.10e-5
          8        9.39e-6     2.27e-3    1.43e-2    7.67e-6    2.21e-6

    Bounds are the order-8 values with ~2x headroom. The far half's order-4 -> 8
    improvement (20x and 14x measured) must exceed 5x: a tower that stopped
    converging -- a wrong lift or sign at one degree -- fails that even when it
    happens to land inside the bound.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for stable tolerance")

    n = 128
    dtype = jnp.float64
    key = jax.random.PRNGKey(5)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )
    softening = 1.0e-3
    leaf_size, theta = 4, 0.4

    def field(mask_row: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        delta = x[None, :] - positions
        dist_sq = jnp.sum(delta * delta, axis=1) + softening * softening
        weights = jnp.where(mask_row, masses * dist_sq ** (-1.5), 0.0)
        return -jnp.sum(weights[:, None] * delta, axis=0)

    def tower(mask: jnp.ndarray, order: int) -> list:
        derivative, levels = field, []
        for k in range(1, order + 1):
            derivative = jax.jacfwd(derivative, argnums=1)
            levels.append(_packed_symmetric(jax.vmap(derivative)(mask, positions), k))
        return levels

    not_self = ~np.eye(n, dtype=bool)
    acc_ref = jax.vmap(field)(jnp.asarray(not_self), positions)
    full_ref = tower(jnp.asarray(not_self), 2)

    errors = {}
    for order in (4, 8):
        fmm = FastMultipoleMethod(
            preset="accurate",
            basis="real",
            theta=theta,
            softening=softening,
            softening_kernel="plummer",
        )
        state = fmm.prepare_state(
            positions, masses, leaf_size=leaf_size, max_order=order
        )
        far_pairs = _far_pair_count(state)
        assert far_pairs > 0, (
            "config must engage the far field, or the far half is zero and its "
            f"oracle compares nothing (got {far_pairs})"
        )
        far_mask_np = _far_source_mask(state, n)
        assert not bool(np.any(np.diagonal(far_mask_np)))
        far_ref = tower(jnp.asarray(far_mask_np), 2)
        near_ref = tower(jnp.asarray(not_self & ~far_mask_np), 2)

        acc, levels = fmm.compute_accelerations(
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=order,
            max_acc_derivative_order=2,
        )
        acc_err = float(jnp.linalg.norm(acc - acc_ref) / jnp.linalg.norm(acc_ref))
        acc_bound = {4: 5.0e-4, 8: 2.0e-5}[order]
        assert acc_err < acc_bound, f"order {order}: acceleration rel-L2 {acc_err:.3e}"
        _, far_levels = fmm._impl.evaluate_prepared_state(
            state, max_acc_derivative_order=2, near_field_derivatives=False
        )

        assert len(levels) == 2 and len(far_levels) == 2
        for k in (1, 2):
            got, far = jnp.asarray(levels[k - 1]), jnp.asarray(far_levels[k - 1])
            assert got.shape == far.shape == full_ref[k - 1].shape

            def rel(a, b):
                return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

            errors[("full", order, k)] = rel(got, full_ref[k - 1])
            errors[("far", order, k)] = rel(far, far_ref[k - 1])
            near_err = rel(got - far, near_ref[k - 1])
            assert (
                near_err < 1.0e-10
            ), f"order {order}, D{k}: near half rel-L2 {near_err:.3e}"

    assert errors[("full", 8, 1)] < 2.0e-5, errors
    assert errors[("full", 8, 2)] < 5.0e-6, errors
    assert errors[("far", 8, 1)] < 5.0e-3, errors
    assert errors[("far", 8, 2)] < 3.0e-2, errors
    for k in (1, 2):
        improvement = errors[("far", 4, k)] / max(errors[("far", 8, k)], 1e-300)
        assert improvement > 5.0, (
            f"far D{k} stopped converging with order: {errors[('far', 4, k)]:.3e} "
            f"at p=4, {errors[('far', 8, k)]:.3e} at p=8 ({improvement:.1f}x)"
        )
    print({key: f"{value:.3e}" for key, value in errors.items()})
