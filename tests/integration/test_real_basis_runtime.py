"""Runtime validation for real-basis rotate+scale M2L path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


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
            preset="accurate", basis="real", theta=0.6, softening=soft
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


@pytest.mark.parametrize(
    ("leaf_size", "dtype", "tolerance", "expect_far_field"),
    [
        # leaf_size=8 is what this test used before. At N=96 it accepts NO M2L
        # pairs, so it compares the near field only -- which is basis-independent
        # code and therefore agrees exactly. Kept, but labelled for what it is.
        pytest.param(8, jnp.float32, 1e-14, False, id="nearfield-only-f32"),
        pytest.param(4, jnp.float32, 2e-6, True, id="farfield-f32"),
        pytest.param(4, jnp.float64, 1e-14, True, id="farfield-f64"),
    ],
)
def test_real_basis_tracks_complex_basis(leaf_size, dtype, tolerance, expect_far_field):
    """Real and solidfmm bases agree to round-off on the same frozen topology.

    This is the genuine independent-basis cross-check: ``real`` and ``solidfmm``
    reach the same force through different expansion algebra, so agreement bounds
    the algebra rather than one implementation against itself.

    Two things about this test were previously not true of it.

    First, it asserted ``rel_l2 < 3e-2`` on a quantity that is **exactly zero**. At
    ``leaf_size=8`` and N=96 the MAC accepts no M2L pairs, so both bases evaluate a
    pure near-field P2P sum through the same basis-independent code -- measured,
    the two outputs are bit-identical. That configuration is kept here as the
    ``nearfield-only`` case, with an assertion that actually reflects it, but on its
    own it never compared the two bases.

    Second, 3e-2 was ~6 orders looser than the real agreement even in the regime
    that does exercise the far field. Measured across 4 PRNG seeds x N in {96, 256}
    at ``leaf_size=4`` (52-1090 M2L pairs):

        float32   2.3e-08 .. 1.89e-07     (~1.6 eps_f32)
        float64   3.5e-17 .. 9.63e-17     (~0.4 eps_f64)

    Bounds are 2e-6 for fp32 (~10x headroom) and 1e-14 for fp64. Both are far
    tighter than 3e-2, so this strengthens the test; nothing is relaxed.

    The far-pair count is asserted, in both directions -- ``> 0`` for the far-field
    cases and ``== 0`` for the near-field-only one -- so neither can silently drift
    into the other regime and start passing for the wrong reason.

    Topology parity is also asserted: both bases must accept the *same* number of
    M2L pairs (measured, they always do). Without that, a difference in the accepted
    interaction set would be indistinguishable from a difference in the algebra, and
    the tolerance above would be measuring the wrong thing.
    """
    if dtype is jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("float64 case requires JAX_ENABLE_X64=1")

    n = 96
    key = jax.random.PRNGKey(321)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype
    )
    masses = jnp.abs(jax.random.normal(key_mass, (n,), dtype=dtype)) + jnp.asarray(
        0.5, dtype=dtype
    )

    fmm_real = _solver_with_capacity(basis="real")
    fmm_complex = _solver_with_capacity(basis="complex")

    far_real = _far_pair_count(
        fmm_real.prepare_state(positions, masses, leaf_size=leaf_size, max_order=4)
    )
    far_complex = _far_pair_count(
        fmm_complex.prepare_state(positions, masses, leaf_size=leaf_size, max_order=4)
    )
    assert far_real == far_complex, (
        "the two bases must accept the same M2L set for this to be a basis "
        f"comparison rather than a topology comparison (real={far_real}, "
        f"complex={far_complex})"
    )
    if expect_far_field:
        assert far_real > 0, (
            "config must engage the far field or this compares basis-independent "
            f"near-field code only (got {far_real} M2L pairs)"
        )
    else:
        assert far_real == 0, (
            "this case is the documented near-field-only regime; it now accepts "
            f"{far_real} M2L pairs, so its exact-agreement bound no longer applies"
        )

    acc_real = np.asarray(
        fmm_real.compute_accelerations(
            positions, masses, leaf_size=leaf_size, max_order=4
        ),
        dtype=np.float64,
    )
    acc_complex = np.asarray(
        fmm_complex.compute_accelerations(
            positions, masses, leaf_size=leaf_size, max_order=4
        ),
        dtype=np.float64,
    )

    rel_l2 = float(
        np.linalg.norm(acc_real - acc_complex) / (np.linalg.norm(acc_complex) + 1.0e-12)
    )
    assert rel_l2 < tolerance, (
        f"real and solidfmm bases disagree beyond round-off: rel-L2 {rel_l2:.3e} > "
        f"{tolerance:.0e} (leaf={leaf_size}, {np.dtype(dtype).name}, "
        f"{far_real} M2L pairs)"
    )


def test_real_basis_acceleration_derivatives_match_complex():
    """Real-basis acceleration derivative towers match the complex path.

    Regression guard for the real L2P derivative tower
    (evaluate_local_real_derivative_tower): with the same tree/interactions the
    real and complex bases must produce identical accelerations AND identical
    packed acceleration-derivative levels.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for stable tolerance")

    # Parity claim (real derivative tower == complex derivative tower on the
    # same tree) holds at any order/N, so a small n + moderate order keeps the
    # regression coverage while cutting the compile cost of this test.
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

    def run(basis: str, k: int):
        fmm = FastMultipoleMethod(preset="accurate", basis=basis, theta=0.4)
        return fmm.compute_accelerations(
            positions, masses, leaf_size=16, max_order=4, max_acc_derivative_order=k
        )

    for k in (1, 2):
        real_out = run("real", k)
        complex_out = run("complex", k)
        acc_real = np.asarray(real_out[0])
        acc_complex = np.asarray(complex_out[0])
        assert (
            np.linalg.norm(acc_real - acc_complex)
            / (np.linalg.norm(acc_complex) + 1.0e-12)
            < 1.0e-6
        )
        levels_real = real_out[1]
        levels_complex = complex_out[1]
        assert len(levels_real) == len(levels_complex) == k
        for dr, dc in zip(levels_real, levels_complex):
            dr_np = np.asarray(dr)
            dc_np = np.asarray(dc)
            assert dr_np.shape == dc_np.shape
            rel = np.linalg.norm(dr_np - dc_np) / (np.linalg.norm(dc_np) + 1.0e-30)
            assert rel < 1.0e-6
