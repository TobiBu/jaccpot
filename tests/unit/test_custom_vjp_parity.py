"""Bit-for-bit verification harness for analytic custom_vjp kernels.

Every analytic reverse rule installed for the differentiable FMM must produce
gradients identical (to round-off) to autodiff through the pure-JAX primal. This
module provides a reusable checker and the per-rule parity tests. A custom rule
can never silently encode a wrong gradient if these stay green.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def assert_vjp_matches(f_custom, f_ref, primals, *, seed=0, rtol=1e-11, atol=1e-11):
    """Assert f_custom and f_ref agree in value AND vjp for random cotangents.

    ``f_custom``/``f_ref`` map ``*primals -> pytree of arrays``; they must be the
    same mathematical function (f_ref = autodiff reference, f_custom = custom_vjp
    wrapped). Checks the forward outputs match and the cotangents w.r.t. every
    primal match for a random output cotangent.
    """
    out_c, vjp_c = jax.vjp(f_custom, *primals)
    out_r, vjp_r = jax.vjp(f_ref, *primals)

    leaves_c = jax.tree_util.tree_leaves(out_c)
    leaves_r = jax.tree_util.tree_leaves(out_r)
    for a, b in zip(leaves_c, leaves_r):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)

    key = jax.random.PRNGKey(seed)
    cot = []
    for leaf in leaves_c:
        key, sub = jax.random.split(key)
        cot.append(jax.random.normal(sub, leaf.shape, dtype=leaf.dtype))
    cot = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(out_c), cot)

    grads_c = vjp_c(cot)
    grads_r = vjp_r(cot)
    for gc, gr in zip(
        jax.tree_util.tree_leaves(grads_c), jax.tree_util.tree_leaves(grads_r)
    ):
        np.testing.assert_allclose(np.asarray(gc), np.asarray(gr), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# Real-basis L2P custom_vjp
# --------------------------------------------------------------------------
from jaccpot.operators.real_harmonics import (  # noqa: E402
    evaluate_local_real,
    evaluate_local_real_with_grad,
)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_real_l2p_custom_vjp_matches_autodiff(order):
    rng = np.random.default_rng(order)
    ncoeff = (order + 1) ** 2
    local = jnp.asarray(rng.normal(size=(ncoeff,)), dtype=jnp.float64)
    delta = jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float64)

    def f_custom(c, d):
        return evaluate_local_real_with_grad(c, d, order=order)

    def f_ref(c, d):
        def phi(dd):
            return evaluate_local_real(c, dd, order=order)

        potential, grad = jax.value_and_grad(phi)(d)
        return grad, potential

    assert_vjp_matches(f_custom, f_ref, (local, delta))


@pytest.mark.parametrize("order", [2, 4])
def test_real_l2p_custom_vjp_batched_matches_autodiff(order):
    """The rule must compose through vmap (as it does in the assembled L2P)."""
    rng = np.random.default_rng(100 + order)
    ncoeff = (order + 1) ** 2
    n = 8
    local = jnp.asarray(rng.normal(size=(ncoeff,)), dtype=jnp.float64)
    deltas = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)

    def f_custom(c, d):
        return jax.vmap(
            lambda dd: evaluate_local_real_with_grad(c, dd, order=order)[0]
        )(d)

    def f_ref(c, d):
        def one(dd):
            return jax.grad(lambda x: evaluate_local_real(c, x, order=order))(dd)

        return jax.vmap(one)(d)

    assert_vjp_matches(f_custom, f_ref, (local, deltas))


# --------------------------------------------------------------------------
# Near-field P2P analytic custom_vjp (symmetric tidal tensor)
# --------------------------------------------------------------------------
from jaccpot.nearfield.near_field import (  # noqa: E402
    _pair_accel_cvjp,
    _pair_accel_masked_accels,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p2p_analytic_custom_vjp_matches_autodiff(seed):
    rng = np.random.default_rng(seed)
    B, Wt, Ws = 2, 5, 7
    tpos = jnp.asarray(rng.normal(size=(B, Wt, 3)), dtype=jnp.float64)
    spos = jnp.asarray(rng.normal(size=(B, Ws, 3)), dtype=jnp.float64)
    smass = jnp.asarray(rng.uniform(0.5, 1.5, size=(B, Ws)), dtype=jnp.float64)
    # Exercise masking: some invalid targets/sources.
    tmask = jnp.asarray(rng.random((B, Wt)) > 0.2)
    smask = jnp.asarray(rng.random((B, Ws)) > 0.2)
    tmask_f = tmask.astype(jnp.float64)
    smask_f = smask.astype(jnp.float64)
    softening_sq = jnp.asarray(1e-2**2, dtype=jnp.float64)
    G = jnp.asarray(1.5, dtype=jnp.float64)

    def f_custom(tp, sp, sm):
        return _pair_accel_cvjp(tp, sp, sm, tmask_f, smask_f, softening_sq, G)

    def f_ref(tp, sp, sm):
        return _pair_accel_masked_accels(tp, sp, sm, tmask, smask, softening_sq, G)

    assert_vjp_matches(f_custom, f_ref, (tpos, spos, smass))


# --------------------------------------------------------------------------
# Fused Pallas z-axis real M2L custom_vjp (fwd=Pallas, bwd=autodiff-of-twin)
# --------------------------------------------------------------------------
from jaccpot.operators.real_harmonics import (  # noqa: E402
    sh_size,
    translate_along_z_m2l_real,
)
from jaccpot.pallas.m2l_core_z_real import (  # noqa: E402
    m2l_core_z_real_pallas_cvjp,
    pallas_m2l_real_supported,
)


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 4])
def test_m2l_core_z_pallas_custom_vjp_matches_twin(order, interpret):
    """Pallas z-M2L custom_vjp reverse == autodiff of the pure-JAX twin.

    ``interpret=True`` exercises the kernel + custom_vjp on CPU CI (float64, tight
    tolerance); ``interpret=False`` runs the real Pallas GPU kernel (float32, the
    kernel's GPU tolerance), skipped off-GPU or when the backend rejects the tile.
    The reverse is autodiff of ``vmap(translate_along_z_m2l_real)`` -- identical to
    ``f_ref`` -- so gradients match to round-off; only the Pallas-vs-twin forward
    is loosened.
    """
    if not interpret and not pallas_m2l_real_supported():
        pytest.skip("real Pallas M2L kernel requires a GPU/TPU backend")
    if interpret and not jax.config.jax_enable_x64:
        pytest.skip("interpret parity requires x64 for a tight tolerance")

    dtype = jnp.float64 if interpret else jnp.float32
    tol = 1.0e-10 if interpret else 1.0e-5
    rng = np.random.default_rng(order)
    ncoeff = sh_size(order)
    mult = jnp.asarray(rng.normal(size=(8, ncoeff)), dtype=dtype)
    radii = jnp.asarray(rng.uniform(2.0, 5.0, size=(8,)), dtype=dtype)

    def f_custom(m, r):
        return m2l_core_z_real_pallas_cvjp(m, r, order, interpret, "triton")

    def f_ref(m, r):
        return jax.vmap(lambda mm, rr: translate_along_z_m2l_real(mm, rr, order=order))(
            m, r
        )

    try:
        assert_vjp_matches(f_custom, f_ref, (mult, radii), rtol=tol, atol=tol)
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        msg = str(exc).lower()
        if not interpret and ("warpgroup" in msg or "ptx" in msg or "triton" in msg):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise


# --------------------------------------------------------------------------
# Fused complex-basis M2L custom_vjp (fwd=Pallas, bwd=autodiff-of-twin)
# --------------------------------------------------------------------------
from jaccpot.operators.complex_ops import (  # noqa: E402
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
)
from jaccpot.pallas.m2l_complex_fused import (  # noqa: E402
    m2l_complex_fused_jax,
    m2l_complex_fused_pallas_cvjp,
    pallas_m2l_complex_fused_supported,
)


def _complex_m2l_case(order, seed=0, n=9):
    """Build a well-separated complex M2L pair batch (mirrors the parity template)."""
    rng = np.random.default_rng(seed)
    c = sh_size(order)
    mult = (rng.standard_normal((n, c)) + 1j * rng.standard_normal((n, c))).astype(
        np.complex128
    )
    deltas = (rng.standard_normal((n, 3)) * 2.0).astype(np.float64)
    deltas[:, 2] += 3.0  # keep |delta| well away from 0 (well-separated pairs)
    deltas = jnp.asarray(deltas)
    r = jnp.sqrt(jnp.sum(deltas * deltas, axis=1))
    bto = complex_rotation_blocks_to_z_solidfmm_batch(
        deltas, order=order, basis="multipole", dtype=jnp.complex128
    )
    bfr = complex_rotation_blocks_from_z_solidfmm_batch(
        deltas, order=order, basis="local", dtype=jnp.complex128
    )
    return jnp.asarray(mult), bto, bfr, r


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 4])
def test_m2l_complex_fused_pallas_custom_vjp_matches_twin(order, interpret):
    """Fused complex-M2L custom_vjp reverse == autodiff of the pure-jnp twin.

    Differentiates w.r.t. all four kernel inputs (multipoles, both rotation-block
    stacks, and the radius). ``interpret=True`` runs on CPU CI; ``interpret=False``
    runs the real Pallas GPU kernel (skipped off Ampere+ or on backend reject).
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")
    if not interpret and not pallas_m2l_complex_fused_supported():
        pytest.skip("fused complex Pallas M2L requires an Ampere+ (sm_80) GPU")

    tol = 1.0e-10 if interpret else 1.0e-8
    mult, bto, bfr, r = _complex_m2l_case(order, seed=order)

    def f_custom(m, bt, bf, rr):
        return m2l_complex_fused_pallas_cvjp(m, bt, bf, rr, order, interpret, "triton")

    def f_ref(m, bt, bf, rr):
        return m2l_complex_fused_jax(m, bt, bf, rr, order=order)

    try:
        assert_vjp_matches(f_custom, f_ref, (mult, bto, bfr, r), rtol=tol, atol=tol)
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        msg = str(exc).lower()
        if not interpret and (
            "warpgroup" in msg or "ptx" in msg or "triton" in msg or "mosaic" in msg
        ):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise


# --------------------------------------------------------------------------
# Fully-fused real-basis M2L custom_vjp (fwd=Pallas, bwd=autodiff-of-twin)
# --------------------------------------------------------------------------
from jaccpot.operators.m2l_real_rot_scale import (  # noqa: E402
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.pallas.m2l_real_fused import (  # noqa: E402
    m2l_real_fused_jax,
    m2l_real_fused_pallas_cvjp,
    pallas_m2l_real_fused_supported,
)


def _real_m2l_case(order, seed=0, n=9):
    """Build a well-separated real M2L pair batch + the runtime's real blocks."""
    rng = np.random.default_rng(seed)
    c = sh_size(order)
    mult = jnp.asarray(rng.standard_normal((n, c)).astype(np.float64))
    deltas = (rng.standard_normal((n, 3)) * 2.0).astype(np.float64)
    deltas[:, 2] += 3.0  # keep |delta| well away from 0 (well-separated pairs)
    deltas = jnp.asarray(deltas)
    r = jnp.linalg.norm(deltas, axis=1)
    bto = real_rotation_blocks_to_z_multipole_batch(
        deltas, order=order, dtype=jnp.float64
    )
    bfr = real_rotation_blocks_from_z_local_batch(
        deltas, order=order, dtype=jnp.float64
    )
    return mult, bto, bfr, r


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("order", [2, 4])
def test_m2l_real_fused_pallas_custom_vjp_matches_twin(order, interpret):
    """Fused real-M2L custom_vjp reverse == autodiff of the pure-jnp twin.

    Differentiates w.r.t. all four kernel inputs (multipoles, both rotation-block
    stacks, and the radius). ``interpret=True`` runs on CPU CI; ``interpret=False``
    runs the real Pallas GPU kernel (skipped off Ampere+ or on backend reject).
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")
    if not interpret and not pallas_m2l_real_fused_supported():
        pytest.skip("fused real Pallas M2L requires an Ampere+ (sm_80) GPU")

    tol = 1.0e-10 if interpret else 1.0e-8
    mult, bto, bfr, r = _real_m2l_case(order, seed=order)

    def f_custom(m, bt, bf, rr):
        return m2l_real_fused_pallas_cvjp(m, bt, bf, rr, order, interpret, "triton")

    def f_ref(m, bt, bf, rr):
        return m2l_real_fused_jax(m, bt, bf, rr, order=order)

    try:
        assert_vjp_matches(f_custom, f_ref, (mult, bto, bfr, r), rtol=tol, atol=tol)
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        msg = str(exc).lower()
        if not interpret and (
            "warpgroup" in msg or "ptx" in msg or "triton" in msg or "mosaic" in msg
        ):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise


# --------------------------------------------------------------------------
# Fused Pallas near-field custom_vjp (fwd=Pallas, bwd=autodiff-of-twin)
# --------------------------------------------------------------------------
from jaccpot.pallas.nearfield_fused_leaf import (  # noqa: E402
    nearfield_fused_leaf_jax,
    nearfield_fused_leaf_pallas_cvjp,
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas_cvjp,
    pallas_nearfield_fused_supported,
)


def _nf_skip_if_needed(interpret, exc):
    msg = str(exc).lower()
    if not interpret and (
        "warpgroup" in msg or "ptx" in msg or "triton" in msg or "mosaic" in msg
    ):
        pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
    raise exc


@pytest.mark.parametrize("interpret", [True, False])
def test_nearfield_fused_leaf_pallas_custom_vjp_matches_twin(interpret):
    """Fused leaf-major near-field (pairs lane) custom_vjp == autodiff of the twin."""
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")
    if not interpret and not pallas_nearfield_fused_supported():
        pytest.skip("fused near-field Pallas requires an Ampere+ (sm_80) GPU")

    tol = 1.0e-9 if interpret else 1.0e-6
    rng = np.random.default_rng(0)
    nl, wt, k = 3, 8, 12
    tpos = jnp.asarray(rng.standard_normal((nl, wt, 3)), dtype=jnp.float64)
    spos = jnp.asarray(rng.standard_normal((nl, k, 3)), dtype=jnp.float64)
    spos = spos.at[:, :, 2].add(2.0)  # modest target/source separation
    smass = jnp.asarray(rng.uniform(0.5, 1.5, (nl, k)), dtype=jnp.float64)
    tmask = jnp.asarray(rng.random((nl, wt)) > 0.2)
    smask = jnp.asarray(rng.random((nl, k)) > 0.2)
    tmask_f = tmask.astype(jnp.float64)
    smask_f = smask.astype(jnp.float64)
    soft = jnp.asarray(1e-2, dtype=jnp.float64)
    G = jnp.asarray(1.5, dtype=jnp.float64)

    def f_custom(tp, sp, sm):
        return nearfield_fused_leaf_pallas_cvjp(
            tp, tmask_f, sp, sm, smask_f, soft, G, None, 1, None, interpret
        )

    def f_ref(tp, sp, sm):
        return nearfield_fused_leaf_jax(
            tp, tmask, sp, sm, smask, softening_sq=soft, G=G
        )

    try:
        assert_vjp_matches(f_custom, f_ref, (tpos, spos, smass), rtol=tol, atol=tol)
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        _nf_skip_if_needed(interpret, exc)


@pytest.mark.parametrize("interpret", [True, False])
def test_nearfield_leafpair_pallas_custom_vjp_matches_twin(interpret):
    """Leaf-pair (prepacked) near-field custom_vjp == autodiff of the twin.

    Exercises the in-kernel leaf-id gather (ids passed as floats through the rule);
    the twin's gather reverse (scatter-add) is what the custom_vjp must reproduce.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")
    if not interpret and not pallas_nearfield_fused_supported():
        pytest.skip("fused near-field Pallas requires an Ampere+ (sm_80) GPU")

    tol = 1.0e-9 if interpret else 1.0e-6
    rng = np.random.default_rng(1)
    ll, w, s = 4, 8, 3
    leaf_positions = jnp.asarray(rng.standard_normal((ll, w, 3)), dtype=jnp.float64)
    leaf_masses = jnp.asarray(rng.uniform(0.5, 1.5, (ll, w)), dtype=jnp.float64)
    leaf_mask = jnp.asarray(rng.random((ll, w)) > 0.2)
    # each target leaf gathers `s` neighbour source leaves (ids in [0, ll))
    source_leaf_ids = jnp.asarray(rng.integers(0, ll, size=(ll, s)), dtype=jnp.int32)
    source_valid = jnp.asarray(rng.random((ll, s)) > 0.2)
    leaf_mask_f = leaf_mask.astype(jnp.float64)
    ids_f = source_leaf_ids.astype(jnp.float64)
    valid_f = source_valid.astype(jnp.float64)
    soft = jnp.asarray(1e-2, dtype=jnp.float64)
    G = jnp.asarray(1.5, dtype=jnp.float64)

    def f_custom(lp, lm):
        return nearfield_leafpair_pallas_cvjp(
            lp, lm, leaf_mask_f, ids_f, valid_f, soft, G, None, 1, None, interpret
        )

    def f_ref(lp, lm):
        return nearfield_leafpair_jax(
            lp, lm, leaf_mask, source_leaf_ids, source_valid, softening_sq=soft, G=G
        )

    try:
        assert_vjp_matches(
            f_custom, f_ref, (leaf_positions, leaf_masses), rtol=tol, atol=tol
        )
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        _nf_skip_if_needed(interpret, exc)


# ---------------------------------------------------------------------------
# Differentiable radix fast-lane near field (production prepacked lane)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interpret", [True, False])
def test_radix_fast_lane_prepacked_accel_cvjp_matches_tiled_twin(interpret):
    """The differentiable prepacked lane must match its TILED pure-JAX fallback.

    The reference here is deliberately the lane's own tiled fallback, not the dense
    twin used by the other near-field parity tests: that dense reference
    materialises a (leaves, W_t, K, 3) difference tensor (~50 TB at the fiducial
    config, "test-scale only" per its docstring), whereas the tiled fallback is
    what production falls back to and what the reverse rule differentiates.

    The forward is the Pallas kernel, so the forward comparison is the load-bearing
    one -- it is what makes the gradient a gradient of the shipped force. On a real
    GPU in fp32 it agrees to ~4e-6..8e-6 (summation reordering); in fp64 interpret
    mode it is at round-off.
    """
    from jaccpot.nearfield import near_field as nf

    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")
    if not interpret and not pallas_nearfield_fused_supported():
        pytest.skip("fused near-field Pallas requires an Ampere+ (sm_80) GPU")

    rng = np.random.default_rng(0)
    num_leaves, width, max_blocks, block_size = 6, 8, 2, 3
    num_particles = num_leaves * width
    dtype = jnp.float64

    leaf_particle_idx = jnp.asarray(
        np.arange(num_particles).reshape(num_leaves, width), nf.INDEX_DTYPE
    )
    leaf_mask = jnp.ones((num_leaves, width), bool)
    positions = jnp.asarray(rng.normal(size=(num_particles, 3)), dtype)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=num_particles), dtype)
    leaf_positions = positions[leaf_particle_idx]
    leaf_masses = masses[leaf_particle_idx]
    source_leaf_ids = jnp.asarray(
        rng.integers(0, num_leaves, size=(num_leaves, max_blocks, block_size)),
        nf.INDEX_DTYPE,
    )
    source_valid = jnp.asarray(rng.random((num_leaves, max_blocks, block_size)) > 0.3)
    soft = jnp.asarray(1e-2, dtype=dtype)
    G = jnp.asarray(1.0, dtype=dtype)

    def f_custom(leaf_pos, leaf_mass):
        return nf._radix_fast_lane_prepacked_accel_cvjp(
            leaf_pos,
            leaf_mass,
            positions,
            source_leaf_ids.astype(dtype),
            source_valid.astype(dtype),
            leaf_mask.astype(dtype),
            leaf_particle_idx.astype(dtype),
            soft,
            G,
            None,
            1,
            None,
            interpret,
            2,
            2,
            1,
            1,
            None,
        )

    def f_ref(leaf_pos, leaf_mass):
        return nf._compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
            positions,
            source_leaf_ids,
            source_valid,
            leaf_pos,
            leaf_mass,
            leaf_mask,
            leaf_particle_idx,
            G=G,
            softening_sq=soft,
            target_leaf_batch_size=2,
            target_block_tile_size=2,
            target_block_tile_scan_unroll=1,
            target_block_batch_scan_unroll=1,
            occupancy_sort=False,
            skip_empty_tiles=False,
            componentwise_pairs=False,
        )

    try:
        assert_vjp_matches(
            f_custom, f_ref, (leaf_positions, leaf_masses), rtol=1e-9, atol=1e-9
        )
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        _nf_skip_if_needed(interpret, exc)
