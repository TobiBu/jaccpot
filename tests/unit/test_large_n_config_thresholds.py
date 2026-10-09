"""Regressions for the radix differentiable-FMM path at large-N config thresholds.

**This module does not exercise ``LargeNPreparedState``.** It was called
``test_large_n_grad_path.py``, which read as though it did. It does not: the large-N
prepare path requires a GPU backend (``can_use_large_n_prepare_path`` returns False when
``jax.default_backend() != "gpu"``), so on CPU every test here runs the ordinary *radix*
grad path at the *configuration thresholds* large N would cross. Measured:
``runtime/_large_n_grad.py`` sits at 0% coverage with these 20 tests passing and none
skipping. The genuinely uncovered large-N reverse path is tracked as F27 in
``docs/refactor_audit_2026-08.md`` and needs a GPU CI leg, not a rename.

Three distinct failures were reachable on the DEFAULT configuration once
``n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD`` (65536) auto-enabled the
grouped/class-major M2L. Two of them -- ``farfield_mode`` left unresolved at
``"auto"`` by the static-fixed-sizing branch of the override resolver, and the
grouped M2L building its classes on the host, which broke every reverse pass --
went with the grouped far field in the 2026-10 cleanup (X3); the resolver checks
below still pin that the far field resolves to the flat pair list about COM
centres at every N. The third is fp32 far-field gradients: see
``test_fp32_farfield_gradient_is_finite``.

The resolver assertions are pure host-side policy checks (no tree build), so they
are instant; the gradient tests use a deliberately deep tree at small N to get a
non-empty M2L list cheaply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.runtime.fmm_constants import _GPU_LARGE_PARTICLE_THRESHOLD


def _solver(**kwargs):
    return FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.6, G=1.0, softening=1e-2, **kwargs
    )


def _clustered(n, dtype=jnp.float64, seed=3):
    """Clustered, flattened (galaxy-like) distribution."""
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = 0.1 * rng.standard_normal(n)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    mass = (np.abs(rng.standard_normal(n)) + 0.5) / n
    return jnp.asarray(pos, dtype), jnp.asarray(mass, dtype)


def _num_far_pairs(state):
    inter = state.interactions
    return int(jnp.sum(inter.counts)) if inter is not None else 0


# --------------------------------------------------------------------------
# 1. Override-resolver policy (host-only, no tree build)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n", [_GPU_LARGE_PARTICLE_THRESHOLD, 4 * _GPU_LARGE_PARTICLE_THRESHOLD]
)
@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_farfield_mode_never_resolves_to_auto_at_large_n(n, backend):
    """``farfield_mode`` must reach the kernels as a concrete mode at every N.

    Regression: the static-fixed-sizing branch (the production default) returned
    ``"auto"`` once the grouped M2L auto-enabled at ``n >= 65536`` on GPU. The
    grouped far field is gone (X3), so the only value left is the flat pair list.
    """
    fmm = _solver()
    overrides = fmm._impl._resolve_runtime_execution_overrides(
        num_particles=n, backend=backend
    )
    assert (
        overrides.farfield_mode == "pair_grouped"
    ), f"farfield_mode={overrides.farfield_mode!r} at n={n} backend={backend}"


def test_static_sizing_does_not_inherit_the_adaptive_grouped_rewrite():
    """Static fixed sizing skips adaptive rewrites -- including auto-grouping."""
    fmm = _solver()
    fmm._impl._static_runtime_fixed_sizing = True
    overrides = fmm._impl._resolve_runtime_execution_overrides(
        num_particles=4 * _GPU_LARGE_PARTICLE_THRESHOLD, backend="gpu"
    )
    assert overrides.farfield_mode == "pair_grouped"
    assert overrides.center_mode == "com"


# --------------------------------------------------------------------------
# 2. fp32 gradients (the precision the large-N production path runs in)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize(
    "delta,label",
    [
        ([0.0, 0.0, 0.0], "exact-zero displacement (single-child COM L2L)"),
        ([0.0, 0.0, 0.7], "z-axis aligned (rho == 0)"),
        ([0.3, -0.2, 0.7], "generic control"),
    ],
)
def test_translation_reverse_is_finite_at_degenerate_displacements(dtype, delta, label):
    """The squared-radius floor must survive the working dtype.

    ``jnp.maximum(r2, 1e-60)`` is what makes ``d sqrt/dr2 = 1/(2r)`` finite at the
    degenerate displacements a fixed-topology FMM genuinely hits -- when the floor
    wins, ``jnp.maximum`` routes a zero cotangent. But ``1e-60`` underflows to
    exactly ``0.0`` in float32, which silently disabled the guard: this exact probe
    returned ``[nan, nan, nan]`` in float32 (float64 stayed finite), which is why
    fp32 far-field gradients were all-NaN.
    """
    from jaccpot.operators.complex_ops import l2l_complex

    order = 4
    ncoeff = (order + 1) ** 2
    cdtype = jnp.complex128 if dtype is jnp.float64 else jnp.complex64
    local = jnp.asarray(np.arange(1, ncoeff + 1) * 0.1, dtype=cdtype)
    d = jnp.asarray(delta, dtype)

    grad = jax.grad(
        lambda dd: jnp.sum(jnp.abs(l2l_complex(local, dd, order=order)) ** 2)
    )(d)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite L2L reverse at {label}"


def test_squared_radius_floor_preserves_float64_and_is_normal_in_float32():
    """float64 keeps the historical constant (bit-identical); float32 gets a normal."""
    from jaccpot.operators.dtypes import squared_radius_floor

    assert squared_radius_floor(jnp.float64) == 1e-60
    # The bug in one line: the legacy constant is not representable in float32.
    assert np.float32(1e-60) == 0.0
    f32_floor = squared_radius_floor(jnp.float32)
    assert f32_floor >= float(np.finfo(np.float32).tiny)
    assert np.float32(f32_floor) > 0.0


def test_fp32_farfield_gradient_is_finite():
    """fp32 gradients must be finite when the far field is active.

    Regression: with a non-empty M2L list the fp32 reverse pass returned all-NaN
    (float64 was fine, and fp32 with an empty M2L list was fine), so gradients
    were unusable at the precision the large-N production path uses.
    """
    positions, masses = _clustered(512, dtype=jnp.float32)
    fmm = FastMultipoleMethod(
        basis="complex",
        use_pallas=False,
        theta=0.6,
        G=1.0,
        softening=1e-2,
        working_dtype=jnp.float32,
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=4)
    assert _num_far_pairs(state) > 0, "config must exercise the far field"

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "fp32 far-field gradient is non-finite"
    assert float(jnp.linalg.norm(grad)) > 0.0
