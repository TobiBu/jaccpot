"""Legacy kwargs transition coverage for jaccpot solver."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(17)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def test_accepts_legacy_expanse_kwargs_with_deprecation_warning():
    positions, masses = _sample_problem(n=32)
    with pytest.warns(DeprecationWarning):
        fmm = FastMultipoleMethod(
            preset="fast",
            expansion_basis="solidfmm",
            complex_rotation="solidfmm",
            mac_type="dehnen",
            fixed_order=4,
            fixed_max_leaf_size=16,
            grouped_interactions=True,
            farfield_mode="pair_grouped",
            nearfield_mode="bucketed",
            nearfield_edge_chunk_size=256,
            theta=0.6,
            softening=1e-3,
            working_dtype=jnp.float32,
        )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
    )
    assert acc.shape == positions.shape


def test_runtime_config_spelling_of_fixed_order_and_leaf_cap_is_silent():
    """The modern spelling of both knobs must not warn.

    ``fixed_order`` and ``fixed_max_leaf_size`` used to be kwarg-only, so warning
    on them pointed callers at a config object that had no field for either. They
    now live on ``RuntimePolicyConfig``; this is the spelling to migrate to.
    """
    positions, masses = _sample_problem(n=32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fmm = FastMultipoleMethod(
            preset="fast",
            basis="solidfmm",
            theta=0.6,
            softening=1e-3,
            working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                runtime=RuntimePolicyConfig(
                    fixed_order=4,
                    fixed_max_leaf_size=16,
                ),
            ),
        )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
    )
    assert acc.shape == positions.shape


def test_the_config_spelling_reaches_the_engine():
    """Not just accepted -- actually threaded through to the evaluate stage.

    ``runtime/fmm_evaluate.py`` reads both off the engine: ``fixed_order`` pins the
    order instead of inferring it from the coefficient count, and
    ``fixed_max_leaf_size`` is the bound behind "too small for prepared tree".
    """
    fmm = FastMultipoleMethod(
        preset="fast",
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(fixed_order=5, fixed_max_leaf_size=32),
        ),
    )
    engine = getattr(fmm, "_impl", fmm)
    assert engine.fixed_order == 5
    assert engine.fixed_max_leaf_size == 32


def test_the_kwarg_spelling_still_warns_but_wins():
    """The legacy kwargs remain accepted, now warn, and override the config.

    They kept working throughout so downstream callers (odisseo passes
    ``fixed_max_leaf_size`` unconditionally) are not broken by the migration. The
    warning is honest now: there is somewhere for it to point.
    """
    with pytest.warns(DeprecationWarning):
        fmm = FastMultipoleMethod(
            preset="fast",
            basis="solidfmm",
            fixed_order=4,
            fixed_max_leaf_size=16,
            advanced=FMMAdvancedConfig(
                runtime=RuntimePolicyConfig(fixed_order=7, fixed_max_leaf_size=64),
            ),
        )
    engine = getattr(fmm, "_impl", fmm)
    assert engine.fixed_order == 4
    assert engine.fixed_max_leaf_size == 16
