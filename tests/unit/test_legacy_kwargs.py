"""Legacy kwargs transition coverage for jaccpot solver."""

import jax
import jax.numpy as jnp
import pytest

from jaccpot import FastMultipoleMethod


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
    positions, masses = _sample_problem(n=64)
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
