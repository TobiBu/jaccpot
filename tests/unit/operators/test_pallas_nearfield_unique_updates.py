"""Regression checks for packed nearfield unique-update helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _load_nearfield_unique_updates_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "jaccpot"
        / "pallas"
        / "nearfield_unique_updates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "jaccpot_pallas_nearfield_unique_updates_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_nearfield_unique_updates_module()
apply_packed_particle_vector_updates_jax = (
    _MODULE.apply_packed_particle_vector_updates_jax
)
nearfield_unique_updates_backend = _MODULE.nearfield_unique_updates_backend
pack_unique_particle_vector_updates = _MODULE.pack_unique_particle_vector_updates


def test_pack_unique_particle_vector_updates_matches_scatter_add():
    base = jnp.zeros((10, 3), dtype=jnp.float32)
    indices = jnp.array(
        [
            [2, 4, 2],
            [7, 4, 0],
        ],
        dtype=jnp.int32,
    )
    values = jnp.array(
        [
            [[1.0, 0.5, -1.0], [0.5, 0.0, 0.25], [2.0, -1.0, 0.5]],
            [[3.0, 0.0, 1.0], [0.5, 1.5, -0.25], [9.0, 9.0, 9.0]],
        ],
        dtype=jnp.float32,
    )
    mask = jnp.array(
        [
            [True, True, True],
            [True, True, False],
        ],
        dtype=bool,
    )

    unique_indices, unique_values, unique_valid = pack_unique_particle_vector_updates(
        indices,
        values,
        mask,
    )
    packed = apply_packed_particle_vector_updates_jax(
        base,
        unique_indices,
        unique_values,
        unique_valid,
    )
    masked_values = jnp.where(mask[..., None], values, 0.0)
    scattered = base.at[indices.reshape(-1)].add(masked_values.reshape(-1, 3))

    assert np.allclose(
        np.asarray(packed), np.asarray(scattered), rtol=1.0e-6, atol=1.0e-6
    )


def test_nearfield_unique_updates_backend_returns_known_value():
    backend = nearfield_unique_updates_backend()
    assert backend in {"jax_set", "pallas"}
