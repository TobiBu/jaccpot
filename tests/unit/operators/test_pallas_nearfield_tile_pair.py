"""Regression checks for fused nearfield tile-pair helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _load_nearfield_tile_pair_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "jaccpot"
        / "pallas"
        / "nearfield_tile_pair.py"
    )
    spec = importlib.util.spec_from_file_location(
        "jaccpot_pallas_nearfield_tile_pair_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_nearfield_tile_pair_module()
nearfield_tile_pair_accel_jax = _MODULE.nearfield_tile_pair_accel_jax
nearfield_tile_pair_backend = _MODULE.nearfield_tile_pair_backend


def test_nearfield_tile_pair_accel_jax_matches_manual_reference():
    target_positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    target_mask = jnp.array([True, True, False, True], dtype=bool)
    source_positions = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    source_masses = jnp.array([2.0, 1.0, 3.0, 5.0], dtype=jnp.float32)
    source_mask = jnp.array([True, True, True, False], dtype=bool)
    softening_sq = jnp.asarray(1.0e-4, dtype=jnp.float32)
    G = jnp.asarray(1.25, dtype=jnp.float32)

    actual = nearfield_tile_pair_accel_jax(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        softening_sq=softening_sq,
        G=G,
    )

    expected = []
    for i in range(target_positions.shape[0]):
        if not bool(target_mask[i]):
            expected.append([0.0, 0.0, 0.0])
            continue
        acc = np.zeros((3,), dtype=np.float32)
        tgt = np.asarray(target_positions[i])
        for j in range(source_positions.shape[0]):
            if not bool(source_mask[j]):
                continue
            src = np.asarray(source_positions[j])
            diff = tgt - src
            dist_sq = float(np.sum(diff * diff) + float(softening_sq))
            inv_r = 1.0 / np.sqrt(dist_sq)
            inv_dist3 = inv_r * inv_r * inv_r
            acc += (-float(G) * inv_dist3 * float(source_masses[j])) * diff
        expected.append(acc)
    expected = np.asarray(expected, dtype=np.float32)

    assert np.allclose(np.asarray(actual), expected, rtol=1.0e-5, atol=1.0e-5)


def test_nearfield_tile_pair_backend_returns_known_value():
    backend = nearfield_tile_pair_backend()
    assert backend in {"jax", "pallas"}
