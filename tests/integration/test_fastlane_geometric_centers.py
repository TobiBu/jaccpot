"""Parity gates for the geometric-centre real radix fast lane (large_n_gpu).

Exercises the opt-in ``JACCPOT_LARGE_N_FASTLANE_GEOMETRIC_CENTERS`` knob, which
selects box/aabb centres for the real-basis production fast lane independently of
``grouped_interactions`` (which stays False, so the streamed near/far payload is
unchanged).

The fast lane requires a GPU (radix + solidfmm + float32) and N above the
production threshold, so the module skips unless a GPU backend is available and
the opt-in ``JACCPOT_RUN_FASTLANE_GPU_TESTS`` env flag is set (these are slow).

Assertions:
  1. the real basis with COM centres tracks the trusted complex fast lane
     (same centres, different coefficient family);
  2. geometric (aabb) centres genuinely change the field (the knob is wired) yet
     stay well-conditioned (finite).

NB: class-cached far-field *grouping* on top of geometric centres was measured NOT
to transfer to the radix binary tree (its box centres do not grid-quantise, so the
distinct-class count overflows any fixed capacity at production N). Grouping is an
octree-only optimisation and is intentionally absent from this lane.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

_RUN = str(os.environ.get("JACCPOT_RUN_FASTLANE_GPU_TESTS", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not (_RUN and jax.default_backend() == "gpu"),
    reason="fast-lane GPU parity tests: set JACCPOT_RUN_FASTLANE_GPU_TESTS=1 on a GPU",
)

_N = int(os.environ.get("JACCPOT_FASTLANE_TEST_N", "65536"))
_SOFT = 0.02
_THETA = 0.8
_LEAF = 256
_ORDER = 4


def _plummer_disk(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A clustered flattened distribution (exercises the far field)."""
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=1.0, size=n).astype(np.float32)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    z = (0.1 * rng.standard_normal(n)).astype(np.float32)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float32)
    mass = (np.abs(rng.standard_normal(n)).astype(np.float32) + 0.5) / n
    return pos, mass


def _build(basis: str):
    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis=basis,
        theta=_THETA,
        G=1.0,
        softening=_SOFT,
        working_dtype="float32",
        use_pallas=True,
    )


def _accel(
    basis: str, pos: jnp.ndarray, mass: jnp.ndarray, geometric: bool
) -> np.ndarray:
    key = "JACCPOT_LARGE_N_FASTLANE_GEOMETRIC_CENTERS"
    prev = os.environ.get(key)
    os.environ[key] = "1" if geometric else "0"
    try:
        fmm = _build(basis)
        return np.asarray(
            jax.block_until_ready(
                fmm.compute_accelerations(pos, mass, leaf_size=_LEAF, max_order=_ORDER)
            )
        )
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


@pytest.fixture(scope="module")
def _field():
    pos_np, mass_np = _plummer_disk(_N)
    return jnp.asarray(pos_np), jnp.asarray(mass_np)


def _rel_median(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    return float(np.median(num / np.maximum(den, 1e-30)))


def test_real_com_tracks_complex(_field):
    """Real basis with COM centres tracks the complex fast lane."""
    pos, mass = _field
    complex_acc = _accel("complex", pos, mass, geometric=False)
    real_acc = _accel("real", pos, mass, geometric=False)
    assert np.isfinite(complex_acc).all() and np.isfinite(real_acc).all()
    assert _rel_median(real_acc, complex_acc) < 3.0e-2


def test_geometric_centers_knob_is_live(_field):
    """aabb centres must change the field (knob wired) yet stay finite."""
    pos, mass = _field
    com = _accel("real", pos, mass, geometric=False)
    aabb = _accel("real", pos, mass, geometric=True)
    assert np.isfinite(aabb).all()
    # aabb noticeably shifts the field vs COM (the knob is live), but stays sane.
    assert _rel_median(aabb, com) > 1.0e-4
