"""``jaccpot.position_precision`` (multi-node plan A3): the schemes' arithmetic.

No force here (``tests/integration/strict/test_runner_position_precision.py`` runs
them in the production runner): the compensated drift keeps ~48 bits of position
under jit, the summed drift is the built-in's terms summed first, the kicks are
the built-in's bit for bit, and the packing round-trips.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.position_precision import (
    CompensatedKDK,
    SummedDriftKDK,
    compensated_state,
    state_float64,
)
from jaccpot.runtime.fmm_state import _velocity_verlet_kick_drifted
from jaccpot.scheme import SplitStepScheme, check_mesh_scheme, check_scheme

N = 64
DT = 5e-4
STEPS = 400


def _state64(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8.0, 8.0, (N, 3)) + 5.0
    v = rng.normal(size=(N, 3))
    return np.stack([x, v], axis=1)


def _acc(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N, 3)) * 30.0


def test_the_schemes_are_rowwise_split_step_schemes():
    for scheme in (SummedDriftKDK(), CompensatedKDK(), CompensatedKDK(True)):
        assert isinstance(scheme, SplitStepScheme)
        assert check_scheme(scheme) is scheme
        assert check_mesh_scheme(scheme) is scheme
        assert scheme.rowwise
    keys = {s.key() for s in (SummedDriftKDK(), CompensatedKDK(), CompensatedKDK(True))}
    assert len(keys) == 3


@pytest.mark.parametrize("velocities", [False, True])
def test_packing_round_trips(velocities):
    s64 = _state64()
    packed = compensated_state(s64, velocities=velocities)
    assert packed.dtype == np.float32
    assert packed.shape == (N, 4 if velocities else 3, 3)
    # rows 0 and 1 are the float32 state every reader reads
    assert np.array_equal(packed[:, 0], s64[:, 0].astype(np.float32))
    assert np.array_equal(packed[:, 1], s64[:, 1].astype(np.float32))
    back = state_float64(packed)
    assert np.max(np.abs(back[:, 0] - s64[:, 0])) <= 2.0**-46 * np.max(
        np.abs(s64[:, 0])
    )
    if velocities:
        assert np.max(np.abs(back[:, 1] - s64[:, 1])) <= 2.0**-46 * np.max(
            np.abs(s64[:, 1])
        )
    # a float32 state packs with zero residuals; a plain state unpacks as itself
    assert not np.any(compensated_state(s64.astype(np.float32))[:, 2])
    assert np.array_equal(state_float64(s64.astype(np.float32)), s64.astype(np.float32))
    with pytest.raises(ValueError):
        compensated_state(packed)


def _drift_many(scheme, state, acc, dt=DT):
    """``STEPS`` drifts (no kicks) under one jit: the residual must survive XLA."""

    @jax.jit
    def run(s):
        def body(c, _):
            return scheme.open_step(c, acc, jnp.float32(dt)), None

        return jax.lax.scan(body, s, None, STEPS)[0]

    return np.asarray(run(state))


def test_the_compensated_drift_keeps_the_bits_float32_drops():
    """Dyadic velocities, accelerations and dt: every increment ``v dt + a dt**2 / 2``
    is exact in float32 however XLA contracts it, so the exact position after
    ``STEPS`` drifts is known; the compensated pair holds it exactly, the float32
    state (summed drift) loses the increment's low bits at every step."""
    rng = np.random.default_rng(3)
    dt = 2.0**-11
    x0 = (rng.uniform(-8.0, 8.0, (N, 3)) + 5.0).astype(np.float32).astype(np.float64)
    v = rng.integers(-255, 256, (N, 3)) * 2.0**-7
    a = rng.integers(-255, 256, (N, 3)) * 2.0**-3
    inc = v * dt + 0.5 * a * dt**2  # exact: 8 + 8 significant bits apart by 2^-12
    want = x0 + STEPS * inc  # exact in float64
    state = np.stack([x0, v], axis=1)
    acc = jnp.asarray(a, jnp.float32)
    packed = _drift_many(
        CompensatedKDK(), jnp.asarray(compensated_state(state)), acc, dt
    )
    got_c = state_float64(packed)[:, 0]
    got_s = _drift_many(SummedDriftKDK(), jnp.asarray(state, jnp.float32), acc, dt)
    err_s = np.max(np.abs(got_s[:, 0].astype(np.float64) - want))
    assert np.array_equal(got_c, want)
    assert err_s > 1e-7, err_s
    # hi is renormalised: the float32 rounding of hi + lo, |lo| <= ulp(hi) / 2
    hi, lo = packed[:, 0].astype(np.float64), packed[:, 2].astype(np.float64)
    assert np.array_equal(packed[:, 0], (hi + lo).astype(np.float32))
    assert np.all(
        np.abs(lo) <= 0.5 * np.spacing(np.abs(packed[:, 0])).astype(np.float64)
    )


def test_the_summed_drift_is_the_builtin_terms_summed_first():
    st = jnp.asarray(_state64(), jnp.float32)
    acc = jnp.asarray(_acc(), jnp.float32)
    dt = jnp.float32(DT)
    got = np.asarray(SummedDriftKDK().open_step(st, acc, dt))
    x, v = st[:, 0], st[:, 1]
    assert np.array_equal(got[:, 0], np.asarray(x + (v * dt + 0.5 * acc * dt**2)))
    assert np.array_equal(got[:, 1], np.asarray(v))


@pytest.mark.parametrize(
    "scheme", [SummedDriftKDK(), CompensatedKDK()], ids=["summed", "compensated_x"]
)
def test_the_kick_is_the_builtin_kick(scheme):
    s64 = _state64()
    st = (
        jnp.asarray(compensated_state(s64))
        if isinstance(scheme, CompensatedKDK)
        else jnp.asarray(s64, jnp.float32)
    )
    a0 = jnp.asarray(_acc(1), jnp.float32)
    a1 = jnp.asarray(_acc(2), jnp.float32)
    dt = jnp.float32(DT)
    got = np.asarray(scheme.close_step(st, a0, a1, dt))
    want = np.asarray(_velocity_verlet_kick_drifted(st[:, :2], a0, a1, dt))
    assert np.array_equal(got[:, :2], want)
    assert np.array_equal(got[:, 2:], np.asarray(st[:, 2:]))


def test_compensated_velocities_accumulate_the_kicks():
    s64 = _state64()
    st = jnp.asarray(compensated_state(s64, velocities=True))
    a0 = jnp.asarray(_acc(1), jnp.float32)
    scheme = CompensatedKDK(velocities=True)

    @jax.jit
    def run(s):
        def body(c, _):
            return scheme.close_step(c, a0, a0, jnp.float32(DT)), None

        return jax.lax.scan(body, s, None, STEPS)[0]

    got = state_float64(np.asarray(run(st)))[:, 1]
    inc = np.asarray(0.5 * (a0 + a0) * jnp.float32(DT), np.float64)
    want = state_float64(np.asarray(st))[:, 1] + STEPS * inc
    assert np.max(np.abs(got - want)) < 1e-11


@pytest.mark.parametrize("rows", [2, 4])
def test_a_wrong_state_shape_is_refused(rows):
    st = jnp.zeros((N, rows, 3), jnp.float32)
    acc = jnp.zeros((N, 3), jnp.float32)
    with pytest.raises(ValueError, match="CompensatedKDK"):
        CompensatedKDK().open_step(st, acc, jnp.float32(DT))
