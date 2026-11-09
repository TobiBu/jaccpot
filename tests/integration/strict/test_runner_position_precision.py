"""The drift order and the compensated state in ``ProductionRunner`` (multi-node plan A3).

On CPU at N = 512 with ``test_production_runner.py``'s operating point:

* the built-in drift is ``x + dt (v + 0.5 dt a)`` (the default since A3) and
  ``GalaxySettings(drift_order="legacy")`` is ``x + v dt + 0.5 a dt**2`` left to
  right (every run before A3), each bit for bit a scheme that writes it out;
* a restart records the order: a legacy run's settings record has no
  ``drift_order`` (a file written before A3 compares equal to it), a file
  continued under the other order warns and names it, under its own order it
  continues bitwise;
* ``CompensatedKDK`` runs its ``[N, 3, 3]`` state through the runner, follows the
  float64-state run closer than the float32 legacy state does, and continues a
  restart bitwise.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.position_precision import CompensatedKDK, compensated_state, state_float64
from jaccpot.production import ProductionRunner, load_restart
from tests.integration.strict.test_production_runner import (  # noqa: F401
    DT,
    SETTINGS,
    _lane,
    _solver,
    _static,
    _system,
)

STEPS = 4
LEGACY = dataclasses.replace(SETTINGS, drift_order="legacy")


@dataclasses.dataclass(frozen=True)
class _KDK:
    """Velocity Verlet with the drift written out in either order."""

    order: str
    rowwise: bool = True

    def key(self):
        return ("test.A3KDK", self.order)

    def open_step(self, state, acceleration, dt):
        x, v = state[:, 0], state[:, 1]
        if self.order == "legacy":
            return state.at[:, 0].set(x + v * dt + 0.5 * acceleration * dt**2)
        return state.at[:, 0].set(x + dt * (v + (0.5 * dt) * acceleration))

    def close_step(self, state_open, acceleration, acceleration_new, dt):
        dt = jnp.asarray(dt, dtype=state_open.dtype)
        v = state_open[:, 1]
        return state_open.at[:, 1].set(v + 0.5 * (acceleration + acceleration_new) * dt)


def _run(state, *, settings=SETTINGS, scheme=None, num_steps=STEPS, segment=None):
    _, masses = _system()
    runner = ProductionRunner(
        _solver(),
        settings=settings,
        segment=segment or _static(steps=STEPS),
        scheme=scheme,
    )
    return runner.run(state, masses, dt=DT, num_steps=num_steps)


def _state64() -> np.ndarray:
    state, _ = _system()
    s64 = np.asarray(state, np.float64)
    # away from the origin, so that float32 rounds the positions coarsely
    s64[:, 0] += 3.0
    return s64


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_the_drift_order_is_what_steps():
    """The default runner is the summed expression and the legacy setting the
    left-to-right one, each bitwise a scheme that writes it; they differ."""
    s32 = jnp.asarray(_state64(), jnp.float32)
    summed = _run(s32)
    legacy = _run(s32, settings=LEGACY)
    assert _same(summed.state, _run(s32, scheme=_KDK("summed")).state)
    assert _same(legacy.state, _run(s32, scheme=_KDK("legacy")).state)
    assert not _same(summed.state, legacy.state)
    # an unknown order: ValueError from the runner, or (JACCPOT_RUNTIME_TYPECHECK=1)
    # the settings' Literal annotation's TypeCheckError first
    with pytest.raises((ValueError, TypeError), match="drift_order"):
        ProductionRunner(
            _solver(), settings=dataclasses.replace(SETTINGS, drift_order="kahan")
        )


def test_a_restart_records_the_drift_order(tmp_path):
    """A legacy run records no ``drift_order`` (a file from before A3 matches it);
    a file continued under the other order warns with the drift-order hint; under
    its own order it continues bitwise."""
    s32 = jnp.asarray(_state64(), jnp.float32)
    path = tmp_path / "restart.npz"
    straight = _run(s32, settings=LEGACY, segment=_static(steps=2))
    _run(
        s32,
        settings=LEGACY,
        num_steps=2,
        segment=_static(steps=2, restart_every=1, restart_path=str(path)),
    )
    _, meta = load_restart(path)
    assert "drift_order" not in meta["fingerprint"]["settings"]
    resumed = ProductionRunner(
        _solver(), settings=LEGACY, segment=_static(steps=2)
    ).run(dt=DT, num_steps=STEPS, restart_from=path)
    assert resumed.restored["bitwise"], resumed.restored
    assert _same(resumed.state, straight.state)
    with pytest.warns(UserWarning, match="drift order"):
        other = ProductionRunner(
            _solver(), settings=SETTINGS, segment=_static(steps=2)
        ).run(dt=DT, num_steps=STEPS, restart_from=path)
    assert other.restored["mismatches"] == ["settings"]


def test_compensated_follows_the_float64_state():
    s64 = _state64()
    want = np.asarray(_run(jnp.asarray(s64)).state, np.float64)
    legacy = np.asarray(
        _run(jnp.asarray(s64, jnp.float32), settings=LEGACY).state, np.float64
    )
    res = np.asarray(
        _run(jnp.asarray(compensated_state(s64)), scheme=CompensatedKDK()).state
    )
    assert res.shape == (s64.shape[0], 3, 3) and res.dtype == np.float32
    assert np.array_equal(
        res[:, 0], (res[:, 0].astype(np.float64) + res[:, 2]).astype(np.float32)
    )
    err_c = np.max(np.abs(state_float64(res)[:, 0] - want[:, 0]))
    err_l = np.max(np.abs(legacy[:, 0] - want[:, 0]))
    # float32 rounds x every step (ulp(3) / 2 = 1.2e-7); the pair keeps ~48 bits.
    # The bound is loose on purpose: a position one float32 ulp apart can move a
    # particle across a MAC or leaf decision, and the force then differs at the
    # expansion's error (3.2e-7 against 1.2e-6 in x after 4 steps); the arithmetic
    # itself is pinned exactly in tests/unit/test_position_precision.py
    assert err_l > 1e-8, err_l
    assert err_c < 0.5 * err_l, (err_c, err_l)


def test_a_compensated_restart_continues_bitwise(tmp_path):
    path = tmp_path / "restart.npz"
    straight = _run(
        jnp.asarray(compensated_state(_state64())),
        scheme=CompensatedKDK(),
        segment=_static(steps=2),
    )
    _run(
        jnp.asarray(compensated_state(_state64())),
        scheme=CompensatedKDK(),
        num_steps=2,
        segment=_static(steps=2, restart_every=1, restart_path=str(path)),
    )
    arrays, _ = load_restart(path)
    assert np.asarray(arrays["state"]).shape[1] == 3
    resumed = ProductionRunner(
        _solver(), settings=SETTINGS, segment=_static(steps=2), scheme=CompensatedKDK()
    ).run(dt=DT, num_steps=STEPS, restart_from=path)
    assert resumed.restored["bitwise"], resumed.restored
    assert np.array_equal(np.asarray(resumed.state), np.asarray(straight.state))
