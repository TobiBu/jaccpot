"""``ProductionRunner(scheme=CompensatedKDK())`` (multi-node plan A3 prototype).

On CPU at N = 512 with ``test_production_runner.py``'s operating point:

* the compensated ``[N, 3, 3]`` state runs through the runner (segments, the
  particle carry, diagnostics) and keeps its shape; its row 0 is the float32
  rounding of ``hi + lo``;
* it follows the float64-state run (the same float32 force: the lane casts the
  positions) closer than the float32 state does;
* a restart file carries the residual row and continues bitwise.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaccpot.position_precision import (
    CompensatedKDK,
    SummedDriftKDK,
    compensated_state,
    state_float64,
)
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


def _run(state, *, scheme=None, num_steps=STEPS, segment=None, **kw):
    _, masses = _system()
    runner = ProductionRunner(
        _solver(),
        settings=SETTINGS,
        segment=segment or _static(steps=STEPS),
        scheme=scheme,
    )
    return runner.run(state, masses, dt=DT, num_steps=num_steps, **kw)


def _state64() -> np.ndarray:
    state, _ = _system()
    s64 = np.asarray(state, np.float64)
    # away from the origin, so that float32 rounds the positions coarsely
    s64[:, 0] += 3.0
    return s64


def test_compensated_follows_the_float64_state():
    s64 = _state64()
    want = np.asarray(_run(jnp.asarray(s64)).state, np.float64)
    plain = np.asarray(_run(jnp.asarray(s64, jnp.float32)).state, np.float64)
    summed = np.asarray(
        _run(jnp.asarray(s64, jnp.float32), scheme=SummedDriftKDK()).state,
        np.float64,
    )
    res = np.asarray(
        _run(jnp.asarray(compensated_state(s64)), scheme=CompensatedKDK()).state
    )
    assert res.shape == (s64.shape[0], 3, 3) and res.dtype == np.float32
    assert np.array_equal(
        res[:, 0], (res[:, 0].astype(np.float64) + res[:, 2]).astype(np.float32)
    )
    got = state_float64(res)
    err_c = np.max(np.abs(got[:, 0] - want[:, 0]))
    err_p = np.max(np.abs(plain[:, 0] - want[:, 0]))
    err_s = np.max(np.abs(summed[:, 0] - want[:, 0]))
    # float32 rounds x every step (ulp(3) / 2 = 1.2e-7); the pair keeps ~48 bits.
    # The bound is loose on purpose: a position one float32 ulp apart can move a
    # particle across a MAC or leaf decision, and the force then differs at the
    # expansion's error (here 3.2e-7 against 1.2e-6 in x after 4 steps); the
    # arithmetic itself is pinned exactly in tests/unit/test_position_precision.py
    assert err_p > 1e-8, err_p
    assert err_s > 1e-8, err_s
    assert err_c < 0.5 * err_p, (err_c, err_p)


def test_a_compensated_restart_continues_bitwise(tmp_path):
    s = jnp.asarray(compensated_state(_state64()))
    path = tmp_path / "restart.npz"
    straight = _run(s, scheme=CompensatedKDK(), segment=_static(steps=2))
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
