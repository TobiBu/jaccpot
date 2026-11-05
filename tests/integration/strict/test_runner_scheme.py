"""``ProductionRunner(scheme=...)``: the integration-scheme seam (multi-node plan I1).

nornax owns the schemes (``nornax.schemes.LeapfrogKDK``); jaccpot takes one
through the structural protocol :class:`jaccpot.scheme.SplitStepScheme` without
importing nornax. These gates use a test-local copy of nornax's KDK (the same
expressions; nornax's own suite runs the real one against this runner,
``tests/integration/test_jaccpot_provider_rollout.py``), on CPU at N = 512 with
``test_production_runner.py``'s operating point, each bitwise:

* the KDK scheme IS the built-in runner: literal and traced parameters, an
  external field with traced parameters, the tree-order carry; through its own
  scan program (the scheme's key is in the compile key);
* a scheme is really what steps: one that halves ``dt`` in both halves is the
  built-in runner at ``dt / 2``;
* a restart records the scheme, continues bitwise under the same one, and says
  so under another;
* a scheme is refused where the runner has no seam for it (the multi-GPU mode,
  the gradient mode, the state carry) and for an object that is not one; a scheme
  that is not rowwise keeps the input order.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.production import ProductionRunner, SegmentConfig, load_restart
from jaccpot.scheme import SplitStepScheme, check_scheme, scheme_record
from tests.integration.strict.test_production_runner import (  # noqa: F401
    DT,
    HALO_M,
    HALO_RS,
    LEAF,
    ORDER,
    SETTINGS,
    _lane,
    _particle_runners,
    _solver,
    _static,
    _system,
    nfw_halo_params,
)


@dataclasses.dataclass(frozen=True)
class _KDK:
    """nornax's ``LeapfrogKDK``, written out: the built-in's expressions."""

    rowwise: bool = True
    dt_factor: float = 1.0

    def key(self):
        return ("test.KDK", self.dt_factor, self.rowwise)

    def _dt(self, dt):
        return dt if self.dt_factor == 1.0 else dt * self.dt_factor

    def open_step(self, state, acceleration, dt):
        dt = self._dt(dt)
        x, v = state[:, 0], state[:, 1]
        return state.at[:, 0].set(x + v * dt + 0.5 * acceleration * dt**2)

    def close_step(self, state_open, acceleration, acceleration_new, dt):
        dt = jnp.asarray(self._dt(dt), dtype=state_open.dtype)
        v = state_open[:, 1]
        return state_open.at[:, 1].set(v + 0.5 * (acceleration + acceleration_new) * dt)


#: one segment of all the steps: one scan program per runner (the segment edges are
#: the runner's, not the scheme's, and test_production_runner.py pins them; the
#: restart test below crosses one)
STEPS = 4


def _run(segment, *, scheme=None, dt=DT, num_steps=STEPS, **kw):
    state, masses = _system()
    params = kw.pop("external_params", None)
    runner = ProductionRunner(
        _solver(), settings=SETTINGS, segment=segment, scheme=scheme, **kw
    )
    res = runner.run(state, masses, dt=dt, num_steps=num_steps, external_params=params)
    return runner, res


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_the_protocol_is_structural():
    """The test scheme (and so nornax's) satisfies the protocol without importing
    anything of jaccpot; the record names it."""
    assert isinstance(_KDK(), SplitStepScheme)
    assert check_scheme(None) is None
    assert scheme_record(None) is None
    record = scheme_record(_KDK())
    assert record["qualname"].endswith("_KDK")
    assert record["key"] == repr(("test.KDK", 1.0, True))


@pytest.mark.parametrize("form", ["static", "traced", "halo", "tree_order"])
def test_the_kdk_scheme_is_the_builtin_runner(form, monkeypatch):
    """The scheme's runner computes what the built-in's does, bit for bit, in a
    scan program of its own (the scheme's key is part of the compile key)."""
    kw: dict = {}
    if form == "static":
        segment = _static(steps=STEPS)
    else:
        segment = SegmentConfig(steps=STEPS, diagnostics_every=0)
    if form == "halo":
        kw = dict(
            external_acceleration_fn=nfw_halo_params,
            external_params={"m": HALO_M, "rs": HALO_RS},
        )
    if form == "tree_order":
        monkeypatch.setenv("JACCPOT_STRICT_CARRY_ORDER", "tree")
    builtin, want = _run(segment, **dict(kw))
    with_scheme, got = _run(segment, scheme=_KDK(), **dict(kw))
    assert _same(got.state, want.state)
    assert _same(got.handle.self_acceleration, want.handle.self_acceleration)
    if form == "halo":
        assert _same(got.handle.acceleration, want.handle.acceleration)
    keys = _particle_runners(with_scheme)
    assert keys and all(("scheme", _KDK().key()) in k for k in keys)
    assert all(k[-1] == (form == "tree_order") for k in keys)
    assert not any("scheme" in k for k in _particle_runners(builtin))


def test_a_scheme_is_what_steps():
    """A scheme that halves dt in both of its halves is the built-in at dt / 2
    (halving a binary float is exact, and so is every product after it): a scheme
    the runner ignored would give the dt run's state instead."""
    _, want = _run(_static(steps=STEPS), dt=DT / 2)
    _, got = _run(_static(steps=STEPS), scheme=_KDK(dt_factor=0.5))
    assert _same(got.state, want.state)


def test_a_scheme_that_is_not_rowwise_keeps_the_input_order(monkeypatch):
    """The tree-order carry hands a scheme permuted rows: a scheme that reads
    other rows keeps the input order, and its run is the input-order run."""
    _, want = _run(_static(steps=STEPS))
    monkeypatch.setenv("JACCPOT_STRICT_CARRY_ORDER", "tree")
    runner, got = _run(_static(steps=STEPS), scheme=_KDK(rowwise=False))
    assert all(k[-1] is False for k in _particle_runners(runner))
    assert _same(got.state, want.state)


def test_a_restart_records_the_scheme(tmp_path):
    """The file says which scheme wrote it; the same scheme continues bitwise, the
    built-in continues with a warning that names it."""
    path = tmp_path / "restart.npz"
    seg = _static(steps=2, restart_every=1, restart_path=str(path))
    _, straight = _run(_static(steps=2), scheme=_KDK(), num_steps=4)
    _, first = _run(seg, scheme=_KDK(), num_steps=2)
    _, meta = load_restart(path)
    assert meta["scheme"] == scheme_record(_KDK())
    resumed = ProductionRunner(
        _solver(), settings=SETTINGS, segment=_static(steps=2), scheme=_KDK()
    ).run(dt=DT, num_steps=4, restart_from=path)
    assert resumed.restored["bitwise"], resumed.restored
    assert _same(resumed.state, straight.state)
    with pytest.warns(UserWarning, match="not bitwise"):
        other = ProductionRunner(
            _solver(), settings=SETTINGS, segment=_static(steps=2)
        ).run(dt=DT, num_steps=2, restart_from=path)
    assert other.restored["mismatches"] == ["scheme"]
    # and the built-in's own files carry no scheme entry
    plain = tmp_path / "plain.npz"
    _run(_static(steps=2, restart_every=1, restart_path=str(plain)), num_steps=2)
    assert "scheme" not in load_restart(plain)[1]


def test_a_scheme_is_refused_where_there_is_no_seam():
    """The multi-GPU mode (I2), the gradient mode, the state carry, a non-scheme."""
    state, masses = _system()
    with pytest.raises(TypeError, match="SplitStepScheme"):
        ProductionRunner(_solver(), settings=SETTINGS, scheme=object())

    @dataclasses.dataclass(frozen=True)
    class _Unhashable(_KDK):
        def key(self):
            return ["a", "list"]

    with pytest.raises(TypeError, match="hashable"):
        ProductionRunner(_solver(), settings=SETTINGS, scheme=_Unhashable())
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("gpus",))
    with pytest.raises(ValueError, match="I2"):
        ProductionRunner(_solver(), settings=SETTINGS, mesh=mesh, scheme=_KDK())
    runner = ProductionRunner(
        _solver(), settings=SETTINGS, segment=_static(steps=2), scheme=_KDK()
    )
    with pytest.raises(ValueError, match="built-in KDK"):
        runner.gradient_plan(state, masses, dt=DT, num_steps=2)
    with pytest.raises(ValueError, match="carry='particles'"):
        _solver().strict_run_v2(
            state=state,
            masses=masses,
            dt=DT,
            num_steps=1,
            refresh_every=1,
            leaf_size=LEAF,
            max_order=ORDER,
            carry="state",
            scheme=_KDK(),
        )
