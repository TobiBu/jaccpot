"""``strict_run_v2(carry="particles")``: the fused scan carries the particles only.

Pinned here (GPU, the fused strict lane):

* the premise: with the default fresh far-pair rebuild a scan step READS no
  leaf of the carried prepared state -- jax's dead-code elimination of the step,
  with only the particle outputs marked used, keeps none of its inputs;
* the particle carry follows the state carry's trajectory (A-vs-A controlled);
* chaining through the returned handle is one long run, and the handle's
  self-gravity is the force at the returned positions;
* the call leaves less resident (no prepared state, no far list kept);
* the handle is refused where it does not belong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.test_strict_run_v2_donation import (
    _FUSED_ENV,
    _LEAF,
    _ORDER,
    _THETA,
    _solver,
    _state,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    ),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
        "JACCPOT_STRICT_CARRY",
    ):
        monkeypatch.delenv(key, raising=False)


def _run(solver, state, masses, *, steps, carry, prepared=None):
    return solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        carry=carry,
    )


def _maxdiff(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def test_a_step_reads_no_carried_state_leaf(monkeypatch):
    """The premise of the particle carry, measured on the state carry's scan body."""
    from dataclasses import replace

    from jax._src.interpreters import partial_eval as pe

    from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

    captured = {}
    orig = jax.lax.scan

    def spy(f, init, xs=None, length=None, **kw):
        captured["f"] = f
        return orig(f, init, xs=xs, length=length, **kw)

    monkeypatch.setattr(jax.lax, "scan", spy)
    state, masses = _state()
    solver = _solver()
    s1, p1, _ = _run(solver, state, masses, steps=1, carry="state")
    monkeypatch.setattr(jax.lax, "scan", orig)
    impl = solver._impl
    assert impl._strict_far_pairs_ride_outside_the_scan(p1)
    p_in = replace(p1, compact_far_pairs=None)
    carry = (
        p_in,
        s1,
        jnp.zeros_like(s1[:, 0]),
        jnp.asarray(True),
        jnp.zeros((len(WALK_NEEDS_FIELDS),), jnp.int32),
    )
    f = captured["f"]
    closed = jax.make_jaxpr(lambda c: f(c, None)[0])(carry)
    n_prep = len(jax.tree_util.tree_leaves(p_in))
    n_prep_out = len(
        jax.tree_util.tree_leaves(jax.eval_shape(lambda c: f(c, None)[0], carry)[0])
    )
    n_out = len(closed.jaxpr.outvars)
    _, used_in = pe.dce_jaxpr(
        closed.jaxpr, [False] * n_prep_out + [True] * (n_out - n_prep_out)
    )
    assert n_prep > 20  # non-vacuous: the state really is a big pytree
    assert not any(used_in[:n_prep])
    assert all(used_in[n_prep:])  # the particles are read


def test_particle_carry_follows_the_state_carry():
    state, masses = _state()
    a, _, _ = _run(_solver(), state, masses, steps=3, carry="state")
    a2, _, _ = _run(_solver(), state, masses, steps=3, carry="state")
    b, handle, _ = _run(_solver(), state, masses, steps=3, carry="particles")
    from jaccpot.runtime.strict_carry import StrictParticleCarry

    assert isinstance(handle, StrictParticleCarry)
    control = _maxdiff(a2, a)
    assert _maxdiff(b, a) <= max(10.0 * control, 1e-6)
    assert bool(jnp.all(jnp.isfinite(b)))


def test_chaining_through_the_handle_is_one_long_run():
    state, masses = _state()
    solver = _solver()
    one, _, _ = _run(solver, state, masses, steps=4, carry="particles")
    one_again, _, _ = _run(_solver(), state, masses, steps=4, carry="particles")
    mid, handle, _ = _run(solver, state, masses, steps=2, carry="particles")
    end, handle2, _ = _run(
        solver, mid, masses, steps=2, carry="particles", prepared=handle
    )
    control = _maxdiff(one_again, one)
    assert _maxdiff(end, one) <= max(10.0 * control, 1e-6)
    # the handle's force is the self-gravity at the returned positions
    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=end[:, 0],
        masses=masses,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
    )
    eager = np.asarray(eval_fn(prepared))
    scale = float(np.max(np.abs(eager)))
    assert _maxdiff(handle2.self_acceleration, eager) <= 1e-4 * scale


def test_the_particle_carry_leaves_less_resident():
    state, masses = _state()

    def resident_after(carry):
        solver = _solver()
        out = _run(solver, state, masses, steps=2, carry=carry)
        jax.block_until_ready(out[0])
        live = sum(x.nbytes for x in jax.live_arrays() if not x.is_deleted())
        del out, solver
        return live

    by_state = resident_after("state")
    by_particles = resident_after("particles")
    # the state carry keeps the prepared state + far list (several MB at 2e4);
    # the handle holds one [N, 3] force
    assert by_particles < by_state - 1_000_000


def test_the_handle_is_refused_where_it_does_not_belong():
    state, masses = _state()
    solver = _solver()
    out, handle, _ = _run(solver, state, masses, steps=1, carry="particles")
    with pytest.raises(ValueError):
        _run(solver, out, masses, steps=1, carry="state", prepared=handle)
    with pytest.raises(ValueError):
        _run(solver, out[:-1], masses[:-1], steps=1, carry="particles", prepared=handle)
