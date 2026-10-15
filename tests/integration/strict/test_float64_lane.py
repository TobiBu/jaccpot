"""The large-N lane's precision modes: float32 / float64 state x float32 / float64 force.

``preset="large_n_gpu"`` keeps ``working_dtype=float32`` (the production force)
unless the caller asks for float64 -- ``precision="fp64"`` or
``working_dtype=jnp.float64`` -- which selects the float64 VALIDATION lane: the same
fused lane, its Pallas kernels in float64 where they are dtype-generic
(``tests/unit/operators/test_pallas_float64_sweep.py``), no time or memory budget.
The state's dtype is the caller's: the drift and kick run in it, and every entry
point returns the state's dtype whatever the force's.

These run on CPU through the pure-JAX fallbacks with ``jax.default_backend``
reporting ``"gpu"``, as ``tests/characterization/test_lane_goldens.py`` does, on its
clustered 512-particle draw. The scans run on cell leaves (the production
partition): on a bucket-leaf tree ``carry="particles"`` cannot run at all -- its
template rebuild reuses the template's ``node_ranges`` and children, which the
particle carry's shape template fills with zeros -- and ``strict_run_v2`` refuses
it (``fused_provider.require_cell_leaves``), in every precision mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the float64 lane needs JAX_ENABLE_X64=1"
)

_N, _LEAF = 512, 16
#: FMM vs direct sum on this draw at p = 4, theta 0.6, cells of 16, softening 1e-2:
#: 4.2e-3 (CPU, 2026-10-15), of which truncation is 1.5e-3 (unsoftened, 1.46e-3);
#: the rest is the softening the unsoftened far field leaves out for pairs a few
#: softening lengths apart, which cell leaves accept. The lane goldens' anchor.
_ANCHOR_REL_L2 = 1.0e-2
_MODES = [
    pytest.param(jnp.float32, jnp.float32, id="state32-force32"),
    pytest.param(jnp.float64, jnp.float32, id="state64-force32"),
    pytest.param(jnp.float32, jnp.float64, id="state32-force64"),
    pytest.param(jnp.float64, jnp.float64, id="state64-force64"),
]


@pytest.fixture(autouse=True)
def _large_n_lane(monkeypatch):
    """The large-N production profile on CPU, with the library's own sizing (as
    ``test_lane_goldens``'s fixture) and carry order."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
        "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
        "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
        "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        "JACCPOT_STRICT_CARRY",
        "JACCPOT_STRICT_CARRY_ORDER",
    ):
        monkeypatch.delenv(key, raising=False)


def _solver(force_dtype, *, order, theta, softening, cells=True):
    from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, TreeConfig

    partition = dict(leaf_partition="cells", leaf_capacity=128) if cells else {}
    return FastMultipoleMethod(
        preset="large_n_gpu",
        theta=theta,
        softening=softening,
        softening_kernel="plummer",
        working_dtype=force_dtype,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=_LEAF, **partition),
            mac_type="dehnen",
        ),
        fixed_order=order,
    )


def _particles(n=_N):
    """``test_fmm_golden``'s clustered draw (four tight blobs), with velocities."""
    from tests.characterization.test_fmm_golden import _make_inputs

    pos, mass = _make_inputs("clustered", n)
    vel = 0.1 * np.random.default_rng(1).normal(size=(n, 3))
    return pos, vel, mass


def _direct(pos, mass, softening):
    """Plummer-softened direct sum, float64, self excluded (``softening`` may be 0)."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=-1) + softening**2
    np.fill_diagonal(r2, 1.0)
    w = mass[None, :] / (r2 * np.sqrt(r2))
    np.fill_diagonal(w, 0.0)
    return -np.einsum("ij,ijk->ik", w, d)


def _direct_verlet(state, mass, softening, *, dt, steps):
    """The fused scan's velocity Verlet with direct-sum forces, float64."""
    pos, vel = state[:, 0].copy(), state[:, 1].copy()
    acc = _direct(pos, mass, softening)
    for _ in range(steps):
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        new = _direct(pos, mass, softening)
        vel = vel + 0.5 * (acc + new) * dt
        acc = new
    return np.stack([pos, vel], axis=1)


def _rel_l2(got, want):
    got = np.asarray(got, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def test_large_n_gpu_defaults_to_float32_and_takes_an_explicit_float64():
    from jaccpot import FastMultipoleMethod

    assert FastMultipoleMethod(preset="large_n_gpu")._impl.working_dtype == (
        jnp.float32
    )
    for kw in ({"precision": "fp64"}, {"working_dtype": jnp.float64}):
        solver = FastMultipoleMethod(preset="large_n_gpu", **kw)
        assert solver._impl.working_dtype == jnp.float64


def test_the_lane_refuses_a_working_dtype_it_has_no_kernels_for():
    from jaccpot.runtime._large_n_nearfield import resolve_large_n_execution_config

    solver = _solver(jnp.float32, order=2, theta=0.6, softening=1e-2)
    resolve_large_n_execution_config(solver._impl, num_particles=_N)
    solver._impl.working_dtype = jnp.dtype(jnp.float16)
    with pytest.raises(ValueError, match="float32 or float64"):
        resolve_large_n_execution_config(solver._impl, num_particles=_N)


def test_float64_force_reaches_below_the_float32_floor():
    """p = 12, theta = 0.4 against the exact (unsoftened) direct sum: 7.9e-10 in
    float64 (CPU, 2026-10-15), where float32 gives 9.2e-6 -- the round-off of the
    closest pairs. Unsoftened because the far field is the unsoftened expansion: at
    softening 1e-2 the softened direct sum is 2.8e-6 away from it at any order and
    precision, which would hide the lane. Bucket leaves: on cells of 16 this draw's
    truncation error is 9.9e-8 at p = 12 (more of the force is far field)."""
    softening = 0.0
    pos, _vel, mass = _particles()
    solver = _solver(jnp.float64, order=12, theta=0.4, softening=softening, cells=False)
    prepared = solver.prepare_state(
        jnp.asarray(pos), jnp.asarray(mass), leaf_size=_LEAF, max_order=12
    )
    acc = solver.evaluate_prepared_state(prepared)
    assert acc.dtype == jnp.float64
    err = _rel_l2(acc, _direct(pos, mass, softening))
    assert err < 1e-8, f"float64 force vs direct sum: rel-L2 {err:.3e}"


@pytest.mark.parametrize("state_dtype, force_dtype", _MODES)
def test_every_entry_point_returns_the_states_dtype(state_dtype, force_dtype):
    """``compute_accelerations``, ``prepare_state`` + ``evaluate_prepared_state`` and
    ``strict_run_v2`` on both carries, at p = 4 with a live far field: the state's
    dtype out, and the FMM's own error against the direct sum in every mode
    (CPU, 2026-10-15: force 4.18e-3, three-step displacement 3.75e-3 to 3.77e-3,
    the precision moving the third digit at most; gate :data:`_ANCHOR_REL_L2`)."""
    softening, dt, steps = 1e-2, 1e-3, 3
    pos, vel, mass = _particles()
    # the float32 draw, so every mode sees the same particles and one reference
    pos, vel, mass = (np.asarray(np.float32(a), np.float64) for a in (pos, vel, mass))
    p, m = jnp.asarray(pos, state_dtype), jnp.asarray(mass, state_dtype)
    ref = _direct(pos, mass, softening)
    kw = dict(order=4, theta=0.6, softening=softening)

    acc = _solver(force_dtype, **kw).compute_accelerations(
        p, m, leaf_size=_LEAF, max_order=4
    )
    assert acc.dtype == state_dtype
    assert _rel_l2(acc, ref) < _ANCHOR_REL_L2

    solver = _solver(force_dtype, **kw)
    acc = solver.evaluate_prepared_state(
        solver.prepare_state(p, m, leaf_size=_LEAF, max_order=4)
    )
    assert acc.dtype == state_dtype
    assert _rel_l2(acc, ref) < _ANCHOR_REL_L2

    state0 = np.stack([pos, vel], axis=1)
    want = _direct_verlet(state0, mass, softening, dt=dt, steps=steps) - state0
    for carry in ("state", "particles"):
        solver = _solver(force_dtype, **kw)
        out, _, _ = solver.strict_run_v2(
            state=jnp.asarray(state0, state_dtype),
            masses=m,
            dt=dt,
            num_steps=steps,
            refresh_every=1,
            leaf_size=_LEAF,
            max_order=4,
            carry=carry,
        )
        assert out.dtype == state_dtype, carry
        assert solver.get_runtime_diagnostics()["strict_fused_mode_active"] is True
        err = _rel_l2(np.asarray(out, np.float64) - state0, want)
        assert err < _ANCHOR_REL_L2, f"{carry}: displacement vs direct {err:.3e}"


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_float64_verlet_is_the_direct_verlet_to_round_off(carry):
    """Float64 state and force where the FMM is exact: theta 1e-3 accepts only pairs
    of one-particle cells (whose p = 2 expansions are exact), everything else is
    the unsoftened near field. The scan's three-step displacement is then the
    float64 direct-sum Verlet's to 1.1e-15 (CPU, 2026-10-15), so no stage of it --
    drift, refresh, force, kick -- runs in float32: the float32 force on the same
    float64 state leaves 1.4e-4 (the closest pairs' round-off), either carry."""
    softening, dt, steps = 0.0, 1e-3, 3
    pos, vel, mass = _particles(n=256)
    state0 = np.stack([pos, vel], axis=1)
    solver = _solver(jnp.float64, order=2, theta=1e-3, softening=softening)
    out, _, _ = solver.strict_run_v2(
        state=jnp.asarray(state0),
        masses=jnp.asarray(mass),
        dt=dt,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=2,
        carry=carry,
    )
    assert out.dtype == jnp.float64
    want = _direct_verlet(state0, mass, softening, dt=dt, steps=steps) - state0
    err = _rel_l2(np.asarray(out) - state0, want)
    assert err < 1e-12, f"displacement vs direct Verlet {err:.3e}"


def _offset_survives(monkeypatch, carry, order):
    """A 1e-12 shift of one particle, below float32's resolution, in a float64
    state under the float32 force: the force cannot see it, so the scan must carry
    it to the end unchanged -- unless a step rounds the state to float32."""
    if order is not None:
        monkeypatch.setenv("JACCPOT_STRICT_CARRY_ORDER", order)
    pos, vel, mass = _particles(n=256)
    pos = np.asarray(np.float32(pos), np.float64)  # the shift is below the force
    shifted = pos.copy()
    shifted[7, 0] += 1e-12
    outs = []
    for p in (pos, shifted):
        solver = _solver(jnp.float32, order=2, theta=0.6, softening=1e-2)
        out, _, _ = solver.strict_run_v2(
            state=jnp.asarray(np.stack([p, vel], axis=1)),
            masses=jnp.asarray(mass),
            dt=1e-3,
            num_steps=2,
            refresh_every=1,
            leaf_size=_LEAF,
            max_order=2,
            carry=carry,
        )
        assert out.dtype == jnp.float64
        outs.append(np.asarray(out))
    delta = outs[1] - outs[0]
    np.testing.assert_allclose(delta[7, 0, 0], 1e-12, rtol=1e-3)
    delta[7, 0, 0] = 0.0
    assert not np.any(delta)


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_a_float64_state_is_never_rounded_to_the_float32_force(monkeypatch, carry):
    _offset_survives(monkeypatch, carry, None)


@pytest.mark.xfail(
    strict=True,
    reason="finding 4 of the S2 design: the tree-order carry reuses the refresh's "
    "working-dtype positions_sorted / masses_sorted as its state rows "
    "(_advance_tree_order), rounding a float64 state to float32 every step; the fix "
    "goes in with the fmm_strict_run.py refactor",
)
def test_the_tree_order_carry_keeps_a_float64_state(monkeypatch):
    _offset_survives(monkeypatch, "particles", "tree")
