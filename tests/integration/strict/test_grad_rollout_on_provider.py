"""``differentiable_rollout`` over ``FusedForceProvider.step`` is the particle carry.

The integrator half of S9's gate G1, independent of the differentiable step: the
rollout's forward over the provider's forward-only ``step``, jitted with the
masses an argument and the start force computed inside, against
``strict_run_v2(carry="particles")`` -- bitwise, for each checkpoint mode
(``None``, ``"step"``, ``"nested"`` in two segments of 2), on

* the pure-JAX near field, the solver's own parameters;
* the same with an external field, a traced ``dt``, traced ``G`` / ``eps2`` and
  traced field parameters (``test_traced_params.py``'s harmonic pull);
* ``mac_type='dehnen_error'`` on the CSR near field (interpret mode), the first
  step seeded with the eager prepare's f_b exactly as ``strict_run_v2`` seeds it.

A forward-only ``step`` has no reverse rule, so no gradient is taken here (that is
track A's ``differentiable_step``); the checkpoint modes change the program's
structure, not its forward, and that is what is pinned.

CPU, N = 512, the production leaf partition (cells), with ``jax.default_backend``
reporting ``"gpu"`` as ``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.fused_topology import NEEDS_FIELDS

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT, STEPS = 1e-2, 1e-3, 4
#: the external field's spring constant (a row-by-row harmonic pull)
K_EXT = 0.37
MODES = [(None, None), ("step", None), ("nested", 2)]

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
)
#: the CSR near-field lane (production's) on CPU: Pallas interpret mode
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)


def _solver(*, mac_type: str = "dehnen", use_pallas: bool = False):
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    criterion = (
        dict(
            adaptive_eps=1e-5,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac_type == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac_type,
        ),
        fixed_order=ORDER,
        **criterion,
    )


def _system():
    """Four tight blobs with velocities (``test_traced_params.py``'s draw)."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    state = jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32)
    return state, jnp.asarray(masses, jnp.float32)


def _harmonic_params(state_rows, params):
    return -params["k"] * state_rows[:, 0]


def _strict(solver, state, masses, **kw):
    out, _, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        carry="particles",
        **kw,
    )
    return np.asarray(out)


def _check_rollouts(
    prov, state, masses, want, *, dt=DT, aux=None, params=None, external_args=None
):
    """Each checkpoint mode's jitted rollout over ``prov.step`` against ``want``.

    The masses, the seed aux, the traced parameters and an array ``dt`` are
    arguments of the jitted program; a Python ``dt`` stays a literal, as in the
    runner."""
    from jaccpot.runtime.grad_rollout import check_rollout, differentiable_rollout

    field = None if external_args is None else _harmonic_params
    traced_dt = not isinstance(dt, float)
    for checkpoint, inner in MODES:

        @jax.jit
        def run(x0, v0, m, dt_in, aux0, params, external_args):
            return differentiable_rollout(
                prov.step,
                x0,
                v0,
                m,
                dt_in if traced_dt else dt,
                STEPS,
                params=params,
                aux=aux0,
                external_fn=field,
                external_args=external_args,
                checkpoint=checkpoint,
                inner_steps=inner,
            )

        out = run(
            state[:, 0],
            state[:, 1],
            masses,
            dt if traced_dt else None,
            aux,
            params,
            external_args,
        )
        check_rollout(out)
        _, bits, needs = out.aux
        assert int(bits) == 0
        assert int(np.asarray(needs)[NEEDS_FIELDS.index("tree_overflow")]) == 0
        got = np.asarray(jnp.stack([out.positions, out.velocities], axis=1))
        differ = int(np.sum(got != want))
        assert differ == 0, (
            f"checkpoint={checkpoint!r}: {differ} of {got.size} elements differ, "
            f"max {float(np.max(np.abs(got - want))):.3e}"
        )


def test_the_rollout_over_the_providers_step_is_the_particle_carry():
    from jaccpot import FusedForceProvider

    state, masses = _system()
    want = _strict(_solver(), state, masses, dt=DT)
    prov, _ = FusedForceProvider.prepare(
        _solver(), state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    _check_rollouts(prov, state, masses, want)


def test_traced_parameters_and_an_external_field_are_the_particle_carrys():
    """``dt``, ``G`` / ``eps2`` and the field's spring constant as arrays, in both:
    ``strict_run_v2``'s ``StepParams`` and the rollout's ``dt`` / ``params`` /
    ``external_args``. The field sees the same ``[N, 2, 3]`` rows."""
    from jaccpot import ForceParams, FusedForceProvider

    state, masses = _system()
    force = ForceParams(G=jnp.asarray(1.0), eps2=jnp.asarray(SOFTENING**2))
    k = {"k": jnp.asarray(K_EXT, jnp.float32)}
    want = _strict(
        _solver(),
        state,
        masses,
        dt=jnp.asarray(DT),
        add_external=True,
        external_acceleration_fn=_harmonic_params,
        external_params=k,
        force_params=force,
    )
    prov, _ = FusedForceProvider.prepare(
        _solver(), state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    _check_rollouts(
        prov,
        state,
        masses,
        want,
        dt=jnp.asarray(DT),
        params=force,
        external_args=k,
    )


def test_the_dehnen_error_rollout_is_the_particle_carry(monkeypatch):
    """The first step's f_b is the eager prepare's eq (16b) prepass in input order
    (``strict_run_v2``'s seed); the rollout threads each step's f_b to the next
    rebuild through the aux, as the runner's carry does."""
    from jaccpot import FusedForceProvider

    for key, val in _CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    state, masses = _system()
    want = _strict(
        _solver(mac_type="dehnen_error", use_pallas=True), state, masses, dt=DT
    )
    solver = _solver(mac_type="dehnen_error", use_pallas=True)
    prov, topo = FusedForceProvider.prepare(
        solver, state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    assert prov.scale_carry
    fb = jnp.asarray(solver._impl._last_force_scale_particles, state.dtype)
    perm = jnp.asarray(topo.prepared.tree.particle_indices)
    seed = jnp.zeros_like(fb).at[perm].set(fb)
    _check_rollouts(prov, state, masses, want, aux=(seed, None, None))
