"""``strict_fused_prepared_eval_fn`` -- audit item **F33**.

Lines 1123-1171 of ``fmm_strict_run.py``, unexercised on either backend. The
method exists for benchmarking: it builds a fused-lane prepared state eagerly and
returns a jitted *eval-only* closure, so the evaluate cost of the strict fused
static-radix lane can be compared like-for-like against functional FMM eval APIs.

A benchmarking entry point is exactly the kind of code that rots unnoticed --
nothing in the suite calls it, and a benchmark that fails to build simply looks
like a slow day. Its two guards are tested because they are the difference
between "this configuration cannot be measured" and a misleading number. (The
multi-GPU fused lane builds its shard plans through it too, via
``capacity_plan.measure_shard_plan``.)

Was ``tests/integration/test_strict_fused_eval_fn.py``; since the 2026-10 cleanup
(Z-a, rule R5) it also holds the fused scan's API, moved-endpoint and
compact-far-pair-cap tests from ``tests/integration/test_fmm.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import NearFieldConfig, TreeConfig
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime._large_n_types import LargeNPreparedState

N_PARTICLES = 512
LEAF_SIZE = 64
MAX_ORDER = 2


def _engine():
    return FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=MAX_ORDER,
    )


def _particles(seed: int = 9):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    return positions, masses


def test_requires_the_large_n_production_profile():
    """The first guard, and the only one a default engine reaches."""
    fmm = FMMEngine(theta=0.6, working_dtype=jnp.float32, fixed_order=MAX_ORDER)
    positions, masses = _particles()
    with pytest.raises(RuntimeError, match="large_n_gpu production profile"):
        fmm.strict_fused_prepared_eval_fn(
            positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )


@pytest.mark.slow
def test_requires_the_particle_count_in_the_profile_set(strict_env, monkeypatch):
    """The second guard: right profile, a particle count the profile set excludes.

    It names the variable and both remedies in its message rather than failing
    obscurely later, which is the behaviour worth pinning -- a benchmark that
    cannot run should say which setting would let it. (The fused-mode switch this
    guard also named went in the 2026-10 cleanup, X6.)
    """
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", "999999")
    fmm = _engine()
    positions, masses = _particles()
    with pytest.raises(RuntimeError) as excinfo:
        fmm.strict_fused_prepared_eval_fn(
            positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )
    message = str(excinfo.value)
    assert "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET" in message
    assert f"N={N_PARTICLES}" in message
    assert "leave it empty to allow all N" in message
    assert not fmm._strict_fused_mode_active


@pytest.mark.slow
def test_returns_a_prepared_state_and_an_eval_closure(strict_env, monkeypatch):
    """The happy path: the closure must evaluate the state it was built for.

    The returned callable is the thing being benchmarked, so it is not enough
    that it exists -- it is checked against ``evaluate_prepared_state`` on the
    same state, which is what a benchmark comparing it to another library would
    implicitly be claiming.
    """
    fmm = _engine()
    positions, masses = _particles()

    prepared, eval_fn = fmm.strict_fused_prepared_eval_fn(
        positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    assert isinstance(prepared, LargeNPreparedState)
    assert callable(eval_fn)
    assert fmm._strict_fused_mode_active, "the method must leave fused mode active"

    from_closure = np.asarray(eval_fn(prepared))
    from_method = np.asarray(fmm.evaluate_prepared_state(prepared))
    assert from_closure.shape == (N_PARTICLES, 3)
    relative = np.linalg.norm(from_closure - from_method) / np.linalg.norm(from_method)
    assert (
        relative <= 1e-6
    ), f"the eval closure disagrees with evaluate_prepared_state by {relative:.3e}"


# ---------------------------------------------------------------------------
# From tests/integration/test_fmm.py (2026-10 cleanup, Z-a, rule R5): the fused scan
# ---------------------------------------------------------------------------


def test_strict_run_v2_api(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    # Pin the fused static-sizing caps to fixed adequate values (mirroring the
    # sibling strict-fused tests below). Without this the fused lane sizes the
    # static neighbour-edge cap from the *first* build's active-edge count, which
    # is sensitive to leaked process-global tree/neighbour construction state
    # from earlier tests on the same xdist worker; an undersized first-build cap
    # then overflows on a later scan step ("neighbor-edge cap exceeded"). Fixed
    # caps make the run deterministic and order-independent.
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")

    key = jax.random.PRNGKey(20260513)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack(
        [
            positions,
            jnp.zeros_like(positions),
        ],
        axis=1,
    )

    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
        softening_kernel="plummer",
    )

    state_out, prepared_out, history = fmm.strict_run_v2(
        state=state0,
        masses=masses,
        dt=1e-3,
        num_steps=3,
        refresh_every=1,
        leaf_size=128,
        max_order=2,
        theta=0.6,
        return_history=False,
    )
    assert prepared_out.tree.build_mode == "static_radix"
    assert history is None
    assert np.asarray(state_out).shape == np.asarray(state0).shape

    diagnostics = fmm.get_runtime_diagnostics()
    assert diagnostics["strict_runner_execute_count"] >= 2


def test_strict_fused_moved_endpoint_matches_fresh_prepare(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "32768")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")

    key = jax.random.PRNGKey(20260620)
    key_pos, key_vel, key_mass = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_pos,
        (512, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    velocities = 0.01 * jax.random.normal(key_vel, (512, 3), dtype=jnp.float32)
    masses = jax.random.uniform(
        key_mass,
        (512,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack([positions, velocities], axis=1)

    kwargs = dict(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
    )
    fmm = FMMEngine(**kwargs, softening_kernel="plummer")
    state_out, prepared_out, history = fmm.strict_run_v2(
        state=state0,
        masses=masses,
        dt=2.0e-4,
        num_steps=1,
        refresh_every=1,
        leaf_size=128,
        max_order=2,
        theta=0.6,
        return_history=False,
    )
    assert history is None
    diagnostics = fmm.get_runtime_diagnostics()
    assert diagnostics["strict_fused_mode_active"] is True
    assert diagnostics["strict_fused_fallback_count"] == 0
    assert diagnostics["strict_self_force_endpoint_evaluations"] == 1

    fresh_fmm = FMMEngine(**kwargs, softening_kernel="plummer")
    fresh = fresh_fmm.prepare_state(
        state_out[:, 0, :], masses, leaf_size=128, max_order=2
    )
    endpoint_acc = np.asarray(fmm.evaluate_prepared_state(prepared_out))
    fresh_acc = np.asarray(fresh_fmm.evaluate_prepared_state(fresh))
    diff = endpoint_acc - fresh_acc
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(fresh_acc), 1.0e-12)
    assert rel_l2 <= 1.0e-4
    assert np.max(np.abs(diff)) <= 5.0e-3


def test_strict_fused_compact_far_pair_cap_fails(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")

    key = jax.random.PRNGKey(20260621)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (2048, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (2048,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )
    state0 = jnp.stack([positions, jnp.zeros_like(positions)], axis=1)

    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
        softening_kernel="plummer",
    )
    with pytest.raises(
        Exception,
        match="JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP|compact far-pair cap exceeded",
    ):
        fmm.strict_run_v2(
            state=state0,
            masses=masses,
            dt=1.0e-4,
            num_steps=1,
            refresh_every=1,
            leaf_size=64,
            max_order=2,
            theta=0.6,
            return_history=False,
        )


# ---------------------------------------------------------------------------
# The walk's MAC geometry is the TREE's, not the engine's call history (S1, J5)
# ---------------------------------------------------------------------------
#
# Until 2026-10-15 the dual walk took its MAC geometry default from
# `_strict_fused_mode_active`, a flag every strict call set and nothing cleared:
# a fresh engine's `compute_accelerations` / `prepare_state` walked the box
# geometry, the same engine after one strict call the COM geometry of the fused
# lane. On an A100 (the S4b pin's configuration, N=2e4) that moved the force's
# rel-L2 against the direct sum from 1.16e-3 to 6.73e-4 -- and the differentiable
# lane, built on a fresh engine, was the less accurate one. The geometry now
# follows the tree (`walk_geometry_default`: COM on a static-radix tree), so
# every entry point walks the same MAC, whatever ran before.
#
# N=2048 at leaf 16 is the size the large-N reverse tests use for the same
# reason: at leaf 64 this set has no far pair and the geometry decides nothing.

_GEOMETRY_N = 2048
_GEOMETRY_LEAF = 16
_GEOMETRY_ORDER = 4


@pytest.fixture
def geometry_env(strict_env, monkeypatch):
    """The large-N lane on CPU, with the payload the differentiable lane accepts."""
    # the prepacked near-field layout production runs at 2e5 (see
    # tests/unit/runtime/test_large_n_grad_reverse_path.py): the gradient plan
    # refuses the per-particle pairs payload small N would otherwise select
    monkeypatch.setenv("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", "0")
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", raising=False)
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", raising=False)


def _geometry_solver():
    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(
        preset="large_n_gpu", G=1.0, softening=1e-2, retain_far_pairs_for_grad=True
    )


def _geometry_particles():
    rng = np.random.default_rng(11)
    positions = jnp.asarray(rng.normal(size=(_GEOMETRY_N, 3)), dtype=jnp.float32)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(_GEOMETRY_N,)), jnp.float32)
    return positions, masses


def _production_force(solver, positions, masses):
    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=positions,
        masses=masses,
        leaf_size=_GEOMETRY_LEAF,
        max_order=_GEOMETRY_ORDER,
    )
    return np.asarray(eval_fn(prepared))


@pytest.mark.slow
def test_a_strict_call_does_not_change_what_the_engine_returns_next(geometry_env):
    """``compute_accelerations`` on one engine, before and after a strict call: the
    same bits. On main the second call walked the COM geometry the strict call
    left switched on, the first the box geometry."""
    solver = _geometry_solver()
    positions, masses = _geometry_particles()
    kwargs = dict(leaf_size=_GEOMETRY_LEAF, max_order=_GEOMETRY_ORDER)
    before = np.asarray(solver.compute_accelerations(positions, masses, **kwargs))
    far = solver.prepare_state(positions, masses, **kwargs).compact_far_pairs
    assert int(np.asarray(far.far_pair_count)) > 0, "vacuous: no far pair"

    _production_force(solver, positions, masses)
    assert solver._impl._strict_fused_mode_active  # still set, as a diagnostic

    after = np.asarray(solver.compute_accelerations(positions, masses, **kwargs))
    np.testing.assert_array_equal(after, before)


@pytest.mark.slow
def test_every_entry_point_walks_the_com_mac_on_a_static_radix_tree(
    geometry_env, monkeypatch
):
    """The strict lane keeps its COM MAC, and a fresh engine's non-strict calls
    walk it too; the environment still overrides both."""
    from jaccpot.runtime import fmm_prepare

    asked: list[str] = []
    resolve = fmm_prepare.resolve_walk_geometry

    def _recording(*args, **kwargs):
        asked.append(fmm_prepare.mac_geometry_mode(kwargs["default_mode"]))
        return resolve(*args, **kwargs)

    monkeypatch.setattr(fmm_prepare, "resolve_walk_geometry", _recording)
    positions, masses = _geometry_particles()
    kwargs = dict(leaf_size=_GEOMETRY_LEAF, max_order=_GEOMETRY_ORDER)

    _geometry_solver().compute_accelerations(positions, masses, **kwargs)
    assert asked and set(asked) == {"com"}, asked
    asked.clear()
    _production_force(_geometry_solver(), positions, masses)
    assert asked and set(asked) == {"com"}, asked

    asked.clear()
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "aabb")
    _production_force(_geometry_solver(), positions, masses)
    assert asked and set(asked) == {"aabb"}, asked


@pytest.mark.slow
def test_the_differentiable_forward_is_the_production_forward(geometry_env):
    """The gradient lane differentiates the force production computes.

    ``differentiable_accelerations`` freezes the lists ``prepare_state`` built on
    a FRESH engine; the production eval is the strict fused lane's. On main the
    first walked the box geometry and the second the COM one, so the gradient was
    taken of a different (there: less accurate) force. They now walk the same
    MAC; what is left is the two evaluation paths' float32 rounding.
    """
    positions, masses = _geometry_particles()
    production = _production_force(_geometry_solver(), positions, masses)

    solver = _geometry_solver()
    state = solver.prepare_state(
        positions, masses, leaf_size=_GEOMETRY_LEAF, max_order=_GEOMETRY_ORDER
    )
    assert isinstance(state, LargeNPreparedState)
    forward = np.asarray(solver.differentiable_accelerations(state, positions, masses))

    scale = float(np.max(np.linalg.norm(production, axis=1)))
    assert np.max(np.abs(forward - production)) <= 1e-5 * scale


@pytest.mark.slow
def test_a_public_refresh_never_takes_the_strict_lanes_traced_path(geometry_env):
    """``refresh_prepared_state(fused_device_mode=True)`` is the same call before and
    after a strict call. On main the strict call's flag also switched it onto the
    fused scan's traced hot path (streamed fast lane or raise, host effects off, the
    input's far list handed back as the carry placeholder); only the strict lane's
    own refreshes take that path now, and they say so (``strict_fused_lane``)."""
    solver = _geometry_solver()
    engine = solver._impl
    positions, masses = _geometry_particles()
    kwargs = dict(leaf_size=_GEOMETRY_LEAF, max_order=_GEOMETRY_ORDER)
    state = solver.prepare_state(positions, masses, fused_device_mode=True, **kwargs)
    moved = positions * jnp.float32(1.01)

    def _refresh():
        attempts = int(engine._strict_fused_fastlane_attempts)
        out = solver.refresh_prepared_state(
            state, moved, masses, fused_device_mode=True, **kwargs
        )
        assert int(engine._strict_fused_fastlane_attempts) == attempts, "hot path"
        return out

    before = _refresh()
    _production_force(solver, positions, masses)
    after = _refresh()

    far_before, far_after = before.compact_far_pairs, after.compact_far_pairs
    count = int(np.asarray(far_before.far_pair_count))
    assert count > 0 and int(np.asarray(far_after.far_pair_count)) == count
    for name in ("sources", "targets"):
        np.testing.assert_array_equal(
            np.asarray(getattr(far_after, name))[:count],
            np.asarray(getattr(far_before, name))[:count],
        )
    np.testing.assert_array_equal(
        np.asarray(solver.evaluate_prepared_state(after)),
        np.asarray(solver.evaluate_prepared_state(before)),
    )
