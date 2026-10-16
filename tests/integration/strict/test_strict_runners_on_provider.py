"""``strict_run_v2``'s runners on the ``FusedForceProvider`` (S2, C3), on CPU.

The scan's compile keys come from the provider's ``key()``, a failed segment is
re-planned through ``provider.replanned`` and the particle carry's handle carries
the provider. ``tests/integration/strict/test_strict_run_v2.py``'s
``TestSegmentRetry`` runs the retry on a card at 2e4; here it runs on CPU at
N = 512 with caps sized on a configuration that has fewer pairs (four blobs)
than the one the scan steps (a clipped Plummer sphere), so the first refresh's
walk overflows, on both carries:

* the retried segment equals a run that was prepared on the right positions,
  bitwise -- the re-planned lists are wider, the pairs and their order the same;
* the retry can be switched off, and then the call raises;
* the handle's provider is the call's force: its step at the returned positions is
  the handle's self-gravity;
* the compile keys keep the layout the benches read;
* the particle carry refuses bucket leaves, whose rebuild reads template values;
* ``mac_type='dehnen_error'``: 1 + 2 chained steps are one call of 3, bitwise, on
  both carries (the CSR near field in interpret mode: the force scale needs it);
* the TREE re-plan (S3): a cell-leaf tree that outgrows its leaf capacity, or a
  level that outgrows the level loops' batch, in the middle of a segment is
  re-planned at the segment's start, and the re-run equals a run started there
  with the grown bound, bitwise.

CPU, the pure-JAX near field (``use_pallas=False``), the production leaf
partition (cells), with ``jax.default_backend`` reporting ``"gpu"`` as
``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.fused_topology import NEEDS_FIELDS

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT, STEPS = 1e-2, 1e-3, 2

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
)


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)


def _solver(*, leaf_capacity: int = 1024, use_pallas: bool = False):
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

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
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _systems():
    """``(blobs, state on a Plummer sphere, masses)``: caps sized on the first
    overflow on the second (provider_hlo_identity.py's retry variants)."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    blobs = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 20.0)
    mu = rng.uniform(-1.0, 1.0, N)
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    state = jnp.asarray(np.stack([plummer, vel], axis=1), jnp.float32)
    return jnp.asarray(blobs, jnp.float32), state, jnp.asarray(masses, jnp.float32)


def _run(solver, state, masses, *, carry, prepared=None, steps=STEPS):
    return solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=DT,
        num_steps=steps,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        carry=carry,
    )


def _caps_from(solver, positions, masses):
    small, _ = solver.strict_fused_prepared_eval_fn(
        positions=positions,
        masses=masses,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    return small


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_an_outgrown_segment_is_replanned_once_and_matches_a_fitting_run(carry):
    blobs, state, masses = _systems()
    ref, _, _ = _run(_solver(), state, masses, carry=carry)

    solver = _solver()
    small = _caps_from(solver, blobs, masses)
    impl = solver._impl
    caps_before = dict(impl._strict_fused_validated_caps)
    before = int(solver.get_runtime_diagnostics()["strict_fused_fallback_count"] or 0)
    got, _, _ = _run(solver, state, masses, carry=carry, prepared=small)
    diag = solver.get_runtime_diagnostics()
    assert int(diag["strict_fused_fallback_count"] or 0) == before + 1
    assert diag["strict_fused_last_fallback_reason"] == "capacity_segment_retry"
    caps_after = impl._strict_fused_validated_caps
    assert (
        caps_after["compact_far_pair_capacity"]
        > caps_before["compact_far_pair_capacity"]
    )
    # the re-planned lists are wider; the pairs, their order and the force are not
    assert np.array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_the_retry_can_be_switched_off(carry, monkeypatch):
    monkeypatch.setenv("JACCPOT_STRICT_SEGMENT_RETRY", "0")
    blobs, state, masses = _systems()
    solver = _solver()
    small = _caps_from(solver, blobs, masses)
    with pytest.raises(RuntimeError, match="capacity saturated"):
        _run(solver, state, masses, carry=carry, prepared=small)


def test_the_handle_carries_the_calls_provider():
    from jaccpot import FusedForceProvider

    _, state, masses = _systems()
    solver = _solver()
    out, handle, _ = _run(solver, state, masses, carry="particles")
    prov = handle.provider
    assert isinstance(prov, FusedForceProvider)
    assert prov.engine is solver._impl and handle.template is prov.template
    # its force at the returned positions is the handle's self-gravity
    acc, (f_b, bits, _) = jax.jit(prov.step)(out[:, 0], masses)
    assert f_b is None and int(bits) == 0
    assert np.array_equal(np.asarray(acc), np.asarray(handle.self_acceleration))
    # the next call continues from the handle as before: one run of 2 + 2 steps
    end, _, _ = _run(solver, out, masses, carry="particles", prepared=handle)
    whole, _, _ = _run(_solver(), state, masses, carry="particles", steps=2 * STEPS)
    assert np.array_equal(np.asarray(end), np.asarray(whole))


def test_a_user_written_scan_over_the_provider_is_the_particle_carry():
    """``FusedForceProvider.step`` is the public surface of the fused lane's force.

    Velocity Verlet written by a caller -- a start force, then a scan of drift,
    ``prov.step`` and kick, in the scan's own expressions -- reproduces
    ``strict_run_v2(carry="particles")`` bit for bit: nothing the lane computes is
    hidden in the runner. The later sessions' integrators (Odisseo's, nornax's)
    build on this.

    The masses are an ARGUMENT of the jitted scan, as they are of the lane's runner.
    Closed over, they are a constant of the program, which XLA compiles differently:
    one float32 rounding of one force component moved at the third step (CPU,
    2026-10-16: particle 305's a_y, -4.7830286 against the lane's -4.783032, |a| =
    178; so 1 of 3072 state elements, 1.5e-8). Steps 1 and 2 were bitwise either
    way; the packed ``[N, 2, 3]`` state instead of ``x`` and ``v`` changes nothing.
    """
    from jaccpot import FusedForceProvider

    _, state, masses = _systems()
    steps = 3
    want, _, _ = _run(_solver(), state, masses, carry="particles", steps=steps)

    x0, v0 = state[:, 0], state[:, 1]
    prov, _ = FusedForceProvider.prepare(
        _solver(), x0, masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    # --- the caller's integrator --------------------------------------------
    dt = jnp.asarray(DT, state.dtype)

    def verlet(carry, m):
        x, v, a, aux = carry
        x = x + v * dt + 0.5 * a * dt**2
        a_new, aux = prov.step(x, m, aux=aux)
        return (x, v + 0.5 * (a + a_new) * dt, a_new, aux), None

    @jax.jit
    def run(carry, m):
        return jax.lax.scan(lambda c, _: verlet(c, m), carry, None, length=steps)[0]

    a0, aux0 = jax.jit(prov.step)(x0, masses)
    x, v, _, (f_b, bits, needs) = run((x0, v0, a0, aux0), masses)
    # -------------------------------------------------------------------------
    assert f_b is None and int(bits) == 0
    assert int(np.asarray(needs)[NEEDS_FIELDS.index("tree_overflow")]) == 0
    got = np.asarray(jnp.stack([x, v], axis=1))
    want = np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{differ} of {got.size} elements differ, max "
        f"{float(np.max(np.abs(got - want))):.3e}"
    )


def test_the_compile_keys_keep_their_layout():
    """What the benches and tests read off the runner cache."""
    _, state, masses = _systems()
    solver = _solver()
    _run(solver, state, masses, carry="state")
    _run(solver, state, masses, carry="particles")
    keys = list(solver._impl._strict_fused_jit_function_cache)
    by_tag = {k[0]: k for k in keys}
    assert set(by_tag) == {
        "strict_velocity_verlet",
        "strict_velocity_verlet_particles",
        "strict_particle_start_force",
    }
    # bench/fused_memory_budget.py selects the state carry's runner by key[6]
    assert by_tag["strict_velocity_verlet"][6] == STEPS
    # the tree-order flag stays last (test_strict_run_v2.py reads key[-1])
    assert by_tag["strict_velocity_verlet_particles"][-1] is False


def test_the_particle_carry_refuses_bucket_leaves():
    """Bucket leaves keep the eager tree's node ranges and children on a rebuild.

    A shape template does not hold them: the particle carry materialised zeros,
    the walk's queue overflowed on every step and the call raised a capacity error
    after its retry (a peak wavefront of 3^11 = 177,147 at N = 512). It is refused
    up front, before any prepare; the state carry, which carries the values, still
    runs.
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    def buckets():
        return FastMultipoleMethod(
            preset="large_n_gpu",
            basis="real",
            theta=THETA,
            G=1.0,
            softening=SOFTENING,
            working_dtype=jnp.float32,
            use_pallas=False,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
                nearfield=NearFieldConfig(mode="auto"),
                mac_type="dehnen",
            ),
            fixed_order=ORDER,
        )

    blobs, _, masses = _systems()
    state = jnp.stack([blobs, jnp.zeros_like(blobs)], axis=1)
    solver = buckets()
    with pytest.raises(ValueError, match="needs cell leaves"):
        _run(solver, state, masses, carry="particles")
    # refused before any prepare: no walk caps were validated
    assert getattr(solver._impl, "_strict_fused_validated_caps", None) is None
    out, _, _ = _run(buckets(), state, masses, carry="state")
    assert bool(jnp.all(jnp.isfinite(out)))


#: the CSR near-field lane on CPU (Pallas interpret mode): the eq (16b) force scale
#: needs its direct kernel
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _dehnen_error_solver():
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        adaptive_eps=1e-5,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
    )


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_chained_dehnen_error_calls_are_one_call(carry, monkeypatch):
    """1 + 2 chained steps are 3 steps in one call, bitwise, on both carries.

    Each refresh takes eq (16a)'s thresholds from the previous evaluation's f_b.
    The particle carry's handle always carried it; the state carry's next call
    used to reseed it from the engine's last eager prepass -- the FIRST call's
    start positions, in that tree's order, scattered through the returned tree's
    permutation.
    """
    for key, val in _CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    blobs, plummer_state, masses = _systems()
    state = plummer_state.at[:, 0].set(blobs)
    whole, _, _ = _run(_dehnen_error_solver(), state, masses, carry=carry, steps=3)
    solver = _dehnen_error_solver()
    mid, returned, _ = _run(solver, state, masses, carry=carry, steps=1)
    end, _, _ = _run(solver, mid, masses, carry=carry, steps=2, prepared=returned)
    got, want = np.asarray(end), np.asarray(whole)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{differ} of {got.size} elements differ, max "
        f"{float(np.max(np.abs(got - want))):.3e}"
    )


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_a_cell_leaf_overflow_inside_the_scan_is_refused(carry, monkeypatch):
    """Cell leaves sized on a uniform draw (capacity 66), the scan stepped on four
    blobs (105 live leaves): a step's rebuild overflows the leaf capacity.

    The traced refresh never lets the walk see that overflow, and on the CSR near
    field (production's) nothing else fails, so the scan ran on with the force
    55 % off (rel-L2). The verdict now folds the tree's flag; with the segment
    retry off the call raises at once, naming what the tree needed against which
    bound, and the remedy.
    """
    for key, val in _CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv("JACCPOT_STRICT_SEGMENT_RETRY", "0")
    blobs, plummer_state, masses = _systems()
    uniform = jnp.asarray(
        np.random.default_rng(2).uniform(-1.0, 1.0, (N, 3)), jnp.float32
    )
    state = plummer_state.at[:, 0].set(blobs)
    solver = _solver(leaf_capacity=66, use_pallas=True)
    small = _caps_from(solver, uniform, masses)
    before = int(solver.get_runtime_diagnostics()["strict_fused_fallback_count"] or 0)
    with pytest.raises(RuntimeError, match="cell-leaf tree overflowed") as err:
        _run(solver, state, masses, carry=carry, prepared=small)
    # 105 live leaves at the blobs; the state carry's steps drift to 119
    assert re.search(
        r"needed 1[01]\d cell leaves, more than TreeConfig\(leaf_capacity=66\)",
        str(err.value),
    )
    assert "JACCPOT_STRICT_SEGMENT_RETRY=0" in str(err.value)
    assert "tree_overflow=1" in str(err.value)
    # not re-planned: no retry ran
    diag = solver.get_runtime_diagnostics()
    assert int(diag["strict_fused_fallback_count"] or 0) == before


# --------------------------------------------------------------------------
# The TREE re-plan (S3, R6): a tree that outgrows its bounds MID-segment is
# re-planned at the segment's start and the segment re-run once.
# --------------------------------------------------------------------------


def _cusp(seed: int = 3) -> np.ndarray:
    """Radii log-uniform over e^-20..1: nested cells, a deep and narrow radix tree
    (N = 512, leaf 16, capacity 128: 107 live leaves, 58 levels, widest level 18)."""
    rng = np.random.default_rng(seed)
    radius = np.exp(-20.0 * rng.uniform(0.0, 1.0, N))
    direction = rng.normal(size=(N, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return radius[:, None] * direction


def _drifting(start: np.ndarray, end: np.ndarray):
    """``(state, masses)``: particles at ``start`` whose free drift reaches ``end``
    at the end of the call's STEPS steps (halfway after the first), with masses so
    small (1e-9 each) that gravity moves nothing measurably. The start fits the
    tree it is prepared on; the steps' rebuilds see the configurations the bounds
    were not sized for."""
    vel = (np.asarray(end) - np.asarray(start)) / (STEPS * DT)
    state = jnp.asarray(np.stack([np.asarray(start), vel], axis=1), jnp.float32)
    return state, jnp.full((N,), 1e-9, jnp.float32)


#: The walk caps of the R6 tests. ``named``: wide enough for every configuration
#: below (never widened, never overflowed), so only the tree is re-planned.
#: ``unnamed``: the library's own. The walk's needs in a segment whose tree
#: overflowed are measured on that broken tree, and these drifts change the
#: configuration completely in two steps, so the tree re-plan's re-run overflows
#: the walk caps the start sized: the segment's SECOND retry, walk-only, re-plans
#: them from the re-run's needs.
_WALK_CAPS = {
    "named": {
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "16384",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "32768",
    },
    "unnamed": {},
}
#: retries a drift runs: the tree re-plan, then (unnamed) the walk's
_RETRIES = {"named": 1, "unnamed": 2}


@pytest.fixture()
def _fresh_level_registry(monkeypatch):
    """An empty level-shape registry: it is process-global and never lowered, so a
    width another test registered for the same tree shape would hide the overflow
    (or the re-plan) this test needs."""
    from jaccpot.runtime import _level_shapes

    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})
    return _level_shapes


def _final_states_equal(got, want):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{differ} of {got.size} elements differ, max "
        f"{float(np.max(np.abs(got - want))):.3e}"
    )


def _fallbacks(solver) -> int:
    return int(solver.get_runtime_diagnostics()["strict_fused_fallback_count"] or 0)


def _reference_run(solver, leaf_capacity, state, masses, *, carry):
    """A fresh solver's run from the same start with the retried run's final walk
    caps (the floors its eager prepare sizes from): it must fit at once."""
    from jaccpot.runtime.capacity_plan import install_walk_caps

    reference = _solver(leaf_capacity=leaf_capacity, use_pallas=True)
    install_walk_caps(reference, solver._impl._strict_fused_validated_caps)
    want, _, _ = _run(reference, state, masses, carry=carry)
    assert _fallbacks(reference) == 0
    return want


@pytest.mark.parametrize("walk_caps", ["named", "unnamed"])
@pytest.mark.parametrize("carry", ["state", "particles"])
def test_a_tree_outgrowing_its_leaf_capacity_mid_segment_is_replanned(
    carry, walk_caps, monkeypatch, _fresh_level_registry
):
    """Prepared on a uniform draw at capacity 66 (64 live leaves, the start fits),
    drifting into four blobs: the first step's rebuild needs 109 leaves, the second
    105. The verdict fails; the segment is re-planned at its start -- capacity
    ``auto_leaf_capacity(109) = 1024``, the depth bound to the deepest tree + 8 --
    and run again (twice with the library's walk caps, the second time for the
    walk); the result equals a run started there at capacity 1024 with the final
    walk caps."""
    for key, val in (_CSR_ON_CPU | _WALK_CAPS[walk_caps]).items():
        monkeypatch.setenv(key, val)
    blobs, _, _ = _systems()
    uniform = np.random.default_rng(2).uniform(-1.0, 1.0, (N, 3))
    state, masses = _drifting(uniform, np.asarray(blobs))

    solver = _solver(leaf_capacity=66, use_pallas=True)
    impl = solver._impl
    before = _fallbacks(solver)
    got, _, _ = _run(solver, state, masses, carry=carry)
    diag = solver.get_runtime_diagnostics()
    assert _fallbacks(solver) == before + _RETRIES[walk_caps]
    assert diag["strict_fused_last_fallback_reason"] == "capacity_segment_retry"
    assert impl._tree_leaf_capacity == 1024  # sticky: later prepares use it

    want = _reference_run(solver, 1024, state, masses, carry=carry)
    _final_states_equal(got, want)


@pytest.mark.parametrize("walk_caps", ["named", "unnamed"])
@pytest.mark.parametrize("carry", ["state", "particles"])
def test_a_level_wider_than_the_level_loops_batch_mid_segment_is_replanned(
    carry, walk_caps, monkeypatch, _fresh_level_registry
):
    """The bound a long galaxy run reaches first (the 25M quarter-orbit probe: the
    widest radix level +10.9 % in 489 steps, against 1.25x the eager widest): the
    leaves fit, a level does not.

    Prepared on a cusp at capacity 128 (107 live leaves, widest level 18: the level
    loops' batch is ceil(1.25 x 18) = 23), drifting into four blobs (113 and 105
    live leaves, widest levels 44 and 46). The re-prepare at the segment's start
    sees the cusp again and would register 23 once more: the batch must come from
    what the segment's trees needed, ceil(1.25 x 46) = 58, or the re-run overflows
    the same way. The leaf capacity stays 128; the result equals a run started
    there with the widened batch and the final walk caps.
    """
    for key, val in (_CSR_ON_CPU | _WALK_CAPS[walk_caps]).items():
        monkeypatch.setenv(key, val)
    blobs, _, _ = _systems()
    state, masses = _drifting(_cusp(), np.asarray(blobs))

    solver = _solver(leaf_capacity=128, use_pallas=True)
    impl = solver._impl
    shape = (2 * 128 - 1, 128 - 1)
    before = _fallbacks(solver)
    got, _, _ = _run(solver, state, masses, carry=carry)
    assert _fallbacks(solver) == before + _RETRIES[walk_caps]
    assert impl._tree_leaf_capacity == 128  # the leaves never overflowed
    assert _fresh_level_registry._WIDTHS[shape] == 58

    # the registry is process-global: a fresh solver started at the same state
    # compiles with the widened batch
    want = _reference_run(solver, 128, state, masses, carry=carry)
    _final_states_equal(got, want)


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_a_walk_retry_after_a_tree_replan_is_the_last(carry, monkeypatch):
    """At most two retries per segment, the second only for the walk: with the
    tree re-planned and then the walk's caps overflowing again (the second re-run
    measures nothing new here -- the walk's re-plan is switched off), the call
    raises and says two retries ran."""
    for key, val in _CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    blobs, _, _ = _systems()
    uniform = np.random.default_rng(2).uniform(-1.0, 1.0, (N, 3))
    state, masses = _drifting(uniform, np.asarray(blobs))
    solver = _solver(leaf_capacity=66, use_pallas=True)
    impl = solver._impl
    replan_walk = impl._replan_walk_caps_from_needs
    calls = []

    def walk_replan_without_growth(needs):
        # the first call (the tree re-plan's) re-plans as usual; later ones claim
        # to re-plan but grow nothing, so the re-run overflows the walk again
        calls.append(np.asarray(needs))
        return replan_walk(needs) if len(calls) == 1 else True

    monkeypatch.setattr(
        impl, "_replan_walk_caps_from_needs", walk_replan_without_growth
    )
    with pytest.raises(RuntimeError, match="capacity saturated") as err:
        _run(solver, state, masses, carry=carry)
    assert "Segment retries run: 2" in str(err.value)
    assert len(calls) == 2  # no third re-plan
    assert impl._tree_leaf_capacity == 1024
