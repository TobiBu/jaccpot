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
* the compile keys keep the layout the benches read.

CPU, the pure-JAX near field (``use_pallas=False``), the production leaf
partition (cells), with ``jax.default_backend`` reporting ``"gpu"`` as
``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _solver():
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=False,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
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
