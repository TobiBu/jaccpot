"""``strict_run_v2``: the segment retry for outgrown walk caps.

Unnamed flat-walk list caps are sized from the counts the eager prepare measured
(``JACCPOT_FLAT_WALK_CAP_HEADROOM`` x count), so a rollout whose counts grow past
the headroom trips the scan's capacity flag. The scan carries the running maximum
of what its walks needed out of the trace; on a failed segment ``strict_run_v2``
re-plans the caps from it, re-prepares from the segment's start and runs it once
more. The scenario here forces that: the state is prepared on a uniform cube
(21k far / 11k near pairs at N = 2e4) and stepped on a Plummer sphere (84k / 55k),
so all three walk capacities overflow on the first refresh. The retried segment
must equal a run that was prepared on the right positions; a named cap, and the
retry switched off, must still raise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    ),
]

_N = 20_000
_LEAF, _ORDER, _THETA = 64, 4, 0.8
_FAR_CAP = "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP"
_NEAR_CAP = "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"
_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    for key in (
        _FAR_CAP,
        _NEAR_CAP,
        "JACCPOT_FLAT_WALK_CAP_HEADROOM",
        "JACCPOT_STRICT_SEGMENT_RETRY",
    ):
        monkeypatch.delenv(key, raising=False)


def _solver():
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=1e-4,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity=2048,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=_ORDER,
    )


def _systems():
    rng = np.random.default_rng(0)
    cube = rng.uniform(-1.0, 1.0, (_N, 3)).astype(np.float32)
    x = rng.uniform(0.0, 1.0, _N)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, _N)
    phi = rng.uniform(0.0, 2.0 * np.pi, _N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (_N, 3))
    masses = jnp.full((_N,), 1.0 / _N, jnp.float32)
    state = jnp.stack(
        [jnp.asarray(plummer, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    return jnp.asarray(cube), state, masses


def _run(solver, state, masses, prepared=None, steps=2):
    out, prepared_out, _ = solver.strict_run_v2(
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
    )
    return np.asarray(jax.block_until_ready(out), np.float64), prepared_out


def _diag(solver):
    return dict(solver.get_runtime_diagnostics() or {})


def test_an_outgrown_segment_is_replanned_once_and_matches_a_fitting_run():
    cube, state, masses = _systems()
    ref_solver = _solver()
    ref, _ = _run(ref_solver, state, masses)
    ref_again, _ = _run(_solver(), state, masses)
    # A-vs-A control: two fitting runs agree to this level
    control = np.max(np.abs(ref_again - ref))

    solver = _solver()
    small, _ = solver.strict_fused_prepared_eval_fn(
        positions=cube, masses=masses, leaf_size=_LEAF, max_order=_ORDER, theta=_THETA
    )
    caps_before = dict(solver._impl._strict_fused_validated_caps)
    before = int(_diag(solver).get("strict_fused_fallback_count") or 0)
    got, _ = _run(solver, state, masses, prepared=small)
    diag = _diag(solver)
    assert int(diag.get("strict_fused_fallback_count") or 0) == before + 1
    assert diag.get("strict_fused_last_fallback_reason") == "capacity_segment_retry"
    caps_after = solver._impl._strict_fused_validated_caps
    assert (
        caps_after["compact_far_pair_capacity"]
        > caps_before["compact_far_pair_capacity"]
    )
    assert caps_after["near_edge_capacity"] > caps_before["near_edge_capacity"]
    assert np.max(np.abs(got - ref)) <= max(10.0 * control, 1e-6)


def test_the_retry_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("JACCPOT_STRICT_SEGMENT_RETRY", "0")
    cube, state, masses = _systems()
    solver = _solver()
    small, _ = solver.strict_fused_prepared_eval_fn(
        positions=cube, masses=masses, leaf_size=_LEAF, max_order=_ORDER, theta=_THETA
    )
    with pytest.raises(RuntimeError) as excinfo:
        _run(solver, state, masses, prepared=small)
    assert "saturated" in str(excinfo.value.__cause__ or excinfo.value)


def test_a_named_cap_is_never_widened(monkeypatch):
    cube, state, masses = _systems()
    probe = _solver()
    probe.strict_fused_prepared_eval_fn(
        positions=cube, masses=masses, leaf_size=_LEAF, max_order=_ORDER, theta=_THETA
    )
    caps = probe._impl._strict_fused_validated_caps
    # caps that fit the cube exactly as named
    monkeypatch.setenv(_FAR_CAP, str(caps["compact_far_pair_capacity"]))
    monkeypatch.setenv(_NEAR_CAP, str(caps["near_edge_capacity"]))
    solver = _solver()
    small, _ = solver.strict_fused_prepared_eval_fn(
        positions=cube, masses=masses, leaf_size=_LEAF, max_order=_ORDER, theta=_THETA
    )
    with pytest.raises(RuntimeError) as excinfo:
        _run(solver, state, masses, prepared=small)
    assert "saturated" in str(excinfo.value.__cause__ or excinfo.value)
    assert int(_diag(solver).get("strict_fused_fallback_count") or 0) == 0
