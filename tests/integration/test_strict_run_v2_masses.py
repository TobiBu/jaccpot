"""``strict_run_v2`` honours the masses of every call.

The compiled runner closed over ``masses`` while its cache key held only their shape
and dtype, so a second call with different masses of the same shape reused the first
call's: every refresh inside the scan rebuilt the multipoles and the near field with
stale masses (the eager prepare and the initial force used the new ones, so the error
started at the first refreshed step). The masses are now a runner argument.
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
_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "1048576",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)


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


def test_a_second_call_with_new_masses_uses_them():
    _, state, masses = _systems()
    heavy = masses.at[: _N // 2].multiply(3.0)
    solver = _solver()
    first, _ = _run(solver, state, masses)
    second, _ = _run(solver, state, heavy)
    fresh, _ = _run(_solver(), state, heavy)
    fresh_again, _ = _run(_solver(), state, heavy)
    control = np.max(np.abs(fresh_again - fresh))
    # the masses matter at this resolution ...
    assert np.max(np.abs(first - fresh)) > 1e3 * max(control, 1e-9)
    # ... and the cached runner now reads the ones it is given
    assert np.max(np.abs(second - fresh)) <= max(10.0 * control, 1e-7)
