"""The fused lane's CPU route runs the lane jaccpot's CPU tests run, bit for bit.

jaccpot's own CPU tests reach the fused lane by making ``jax.default_backend``
report ``"gpu"`` for the whole process (``strict_env``), and with the
``dehnen_error`` MAC by setting the interpret-mode CSR switches in the
environment. ``RuntimePolicyConfig(cpu_fused_lane=True)`` is the supported form of
that, per engine (``jaccpot.runtime._lane_platform``). Pinned here:

* the runner on the route, with no monkeypatch and no environment, computes the
  monkeypatched run's state bit for bit -- ``dehnen_error`` with ``use_pallas=True``
  (which also needs the scoped interpret-mode CSR switches) and the geometric MAC
  with the pure-JAX near field;
* without the flag the lane refuses a CPU with an error naming the flag.

N = 256, 2 steps (CPU, ~1-2 min a case).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._lane_platform import CPU_ROUTE_PALLAS_ENV

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="the fused lane's CPU route"
)

N, STEPS, DT = 256, 2, 1e-3

_ENV_UNSET = (
    *CPU_ROUTE_PALLAS_ENV,
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)


def _settings(mac: str):
    from jaccpot.production import GalaxySettings

    return GalaxySettings(
        leaf_size=16,
        max_order=4,
        theta=0.6,
        cell_min_level=0,
        mac_type=mac,
        adaptive_eps=1e-4 if mac == "dehnen_error" else None,
    )


def _solver(mac: str, *, route: bool):
    from jaccpot import FastMultipoleMethod

    settings = _settings(mac)
    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = dataclasses.replace(
        base,
        tree=dataclasses.replace(
            base.tree, leaf_target=settings.leaf_size, cell_min_level=0
        ),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=settings.max_order, cpu_fused_lane=route
        ),
        mac_type=mac,
    )
    criterion = (
        dict(
            adaptive_eps=1e-4,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="galaxy",
        theta=settings.theta,
        G=1.0,
        softening=1e-2,
        precision="fp32",
        use_pallas=mac == "dehnen_error",
        advanced=advanced,
        **criterion,
    )


def _system():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-1.0, 1.0, (N, 3))
    vel = 0.05 * rng.normal(size=(N, 3))
    state = jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32)
    return state, jnp.full((N,), 1.0 / N, jnp.float32)


def _run(mac: str, *, route: bool) -> np.ndarray:
    from jaccpot.production import ProductionRunner

    state, masses = _system()
    runner = ProductionRunner(_solver(mac, route=route), settings=_settings(mac))
    return np.asarray(runner.run(state, masses, dt=DT, num_steps=STEPS).state)


@pytest.mark.parametrize("mac", ["dehnen_error", "dehnen"])
def test_route_matches_the_monkeypatched_lane_bitwise(mac, monkeypatch):
    assert jax.default_backend() == "cpu"
    routed = _run(mac, route=True)
    with monkeypatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        if mac == "dehnen_error":
            for key, value in CPU_ROUTE_PALLAS_ENV.items():
                patch.setenv(key, value)
        patched = _run(mac, route=False)
    assert np.isfinite(routed).all()
    np.testing.assert_array_equal(routed, patched)


def test_without_the_flag_a_cpu_is_refused_naming_it():
    from jaccpot.runtime.fused_provider import FusedForceProvider

    state, masses = _system()
    with pytest.raises(RuntimeError, match="cpu_fused_lane=True"):
        FusedForceProvider.prepare(
            _solver("dehnen", route=False),
            state[:, 0],
            masses,
            leaf_size=16,
            max_order=4,
        )
