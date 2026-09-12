"""The gradient path must not touch the Pallas cascades (plan sub-10ms, Phase 6 fallout).

``pallas_call`` carries no autodiff rule, so JAX falls back to its generic JVP
rule, tries to differentiate the kernel body, and dies on ``program_id`` (no grid
context under that trace). Phase 6 made the per-level Pallas M2M/L2L cascades and
the leaf P2M default-ON, which broke ``jax.grad`` everywhere Pallas lowers.

It broke INVISIBLY: Pallas does not lower on CPU, so the gate resolves False there
and every existing gradient test silently exercised the loop cascades.
``JACCPOT_CASCADE_PALLAS_INTERPRET=1`` is the lever that makes the Pallas path
reachable on CPU, which is what lets this test run in ordinary CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.autodiff import direct_sum_gravitational_acceleration
from jaccpot.runtime._level_shapes import pallas_cascades_enabled
from jaccpot.runtime.grad_options import grad_option_overrides, resolve_grad_options

_SOFT = 1e-2


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.fixture
def pallas_cascades_on(monkeypatch):
    """Force the Pallas cascades reachable on CPU (interpret mode)."""
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS", "1")
    monkeypatch.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
    assert pallas_cascades_enabled(), "fixture is vacuous: the Pallas path is not reachable"


def test_the_override_turns_the_gate_off_and_restores_it(pallas_cascades_on):
    """Non-vacuous both ways: on outside the block, off inside, on again after."""
    options = resolve_grad_options(None, num_particles=256, supports_fast_lane=True)
    assert options.cascade_pallas is False
    assert pallas_cascades_enabled()
    with grad_option_overrides(options):
        assert not pallas_cascades_enabled()
    assert pallas_cascades_enabled()


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_grad_runs_and_matches_the_direct_sum_with_the_cascades_default_on(pallas_cascades_on):
    """``jax.grad`` through the FMM, with the Pallas cascades enabled globally.

    Before the override this raised ``AssertionError`` from
    ``jax/_src/pallas/core.py::axis_frame``.
    """
    n = 512
    pos = jnp.asarray(_plummer(n, 3), jnp.float64)
    mass = jnp.asarray(np.full(n, 1.0 / n), jnp.float64)
    fmm = FastMultipoleMethod(theta=0.5, softening=_SOFT, G=1.0)
    state = fmm.prepare_state(pos, mass, max_order=4, leaf_size=32)

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, mass) ** 2)

    got = np.asarray(jax.grad(loss)(pos))
    ref = np.asarray(
        jax.grad(
            lambda p: jnp.sum(
                direct_sum_gravitational_acceleration(p, mass, G=1.0, softening=_SOFT) ** 2
            )
        )(pos)
    )
    assert np.all(np.isfinite(got))
    assert np.linalg.norm(got) > 0  # non-vacuous
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-5, f"gradient disagrees with the direct-sum oracle: rel-L2 {rel:.3e}"
