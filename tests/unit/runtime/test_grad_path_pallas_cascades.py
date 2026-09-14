"""The gradient path runs the Pallas cascades and reaches their reverse kernels (plan fast-gradients).

History. Phase 6 of the sub-10 ms plan made the per-level Pallas M2M/L2L cascades
and the leaf P2M default-ON while they had no autodiff rule; ``pallas_call``'s
generic JVP rule dies on ``program_id`` (no grid context), so ``jax.grad`` broke
everywhere Pallas lowers -- invisibly, because Pallas does not lower on CPU and the
gradient suite runs there. The first fix (d06dee0) forced the gradient path onto
the pure-JAX loops. The kernels now carry ``custom_vjp`` rules whose reverse is a
Pallas kernel of its own, so the gate stays ON under ``jax.grad`` and
``GradConfig(cascade_pallas=False)`` is the A/B switch back to the loops.

``JACCPOT_CASCADE_PALLAS_INTERPRET=1`` is what makes the Pallas path reachable on
CPU, so this runs in ordinary CI. The reverse kernels are COUNTED, not assumed:
a test that passes with the loops silently substituted would be decoration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod, GradConfig
from jaccpot.autodiff import direct_sum_gravitational_acceleration
from jaccpot.pallas import cascade_real_level, p2m_real_leaf
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
    assert (
        pallas_cascades_enabled()
    ), "fixture is vacuous: the Pallas path is not reachable"


@pytest.fixture
def reverse_counters(monkeypatch):
    """Count calls of the three reverse kernels (the bwd rules look them up at call time)."""
    counts = {"m2m": 0, "l2l": 0, "p2m": 0}

    def wrap(module, name, key):
        real = getattr(module, name)

        def counted(*a, **k):
            counts[key] += 1
            return real(*a, **k)

        monkeypatch.setattr(module, name, counted)

    wrap(cascade_real_level, "m2m_real_levels_reverse_pallas", "m2m")
    wrap(cascade_real_level, "l2l_real_levels_reverse_pallas", "l2l")
    wrap(p2m_real_leaf, "p2m_real_leaves_reverse_pallas", "p2m")
    return counts


def test_the_gate_stays_on_under_grad_and_the_config_turns_it_off(pallas_cascades_on):
    """Non-vacuous both ways: on by default inside the block, off with the config, restored after."""
    options = resolve_grad_options(None, num_particles=256, supports_fast_lane=True)
    assert options.cascade_pallas is True
    with grad_option_overrides(options):
        assert pallas_cascades_enabled()
    off = resolve_grad_options(
        GradConfig(cascade_pallas=False), num_particles=256, supports_fast_lane=True
    )
    assert off.cascade_pallas is False
    with grad_option_overrides(off):
        assert not pallas_cascades_enabled()
    assert pallas_cascades_enabled()


def _system(n=1024, seed=3):
    pos = jnp.asarray(_plummer(n, seed), jnp.float64)
    mass = jnp.asarray(np.full(n, 1.0 / n), jnp.float64)
    return pos, mass


def _far_pairs(state) -> int:
    """M2L pairs in the frozen topology. ZERO means the multipoles are dead code:
    JAX never calls a custom_vjp's reverse for an output nothing depends on, and
    the near field alone reproduces the direct sum -- the inertness trap of
    ``test_grad_golden_leaf_size_is_what_makes_the_far_field_nonempty``. At
    N=512 / leaf 32 / theta 0.5 that is exactly what happened, which is why the
    first version of this test (d06dee0) could only fail through JVP tracing."""
    inter = state.interactions
    if inter is not None:
        return int(jnp.sum(inter.counts))
    dual = state.dual_tree_result
    return int(jnp.sum(dual.far_pair_count)) if dual is not None else 0


_LEAF = 8  # deep enough for a non-empty M2L list at N=1024


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_grad_matches_the_direct_sum_through_the_reverse_pallas_kernels(
    pallas_cascades_on, reverse_counters
):
    """``jax.grad`` through the FMM with the Pallas cascades ON runs their reverse kernels and is exact.

    Before the custom_vjp seams this raised ``AssertionError`` from
    ``jax/_src/pallas/core.py::axis_frame``; with the loops substituted it would
    pass but the counters below would stay at zero.
    """
    pos, mass = _system()
    fmm = FastMultipoleMethod(theta=0.5, softening=_SOFT, G=1.0)
    state = fmm.prepare_state(pos, mass, max_order=4, leaf_size=_LEAF)
    assert (
        _far_pairs(state) > 0
    ), "vacuous configuration: no far pairs, the cascades are dead code"

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, mass) ** 2)

    got = np.asarray(jax.grad(loss)(pos))
    ref = np.asarray(
        jax.grad(
            lambda p: jnp.sum(
                direct_sum_gravitational_acceleration(p, mass, G=1.0, softening=_SOFT)
                ** 2
            )
        )(pos)
    )
    assert reverse_counters == {"m2m": 1, "l2l": 1, "p2m": 1}, reverse_counters
    assert np.all(np.isfinite(got))
    assert np.linalg.norm(got) > 0
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert (
        rel < 1e-5
    ), f"gradient disagrees with the direct-sum oracle: rel-L2 {rel:.3e}"


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_pallas_and_pure_jax_reverses_agree(pallas_cascades_on, reverse_counters):
    """The A/B switch: ``GradConfig(cascade_pallas=False)`` runs the loops and gives the same gradient."""
    pos, mass = _system(seed=5)
    fmm = FastMultipoleMethod(theta=0.5, softening=_SOFT, G=1.0)
    state = fmm.prepare_state(pos, mass, max_order=4, leaf_size=_LEAF)
    assert _far_pairs(state) > 0

    def loss(p, cfg):
        return jnp.sum(
            fmm.differentiable_accelerations(state, p, mass, grad_config=cfg) ** 2
        )

    g_loops = np.asarray(jax.grad(loss)(pos, GradConfig(cascade_pallas=False)))
    assert reverse_counters == {"m2m": 0, "l2l": 0, "p2m": 0}
    g_pallas = np.asarray(jax.grad(loss)(pos, GradConfig(cascade_pallas=True)))
    assert reverse_counters == {"m2m": 1, "l2l": 1, "p2m": 1}
    rel = np.linalg.norm(g_pallas - g_loops) / np.linalg.norm(g_loops)
    assert rel < 1e-9, f"Pallas reverse vs loop reverse rel-L2 {rel:.3e}"
