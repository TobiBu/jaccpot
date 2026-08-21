"""The large-N reverse path, exercised at last -- audit F27.

``runtime/_large_n_grad.py`` sat at **0% coverage**, and the reason recorded in
``docs/refactor_audit_2026-08.md`` (F27) and in
``tests/unit/test_large_n_config_thresholds.py``'s docstring was that it "needs a
GPU CI leg". That turned out to be wrong, and measurably so: on 2026-08-21 the
whole suite was run on an A100 and ``_large_n_grad.py`` stayed at **0/73
statements**. A GPU makes the lane *reachable*; nothing in the suite reached for
it.

Three gates have to be open at once, and each one closes silently:

1. ``can_use_large_n_prepare_path`` (``_large_n_pipeline.py``) needs
   ``preset="large_n_gpu"``, a GPU backend, a radix tree and the solidfmm
   basis/rotation. It has **no particle-count threshold**, which is what makes
   this test affordable -- the lane is enterable at N=2048.
2. The near-field payload must be the **prepacked source-leaf-id** layout.
   ``prepare_large_n_grad_plan`` rejects the materialized per-particle "pairs"
   layout outright, and that layout is what small N selects: the choice is
   ``est_payload_mb <= JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB`` (default 1024)
   in ``_large_n_pipeline.py``, so a small problem materializes the pairs payload
   and only a big one falls back to prepacked. The source comment there records
   the crossover: "the per-particle 'pairs' layout at N=65536, the prepacked
   layout at N=200000".

   Setting that cap to 0 selects the prepacked layout at small N. This is not a
   trick to make a test pass -- it is the *same layout production runs at
   N=200000*, reached through the knob that already decides it, so the code under
   test is the code that ships. The alternative was a 200k-particle GPU test.
3. ``retain_far_pairs_for_grad=True``, or the frozen M2L list is discarded at
   prepare time and the far field would be differentiated as a constant.

And one trap that is not a gate at all: at ``leaf_size=64`` this configuration
has **zero far pairs**, so the FMM degenerates to a direct sum and every accuracy
assertion passes at float32 round-off while testing nothing about P2M/M2M/M2L/L2L
in reverse. Measured: 0 far pairs at N=2048/leaf 64 and forward rel-L2 5.7e-07,
against 1130 far pairs at N=2048/leaf 16 and forward rel-L2 6.3e-04 -- three
orders of magnitude, which is the difference between a test and a decoration. So
``test_the_far_field_is_load_bearing`` is an inertness gate, in the same spirit as
``test_grad_golden_leaf_size_is_what_makes_the_far_field_nonempty``.

MEASURED 2026-08-21, A100-PCIE-40GB, jax 0.10.2, N=2048, leaf 16, order 4,
fp32 working dtype, 1130 active far pairs::

    forward  rel-L2  6.312e-04
    d/dpos   rel-L2  2.164e-05
    d/dmass  rel-L2  5.512e-04

Tolerances below carry ~5x headroom on the worst of those. Costs about 25 s to
prepare and 115 s for the reverse compile, so the fixture is module-scoped and
the expensive tests are marked ``slow`` by node id in ``tests/slow_tests.txt``.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.autodiff import direct_sum_gravitational_acceleration
from jaccpot.runtime._large_n_types import LargeNPreparedState

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason=(
        "the large-N prepare path is gated on a GPU backend by "
        "can_use_large_n_prepare_path, so this whole module is inert on CPU"
    ),
)

_N = 2048
_LEAF = 16
_ORDER = 4
_G = 1.0
_SOFTENING = 1e-2
_PAYLOAD_CAP_ENV = "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB"

# 5x the worst measured relative error (5.5e-04), which leaves room for GPU
# reduction nondeterminism without letting a real regression through.
_RTOL = 3e-3


def _problem(seed: int = 11) -> tuple[Any, Any, Any]:
    """Build a small fp32 problem plus a random adjoint weight.

    Parameters
    ----------
    seed : int, optional
        Seed for positions, masses and the loss weight, by default 11.

    Returns
    -------
    tuple[Any, Any, Any]
        Positions ``(N, 3)``, masses ``(N,)`` and a weight ``(N, 3)``, all
        float32 -- the working dtype the radix fast lane requires.
    """
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(_N, 3)), dtype=jnp.float32)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(_N,)), dtype=jnp.float32)
    # A random adjoint rather than sum(a**2), so no single particle dominates the
    # gradient and a sign error in one component cannot cancel.
    weight = jnp.asarray(rng.normal(size=(_N, 3)), dtype=jnp.float32)
    return positions, masses, weight


def _solver() -> FastMultipoleMethod:
    """Construct the one configuration that reaches the large-N reverse path.

    Returns
    -------
    FastMultipoleMethod
        Solver with ``preset="large_n_gpu"`` and the far-pair retention the
        reverse path requires.
    """
    return FastMultipoleMethod(
        preset="large_n_gpu",
        G=_G,
        softening=_SOFTENING,
        retain_far_pairs_for_grad=True,
    )


def _relative_l2(got: Any, want: Any) -> float:
    """Relative L2 difference, accumulated in float64.

    Parameters
    ----------
    got : Any
        Array under test.
    want : Any
        Reference array.

    Returns
    -------
    float
        ``||got - want|| / ||want||``.
    """
    got_arr = np.asarray(got, dtype=np.float64)
    want_arr = np.asarray(want, dtype=np.float64)
    return float(
        np.linalg.norm(got_arr - want_arr) / (np.linalg.norm(want_arr) + 1e-30)
    )


@pytest.fixture(scope="module")
def prepacked_state() -> Iterator[tuple[Any, Any, Any, Any]]:
    """A large-N state on the prepacked near-field layout.

    The payload cap is forced to 0 for the duration of ``prepare_state`` only:
    the layout is baked into the state there, so the reverse pass needs no
    environment of its own.

    Yields
    ------
    tuple[Any, Any, Any, Any]
        Solver, prepared state, positions, masses.
    """
    positions, masses, _ = _problem()
    fmm = _solver()
    previous = os.environ.get(_PAYLOAD_CAP_ENV)
    os.environ[_PAYLOAD_CAP_ENV] = "0"
    try:
        state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
    finally:
        if previous is None:
            os.environ.pop(_PAYLOAD_CAP_ENV, None)
        else:
            os.environ[_PAYLOAD_CAP_ENV] = previous
    yield fmm, state, positions, masses


@pytest.fixture(scope="module")
def reverse_pass(prepacked_state) -> dict[str, Any]:
    """Run the reverse pass once and reuse it across assertions.

    One ``jax.grad`` compile is ~115 s, so every assertion below reads this.

    Parameters
    ----------
    prepacked_state : tuple
        Fixture value: solver, state, positions, masses.

    Returns
    -------
    dict[str, Any]
        Forward accelerations and position/mass gradients for both the FMM and
        the direct-sum reference.
    """
    fmm, state, positions, masses = prepacked_state
    _, _, weight = _problem()

    def fmm_loss(pos: Any, mass: Any) -> Any:
        return jnp.sum(weight * fmm.differentiable_accelerations(state, pos, mass))

    def reference_loss(pos: Any, mass: Any) -> Any:
        return jnp.sum(
            weight
            * direct_sum_gravitational_acceleration(
                pos, mass, G=_G, softening=_SOFTENING
            )
        )

    grad_positions, grad_masses = jax.grad(fmm_loss, argnums=(0, 1))(positions, masses)
    ref_positions, ref_masses = jax.grad(reference_loss, argnums=(0, 1))(
        positions, masses
    )
    return {
        "accelerations": fmm.differentiable_accelerations(state, positions, masses),
        "reference_accelerations": direct_sum_gravitational_acceleration(
            positions, masses, G=_G, softening=_SOFTENING
        ),
        "grad_positions": grad_positions,
        "grad_masses": grad_masses,
        "reference_grad_positions": ref_positions,
        "reference_grad_masses": ref_masses,
    }


def test_the_lane_under_test_is_actually_the_large_n_lane(prepacked_state) -> None:
    """``prepare_state`` must return a ``LargeNPreparedState``, not a radix one.

    Without this the module would silently test the ordinary radix grad path --
    which is exactly what ``test_large_n_config_thresholds.py`` does, and why
    F27 stayed at 0% while 20 tests named for the large-N path passed.
    """
    _, state, _, _ = prepacked_state
    assert isinstance(state, LargeNPreparedState), (
        f"expected LargeNPreparedState, got {type(state).__name__}: the large-N "
        "prepare path declined and this module is testing a different lane"
    )


def test_the_far_field_is_load_bearing(prepacked_state) -> None:
    """There must be far pairs, or the FMM is a direct sum wearing its clothes.

    At ``leaf_size=64`` this same configuration yields **zero** far pairs and the
    accuracy assertions below pass at float32 round-off (5.7e-07) while covering
    nothing in P2M/M2M/M2L/L2L. ``leaf_size=16`` gives 1130.
    """
    _, state, _, _ = prepacked_state
    far_pairs = state.compact_far_pairs
    assert far_pairs is not None, "compact_far_pairs was discarded"
    active = int(np.asarray(far_pairs.far_pair_count))
    assert active > 0, (
        "zero active far pairs: the tree is too shallow for this leaf size, so "
        "the far-field reverse is not being exercised at all"
    )


def test_reverse_pass_is_finite(reverse_pass) -> None:
    """Neither gradient may contain a NaN or an inf."""
    for name in ("grad_positions", "grad_masses"):
        assert bool(
            jnp.all(jnp.isfinite(reverse_pass[name]))
        ), f"{name} contains a non-finite entry"


def test_forward_matches_the_direct_sum(reverse_pass) -> None:
    """The forward pass anchors the reverse: measured rel-L2 6.3e-04."""
    error = _relative_l2(
        reverse_pass["accelerations"], reverse_pass["reference_accelerations"]
    )
    assert error < _RTOL, f"forward rel-L2 {error:.3e} exceeds {_RTOL:.0e}"


def test_position_gradient_matches_the_direct_sum(reverse_pass) -> None:
    """d/dpositions against an exact O(N^2) adjoint: measured rel-L2 2.2e-05."""
    error = _relative_l2(
        reverse_pass["grad_positions"], reverse_pass["reference_grad_positions"]
    )
    assert error < _RTOL, f"d/dpos rel-L2 {error:.3e} exceeds {_RTOL:.0e}"


def test_mass_gradient_matches_the_direct_sum(reverse_pass) -> None:
    """d/dmasses is the one that catches a far field frozen into a constant.

    If the far field were treated as constant -- the failure
    ``prepare_large_n_grad_plan`` raises about when ``compact_far_pairs`` is
    missing -- the mass sensitivity through P2M/M2M/M2L/L2L would be zero and
    this comparison against the exact adjoint is what notices. Measured rel-L2
    5.5e-04.
    """
    error = _relative_l2(
        reverse_pass["grad_masses"], reverse_pass["reference_grad_masses"]
    )
    assert error < _RTOL, f"d/dmass rel-L2 {error:.3e} exceeds {_RTOL:.0e}"


def test_the_pairs_layout_is_rejected_rather_than_silently_dropped() -> None:
    """Without the cap override, small N materializes the pairs layout and must reject.

    This is the guard that makes the payload-cap override in this module honest:
    the unsupported layout raises instead of quietly differentiating a different
    force than the forward computed.
    """
    positions, masses, _ = _problem()
    fmm = _solver()
    state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
    assert int(state.radix_fast_payload.source_particle_ids.size) > 0, (
        "expected the materialized per-particle payload at this size; the "
        "layout crossover moved and this test no longer covers the rejection"
    )
    with pytest.raises(NotImplementedError, match="PREPACKED"):
        fmm.differentiable_accelerations(state, positions, masses)


def test_discarded_far_pairs_are_rejected_rather_than_differentiated_as_constant() -> (
    None
):
    """Without ``retain_far_pairs_for_grad`` the reverse must refuse, not freeze the far field."""
    positions, masses, _ = _problem()
    fmm = FastMultipoleMethod(preset="large_n_gpu", G=_G, softening=_SOFTENING)
    previous = os.environ.get(_PAYLOAD_CAP_ENV)
    os.environ[_PAYLOAD_CAP_ENV] = "0"
    try:
        state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
    finally:
        if previous is None:
            os.environ.pop(_PAYLOAD_CAP_ENV, None)
        else:
            os.environ[_PAYLOAD_CAP_ENV] = previous
    with pytest.raises(RuntimeError, match="compact_far_pairs"):
        fmm.differentiable_accelerations(state, positions, masses)
