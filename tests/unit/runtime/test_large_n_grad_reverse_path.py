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

   The backend half of that gate is **profile selection, not a hardware
   requirement**, and this module now opens it with a fixture rather than
   skipping (see ``_open_the_backend_gate``). Until 2026-08-25 it skipped, so
   the 91% this file earns was a GPU-only number and ``_large_n_grad.py``
   measured **0%** on every machine that can run the suite -- which is what kept
   F11's split deferred. It runs on CPU in ~19 s and agrees with the direct sum
   to 2.2e-05 / 5.5e-04.
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

This module owns the large-N reverse property (the 2026-10 cleanup's rule R6), so
two former modules are classes here, each with its own lane setup:

* ``TestEveryFastPallasKernel`` (was ``test_grad_path_fast_kernels_large_n.py``):
  the same lane with every Pallas kernel opened on CPU through its interpret flag,
  the five reverse kernels COUNTED;
* ``TestPallasCascades`` (was ``test_grad_path_pallas_cascades.py``): the default
  solver's gradient through the Pallas M2M/L2L cascades and the leaf P2M, on the
  real (CPU) backend profile;
* ``TestSofteningKernelsOnTheCsrLane``: the solver's softening kernel reaches the
  differentiable CSR near field (forward and reverse), checked through the public
  entries for every kernel.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    GradConfig,
    TreeConfig,
)
from jaccpot.autodiff import direct_sum_gravitational_acceleration
from jaccpot.pallas import (
    cascade_real_level,
    l2p_real,
    m2l_real_csr_lanes,
    nearfield_leafpair_csr,
    nearfield_leafpair_csr_direct_vjp,
    p2m_real_leaf,
)
from jaccpot.runtime._large_n_types import LargeNPreparedState
from jaccpot.runtime._level_shapes import pallas_cascades_enabled
from jaccpot.runtime.grad_options import grad_option_overrides, resolve_grad_options
from jaccpot.softening import DEFAULT_SOFTENING_KERNEL, SOFTENING_KERNELS

#: The real backend query, captured before `_open_the_backend_gate` patches it, for
#: the tests that must run on the CPU's own profile (`TestPallasCascades`).
_REAL_DEFAULT_BACKEND = jax.default_backend


@pytest.fixture(scope="module", autouse=True)
def _open_the_backend_gate():
    """Open ``can_use_large_n_prepare_path``'s backend check on CPU.

    That check is *profile selection*, not a hardware requirement: it decides
    which lane to enter, and the lane it guards runs on CPU. This module used to
    be skipped outright when the backend was not a GPU, which meant
    ``_large_n_grad.py`` measured 0% on every machine that can actually run this
    suite -- the number F27 was written about, and the number that kept F11's
    split deferred.

    Measured on CPU at N=2048 / leaf 16 / order 4: the reverse pass completes in
    ~19 s and the gradients match the direct sum at **2.2e-05** (positions) and
    **5.5e-04** (masses). That second figure is the *same* "worst measured
    relative error" ``_RTOL`` was derived from, so the tolerance transfers rather
    than being re-tuned for CPU.

    Module-scoped, because the fixtures below are, and the gate has to be open
    before ``prepare_state`` reads it.
    """
    patcher = pytest.MonkeyPatch()
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()


_N = 2048
_LEAF = 16
_ORDER = 4
_G = 1.0
_SOFTENING = 1e-2
_PAYLOAD_CAP_ENV = "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB"

# 5x the worst measured relative error (5.5e-04), which leaves room for GPU
# reduction nondeterminism without letting a real regression through.
_RTOL = 3e-3

# A module- or class-scoped fixture is built once per xdist WORKER that requests it,
# and CI runs `-n 2 --dist loadgroup`: unpinned, the ~25 s prepare and ~115 s reverse
# compile behind `reverse_pass` ran on both workers. Each fixture's users share a
# group, so `loadgroup` sends them to one worker; the groups still run in parallel.
_PREPACKED_STATE_USERS = pytest.mark.xdist_group("large_n_reverse_prepacked_state")


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


@_PREPACKED_STATE_USERS
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


@_PREPACKED_STATE_USERS
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


@_PREPACKED_STATE_USERS
def test_forward_matches_the_direct_sum(reverse_pass) -> None:
    """The forward pass anchors the reverse: measured rel-L2 6.3e-04."""
    error = _relative_l2(
        reverse_pass["accelerations"], reverse_pass["reference_accelerations"]
    )
    assert error < _RTOL, f"forward rel-L2 {error:.3e} exceeds {_RTOL:.0e}"


@_PREPACKED_STATE_USERS
def test_position_gradient_matches_the_direct_sum(reverse_pass) -> None:
    """d/dpositions against an exact O(N^2) adjoint: measured rel-L2 2.2e-05."""
    error = _relative_l2(
        reverse_pass["grad_positions"], reverse_pass["reference_grad_positions"]
    )
    assert error < _RTOL, f"d/dpos rel-L2 {error:.3e} exceeds {_RTOL:.0e}"


@_PREPACKED_STATE_USERS
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


def test_the_pairs_layout_is_rejected_rather_than_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the cap override, small N materializes the pairs layout and must reject.

    This is the guard that makes the payload-cap override in this module honest:
    the unsupported layout raises instead of quietly differentiating a different
    force than the forward computed.

    The CSR near-field lane is switched off: it is the default where it lowers
    (an Ampere GPU), and its prepare shrinks the rectangle payload to a one-block
    placeholder, so on an A100 no pairs layout exists to reject (the premise
    below failed there, 2026-10-15). Off, the same N materializes it on both.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Switches the CSR near-field lane off.
    """
    monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "0")
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
    """Without ``retain_far_pairs_for_grad`` the reverse must refuse, not freeze the far field.

    The ``lbvh`` tree discards the far-pair list. The ``static_radix`` tree that
    ``large_n_gpu`` builds since the 2026-10 cleanup (D2) keeps it for its strict
    lane, so there the reverse runs through the real far field instead.
    """
    positions, masses, _ = _problem()
    states = {}
    for mode in ("lbvh", "static_radix"):
        fmm = FastMultipoleMethod(
            preset="large_n_gpu",
            G=_G,
            softening=_SOFTENING,
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode=mode)),
        )
        previous = os.environ.get(_PAYLOAD_CAP_ENV)
        os.environ[_PAYLOAD_CAP_ENV] = "0"
        try:
            states[mode] = (
                fmm,
                fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER),
            )
        finally:
            if previous is None:
                os.environ.pop(_PAYLOAD_CAP_ENV, None)
            else:
                os.environ[_PAYLOAD_CAP_ENV] = previous
    fmm, state = states["lbvh"]
    assert state.compact_far_pairs is None
    with pytest.raises(RuntimeError, match="compact_far_pairs"):
        fmm.differentiable_accelerations(state, positions, masses)
    assert states["static_radix"][1].compact_far_pairs is not None


# ---------------------------------------------------------------------------
# Every fast Pallas kernel, on CPU (was test_grad_path_fast_kernels_large_n.py).
#
# Plan ``fast-gradients-for-the-sub10ms-fmm``. The sub-10 ms forward at N = 2x10^5
# runs six Pallas kernels: the leaf P2M, the per-level M2M and L2L cascades, the
# pair-per-lane M2L over the far-pair CSR, the particle-major L2P and the CSR near
# field's direct layout. Each carries a ``custom_vjp`` whose reverse is a Pallas
# kernel of its own (the L2P's and the direct layout's since plan S8: before it, the
# gradient lane ran the leaf-major jnp L2P and the near field's table layout). This
# opens every one of them on CPU through its interpret flag, differentiates the
# production ``large_n_gpu`` path (the gates above: backend patched, prepacked
# near-field layout forced, far pairs retained) and checks the gradient against
# the direct-sum oracle -- and COUNTS the six reverse kernels, because a path that
# quietly fell back to a pure-JAX lane would pass the accuracy check while
# measuring nothing.
#
# Found while writing it (Phase 1 of the plan, GPU): the M2L CSR lanes kernel had
# no gradient guard at all, so on an Ampere card ``jax.grad`` still died in
# ``program_id`` after the Phase 0 fix -- the CPU tests never reached it because
# the CSR M2L does not lower there. ``JACCPOT_M2L_CSR_INTERPRET=1`` reaches it.
# ---------------------------------------------------------------------------

_INTERPRET_FLAGS = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    _PAYLOAD_CAP_ENV: "0",  # the prepacked layout, as at N=200k
}


def _count_calls(monkeypatch, counts, module, name, key) -> None:
    """Wrap ``module.name`` so each call increments ``counts[key]``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Undoes the wrap at teardown.
    counts : dict
        The counters.
    module : module
        Where the kernel is looked up at call time.
    name : str
        The kernel's name.
    key : str
        Its counter.
    """
    real = getattr(module, name)

    def counted(*args, **kwargs):
        counts[key] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


@pytest.mark.xdist_group("large_n_reverse_every_pallas_kernel")
class TestEveryFastPallasKernel:
    @pytest.fixture(scope="class", autouse=True)
    def _every_pallas_lane_on_cpu(self) -> Iterator[None]:
        patcher = pytest.MonkeyPatch()
        for key, value in _INTERPRET_FLAGS.items():
            patcher.setenv(key, value)
        yield
        patcher.undo()

    @pytest.fixture(scope="class")
    def prepared(self, _every_pallas_lane_on_cpu) -> tuple[Any, Any, Any, Any, Any]:
        positions, masses, weight = _problem()
        fmm = FastMultipoleMethod(
            preset="large_n_gpu",
            G=_G,
            softening=_SOFTENING,
            retain_far_pairs_for_grad=True,
            use_pallas=True,
        )
        state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
        assert isinstance(state, LargeNPreparedState)
        assert (
            int(getattr(state.radix_fast_payload, "source_particle_ids").size) == 0
        )  # prepacked
        assert state.compact_far_pairs is not None
        return fmm, state, positions, masses, weight

    @pytest.fixture
    def reverse_counters(self, monkeypatch):
        counts = {k: 0 for k in ("p2m", "m2m", "l2l", "m2l", "l2p", "near")}
        for module, name, key in (
            (p2m_real_leaf, "p2m_real_leaves_reverse_pallas", "p2m"),
            (cascade_real_level, "m2m_real_levels_reverse_pallas", "m2m"),
            (cascade_real_level, "l2l_real_levels_reverse_pallas", "l2l"),
            (m2l_real_csr_lanes, "m2l_real_csr_lanes_reverse_pallas", "m2l"),
            (l2p_real, "l2p_real_particles_reverse_pallas", "l2p"),
            (
                nearfield_leafpair_csr_direct_vjp,
                "nearfield_leafpair_csr_sorted_direct_reverse_pallas",
                "near",
            ),
        ):
            _count_calls(monkeypatch, counts, module, name, key)
        return counts

    @pytest.fixture
    def forward_counters(self, monkeypatch):
        counts = {k: 0 for k in ("m2l_lanes", "near_direct", "near_table")}
        for module, name, key in (
            (m2l_real_csr_lanes, "m2l_real_csr_lanes_pallas", "m2l_lanes"),
            (
                nearfield_leafpair_csr_direct_vjp,
                "nearfield_leafpair_csr_sorted_direct_pallas_cvjp",
                "near_direct",
            ),
            (nearfield_leafpair_csr, "nearfield_leafpair_csr_pallas", "near_table"),
        ):
            _count_calls(monkeypatch, counts, module, name, key)
        return counts

    def test_the_far_field_is_load_bearing(self, prepared):
        _, state, *_ = prepared
        assert int(np.asarray(state.compact_far_pairs.far_pair_count)) > 0

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_grad_through_all_six_reverse_kernels_matches_the_direct_sum(
        self, prepared, reverse_counters, forward_counters
    ):
        fmm, state, positions, masses, weight = prepared

        def loss(p, m):
            return jnp.sum(weight * fmm.differentiable_accelerations(state, p, m))

        def oracle(p, m):
            return jnp.sum(
                weight
                * direct_sum_gravitational_acceleration(
                    p, m, G=_G, softening=_SOFTENING
                )
            )

        forward = np.asarray(fmm.differentiable_accelerations(state, positions, masses))
        # the production layout: the direct near field, never the table one
        assert (
            forward_counters["m2l_lanes"] >= 1
            and forward_counters["near_direct"] >= 1
            and forward_counters["near_table"] == 0
        ), forward_counters
        assert reverse_counters == {k: 0 for k in reverse_counters}
        got_p, got_m = jax.grad(loss, argnums=(0, 1))(positions, masses)
        assert all(v == 1 for v in reverse_counters.values()), reverse_counters
        want_p, want_m = jax.grad(oracle, argnums=(0, 1))(positions, masses)
        assert np.all(np.isfinite(np.asarray(got_p))) and np.all(
            np.isfinite(np.asarray(got_m))
        )
        rel_p = _relative_l2(got_p, want_p)
        rel_m = _relative_l2(got_m, want_m)
        assert rel_p < _RTOL, f"d/dpos rel-L2 {rel_p:.3e}"
        assert rel_m < _RTOL, f"d/dmass rel-L2 {rel_m:.3e}"
        ref_fwd = np.asarray(
            direct_sum_gravitational_acceleration(
                positions, masses, G=_G, softening=_SOFTENING
            )
        )
        assert _relative_l2(forward, ref_fwd) < 5e-3


# ---------------------------------------------------------------------------
# The softening kernel on the differentiable CSR near field.
#
# The differentiable CSR call passes its arguments positionally (a custom_vjp
# primal) and dropped the trailing ``softening_kernel``, so the gradient lane's
# forward AND reverse ran the default kernel whatever the solver's: 4.2e-2 off the
# production forward with plummer (review 2026-10-13). Every kernel-level parity
# test passed -- each kernel is right; the DISPATCH dropped the argument. So this
# compares the two public entries, ``differentiable_accelerations`` and
# ``evaluate_prepared_state``, for every kernel in ``jaccpot.softening``, at a
# softening where the kernels differ.
#
# Measured on CPU (N=1024, leaf 8, order 4, eps 0.05, 2444 far pairs) before the
# fix: forward vs production 5.96e-02 (plummer), 1.02e-02 (wendland_c2), 3.4e-07
# (ferrers3, the default); d/dpos vs the plummer direct-sum gradient 0.380.
#
# Since plan S8 the two entries run the same kernels -- the near field's direct
# layout (the gradient lane through its custom_vjp) and the particle-major L2P (its
# Pallas kernel in interpret mode here, as on a GPU) -- so the forwards are BITWISE
# equal, for every kernel. On the base they differed by the layouts (the gradient
# lane's table near field and leaf-major L2P): 3.4e-07 with ferrers3.
#
# Both are compiled programs here, as the production one is: run eagerly, the
# custom_vjp's forward is a program of its own in which the softening is an
# argument, and in interpret mode XLA then cannot fold it into the kernel body --
# one float32 rounding on 1471 of 3072 elements with ferrers3, 1334 with
# wendland_c2, none with plummer (the rewrite ``docs/fused_provider.md`` describes
# for traced parameters). On a GPU the kernel reads the softening as an operand
# either way.
# ---------------------------------------------------------------------------


def _jitted_force(fmm: Any, state: Any, positions: Any, masses: Any) -> Any:
    """The differentiable force compiled with the state and its plan as arguments.

    Parameters
    ----------
    fmm : Any
        The solver.
    state : Any
        Its prepared large-N state.
    positions : Any
        Positions in the caller's order.
    masses : Any
        Masses, same order.

    Returns
    -------
    Any
        ``(N, 3)`` accelerations in the caller's order.
    """
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

    plan = prepare_large_n_grad_plan(fmm, state)

    @jax.jit
    def force(st, pl, p, m):
        return fmm.differentiable_accelerations(st, p, m, grad_plan=pl)

    return force(state, plan, positions, masses)


_KERNEL_N = 1024
_KERNEL_LEAF = 8  # 2444 far pairs: both forwards carry a far field
_KERNEL_SOFTENING = 0.05  # direct sums vs ferrers3: wendland_c2 1.0e-2, plummer 5.9e-2
_KERNEL_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    # the GPU's L2P: the XLA one's program rounds differently with its locals
    # computed in the same program (23 of 3072 elements at 1.5e-5 here)
    "JACCPOT_L2P_KERNEL": "interpret",
    _PAYLOAD_CAP_ENV: "0",  # the prepacked layout the grad plan needs
}
# The plummer gradient against its direct-sum oracle: d/dpos 7.6e-04, d/dmass
# 3.8e-03 measured, the FMM's own error at leaf 8 (ferrers3 gives 6.9e-04 /
# 3.4e-03 here). With the default kernel's reverse: 0.38 / 0.19.
_KERNEL_GRAD_RTOL = 1e-2


@pytest.mark.xdist_group("large_n_reverse_softening_kernels")
class TestSofteningKernelsOnTheCsrLane:
    @pytest.fixture(scope="class", autouse=True)
    def _csr_lane_on_cpu(self) -> Iterator[None]:
        patcher = pytest.MonkeyPatch()
        for key, value in _KERNEL_ENV.items():
            patcher.setenv(key, value)
        yield
        patcher.undo()

    @pytest.fixture(scope="class")
    def problem(self) -> tuple[Any, Any, Any]:
        rng = np.random.default_rng(11)
        positions = jnp.asarray(rng.normal(size=(_KERNEL_N, 3)), jnp.float32)
        masses = jnp.asarray(rng.uniform(0.5, 1.5, size=_KERNEL_N), jnp.float32)
        weight = jnp.asarray(rng.normal(size=(_KERNEL_N, 3)), jnp.float32)
        return positions, masses, weight

    @pytest.fixture(scope="class")
    def solvers(self, _csr_lane_on_cpu, problem) -> Any:
        """One solver and prepared state per kernel, built on first use."""
        cache: dict[str, tuple[Any, Any]] = {}
        positions, masses, _ = problem

        def get(kernel: str) -> tuple[Any, Any]:
            if kernel not in cache:
                fmm = FastMultipoleMethod(
                    preset="large_n_gpu",
                    G=_G,
                    softening=_KERNEL_SOFTENING,
                    softening_kernel=kernel,
                    retain_far_pairs_for_grad=True,
                    use_pallas=True,
                )
                state = fmm.prepare_state(
                    positions, masses, leaf_size=_KERNEL_LEAF, max_order=_ORDER
                )
                assert isinstance(state, LargeNPreparedState)
                assert int(np.asarray(state.compact_far_pairs.far_pair_count)) > 0
                cache[kernel] = (fmm, state)
            return cache[kernel]

        return get

    @staticmethod
    def _direct(problem, kernel: str) -> Any:
        """The fp64 direct sum with ``kernel``."""
        positions, masses, _ = problem
        return direct_sum_gravitational_acceleration(
            positions.astype(jnp.float64),
            masses.astype(jnp.float64),
            G=_G,
            softening=_KERNEL_SOFTENING,
            softening_kernel=kernel,
        )

    @pytest.fixture
    def csr_counters(self, monkeypatch) -> dict[str, int]:
        counts = {"table": 0, "direct": 0, "direct_cvjp": 0, "reverse": 0}
        for module, name, key in (
            (nearfield_leafpair_csr, "nearfield_leafpair_csr_pallas", "table"),
            (
                nearfield_leafpair_csr,
                "nearfield_leafpair_csr_sorted_direct_pallas",
                "direct",
            ),
            (
                nearfield_leafpair_csr_direct_vjp,
                "nearfield_leafpair_csr_sorted_direct_pallas_cvjp",
                "direct_cvjp",
            ),
            (
                nearfield_leafpair_csr_direct_vjp,
                "nearfield_leafpair_csr_sorted_direct_reverse_pallas",
                "reverse",
            ),
        ):
            _count_calls(monkeypatch, counts, module, name, key)
        return counts

    def test_the_kernels_differ_at_this_softening(self, problem):
        """Non-vacuous: a dropped kernel must move the force far past round-off."""
        refs = {k: np.asarray(self._direct(problem, k)) for k in SOFTENING_KERNELS}
        for kernel in SOFTENING_KERNELS:
            if kernel != DEFAULT_SOFTENING_KERNEL:
                gap = _relative_l2(refs[kernel], refs[DEFAULT_SOFTENING_KERNEL])
                assert gap > 1e-3, (kernel, gap)

    @pytest.mark.parametrize("kernel", SOFTENING_KERNELS)
    def test_the_differentiable_forward_is_the_production_forward(
        self, kernel, solvers, problem, csr_counters
    ):
        positions, masses, _ = problem
        fmm, state = solvers(kernel)
        production = np.asarray(fmm.evaluate_prepared_state(state))
        assert csr_counters["direct"] >= 1, csr_counters
        differentiable = np.asarray(_jitted_force(fmm, state, positions, masses))
        # the gradient lane ran production's direct layout under its custom_vjp
        assert csr_counters["direct_cvjp"] >= 1, csr_counters
        assert csr_counters["table"] == 0, csr_counters
        differ = int(
            np.sum(differentiable.view(np.uint32) != production.view(np.uint32))
        )
        assert differ == 0, (
            f"{kernel}: {differ} of {production.size} elements differ, max "
            f"{float(np.max(np.abs(differentiable - production))):.3e}"
        )
        reference = np.asarray(self._direct(problem, kernel))
        assert _relative_l2(differentiable, reference) < _RTOL

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_reverse_runs_the_solvers_kernel(self, solvers, problem, csr_counters):
        """d/dpos and d/dmass with plummer, the kernel furthest from the default."""
        positions, masses, weight = problem
        fmm, state = solvers("plummer")

        def loss(p, m):
            return jnp.sum(weight * fmm.differentiable_accelerations(state, p, m))

        def oracle(p, m):
            return jnp.sum(
                weight.astype(jnp.float64)
                * direct_sum_gravitational_acceleration(
                    p,
                    m,
                    G=_G,
                    softening=_KERNEL_SOFTENING,
                    softening_kernel="plummer",
                )
            )

        got_p, got_m = jax.grad(loss, argnums=(0, 1))(positions, masses)
        assert csr_counters["reverse"] == 1, csr_counters
        want_p, want_m = jax.grad(oracle, argnums=(0, 1))(
            positions.astype(jnp.float64), masses.astype(jnp.float64)
        )
        rel_p = _relative_l2(got_p, want_p)
        rel_m = _relative_l2(got_m, want_m)
        assert rel_p < _KERNEL_GRAD_RTOL, f"d/dpos rel-L2 {rel_p:.3e}"
        assert rel_m < _KERNEL_GRAD_RTOL, f"d/dmass rel-L2 {rel_m:.3e}"


# ---------------------------------------------------------------------------
# The Pallas cascades under jax.grad (was test_grad_path_pallas_cascades.py).
#
# Phase 6 of the sub-10 ms plan made the per-level Pallas M2M/L2L cascades and the
# leaf P2M default-ON while they had no autodiff rule; ``pallas_call``'s generic
# JVP rule dies on ``program_id`` (no grid context), so ``jax.grad`` broke
# everywhere Pallas lowers -- invisibly, because Pallas does not lower on CPU and
# the gradient suite runs there. The first fix (d06dee0) forced the gradient path
# onto the pure-JAX loops. The kernels now carry ``custom_vjp`` rules whose reverse
# is a Pallas kernel of its own, so the gate stays ON under ``jax.grad`` and
# ``GradConfig(cascade_pallas=False)`` is the A/B switch back to the loops.
#
# ``JACCPOT_CASCADE_PALLAS_INTERPRET=1`` makes the Pallas path reachable on CPU,
# so this runs in ordinary CI. The reverse kernels are COUNTED, not assumed. These
# run the default solver on the real backend profile, so the module's backend
# gate is closed again for each of them.
# ---------------------------------------------------------------------------

_CASCADE_SOFT = 1e-2
_CASCADE_LEAF = 8  # deep enough for a non-empty M2L list at N=1024


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _plummer_system(n=1024, seed=3):
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


class TestPallasCascades:
    @pytest.fixture
    def pallas_cascades_on(self, monkeypatch):
        """The real backend profile, with the Pallas cascades reachable on CPU."""
        monkeypatch.setattr(jax, "default_backend", _REAL_DEFAULT_BACKEND)
        monkeypatch.setenv("JACCPOT_CASCADE_PALLAS", "1")
        monkeypatch.setenv("JACCPOT_CASCADE_PALLAS_INTERPRET", "1")
        assert (
            pallas_cascades_enabled()
        ), "fixture is vacuous: the Pallas path is not reachable"

    @pytest.fixture
    def reverse_counters(self, monkeypatch):
        """Count the three reverse kernels (the bwd rules look them up per call)."""
        counts = {"m2m": 0, "l2l": 0, "p2m": 0}
        for module, name, key in (
            (cascade_real_level, "m2m_real_levels_reverse_pallas", "m2m"),
            (cascade_real_level, "l2l_real_levels_reverse_pallas", "l2l"),
            (p2m_real_leaf, "p2m_real_leaves_reverse_pallas", "p2m"),
        ):
            _count_calls(monkeypatch, counts, module, name, key)
        return counts

    def test_the_gate_stays_on_under_grad_and_the_config_turns_it_off(
        self, pallas_cascades_on
    ):
        """Non-vacuous both ways: on inside the block, off with the config, restored."""
        options = resolve_grad_options(None, num_particles=256, supports_fast_lane=True)
        assert options.cascade_pallas is True
        with grad_option_overrides(options):
            assert pallas_cascades_enabled()
        off = resolve_grad_options(
            GradConfig(cascade_pallas=False),
            num_particles=256,
            supports_fast_lane=True,
        )
        assert off.cascade_pallas is False
        with grad_option_overrides(off):
            assert not pallas_cascades_enabled()
        assert pallas_cascades_enabled()

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_grad_matches_the_direct_sum_through_the_reverse_pallas_kernels(
        self, pallas_cascades_on, reverse_counters
    ):
        """``jax.grad`` with the Pallas cascades ON runs their reverses and is exact.

        Before the custom_vjp seams this raised ``AssertionError`` from
        ``jax/_src/pallas/core.py::axis_frame``; with the loops substituted it would
        pass but the counters below would stay at zero. ``GradConfig(cascade_pallas=
        False)`` runs the pure-JAX loops instead, counted at zero, and gives the same
        gradient. One system, one prepare: the two arms and the direct-sum oracle
        used to be two tests (seeds 3 and 5) with the same solver, softening, order,
        leaf size, N and loss.
        """
        pos, mass = _plummer_system()
        fmm = FastMultipoleMethod(theta=0.5, softening=_CASCADE_SOFT, G=1.0)
        state = fmm.prepare_state(pos, mass, max_order=4, leaf_size=_CASCADE_LEAF)
        assert (
            _far_pairs(state) > 0
        ), "vacuous configuration: no far pairs, the cascades are dead code"

        def loss(p, cfg):
            return jnp.sum(
                fmm.differentiable_accelerations(state, p, mass, grad_config=cfg) ** 2
            )

        g_loops = np.asarray(jax.grad(loss)(pos, GradConfig(cascade_pallas=False)))
        assert reverse_counters == {"m2m": 0, "l2l": 0, "p2m": 0}, reverse_counters
        # The default config: the fixture's JACCPOT_CASCADE_PALLAS=1 resolves it ON.
        got = np.asarray(jax.grad(loss)(pos, None))
        assert reverse_counters == {"m2m": 1, "l2l": 1, "p2m": 1}, reverse_counters
        ref = np.asarray(
            jax.grad(
                lambda p: jnp.sum(
                    direct_sum_gravitational_acceleration(
                        p, mass, G=1.0, softening=_CASCADE_SOFT
                    )
                    ** 2
                )
            )(pos)
        )
        assert np.all(np.isfinite(got))
        assert np.linalg.norm(got) > 0
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert (
            rel < 1e-5
        ), f"gradient disagrees with the direct-sum oracle: rel-L2 {rel:.3e}"
        rel = np.linalg.norm(got - g_loops) / np.linalg.norm(g_loops)
        assert rel < 1e-9, f"Pallas reverse vs loop reverse rel-L2 {rel:.3e}"


# ---------------------------------------------------------------------------
# The forward this module differentiates is the production force (plan S1; the GPU
# pin is G1 in bench/dce_pins.py).
#
# ``differentiable_accelerations`` re-evaluates the lists ``prepare_state`` froze, so
# it is the production force only if that walk tested the MAC on the same geometry
# as the fused lane's. Since bc62978 the geometry followed the engine's
# ``_strict_fused_mode_active``, which only a strict call sets: a FRESH engine's
# ``prepare_state`` walked the box MAC and the fused eval the centre-of-mass one.
# ``docs/fast_gradients_2026-09.md``'s 1.8e-11 was measured on the branch before
# that commit; on main the two lanes were 1.46e-3 apart at the operating point of
# bench/grad_step_profile.py (2e5), the differentiable forward the less accurate
# (aggL2 1.60e-3 against 1.27e-3). Here, on CPU (~70 s): 5.3e-3 apart with 2640
# against 2168 far pairs, and 5.5e-8 with the same geometry
# (JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY=com).
#
# Both call orders: a fresh solver (what the bench does) and the production solver
# after its strict call.
#
# With the same lists the two lanes differed by fp32 summation order only (5.5e-8
# here, 1.7e-7 at 2e5 on an A100): the gradient lane's near field was the table
# layout and its L2P the leaf-major jnp one. Since plan S8 it runs production's
# kernels (the direct near layout and the particle-major L2P under their reverses),
# so on the Pallas lane -- the GPU's, interpret mode here -- the forwards are
# BITWISE equal. (On the pure-JAX route the XLA L2P rounds differently with its
# locals computed in the same program: 23 of 3072 elements at N=1024.)
# ---------------------------------------------------------------------------

_FUSED_ENV = {
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "1048576",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "auto",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "262144",
    _PAYLOAD_CAP_ENV: "0",  # the prepacked layout the gradient lane requires
    # the GPU lane's near field and L2P, in interpret mode
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
}
_FUSED_THETA = 0.8


def _fused_solver() -> FastMultipoleMethod:
    """The fused lane's configuration at test size, far pairs retained.

    Returns
    -------
    FastMultipoleMethod
        A new solver: real basis, fp32, static radix tree, theta 0.8, order 4.
    """
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=_FUSED_THETA,
        G=_G,
        softening=_SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=_LEAF),
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
            mac_type="dehnen",
        ),
        fixed_order=_ORDER,
    )


@pytest.mark.xdist_group("large_n_forward_is_the_fused_force")
class TestTheForwardIsTheFusedForce:
    @pytest.fixture(scope="class")
    def forces(self) -> dict[str, Any]:
        """The fused eval, then the differentiable forward on a fresh solver and on
        the production one.

        Returns
        -------
        dict[str, Any]
            The three force arrays, keyed ``production``, ``fresh`` and
            ``same_engine``, and the far-pair count of each lane's walk.
        """
        patcher = pytest.MonkeyPatch()
        for key, value in _FUSED_ENV.items():
            patcher.setenv(key, value)
        try:
            positions = jnp.asarray(_plummer(_N, seed=0), jnp.float32)
            masses = jnp.full((_N,), 1.0 / _N, jnp.float32)
            production = _fused_solver()
            prepared, eval_fn = production.strict_fused_prepared_eval_fn(
                positions=positions,
                masses=masses,
                leaf_size=_LEAF,
                max_order=_ORDER,
                theta=_FUSED_THETA,
            )
            out = {
                "production": np.asarray(eval_fn(prepared)),
                "far_pairs_production": int(
                    np.asarray(prepared.compact_far_pairs.far_pair_count)
                ),
            }
            for arm, fmm in (("fresh", _fused_solver()), ("same_engine", production)):
                state = fmm.prepare_state(
                    positions,
                    masses,
                    leaf_size=_LEAF,
                    max_order=_ORDER,
                    theta=_FUSED_THETA,
                )
                assert isinstance(state, LargeNPreparedState)
                # compiled, as the fused eval is (see _jitted_force)
                out[arm] = np.asarray(_jitted_force(fmm, state, positions, masses))
                out[f"far_pairs_{arm}"] = int(
                    np.asarray(state.compact_far_pairs.far_pair_count)
                )
        finally:
            patcher.undo()
        return out

    def test_the_far_field_is_load_bearing(self, forces):
        assert forces["far_pairs_production"] > 0

    @pytest.mark.parametrize("arm", ["fresh", "same_engine"])
    def test_the_differentiable_forward_is_the_fused_force(self, forces, arm):
        """Bitwise, in either call order (a geometry split is 5.3e-3 here)."""
        got, want = forces[arm], forces["production"]
        differ = int(np.sum(got.view(np.uint32) != want.view(np.uint32)))
        assert differ == 0, (
            f"{arm}: {differ} of {want.size} elements of the differentiable forward "
            f"differ from the fused eval (rel-L2 {_relative_l2(got, want):.3e}), with "
            f"{forces[f'far_pairs_{arm}']} far pairs against the fused lane's "
            f"{forces['far_pairs_production']}"
        )
