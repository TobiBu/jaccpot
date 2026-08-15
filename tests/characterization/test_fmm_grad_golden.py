"""Golden-output characterization oracle for the differentiable FMM gradient.

Sibling of :mod:`tests.characterization.test_fmm_golden`, which snapshots the
forward accelerations. This one snapshots the **reverse pass**, because the forward
oracle cannot see it: a change that preserves potentials and accelerations exactly
while breaking the VJP is invisible to it, and that is precisely the failure
``NUMERICS_AND_JAX.md`` §1 calls "a broken change".

``NUMERICS_AND_JAX.md`` §3 requires this before refactoring thinly-guarded code
("extend the characterization suite first, in its own commit ... **including a
gradient golden, not only a forward one**") and §6 item 2 asks for
``tests/characterization/`` to be verified unmoved "including gradient goldens".
Until this file existed there were none.

**Why a golden and not another gradient-correctness test.** The suite already has
good gradient *correctness* coverage --
``tests/unit/test_gradient_correctness.py`` (FD vs AD),
``tests/integration/test_grad_fmm_vs_directsum.py`` (grad(FMM) vs grad(direct sum)),
``tests/unit/test_custom_vjp_parity.py`` (analytic rules vs autodiff). Those bound
the gradient to ~1e-4..1e-9. A refactor that reassociates a reduction moves the
last two or three digits, which no 1e-4 comparison can see. A golden pinned at
float64 round-off can.

On every run it re-computes and asserts:

1. **Inertness gate** -- gradients match the committed ``.npz`` under
   ``tests/characterization/golden_grad/`` to float64 round-off. A structural
   refactor MUST NOT move these. It has two forms:

   a. **per particle**, on every device: the relative error of each particle's
      gradient *vector* is under ``INERT_PER_PARTICLE_RTOL`` (1e-12). This is the
      portable bound, and it is the one that means something on a GPU -- see that
      constant for why an elementwise relative tolerance is the wrong norm for a
      vector quantity, and ``docs/g9_grad_golden_gpu_diagnosis.md`` for the
      cross-device failure that established it.
   b. **elementwise**, on CPU only: the stricter
      ``rtol=atol=1e-12`` form, which additionally catches an error confined to one
      small component of one particle. The goldens are CPU-generated and hold there
      with ~130x margin, so this is asserted where it is a real claim rather than
      widened into a formality elsewhere.

2. **Physics anchor** -- each gradient agrees with ``jax.grad`` of the
   **direct O(N^2) sum** (:func:`direct_sum_gravitational_acceleration`, the
   documented gradient oracle) to a measured per-order bound, so a regenerated
   golden can never silently snapshot a wrong gradient.

:func:`test_per_particle_gate_still_rejects_a_scaled_reverse_rule` guards gate 1a
against becoming a formality, by mutation rather than by argument.

Regenerate intentionally with ``JACCPOT_REGEN_GOLDEN=1 pytest ...`` -- the same
switch the forward oracle uses, so one command refreshes both -- then commit the
new ``.npz`` files.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.autodiff import (  # noqa: E402
    direct_sum_gravitational_acceleration,
)

GOLDEN_DIR = Path(__file__).parent / "golden_grad"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

G_CONST = 1.0
SOFTENING = 1.0e-2

# Inertness tolerance: golden vs recompute must agree to float64 round-off. Same
# values as the forward oracle, for the same reason.
INERT_RTOL = 1.0e-12
INERT_ATOL = 1.0e-12

# The device-portable inertness bound: the relative error of each particle's
# gradient, measured per particle rather than per array element.
#
# Why a second statistic exists. ``grad_positions`` is a 3-vector per particle, and
# its round-off is proportional to the magnitude of the *vector* -- the three
# components are formed by summing terms of the same size, so they inherit the same
# absolute error. An elementwise relative tolerance divides that shared absolute
# error by each component's own value, so a component that happens to be small
# relative to its own vector reports a huge relative error while nothing is wrong.
#
# That is what audit G.9's cross-device failure turned out to be
# (``docs/g9_grad_golden_gpu_diagnosis.md``). On an A100, ``clu_real_n128_p4``
# particle 57 has gradient (-1.689e5, +8.747e4, -2.619e2): the 6th largest vector
# of 128, with a z-component 726x smaller than the vector norm. The absolute drift
# is ~2e-9 on all three components; divided by |z| it reads 7.94e-12 and breaks the
# 1e-12 gate, while divided by the vector norm it is 1.09e-14. Diagnosed as
# reassociation, not a defect: the accepted M2L set is bit-identical across devices,
# ``--xla_gpu_deterministic_ops=true`` changes the number only in its 5th digit, and
# the particle sits 6-7 orders away from any transverse-degeneracy guard.
#
# Measured max per-particle relative error, all six cases, jax 0.10.2:
#
#     grad_positions   CPU 1.4e-15 (6 ULP)    A100 1.7e-14 (77 ULP)
#     grad_masses      CPU 2.6e-16 (1 ULP)    A100 2.0e-13 (898 ULP)
#
# 1e-12 keeps 58x margin on positions and 5x on masses over the worst A100 case.
# For ``grad_masses`` this statistic *is* the elementwise one -- a scalar per
# particle has no components to normalise across -- so nothing is loosened there;
# it is written once and applied to both arrays for symmetry.
INERT_PER_PARTICLE_RTOL = 1.0e-12

# Physics anchor: grad(FMM) vs grad(direct sum). Keyed by expansion order, because
# the gradient inherits the forward truncation error and a single bound would waste
# the sharpness available at order 4 -- the same argument that made the forward
# oracle's anchor per-basis rather than global.
#
# Measured relative L2 across the six cases below (grad w.r.t. positions / masses):
#
#     order 2:  7.84e-04 / 2.62e-03
#     order 4:  1.74e-04 .. 2.63e-04 / 5.50e-04 .. 1.56e-03
#
# The mass gradient is consistently the looser of the two (up to ~2x the forward
# force error, versus ~0.3x for positions), so it gets its own bound rather than
# being folded into a single number that the position bound would then not police.
# Bounds keep ~6-8x headroom on every entry, and order 4's are 2.5x tighter than
# order 2's, so raising the expansion order still has to buy something.
ANCHOR_REL_L2_BY_ORDER = {
    2: {"positions": 5.0e-3, "masses": 2.0e-2},
    4: {"positions": 2.0e-3, "masses": 1.0e-2},
}

# (id, distribution, N, basis, order). Deliberately smaller than the forward grid:
# each case compiles a full reverse-mode program at ~6-23 s and ~1.1-1.5 GB peak,
# and the cost is the retained executable, not the tape (see the
# `_bound_diff_fmm_compile_cache` note in tests/conftest.py -- this module is
# registered there). The axes kept are the ones a refactor can break:
#
#   * basis real vs complex -- the two share this grad path through different
#     operator families, and their gradients agree to all printed digits, so a
#     divergence is a real signal
#   * order 2 vs 4 -- convergence: the anchor must tighten with order, or the
#     far-field reverse is not actually being exercised
#   * uniform vs clustered -- the clustered case is what stresses the MAC
#   * N 128 vs 256 -- the N=256 uniform system is bit-identical to the forward
#     oracle's `uni_*_n256_*` particles (same PRNG key), so the two goldens
#     constrain the same system from both directions
CASES = [
    ("uni_real_n128_p2", "uniform", 128, "real", 2),
    ("uni_real_n128_p4", "uniform", 128, "real", 4),
    ("uni_complex_n128_p2", "uniform", 128, "complex", 2),
    ("uni_complex_n128_p4", "uniform", 128, "complex", 4),
    ("clu_real_n128_p4", "clustered", 128, "real", 4),
    ("uni_real_n256_p4", "uniform", 256, "real", 4),
]

# leaf_size=4 is not a performance choice and must not be "tidied" upward. At the
# default leaf size these systems are too shallow for the MAC to accept any box
# pair, the M2L interaction list is EMPTY, and the whole far-field reverse
# (M2M/M2L/L2L) is never traced -- the golden would silently degrade into a
# near-field-only snapshot while still passing.
# `test_grad_golden_leaf_size_is_what_makes_the_far_field_nonempty` pins that, and
# the count is asserted in the golden test itself.
LEAF_SIZE = 4


def _make_inputs(distribution: str, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic positions in a bounded box + positive masses.

    Mirrors ``test_fmm_golden._make_inputs`` exactly, including the PRNG key, so a
    shared ``(distribution, n)`` produces the same particles in both oracles.
    Duplicated rather than imported because the forward oracle's helper returns
    NumPy for its NumPy reference sum, while this one stays in JAX for ``jax.grad``.
    """
    key = jax.random.PRNGKey(0xC0FFEE)
    k_pos, k_mass, k_blob = jax.random.split(key, 3)
    dtype = jnp.float64
    if distribution == "uniform":
        positions = jax.random.uniform(
            k_pos, (n, 3), dtype=dtype, minval=-1.0, maxval=1.0
        )
    elif distribution == "clustered":
        n_blobs = 4
        centers = jax.random.uniform(
            k_blob, (n_blobs, 3), dtype=dtype, minval=-0.8, maxval=0.8
        )
        assign = jax.random.randint(k_pos, (n,), 0, n_blobs)
        jitter = 0.08 * jax.random.normal(k_mass, (n, 3), dtype=dtype)
        positions = jnp.clip(centers[assign] + jitter, -1.0, 1.0)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown distribution {distribution!r}")
    masses = jnp.abs(jax.random.normal(k_mass, (n,), dtype=dtype)) + 0.5
    return positions, masses


def _cotangent(n: int) -> jnp.ndarray:
    """Fixed output cotangent ``[n, 3]`` for the scalar loss.

    The loss is ``sum(w * a)``, a plain VJP with a constant cotangent, rather than
    the ``sum(a**2)`` the correctness tests use. ``sum(a**2)`` weights each
    component by its own value, so a component that is small in the forward pass
    contributes little to the gradient even when its reverse rule is wrong. A fixed
    random ``w`` weights every component of the reverse pass comparably, which is
    what a tripwire wants.
    """
    return jax.random.normal(jax.random.PRNGKey(7), (n, 3), dtype=jnp.float64)


def _num_far_pairs(state) -> int:
    """Number of accepted M2L interactions in the frozen topology.

    Zero means the far-field reverse path is never traced. Same helper as
    ``tests/unit/test_gradient_correctness.py``; kept local so this module has no
    cross-tier import.
    """
    interactions = state.interactions
    if interactions is not None:
        return int(jnp.sum(interactions.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


def _fmm_gradients(basis: str, order: int, positions, masses):
    """``grad`` of ``sum(w * accel)`` w.r.t. positions and masses, plus the M2L count.

    Uses the fixed-topology contract: ``prepare_state`` is called once,
    concretely, outside the traced region (it is not traceable), and
    ``differentiable_accelerations`` re-evaluates the numeric pipeline on the live
    inputs. ``use_pallas=False`` keeps this on the pure-JAX path so the golden is
    reproducible on CPU CI.
    """
    fmm = FastMultipoleMethod(
        basis=basis,
        use_pallas=False,
        theta=0.5,
        G=G_CONST,
        softening=SOFTENING,
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=LEAF_SIZE)
    weights = _cotangent(int(positions.shape[0]))

    def loss(pos, mass):
        return jnp.sum(weights * fmm.differentiable_accelerations(state, pos, mass))

    grad_positions, grad_masses = jax.grad(loss, argnums=(0, 1))(positions, masses)
    return grad_positions, grad_masses, _num_far_pairs(state)


def _direct_sum_gradients(positions, masses):
    """The same loss differentiated through the direct O(N^2) sum.

    ``direct_sum_gravitational_acceleration`` is the documented gradient oracle
    (ARCHITECTURE §6: "``grad(FMM)`` must match ``grad(direct-sum)`` to FMM force
    accuracy"), so this is an independent reference rather than a second opinion
    from the same code.
    """
    weights = _cotangent(int(positions.shape[0]))

    def loss(pos, mass):
        return jnp.sum(
            weights
            * direct_sum_gravitational_acceleration(
                pos, mass, G=G_CONST, softening=SOFTENING
            )
        )

    return jax.grad(loss, argnums=(0, 1))(positions, masses)


def _relative_l2(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300))


def _per_particle_relative_error(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Relative error of each particle's gradient, normalised by its own magnitude.

    For ``grad_positions`` (shape ``[n, 3]``) that magnitude is the 3-vector norm,
    so the three components share one denominator instead of each dividing the
    shared absolute round-off by its own value. For ``grad_masses`` (shape ``[n]``)
    it reduces to the ordinary elementwise relative error. See
    :data:`INERT_PER_PARTICLE_RTOL`.

    Parameters
    ----------
    got : np.ndarray
        Recomputed gradient, shape ``[n, 3]`` or ``[n]``.
    want : np.ndarray
        Committed golden of the same shape.

    Returns
    -------
    np.ndarray
        Per-particle relative error, shape ``[n]``.

    Raises
    ------
    AssertionError
        If ``got`` and ``want`` do not have the same shape.
    """
    if got.shape != want.shape:
        raise AssertionError(f"shape mismatch: {got.shape} vs {want.shape}")
    # axis=() for a 1-D input, which numpy treats as "reduce nothing" -- each
    # element is already its own particle. Do NOT collapse the empty tuple to None:
    # that would sum the whole array to a scalar.
    axis = tuple(range(1, want.ndim))
    magnitude = np.sqrt(np.sum(np.square(want), axis=axis))
    residual = np.sqrt(np.sum(np.square(got - want), axis=axis))
    # 1e-300 rather than 0 so an all-zero particle yields 0.0, not a NaN. A genuine
    # zero gradient with a nonzero residual still reports a huge error, which is
    # the intended behaviour.
    return residual / np.maximum(magnitude, 1e-300)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="gradient golden characterization requires float64 (JAX_ENABLE_X64=1)",
)
@pytest.mark.parametrize(
    ("case_id", "distribution", "n", "basis", "order"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fmm_grad_golden(
    case_id: str, distribution: str, n: int, basis: str, order: int
) -> None:
    """Reverse pass of the fixed-topology FMM is inert and physically anchored."""
    positions, masses = _make_inputs(distribution, n)
    grad_positions, grad_masses, num_far_pairs = _fmm_gradients(
        basis, order, positions, masses
    )

    # Vacuity gate, before anything else: with an empty M2L list this case would
    # snapshot a near-field-only gradient and still pass both gates below.
    assert num_far_pairs > 0, (
        f"{case_id}: no M2L pairs were accepted, so the far-field reverse path is "
        f"not exercised and this golden would be vacuous (leaf_size={LEAF_SIZE})"
    )

    grad_positions = np.asarray(grad_positions, dtype=np.float64)
    grad_masses = np.asarray(grad_masses, dtype=np.float64)
    assert np.all(np.isfinite(grad_positions))
    assert np.all(np.isfinite(grad_masses))

    # Physics anchor: never trust a golden that is grossly wrong.
    reference_positions, reference_masses = _direct_sum_gradients(positions, masses)
    bounds = ANCHOR_REL_L2_BY_ORDER[order]
    for label, got, want in (
        ("positions", grad_positions, np.asarray(reference_positions)),
        ("masses", grad_masses, np.asarray(reference_masses)),
    ):
        rel_l2 = _relative_l2(got, want)
        assert rel_l2 < bounds[label], (
            f"{case_id}: grad(FMM) vs grad(direct sum) w.r.t. {label} is "
            f"rel-L2 {rel_l2:.3e}, over the order-{order} anchor {bounds[label]}"
        )

    path = GOLDEN_DIR / f"{case_id}.npz"
    if REGEN or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, grad_positions=grad_positions, grad_masses=grad_masses
        )
        if not REGEN:
            pytest.skip(f"generated missing gradient golden {path.name} (commit it)")
        return

    golden = np.load(path)

    # Gate 1a, every device: the per-particle relative error. This is the bound that
    # is portable across devices, and it is tight -- see INERT_PER_PARTICLE_RTOL.
    for label, got in (
        ("grad_positions", grad_positions),
        ("grad_masses", grad_masses),
    ):
        drift = _per_particle_relative_error(got, golden[label])
        worst = int(np.argmax(drift))
        assert drift[worst] < INERT_PER_PARTICLE_RTOL, (
            f"{case_id}: {label} drifted from the committed golden -- "
            f"per-particle relative error {drift[worst]:.3e} > "
            f"{INERT_PER_PARTICLE_RTOL:.0e} at particle {worst} "
            f"(backend={jax.default_backend()})"
        )

    # Gate 1b, CPU only: the stricter elementwise form, which additionally catches
    # an error confined to one small component of one particle -- a sensitivity
    # gate 1a gives up below ~1e-10 on that component. The goldens were generated
    # on CPU and hold there with ~130x margin (measured 7.8e-15 against 1e-12), so
    # this is asserted where it is a real claim.
    #
    # This is NOT the same as loosening the gate off CPU: off-CPU coverage is gate
    # 1a at a full 1e-12 on a physically meaningful statistic, not a widened
    # elementwise band. G.9 option (2), not option (3).
    if jax.default_backend() != "cpu":
        return
    for label, got in (
        ("grad_positions", grad_positions),
        ("grad_masses", grad_masses),
    ):
        np.testing.assert_allclose(
            got,
            golden[label],
            rtol=INERT_RTOL,
            atol=INERT_ATOL,
            err_msg=f"{case_id}: {label} drifted from the committed golden",
        )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="requires float64 (JAX_ENABLE_X64=1)",
)
def test_grad_golden_leaf_size_is_what_makes_the_far_field_nonempty() -> None:
    """``LEAF_SIZE`` is load-bearing: the default leaf size gives zero M2L pairs.

    This is the trap the vacuity gate in :func:`test_fmm_grad_golden` guards, pinned
    directly so the reason ``LEAF_SIZE = 4`` cannot be "cleaned up" without a test
    going red. Raising it to 16 on the smallest case empties the interaction list,
    at which point the gradient golden would still pass while covering only the
    near field.
    """
    positions, masses = _make_inputs("uniform", 128)
    fmm = FastMultipoleMethod(
        basis="real", use_pallas=False, theta=0.5, G=G_CONST, softening=SOFTENING
    )
    deep = fmm.prepare_state(positions, masses, max_order=2, leaf_size=LEAF_SIZE)
    shallow = fmm.prepare_state(positions, masses, max_order=2, leaf_size=16)
    assert _num_far_pairs(deep) > 0
    assert _num_far_pairs(shallow) == 0, (
        "if a shallow tree now accepts M2L pairs, the vacuity gate is no longer "
        "guarding anything and LEAF_SIZE can be revisited"
    )


def test_per_particle_gate_still_rejects_a_scaled_reverse_rule() -> None:
    """The portable gate must not have become a formality.

    :data:`INERT_PER_PARTICLE_RTOL` normalises by the particle's gradient magnitude
    rather than by each component, which is what makes it device-portable. That
    trade is only acceptable if it still rejects the regression this module exists
    to catch, so the mutation is applied here rather than argued about: scaling the
    analytic real L2P reverse rule by ``1 + 1e-6`` is the perturbation
    ``ARCHITECTURE.md`` section 9 records as invisible to the *forward* golden and
    fatal to this one.

    Applied to the committed goldens it moves the gradient 1e-6 relative, six
    orders above the bound. Also checked: a perturbation confined to a single
    component of a single particle, which is the case the per-particle norm is
    least sensitive to, is still caught at 1e-9 -- and 1e-9 is two orders below the
    A100's own reproducibility floor at that element (7.9e-12), so nothing
    detectable is being given up off CPU.
    """
    paths = sorted(GOLDEN_DIR.glob("*.npz"))
    assert paths, "no gradient goldens found"

    for path in paths:
        golden = np.load(path)["grad_positions"]

        scaled = golden * (1.0 + 1.0e-6)
        drift = _per_particle_relative_error(scaled, golden).max()
        assert drift > INERT_PER_PARTICLE_RTOL, (
            f"{path.stem}: a 1e-6 scaling of the reverse rule slips through the "
            f"per-particle gate ({drift:.3e} <= {INERT_PER_PARTICLE_RTOL:.0e}); it "
            "has stopped being a tripwire"
        )

    # The single-component case, on the case that motivated the gate.
    golden = np.load(GOLDEN_DIR / "clu_real_n128_p4.npz")["grad_positions"]
    smallest = np.unravel_index(
        np.argmax(np.linalg.norm(golden, axis=1, keepdims=True) / np.abs(golden)),
        golden.shape,
    )
    for relative in (1.0e-6, 1.0e-9):
        mutated = golden.copy()
        mutated[smallest] *= 1.0 + relative
        drift = _per_particle_relative_error(mutated, golden).max()
        assert drift > INERT_PER_PARTICLE_RTOL, (
            f"a {relative:.0e} perturbation of the smallest component of particle "
            f"{smallest[0]} slips through the per-particle gate ({drift:.3e})"
        )
