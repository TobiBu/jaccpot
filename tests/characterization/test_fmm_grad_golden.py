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
   ``tests/characterization/golden_grad/`` to float64 round-off
   (``rtol=1e-12, atol=1e-12``). A structural refactor MUST NOT move these.
2. **Physics anchor** -- each gradient agrees with ``jax.grad`` of the
   **direct O(N^2) sum** (:func:`differentiable_gravitational_acceleration`, the
   documented gradient oracle) to a measured per-order bound, so a regenerated
   golden can never silently snapshot a wrong gradient.

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
    differentiable_gravitational_acceleration,
)

GOLDEN_DIR = Path(__file__).parent / "golden_grad"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

G_CONST = 1.0
SOFTENING = 1.0e-2

# Inertness tolerance: golden vs recompute must agree to float64 round-off. Same
# values as the forward oracle, for the same reason.
INERT_RTOL = 1.0e-12
INERT_ATOL = 1.0e-12

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

    ``differentiable_gravitational_acceleration`` is the documented gradient oracle
    (ARCHITECTURE §6: "``grad(FMM)`` must match ``grad(direct-sum)`` to FMM force
    accuracy"), so this is an independent reference rather than a second opinion
    from the same code.
    """
    weights = _cotangent(int(positions.shape[0]))

    def loss(pos, mass):
        return jnp.sum(
            weights
            * differentiable_gravitational_acceleration(
                pos, mass, G=G_CONST, softening=SOFTENING
            )
        )

    return jax.grad(loss, argnums=(0, 1))(positions, masses)


def _relative_l2(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300))


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
