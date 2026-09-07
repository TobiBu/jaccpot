"""Shape contracts for the ``delta``/``direction`` family in ``complex_ops``.

Every test here fails on the commit before the annotations landed, and it fails by
*passing*: the call returns a plausible number instead of raising. That is the
whole reason the family was worth a PR. JAX clamps an out-of-bounds index rather
than raising, so ``delta[2]`` on a length-2 array returns ``delta[1]`` and each of
these functions quietly computes the answer for ``(x, y, y)``.

The first test pins that mechanism directly, so if JAX ever changes it the reason
these annotations exist is re-measured rather than assumed.
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaccpot.operators.complex_ops import (
    evaluate_local_complex,
    evaluate_local_complex_derivative_tower,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_with_grad,
    evaluate_local_complex_with_grad_analytic,
    regular_solid_harmonic_directional_derivative,
    regular_solid_harmonic_directional_derivative_order,
    rotate_complex_local_from_z_solidfmm,
    rotate_complex_local_to_z_solidfmm,
    rotate_complex_multipole_from_z_solidfmm,
    rotate_complex_multipole_to_z_solidfmm,
)
from jaccpot.operators.real_harmonics import sh_size

ORDER = 4
DELTA = jnp.asarray([0.35, -0.25, 0.45])
DIRECTION = jnp.asarray([0.31, -0.44, 0.21])


def _local(order: int = ORDER) -> jnp.ndarray:
    """Return a deterministic packed complex expansion.

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    jnp.ndarray
        Packed complex coefficients, length ``sh_size(order)``.
    """
    rng = np.random.default_rng(0)
    size = sh_size(order)
    return jnp.asarray(rng.normal(size=size) + 1j * rng.normal(size=size))


def _call(name: str, delta: jnp.ndarray):
    """Invoke one member of the family with ``delta``.

    Collected here rather than parametrized inline because the eleven signatures
    differ in argument order and in which keywords they require, and the point of
    the test is the shared ``delta`` contract, not those differences.

    Parameters
    ----------
    name : str
        Which family member to call.
    delta : jnp.ndarray
        The displacement to pass.

    Returns
    -------
    Any
        Whatever the named function returns.

    Raises
    ------
    KeyError
        If ``name`` is not a member of the family.
    """
    local = _local()
    table = {
        "evaluate_local_complex": lambda: evaluate_local_complex(
            local, delta, order=ORDER
        ),
        "evaluate_local_complex_with_grad": lambda: evaluate_local_complex_with_grad(
            local, delta, order=ORDER
        ),
        "evaluate_local_complex_derivative_tower": (
            lambda: evaluate_local_complex_derivative_tower(
                local, delta, order=ORDER, max_derivative_order=1
            )
        ),
        "evaluate_local_complex_with_grad_analytic": (
            lambda: evaluate_local_complex_with_grad_analytic(local, delta, order=ORDER)
        ),
        "evaluate_local_complex_grad_analytic": (
            lambda: evaluate_local_complex_grad_analytic(local, delta, order=ORDER)
        ),
        "regular_solid_harmonic_directional_derivative": (
            lambda: regular_solid_harmonic_directional_derivative(
                delta, DIRECTION[: delta.shape[0]], order=ORDER
            )
        ),
        "regular_solid_harmonic_directional_derivative_order": (
            lambda: regular_solid_harmonic_directional_derivative_order(
                delta, DIRECTION[: delta.shape[0]], order=ORDER, derivative_order=2
            )
        ),
        "rotate_complex_multipole_to_z_solidfmm": (
            lambda: rotate_complex_multipole_to_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_multipole_from_z_solidfmm": (
            lambda: rotate_complex_multipole_from_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_local_to_z_solidfmm": (
            lambda: rotate_complex_local_to_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_local_from_z_solidfmm": (
            lambda: rotate_complex_local_from_z_solidfmm(local, delta, order=ORDER)
        ),
    }
    return table[name]()


FAMILY = (
    "evaluate_local_complex",
    "evaluate_local_complex_with_grad",
    "evaluate_local_complex_derivative_tower",
    "evaluate_local_complex_with_grad_analytic",
    "evaluate_local_complex_grad_analytic",
    "regular_solid_harmonic_directional_derivative",
    "regular_solid_harmonic_directional_derivative_order",
    "rotate_complex_multipole_to_z_solidfmm",
    "rotate_complex_multipole_from_z_solidfmm",
    "rotate_complex_local_to_z_solidfmm",
    "rotate_complex_local_from_z_solidfmm",
)


def test_short_delta_is_silently_the_wrong_physics_without_the_contract() -> None:
    """Pin the mechanism that makes the whole family worth annotating.

    A length-2 ``delta`` does not raise inside JAX; it is read as ``(x, y, y)``.
    This asserts that equality on the raw harmonics, so the annotations below rest
    on a measured fact rather than on a claim about indexing.
    """
    from jaccpot.operators.complex_harmonics import complex_R_solidfmm

    short = complex_R_solidfmm(DELTA[:2], order=ORDER)
    clamped = complex_R_solidfmm(
        jnp.asarray([DELTA[0], DELTA[1], DELTA[1]]), order=ORDER
    )
    full = complex_R_solidfmm(DELTA, order=ORDER)

    assert jnp.allclose(short, clamped), "short delta is not read as (x, y, y)"
    assert not jnp.allclose(short, full), "the truncation has to change the answer"


@pytest.mark.parametrize("name", FAMILY)
def test_length_two_delta_is_rejected(name: str) -> None:
    """A truncated spatial vector must raise rather than return a number.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    with pytest.raises(TypeCheckError):
        _call(name, DELTA[:2])


@pytest.mark.parametrize("name", FAMILY)
def test_full_length_delta_still_works(name: str) -> None:
    """The contract must not reject the calls these functions have always taken.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    assert _call(name, DELTA) is not None


@pytest.mark.parametrize("name", FAMILY)
def test_batched_delta_is_rejected(name: str) -> None:
    """A ``(batch, 3)`` delta belongs to the ``*_batch`` variants, not to these.

    Passing one here used to broadcast into a result of the wrong rank rather than
    raising, which is the same failure wearing a different shape.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    with pytest.raises(TypeCheckError):
        _call(name, jnp.stack([DELTA, DELTA]))


def test_direction_is_shaped_independently_of_delta() -> None:
    """``direction`` carries its own contract, not one inherited from ``delta``.

    Both directional-derivative functions take two spatial vectors, so a test that
    only ever truncated ``delta`` would leave the second annotation unexercised --
    it would pass with ``direction`` still bare.
    """
    with pytest.raises(TypeCheckError):
        regular_solid_harmonic_directional_derivative(DELTA, DIRECTION[:2], order=ORDER)
    with pytest.raises(TypeCheckError):
        regular_solid_harmonic_directional_derivative_order(
            DELTA, DIRECTION[:2], order=ORDER, derivative_order=2
        )


def test_batch_variants_keep_taking_batched_deltas() -> None:
    """The ``*_batch`` siblings are deliberately NOT annotated ``'3'``.

    They take ``(batch, 3)`` and ``vmap`` the scalar function over it, so the
    per-example slice the annotation guards is exactly what they feed in. This
    pins that the PR left them alone; annotating them ``'3'`` would break every
    batched caller, and the assertion is cheap insurance against a later sweep
    doing it by pattern-match.
    """
    from jaccpot.operators.complex_ops import (
        evaluate_local_complex_grad_analytic_batch,
        regular_solid_harmonic_directional_derivative_batch,
    )

    deltas = jnp.stack([DELTA, DELTA + 0.1, DELTA - 0.2])
    grads = evaluate_local_complex_grad_analytic_batch(_local(), deltas, order=ORDER)
    assert grads.shape == (3, 3)

    packed = regular_solid_harmonic_directional_derivative_batch(
        deltas, jnp.stack([DIRECTION] * 3), order=ORDER
    )
    assert packed.shape[0] == 3


# ---------------------------------------------------------------------------
# The rest of the spatial-vector family, added when the remaining 24 functions
# were annotated. The single-vector cases repeat the mechanism the tests above
# pin for the first eleven, so they are covered by one parametrised sweep rather
# than eleven near-copies. What gets its own test is what is NEW here: the
# batched `"_ 3"` contract, the measured 1-D `local` on a batch function, and the
# two functions whose reach differs from the rest.
# ---------------------------------------------------------------------------

REMAINDER_SINGLE = (
    "regular_solid_harmonic_gradient_coefficients",
    "m2m_complex",
    "l2l_complex",
    "m2l_complex_reference",
)

REMAINDER_BATCHED = (
    "l2l_complex_batch",
    "m2l_complex_reference_batch",
    "regular_solid_harmonic_directional_derivative_batch",
)


def _call_remainder_single(name, delta):
    """Call one newly annotated single-vector function.

    Parameters
    ----------
    name : str
        Function name in `jaccpot.operators.complex_ops`.
    delta : Array
        The displacement to pass.

    Returns
    -------
    Any
        Whatever the function returns.
    """
    import jaccpot.operators.complex_ops as module

    fn = getattr(module, name)
    if name == "regular_solid_harmonic_gradient_coefficients":
        return fn(delta, order=ORDER)
    if name == "m2m_complex":
        return fn(_local(), delta, order=ORDER)
    return fn(_local(), delta, order=ORDER)


@pytest.mark.parametrize("name", REMAINDER_SINGLE)
def test_remainder_rejects_a_length_two_delta(name):
    """Same clamping mechanism, the functions the first PR did not reach."""
    with pytest.raises(TypeCheckError):
        _call_remainder_single(name, jnp.asarray([1.0, 2.0]))


@pytest.mark.parametrize("name", REMAINDER_SINGLE)
def test_remainder_still_takes_a_full_delta(name):
    """The control for the sweep above."""
    _call_remainder_single(name, DELTA)


@pytest.mark.parametrize("name", REMAINDER_BATCHED)
def test_batched_family_rejects_an_unbatched_delta(name):
    """`"_ 3"` asserts rank 2, so a bare `(3,)` is now caught.

    This is the half the first PR could not add: it left the `*_batch` siblings
    bare precisely to avoid annotating them `"3"`, which would have broken every
    batched caller. `"_ 3"` closes the hole from the other side.
    """
    import jaccpot.operators.complex_ops as module

    fn = getattr(module, name)
    with pytest.raises(TypeCheckError):
        if name == "regular_solid_harmonic_directional_derivative_batch":
            fn(DELTA, DELTA, order=ORDER)
        elif name == "l2l_complex_batch":
            fn(jnp.stack([_local()]), DELTA, order=ORDER)
        else:
            fn(jnp.stack([_local()]), DELTA, order=ORDER)


@pytest.mark.parametrize("name", REMAINDER_BATCHED)
def test_batched_family_still_takes_a_batch(name):
    """The control, and it is the assertion the earlier PR's pin cared about."""
    import jaccpot.operators.complex_ops as module

    fn = getattr(module, name)
    deltas = jnp.stack([DELTA, DELTA + 0.1])
    if name == "regular_solid_harmonic_directional_derivative_batch":
        fn(deltas, deltas, order=ORDER)
    else:
        fn(jnp.stack([_local()] * 2), deltas, order=ORDER)


def test_a_two_component_batch_is_rejected():
    """`(batch, 2)` is the batched form of the silent-physics bug."""
    from jaccpot.operators.complex_ops import l2l_complex_batch

    with pytest.raises(TypeCheckError):
        l2l_complex_batch(
            jnp.stack([_local()] * 2),
            jnp.zeros((2, 2)),
            order=ORDER,
        )


def test_local_stays_one_dimensional_on_a_batch_function():
    """The measured rank that a pattern-match would have got wrong.

    `evaluate_local_complex_grad_analytic_batch` takes ONE expansion and vmaps it
    against many deltas, so `local` is `(ncoeff,)` and not `(batch, ncoeff)`.
    Annotating it `"_ _"` would have rejected every real caller; this pins the 1-D
    form as the contract.
    """
    from jaccpot.operators.complex_ops import evaluate_local_complex_grad_analytic_batch

    deltas = jnp.stack([DELTA, DELTA + 0.1, DELTA - 0.2])
    grads = evaluate_local_complex_grad_analytic_batch(_local(), deltas, order=ORDER)
    assert grads.shape == (3, 3)

    with pytest.raises(TypeCheckError):
        evaluate_local_complex_grad_analytic_batch(
            jnp.stack([_local()] * 3), deltas, order=ORDER
        )


def test_the_padded_rotation_builders_reject_a_short_delta():
    """The invariant half: `Float[Array, '3']` bites here in either check mode."""
    from jaccpot.operators.complex_ops import (
        _complex_rotation_blocks_to_z_solidfmm_padded,
    )

    with pytest.raises(TypeCheckError):
        _complex_rotation_blocks_to_z_solidfmm_padded(
            jnp.asarray([0.3, -0.4]), **_ROT_KW
        )


@pytest.mark.skipif(
    bool(os.environ.get("JACCPOT_RUNTIME_TYPECHECK")),
    reason=(
        "numpy acceptance here is a DEFAULT-MODE property only. Under the "
        "package-wide import hook, `_transverse_degeneracy_jvp`'s own "
        "`builder_without_unresolvable(deltas: Array)` becomes enforced and rejects "
        "numpy before its `jnp.asarray` runs -- so the asymmetry this test records "
        "disappears, and not because of anything in `complex_ops`."
    ),
)
def test_the_padded_rotation_builders_take_a_numpy_delta_in_default_mode():
    """Their reach differs from the rest of the family, and only in default mode.

    `without_unresolvable_transverse_jvp` calls `jnp.asarray(deltas)` before
    delegating, so `Float[Array, '3']` on the wrapped builder sits BEHIND a coercion
    and does not narrow away numpy callers the way the eleven functions the first PR
    annotated do. Pinned so the asymmetry is not "fixed" into a false uniformity --
    and skipped under the import hook, where an unrelated annotation makes it moot.
    """
    import numpy as np

    from jaccpot.operators.complex_ops import (
        _complex_rotation_blocks_to_z_solidfmm_padded,
    )

    out = _complex_rotation_blocks_to_z_solidfmm_padded(
        np.asarray([0.3, -0.4, 1.2]), **_ROT_KW
    )
    assert out.shape[0] > 0


# ---------------------------------------------------------------------------
# The functions that were SILENTLY ACCEPTING a length-2 delta before this
# annotation, measured one at a time rather than assumed. This is the group that
# matters: for the public `m2m_complex` / `l2l_complex` / `m2l_complex_reference`
# and most of the batched family, a bad delta was ALREADY rejected -- transitively,
# because they delegate into the eleven functions the first PR annotated, or by
# `vmap`'s own inconsistent-sizes error. Annotating those moves the complaint to the
# boundary and names the right parameter; it closes no hole, and the tests above are
# boundary pins rather than protection.
#
# These twelve are the protection. Every one returned an answer for `(x, y, y)`.
# ---------------------------------------------------------------------------

_ROT_KW = {"order": ORDER, "basis": "multipole", "dtype": jnp.complex128}


def _previously_silent():
    """Build the call table for the functions that used to accept a short delta.

    Returns
    -------
    list of tuple
        ``(id, callable taking one delta)`` pairs.
    """
    import jaccpot.operators.complex_ops as m

    return [
        (
            "gradient_coefficients",
            lambda d: m.regular_solid_harmonic_gradient_coefficients(d, order=ORDER),
        ),
        (
            "gradient_coefficients_preserve_dtype",
            lambda d: m.regular_solid_harmonic_gradient_coefficients_preserve_dtype(
                d, order=ORDER
            ),
        ),
        (
            "grad_analytic_preserve_dtype",
            lambda d: m.evaluate_local_complex_grad_analytic_preserve_dtype(
                _local(), d, order=ORDER
            ),
        ),
        (
            "grad_order4_unrolled",
            lambda d: m.evaluate_local_complex_grad_order4_unrolled(
                _local(), d, order=ORDER
            ),
        ),
        ("order4_scalars", lambda d: m._regular_solid_harmonic_order4_scalars(d)),
        ("angles_from_delta", lambda d: m._angles_from_delta_solidfmm(d)),
        (
            "harmonic_derivative_coefficients",
            lambda d: m._build_complex_harmonic_derivative_coefficients(
                d, order=ORDER, max_derivative_order=1
            ),
        ),
        (
            "rotation_to_z",
            lambda d: m._complex_rotation_blocks_to_z_solidfmm(d, **_ROT_KW),
        ),
        (
            "rotation_from_z",
            lambda d: m._complex_rotation_blocks_from_z_solidfmm(d, **_ROT_KW),
        ),
        (
            "rotation_to_z_padded",
            lambda d: m._complex_rotation_blocks_to_z_solidfmm_padded(d, **_ROT_KW),
        ),
        (
            "rotation_from_z_padded",
            lambda d: m._complex_rotation_blocks_from_z_solidfmm_padded(d, **_ROT_KW),
        ),
    ]


@pytest.mark.parametrize(
    "call",
    [c for _, c in _previously_silent()],
    ids=[i for i, _ in _previously_silent()],
)
def test_previously_silent_functions_now_reject_a_short_delta(call):
    """Each of these returned the answer for ``(x, y, y)`` before the annotation."""
    with pytest.raises(TypeCheckError):
        call(jnp.asarray([1.0, 2.0]))


@pytest.mark.parametrize(
    "call",
    [c for _, c in _previously_silent()],
    ids=[i for i, _ in _previously_silent()],
)
def test_previously_silent_functions_still_take_a_full_delta(call):
    """The control: the rejection above must not be rejecting everything."""
    call(DELTA)


def test_the_batched_rotation_builders_now_reject_a_two_component_batch():
    """`complex_rotation_blocks_*_solidfmm_batch` accepted `(batch, 2)` silently."""
    from jaccpot.operators.complex_ops import (
        complex_rotation_blocks_from_z_solidfmm_batch,
        complex_rotation_blocks_to_z_solidfmm_batch,
    )

    bad = jnp.zeros((2, 2))
    for fn in (
        complex_rotation_blocks_to_z_solidfmm_batch,
        complex_rotation_blocks_from_z_solidfmm_batch,
    ):
        with pytest.raises(TypeCheckError):
            fn(bad, **_ROT_KW)
        fn(jnp.stack([DELTA, DELTA + 0.1]), **_ROT_KW)


def test_the_rotation_blocks_must_be_square():
    """`blockdim` repeated is what asserts squareness, and nothing else checks it.

    The cached-blocks M2L takes `(batch, cascade, blockdim, blockdim)` tensors --
    measured `(17, 3, 5, 5)`, `(17, 4, 7, 7)` and `(17, 5, 9, 9)`, so the trailing
    pair agrees at three distinct extents. A non-square block used to reach a matmul
    and fail there, with a message naming neither parameter. Repeating the name also
    ties the `to_z` and `from_z` tensors to each other.
    """
    from jaccpot.operators.complex_ops import m2l_complex_reference_batch_cached_blocks

    batch, cascade, dim = 2, ORDER + 1, 2 * ORDER + 1
    multipoles = jnp.stack([_local()] * batch)
    deltas = jnp.stack([DELTA, DELTA + 0.1])
    square = jnp.zeros((batch, cascade, dim, dim), dtype=jnp.complex128)

    m2l_complex_reference_batch_cached_blocks(
        multipoles, deltas, square, square, order=ORDER
    )

    oblong = jnp.zeros((batch, cascade, dim, dim + 1), dtype=jnp.complex128)
    with pytest.raises(TypeCheckError):
        m2l_complex_reference_batch_cached_blocks(
            multipoles, deltas, oblong, square, order=ORDER
        )

    # And the two tensors must agree with each other, not merely each be square.
    other = jnp.zeros((batch, cascade, dim + 2, dim + 2), dtype=jnp.complex128)
    with pytest.raises(TypeCheckError):
        m2l_complex_reference_batch_cached_blocks(
            multipoles, deltas, square, other, order=ORDER
        )


# ---------------------------------------------------------------------------
# The nine parameters the pilot's LAST sweep of this module supports annotating,
# and one test for the family it does NOT support.
#
# Every assertion below corresponds to a perturbation the pilot reported ACCEPTED
# and that a rank or literal annotation actually closes. The z-translation family
# is represented by a single test in the other direction: it pins that a short
# coefficient buffer is still accepted, because no axis spec can express
# `sh_size(order)` against a static `order`, and pretending otherwise would hide a
# real hole behind a green test.
# ---------------------------------------------------------------------------


def test_a_batched_coefficient_block_must_stay_two_dimensional():
    """`enforce_conjugate_symmetry_batch` accepted a flattened and a 3-D block."""
    from jaccpot.operators.complex_ops import enforce_conjugate_symmetry_batch

    good = jnp.zeros((4, 9), dtype=jnp.complex128)
    enforce_conjugate_symmetry_batch(good, order=2)

    for bad in (good[None], good.reshape(-1)):
        with pytest.raises(TypeCheckError):
            enforce_conjugate_symmetry_batch(bad, order=2)


def test_complex_dot_rejects_an_unsqueezed_operand():
    """Both operands accepted an extra leading axis, independently."""
    from jaccpot.operators.complex_ops import complex_dot

    a = jnp.zeros((16,), dtype=jnp.complex128)
    complex_dot(a, a, order=3, conjugate_left=True)

    with pytest.raises(TypeCheckError):
        complex_dot(a[None], a, order=3, conjugate_left=True)
    with pytest.raises(TypeCheckError):
        complex_dot(a, a[None], order=3, conjugate_left=True)


def test_complex_dot_still_takes_operands_of_different_lengths():
    """They are NOT tied to each other, and that is deliberate.

    Both operands are sliced to `sh_size(order)` independently, so a longer buffer
    on either side is legitimate. A shared axis name would have looked tidier and
    rejected callers that work -- the measurement showed `(16,)` and `(25,)` in
    different calls, never a requirement that the two agree.
    """
    from jaccpot.operators.complex_ops import complex_dot

    short = jnp.zeros((16,), dtype=jnp.complex128)
    long = jnp.zeros((25,), dtype=jnp.complex128)
    complex_dot(short, long, order=3, conjugate_left=True)
    complex_dot(long, short, order=3, conjugate_left=True)


def test_the_velocity_contraction_takes_a_three_vector():
    """`velocity` is the `delta`-family literal again: `(1, 3)` was accepted."""
    from jaccpot.operators.complex_ops import (
        contract_spatial_derivative_with_velocity,
    )

    packed = jnp.zeros((6,))
    velocity = jnp.asarray([1.0, 2.0, 3.0])
    contract_spatial_derivative_with_velocity(packed, velocity, order=1)

    with pytest.raises(TypeCheckError):
        contract_spatial_derivative_with_velocity(packed, velocity[None], order=1)
    with pytest.raises(TypeCheckError):
        contract_spatial_derivative_with_velocity(packed, velocity[:2], order=1)


def test_a_rotation_block_tuple_must_hold_matrices():
    """`tuple[Inexact[Array, '_ _'], ...]`, and what it does NOT guarantee.

    beartype checks a VARIADIC container by sampling -- its default O(1) strategy
    inspects roughly one element per call, measured 8/40, 11/40 and 14/40 for a
    single bad element at each of three positions. A fixed-length
    `tuple[X, X, X]` is checked exhaustively (40/40 at every position); only the
    `...` form samples.

    So this asserts the reliable part: a tuple whose elements are systematically
    the wrong rank is rejected every time, because whichever element gets sampled
    is bad. A single corrupted element is caught only sometimes, which is why the
    first version of this test was flaky in the full suite and passed standalone.

    The elements are square but of DIFFERENT sizes -- `2*ell + 1` a side, per
    degree -- so no axis is shared between them. A shared name would assert every
    degree had the same block size, which is false by construction.
    """
    from jaccpot.operators.complex_ops import _blocks_to_padded_array

    blocks = tuple(
        jnp.zeros((2 * ell + 1, 2 * ell + 1), dtype=jnp.complex128) for ell in range(3)
    )
    _blocks_to_padded_array(blocks, order=2, dtype=jnp.complex128)

    all_flattened = tuple(b.reshape(-1) for b in blocks)
    for _ in range(8):  # sampling is random; a systematic error is caught regardless
        with pytest.raises(TypeCheckError):
            _blocks_to_padded_array(all_flattened, order=2, dtype=jnp.complex128)


# `test_a_short_coefficient_buffer_is_still_silently_accepted` USED TO LIVE HERE.
# It pinned the hole this module's note described -- a coefficient buffer one entry
# short returning wrong numbers -- and was written to FAIL the day a length check
# landed, so that the fix could not inherit a quietly passing test. It did exactly
# that: PR #281 added `_require_packed_length` and the pin failed with
# `ValueError: multipole is too short for order 4: got length 24, need at least 25`.
# Removed rather than inverted, because #281 ships
# `test_complex_ops_packed_length_contracts.py` and duplicating 200 lines of coverage
# buys nothing.
