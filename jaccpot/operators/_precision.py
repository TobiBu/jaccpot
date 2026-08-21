"""One place to pin fp32 matmul precision for the expansion operators.

WHY THIS EXISTS. XLA lowers fp32 matmuls on Ampere+ to **TF32** (~10-bit
mantissa) by default. The rotation/translation algebra in this package is built
from small dense matmuls -- Wigner rotations, the involutory B-matrices, the
align-to-z blocks -- and at TF32 that caps M2L relative accuracy at ~6e-04 from
order 4 upwards, *regardless of expansion order* (measured against a float64
reference, real basis: 5.7e-04 at p=4, 5.7e-04 at p=6, 5.6e-04 at p=8; under
highest precision the same cases give 1.5e-06 / 2.4e-06 / 1.8e-06). Raising the
order past 4 therefore bought nothing in fp32 before this was pinned.

WHY A DECORATOR RATHER THAN ``precision=``. Most of these matmuls are written
with the ``@`` operator, which takes no ``precision`` argument -- 53 of them
across 19 functions. Rewriting dense matrix algebra into
``jnp.matmul(a, b, precision=...)`` calls would obscure the math it expresses, and
would silently miss any ``@`` added later. Entering
:func:`jax.default_matmul_precision` for the whole function body covers every
matmul inside it, including ``@``, and keeps the algebra readable.

The context is entered when the body runs, which under ``jax.jit`` is trace time,
so the precision is baked into the jaxpr -- there is no per-call cost.

COST. Full-precision fp32 matmul is slower than TF32 (XLA emits a multi-pass
algorithm). These are small blocks, not the large GEMMs, but if a future profile
shows this on the critical path the honest fix is to make it preset-dependent
(ACCURATE -> highest, FAST -> default) rather than to quietly drop back to TF32
and reintroduce a ~6e-04 accuracy floor.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import jax
from jax import lax

#: The precision these operators are pinned to, also usable as an explicit
#: ``precision=`` argument for ``jnp.einsum``/``jnp.dot`` call sites.
HIGHEST = lax.Precision.HIGHEST

_F = TypeVar("_F", bound=Callable[..., Any])

__all__ = ["HIGHEST", "highest_matmul_precision"]


def highest_matmul_precision(fn: _F) -> _F:
    """Run ``fn``'s body with fp32 matmuls pinned to full precision (not TF32).

    Apply to any function whose body contains ``@`` matmuls on expansion
    coefficients or rotation blocks. See the module docstring for the measured
    accuracy this protects.

    Parameters
    ----------
    fn : _F
        The function to wrap. Only matmuls executed *inside* ``fn``'s body are
        covered — a callable that ``fn`` merely returns for someone else to call
        runs outside the context and stays at the default precision.

    Returns
    -------
    _F
        ``fn`` wrapped by :func:`functools.wraps`, so name, docstring, and
        ``__wrapped__`` survive. The ``TypeVar`` is what preserves the signature
        for type checkers; at runtime this is an ordinary ``*args, **kwargs``
        wrapper, so it is not itself a ``jax`` transformation and composes with
        ``jit``/``grad`` in either order.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with jax.default_matmul_precision("highest"):
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
