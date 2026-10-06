"""The ``jnp.searchsorted`` method of the hot paths, chosen per backend.

On a GPU the default ``jnp.searchsorted`` is a while loop that launches one small
kernel per bisection step inside the fused step; ``method="scan_unrolled"`` is the
same search as straight-line code XLA fuses (76a5ade: one card at 2e5 10.80 ->
10.12 ms, four cards at 8e5 22.1 -> 21.0 ms, forces identical).

On the CPU backend of jax 0.11.2 the unrolled form sends LLVM's loop vectorizer into
a recursion (``llvm::vputils::onlyFirstLaneUsed``) whose compile time grows ~4x per
doubling of the searched array: 13.5 s against 0.3 s for the other methods at 32k
entries, and 44 minutes for one unit test that compiles in 35 s on 0.10.2. A CPU
has no launch overhead to save, so it gets the plain ``"scan"``. The result is the
same integer index either way.
"""

from __future__ import annotations

import jax

__all__ = ["searchsorted_method"]


def searchsorted_method() -> str:
    """``"scan_unrolled"`` on accelerators, ``"scan"`` on the CPU backend.

    Read at trace time from the default backend.

    Returns
    -------
    str
        The ``method`` to pass to ``jnp.searchsorted``.
    """
    return "scan" if jax.default_backend() == "cpu" else "scan_unrolled"
