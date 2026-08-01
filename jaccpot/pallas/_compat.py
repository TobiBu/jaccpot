"""Pallas API compatibility across the JAX 0.9.1 ``pallas_call`` change.

JAX <= 0.9.0.x selected the GPU lowering with ``pallas_call(backend="triton")``.
In 0.9.1 that keyword was **removed** and the choice is implied by the type of
``compiler_params``: :class:`jax.experimental.pallas.triton.CompilerParams` means
Triton, ``pallas.mosaic_gpu.CompilerParams`` means Mosaic-GPU. Passing
``backend=`` to 0.9.1 raises::

    TypeError: pallas_call() got an unexpected keyword argument 'backend'

Neither the removal nor its replacement is mentioned in the JAX changelog, so the
boundary here is measured: ``backend`` is present in ``pallas_call``'s signature
on 0.9.0.1 and absent on 0.9.1.

This module exists so the M2L kernels can support both without version branches at
each call site, and so the fix is in one place when the floor eventually rises past
0.9.1 and the shim can be deleted.

Why the kernels name a backend rather than taking the default: Mosaic-GPU
**rejects** them. The real M2L uses small per-(pair, coeff) tiles its TMA copies
cannot express, and the complex kernel needs fp64. Naming Triton also keeps them
correct if a caller sets ``JAX_PALLAS_USE_MOSAIC_GPU``, which on 0.9.0.x flips the
default lowering for any ``pallas_call`` that does not name a backend.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from jax.experimental.pallas import pallas_call
from jax.experimental.pallas import triton as plgpu

#: Whether the installed JAX still accepts ``pallas_call(backend=...)``
#: (True on <= 0.9.0.x, False from 0.9.1 on).
PALLAS_CALL_TAKES_BACKEND = "backend" in inspect.signature(pallas_call).parameters

__all__ = ["PALLAS_CALL_TAKES_BACKEND", "pallas_backend_kwargs"]


def pallas_backend_kwargs(
    backend: Optional[str], interpret: bool = False
) -> dict[str, Any]:
    """``pallas_call`` kwargs that select ``backend``, on either JAX API.

    Parameters
    ----------
    backend : Optional[str]
        The Pallas GPU lowering, as these kernels have always spelled it:
        ``"triton"``, or ``None`` to leave the choice to JAX. Only Triton is
        supported here -- see the module docstring.
    interpret : bool
        Whether the caller passes ``interpret=True``. Interpret mode runs CPU
        semantics with no backend lowering, so no backend kwarg is emitted --
        exactly what the call sites did before by passing ``backend=None`` there.

    Returns
    -------
    dict[str, Any]
        Splat into the ``pallas_call``: ``{"backend": ...}`` on the old API,
        ``{"compiler_params": ...}`` on the new one, ``{}`` under interpret.

    Raises
    ------
    NotImplementedError
        If a non-Triton backend is requested on the new API, where it could not be
        honoured. Better to fail loudly than to lower a kernel Mosaic cannot run.
    """

    if interpret:
        return {}
    if PALLAS_CALL_TAKES_BACKEND:
        return {"backend": backend}
    if backend in (None, "triton"):
        # Naming Triton's CompilerParams IS how the backend is chosen from 0.9.1
        # on; an empty instance carries no tuning, matching what ``backend=``
        # conveyed by itself.
        return {"compiler_params": plgpu.CompilerParams()}
    raise NotImplementedError(
        f"backend={backend!r} cannot be selected on this JAX: pallas_call no "
        "longer takes a `backend` kwarg (removed in 0.9.1) and only Triton is "
        "supported by jaccpot's Pallas kernels. Pass backend='triton' or None."
    )
