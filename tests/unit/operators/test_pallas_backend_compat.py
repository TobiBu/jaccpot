"""The pallas_call backend-selection shim, across the JAX 0.9.1 API change.

``pallas_call(backend="triton")`` was removed in JAX 0.9.1 and replaced by
implying the backend from the ``compiler_params`` dataclass type. These tests pin
the shim's contract on *both* sides of that change, so a future JAX bump cannot
quietly turn Triton selection into "whatever the default is" -- which for these
kernels would mean Mosaic-GPU, which rejects them.

No GPU needed: the shim only inspects a signature and builds kwargs.
"""

from __future__ import annotations

import pytest

from jaccpot.pallas import _compat
from jaccpot.pallas._compat import PALLAS_CALL_TAKES_BACKEND, pallas_backend_kwargs


def test_interpret_emits_no_backend_kwarg():
    """Interpret mode runs CPU semantics -- naming a backend is meaningless.

    Matches what the call sites did before the port (``backend=None`` there).
    """
    assert pallas_backend_kwargs("triton", interpret=True) == {}
    assert pallas_backend_kwargs(None, interpret=True) == {}


def test_selects_triton_on_whichever_api_is_installed():
    """The kwargs must actually select Triton, by either mechanism."""
    kwargs = pallas_backend_kwargs("triton")
    if PALLAS_CALL_TAKES_BACKEND:  # JAX <= 0.9.0.x
        assert kwargs == {"backend": "triton"}
    else:  # JAX >= 0.9.1
        from jax.experimental.pallas import triton as plgpu

        assert set(kwargs) == {"compiler_params"}
        assert isinstance(kwargs["compiler_params"], plgpu.CompilerParams)


def test_kwargs_are_accepted_by_the_installed_pallas_call():
    """Whatever the shim emits must be a real ``pallas_call`` keyword.

    This is the assertion that would have caught the original breakage: the port
    is only correct if the emitted names exist in the installed signature.
    """
    import inspect

    from jax.experimental.pallas import pallas_call

    accepted = set(inspect.signature(pallas_call).parameters)
    for backend in ("triton", None):
        assert (
            set(pallas_backend_kwargs(backend)) <= accepted
        ), f"shim emitted a kwarg pallas_call does not accept for {backend!r}"


def test_non_triton_backend_fails_loudly_on_the_new_api(monkeypatch):
    """Mosaic-GPU cannot run these kernels, so it must raise, not fall through."""
    monkeypatch.setattr(_compat, "PALLAS_CALL_TAKES_BACKEND", False)
    with pytest.raises(NotImplementedError, match="mosaic_gpu"):
        pallas_backend_kwargs("mosaic_gpu")


def test_old_api_passes_any_backend_through(monkeypatch):
    """On the old API the kwarg is forwarded verbatim -- no behaviour change."""
    monkeypatch.setattr(_compat, "PALLAS_CALL_TAKES_BACKEND", True)
    assert pallas_backend_kwargs("mosaic_gpu") == {"backend": "mosaic_gpu"}
    assert pallas_backend_kwargs(None) == {"backend": None}
