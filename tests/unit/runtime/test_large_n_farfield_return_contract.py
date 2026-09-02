"""``evaluate_large_n_farfield`` returns Optionals, and its hint must say so.

The adapter was annotated ``tuple[Array, Array, Array]``, which was wrong on two
of the three slots: the delegate returns ``(gradients, None, derivative_outputs)``
when ``return_potential`` is false, and ``derivative_outputs`` is ``None``
whenever ``max_acc_derivative_order == 0`` -- which this adapter pins
unconditionally, so the third element is ``None`` on every call.

Nothing tripped over it because the function has no caller inside ``jaccpot/``,
``tests/`` or ``bench/`` (only ``examples/benchmark_gpu_radix_worker.py`` times
it, and it discards the result without unpacking). This test is that missing
caller: it pins the element types on a real ``LargeNPreparedState`` so the
annotation cannot drift back.

The lane refuses to dispatch off a GPU (``can_use_large_n_prepare_path`` gates on
``jax.default_backend()``), so the backend probe is stubbed the way the sibling
large-N tests stub it. Everything past that gate is backend-independent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig
from jaccpot.runtime._large_n_farfield import evaluate_large_n_farfield
from jaccpot.runtime._large_n_types import LargeNPreparedState
from jaccpot.runtime.fmm import FMMEngine

# Compile-bound: a full large-N prepare plus two farfield evaluations.
pytestmark = pytest.mark.slow

N = 192
LEAF = 32
ORDER = 2


@pytest.fixture(scope="module")
def prepared_state(request: pytest.FixtureRequest) -> LargeNPreparedState:
    """Build a real large-N prepared state, with the GPU gate stubbed out.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Used to register the stub's undo. Module-scoped, so the function-scoped
        ``monkeypatch`` fixture is not available and one is driven by hand.

    Returns
    -------
    LargeNPreparedState
        A prepared state from the real large-N path, not a double.
    """
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    rng = np.random.default_rng(0)
    positions = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(N, 3)),
        dtype=jnp.float32,
    )
    masses = jnp.ones((N,), dtype=jnp.float32)

    engine = FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        working_dtype=jnp.float32,
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        fixed_order=ORDER,
    )
    state = engine.prepare_state(
        positions,
        masses,
        leaf_size=LEAF,
        max_order=ORDER,
    )
    assert isinstance(state, LargeNPreparedState), (
        "the large-N prepare path declined; this test would otherwise assert "
        "the return contract of a different lane"
    )
    return state


@pytest.mark.parametrize("return_potential", [False, True])
def test_farfield_returns_gradient_potential_and_no_derivatives(
    prepared_state: LargeNPreparedState,
    return_potential: bool,
) -> None:
    """The three slots are gradient, optional potential, and always-``None``.

    Parameters
    ----------
    prepared_state : LargeNPreparedState
        The real prepared state to evaluate against.
    return_potential : bool
        Both values are exercised: it is what makes the second slot optional.
    """
    result = evaluate_large_n_farfield(
        prepared_state,
        return_potential=return_potential,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    gradients, potentials, derivatives = result

    assert isinstance(gradients, jax.Array)
    assert gradients.shape == (N, 3)

    if return_potential:
        assert isinstance(potentials, jax.Array)
        assert potentials.shape == (N,)
    else:
        assert potentials is None

    # `max_acc_derivative_order` is pinned to 0 inside the adapter, so this slot
    # is None for both values of `return_potential` -- not just the false one.
    assert derivatives is None
