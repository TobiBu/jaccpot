"""No path may hand the traversal an unresolved MAC type.

``"dehnen_error"`` is a **jaccpot-level** policy: the Dehnen (2014) §5
mass-dependent MAC, layered on top of the geometric ``"dehnen"`` acceptance test.
yggdrax has never heard of it -- its ``MACType`` is
``Literal["bh", "engblom", "dehnen"]`` and its traversal raises
``ValueError("Unknown mac_type: ...")`` for anything else. So every jaccpot path
that reaches the traversal has to translate first, via
:meth:`PolicyMixin._mac_type_for_traversal`.

Two of the three call sites did. ``prepare_downward_sweep`` did not: it took
``self.mac_type`` raw, so with ``mac_type="dehnen_error"`` the unresolved string
travelled all the way to the traversal boundary from five production entry points
(``fmm_derivatives``, ``_large_n_grad``, ``fmm_evaluate``,
``fmm_prepare._prepare_state_uncaught`` and the public sweep). It did not blow up
in practice only because those paths pass ``interactions`` in and so skip the
rebuild -- a latent failure, not a safe one.

These tests pin the property rather than the one line that was wrong: whatever a
caller asks for, what reaches the traversal is a value yggdrax accepts.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import MACType

from jaccpot.runtime import _fmm_impl as fmm_impl_private
from jaccpot.runtime import fmm_sweeps

#: Exactly the values yggdrax's traversal branches on. Derived from its own
#: ``MACType`` so this test tracks the dependency instead of duplicating it.
YGGDRAX_ACCEPTED = frozenset(MACType.__args__)


#: ``mac_type="dehnen_error"`` is refused without one: it is the relative
#: force-accuracy target of eq (16a), and the theta-derived default is a far looser
#: tail_proxy heuristic. That refusal is deliberate (STYLE_GUIDE §9), so supply it.
PAPER_EPS = 1e-3


def _solver(mac_type: str):
    kwargs = {}
    if mac_type == "dehnen_error":
        kwargs["adaptive_eps"] = PAPER_EPS
    return fmm_impl_private.FastMultipoleMethod(
        expansion_basis="solidfmm",
        mac_type=mac_type,
        **kwargs,
    )


def _capture_traversal_mac_type(monkeypatch, solver, **sweep_kwargs) -> list[str]:
    """Run ``prepare_downward_sweep`` and record what the traversal was given.

    Uses a **real** tree and upward sweep, built once per solver via
    ``prepare_state``. An earlier draft passed placeholder objects, since the fake
    sweep raises before reading them -- but ``prepare_downward_sweep`` declares
    ``tree: Tree`` and ``upward_data: TreeUpwardData``, so that would have made
    this test violate a contract while asserting about contracts. It is also
    exactly the looseness F40 is about.
    """
    seen: list[str] = []

    # Build the real artifacts FIRST. `prepare_state` runs the downward sweep too,
    # so patching before this point makes the fake fire during setup -- capturing
    # prepare_state's (correctly resolved) value and raising outside the
    # `pytest.raises` block.
    tree, upward = _real_tree_and_upward(solver)

    def _fake_sweep(*args, **kwargs):
        seen.append(kwargs["mac_type"])
        raise _Stop

    monkeypatch.setattr(
        fmm_sweeps, "_prepare_solidfmm_downward_sweep", _fake_sweep, raising=True
    )
    with pytest.raises(_Stop):
        solver.prepare_downward_sweep(tree, upward, theta=0.5, **sweep_kwargs)
    return seen


class _Stop(Exception):
    """Sentinel: stop once the traversal-facing value has been captured."""


def _real_tree_and_upward(solver):
    """A real ``Tree`` and ``TreeUpwardData`` from the solver's own prepare path."""
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (128, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, (128,)))
    state = solver.prepare_state(positions, masses, leaf_size=8, max_order=2)
    return state.tree, state.upward


@pytest.mark.parametrize("requested", sorted(YGGDRAX_ACCEPTED | {"dehnen_error"}))
def test_downward_sweep_hands_the_traversal_a_value_yggdrax_accepts(
    monkeypatch, requested
):
    """Whatever the solver is configured with, the traversal gets a legal value."""
    (seen,) = _capture_traversal_mac_type(monkeypatch, _solver(requested))
    assert seen in YGGDRAX_ACCEPTED, (
        f"mac_type={requested!r} reached the traversal as {seen!r}, which yggdrax "
        f"rejects with ValueError('Unknown mac_type')"
    )


def test_dehnen_error_resolves_to_the_geometric_dehnen_test(monkeypatch):
    """And specifically: the §5 policy resolves to the geometric ``"dehnen"``.

    Not just "something legal" -- the *right* legal value, since picking ``"bh"``
    would silently change the acceptance criterion and the force.
    """
    (seen,) = _capture_traversal_mac_type(monkeypatch, _solver("dehnen_error"))
    assert seen == "dehnen"


def test_an_explicit_dehnen_error_override_is_resolved_too(monkeypatch):
    """The override argument needs the same treatment as the default.

    ``prepare_downward_sweep(mac_type=...)`` bypasses ``self.mac_type`` entirely,
    so resolving only the default branch would leave the same hole open for any
    caller that passes the value explicitly.
    """
    (seen,) = _capture_traversal_mac_type(
        monkeypatch, _solver("bh"), mac_type="dehnen_error"
    )
    assert seen == "dehnen"


@pytest.mark.parametrize("requested", sorted(YGGDRAX_ACCEPTED | {"dehnen_error"}))
def test_the_sweep_resolves_exactly_as_base_mac_type_does(monkeypatch, requested):
    """``prepare_downward_sweep`` must resolve exactly as ``_base_mac_type`` does.

    The bug was an inconsistency, not a wrong formula: ``fmm_prepare`` and
    ``fmm_strict_run`` both went through ``_base_mac_type()`` while this path did
    not. Asserting they agree is what stops them drifting apart again.

    Parametrised rather than looped: ``monkeypatch`` lasts for the whole test, and
    a second iteration would build its solver state with the fake sweep still
    installed.
    """
    solver = _solver(requested)
    (seen,) = _capture_traversal_mac_type(monkeypatch, solver)
    assert seen == solver._base_mac_type()


def test_the_policy_flag_still_sees_the_unresolved_value():
    """Resolution must not hide the request from the policy layer.

    ``_uses_dehnen_error_policy()`` reads ``self.mac_type`` and is what switches
    the mass-dependent machinery on. If resolution had been done by overwriting
    ``self.mac_type``, the traversal would be right and the policy would be off --
    a much worse bug than the one being fixed.
    """
    solver = _solver("dehnen_error")
    assert solver.mac_type == "dehnen_error"
    assert solver._uses_dehnen_error_policy() is True
    assert solver._base_mac_type() == "dehnen"
