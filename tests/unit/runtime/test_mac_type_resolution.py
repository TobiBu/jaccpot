"""No path may hand the traversal an unresolved MAC type.

``"dehnen_error"`` and ``"dehnen_theta"`` are **jaccpot-level** policies: the
Dehnen (2014) §5 mass-dependent MAC, layered on top of the geometric ``"dehnen"``
acceptance test. yggdrax has never heard of either -- its ``MACType`` is
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

They sweep ``ALL_REQUESTABLE`` rather than naming one policy, because the
jaccpot-level names are a *set* that has already grown once. ``_base_mac_type``
used to key off ``_uses_dehnen_error_policy()`` -- true for both names -- and was
rewritten to call ``_mac_type_for_traversal``, which at first mapped only
``dehnen_error``. The two ``in`` tests fell out of step, ``dehnen_theta`` reached
the traversal unmapped, and yggdrax's runtime type check rejected it. Add any
future jaccpot-level MAC name to ``JACCPOT_POLICY_MAC_TYPES`` and these tests will
demand the resolver learn it.
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

#: The jaccpot-level MAC names that are *not* in yggdrax's set and therefore have
#: to be translated. Kept as a set rather than a literal because
#: ``_mac_type_for_traversal`` and ``_uses_dehnen_error_policy`` are two
#: independent ``in`` tests over the same names, and they must agree.
JACCPOT_POLICY_MAC_TYPES = frozenset({"dehnen_error", "dehnen_theta"})

#: Every value a caller may legally pass, which is what these tests sweep.
ALL_REQUESTABLE = YGGDRAX_ACCEPTED | JACCPOT_POLICY_MAC_TYPES

#: What each requestable name must reach the traversal as. Not just "something
#: legal" -- the *right* legal value, since picking ``"bh"`` for a §5 policy would
#: silently change the acceptance criterion and the force. ``dehnen_theta`` carries
#: the same §5 criterion pre-applied (folded into one effective opening angle per
#: node, a rescaled ``geometry.radius``) instead of into a pair policy, so the
#: *geometric* test underneath is unchanged and it too must be handed ``"dehnen"``
#: -- and since no pair policy is installed in that mode, nothing downstream would
#: re-map it. Keyed exactly by ``ALL_REQUESTABLE`` (asserted), so a new name has to
#: say where it resolves.
EXPECTED_TRAVERSAL_MAC_TYPE = {
    **{name: name for name in YGGDRAX_ACCEPTED},
    **{name: "dehnen" for name in JACCPOT_POLICY_MAC_TYPES},
}

# One (tree, upward) pair serves every capture below, so it is module-scoped and its
# users share an xdist group (CI: `-n 2 --dist loadgroup`), built on one worker.
_TREE_AND_UPWARD_USERS = pytest.mark.xdist_group("mac_type_resolution_tree_and_upward")


#: The §5 policies are refused without one: ``adaptive_eps`` is the relative
#: force-accuracy target of eq (16a), and the theta-derived default is a far looser
#: tail_proxy heuristic. That refusal is deliberate (STYLE_GUIDE §9), so supply it.
PAPER_EPS = 1e-3


def _solver(mac_type: str):
    kwargs = {}
    if mac_type in JACCPOT_POLICY_MAC_TYPES:
        kwargs["adaptive_eps"] = PAPER_EPS
    return fmm_impl_private.FMMEngine(
        expansion_basis="solidfmm",
        mac_type=mac_type,
        **kwargs,
    )


def _capture_traversal_mac_type(
    monkeypatch, solver, tree_and_upward, **sweep_kwargs
) -> list[str]:
    """Run ``prepare_downward_sweep`` and record what the traversal was given.

    Uses a **real** tree and upward sweep (the ``tree_and_upward`` fixture). An
    earlier draft passed placeholder objects, since the fake sweep raises before
    reading them -- but ``prepare_downward_sweep`` declares ``tree: Tree`` and
    ``upward_data: TreeUpwardData``, so that would have made this test violate a
    contract while asserting about contracts. It is also exactly the looseness F40
    is about. The fake raises before reading either, so one pair, built by a
    ``bh`` solver, serves every requested MAC type.
    """
    seen: list[str] = []
    tree, upward = tree_and_upward

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


@pytest.fixture(scope="module")
def tree_and_upward():
    """A real ``Tree`` and ``TreeUpwardData`` from a solver's own prepare path.

    Built FIRST and outside any test's patch: ``prepare_state`` runs the downward
    sweep too, so building it with the fake installed would fire the fake during
    setup -- capturing prepare_state's (correctly resolved) value and raising
    outside the ``pytest.raises`` block.
    """
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (128, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, (128,)))
    state = _solver("bh").prepare_state(positions, masses, leaf_size=8, max_order=2)
    return state.tree, state.upward


def test_the_expected_resolution_names_every_requestable_value():
    """The table below the sweep must not silently skip a name."""
    assert set(EXPECTED_TRAVERSAL_MAC_TYPE) == ALL_REQUESTABLE


@_TREE_AND_UPWARD_USERS
@pytest.mark.parametrize("requested", sorted(ALL_REQUESTABLE))
def test_downward_sweep_hands_the_traversal_a_value_yggdrax_accepts(
    monkeypatch, tree_and_upward, requested
):
    """Whatever the solver is configured with, the traversal gets a legal value.

    The right legal value (``EXPECTED_TRAVERSAL_MAC_TYPE``: the §5 policies resolve
    to the geometric ``"dehnen"``), and exactly the one ``_base_mac_type`` resolves
    to. That last is the bug this file was written for, an inconsistency rather
    than a wrong formula: ``fmm_prepare`` and ``fmm_strict_run`` both went through
    ``_base_mac_type()`` while this path did not. Asserting they agree is what stops
    them drifting apart again.

    Parametrised rather than looped: ``monkeypatch`` lasts for the whole test, and
    a second iteration would run with the fake sweep still installed. (Until
    2026-10-12 the ``"dehnen"`` target and the ``_base_mac_type`` agreement were
    three more tests, seven more solver prepares.)
    """
    solver = _solver(requested)
    (seen,) = _capture_traversal_mac_type(monkeypatch, solver, tree_and_upward)
    assert seen in YGGDRAX_ACCEPTED, (
        f"mac_type={requested!r} reached the traversal as {seen!r}, which yggdrax "
        f"rejects with ValueError('Unknown mac_type')"
    )
    assert seen == EXPECTED_TRAVERSAL_MAC_TYPE[requested], (requested, seen)
    assert seen == solver._base_mac_type(), (requested, seen)


def test_the_resolver_and_the_policy_flag_cover_the_same_names():
    """``_mac_type_for_traversal`` and ``_uses_dehnen_error_policy`` must agree.

    These are two independent ``in`` tests over the same set of jaccpot-level
    names, in two different modules, and nothing but this test couples them. When
    they disagree the failure is asymmetric and both directions are bad:

    * a name the policy flag knows but the resolver does not reaches yggdrax
      unmapped and is rejected -- loud, and how this was found;
    * a name the resolver maps but the policy flag does not runs the plain
      geometric MAC while the caller believes the mass-dependent criterion is
      active -- silent, and it makes the solver *faster*, so no cost measurement
      can see it.

    Asserted against a solver instance rather than the raw functions so the
    staticmethod and the bound method are both exercised as callers reach them.
    """
    solver = _solver("bh")
    for name in JACCPOT_POLICY_MAC_TYPES:
        assert solver._mac_type_for_traversal(name) == "dehnen", (
            f"{name!r} is a jaccpot-level policy but _mac_type_for_traversal "
            f"passes it through to yggdrax unchanged"
        )
        assert _solver(name)._uses_dehnen_error_policy() is True, (
            f"{name!r} is translated for the traversal but does not switch the "
            f"mass-dependent machinery on -- the silent direction"
        )
    for name in YGGDRAX_ACCEPTED:
        assert solver._mac_type_for_traversal(name) == name
        assert _solver(name)._uses_dehnen_error_policy() is False


@_TREE_AND_UPWARD_USERS
def test_an_explicit_dehnen_error_override_is_resolved_too(
    monkeypatch, tree_and_upward
):
    """The override argument needs the same treatment as the default.

    ``prepare_downward_sweep(mac_type=...)`` bypasses ``self.mac_type`` entirely,
    so resolving only the default branch would leave the same hole open for any
    caller that passes the value explicitly.
    """
    (seen,) = _capture_traversal_mac_type(
        monkeypatch, _solver("bh"), tree_and_upward, mac_type="dehnen_error"
    )
    assert seen == "dehnen"


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
