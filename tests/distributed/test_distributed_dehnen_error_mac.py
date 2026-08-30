"""The Dehnen (2014) section 5 mass-dependent MAC on the multi-GPU lane.

``mac_type="dehnen_error"`` is a jaccpot-level *policy*, not one of yggdrax's three
geometric literals: the criterion is evaluated pair-by-pair through a solver-owned
``pair_policy`` and the traversal underneath still runs the geometric ``"dehnen"``
test, whose verdict paper mode discards. This tier pins that the policy is really
installed on the distributed self walk and really deciding, which is the part that
fails silently:

* a lane that forgot the policy runs the geometric MAC, is **faster**, and reports
  the ``mac_type`` the caller asked for;
* a lane that installs the policy but no force scale runs eq (16a) against
  ``eps * 1`` instead of ``eps * min_b f_b`` -- a different criterion, accepting far
  more, again faster (trap 14 in ``docs/dehnen_mass_mac_status_and_plan.md``).

Both are *cheaper* than the correct behaviour, so no cost measurement can see
either. Every test below therefore asserts on something that separates them: the
accept mask against the geometric arm at the same ``theta``, the direction of the
response to ``eps``, and the range of the force scale that was actually installed.

The cross walk is **not** in scope here: it stays geometric, so at ndev >= 2 part of
the near field is still chosen by ``theta``. See ``jaccpot/distributed/_force_scale.py``.

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        pytest tests/distributed/test_distributed_dehnen_error_mac.py -o addopts="" -q
"""

from __future__ import annotations

import numpy as np
import pytest
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import DistributedFMMConfig, distributed_fmm_accelerations

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed FMM needs >= 2 devices"
)

#: Pinned rather than inherited. ``DistributedFMMConfig.leaf_size`` defaults to a
#: production 64, at which this tier's per-device particle count gives a tree with
#: no far field at all -- the walk accepts nothing, the run degenerates to direct
#: summation, and every assertion here passes while measuring nothing (trap 11).
#: Measured on this IC at theta 0.5: leaf 32 -> 2 far pairs, leaf 8 -> ~1250.
_LEAF = 8
_THETA = 0.5
_ORDER = 4
_PER_DEVICE = 1024
_SOFTENING = 0.02

#: An ``eps`` at which the criterion accepts a real far field on this IC. Tight
#: ``eps`` at small N accepts *nothing* -- eq (16a)'s threshold is
#: ``eps * min_b |a_b|`` over the target cell, so the least-accelerated particle
#: sets the budget (trap 5). The vacuity guards below fail rather than pass if this
#: drifts out of the usable window.
_EPS = 1.0e-2
#: A tighter one, used only for the direction-of-response check.
_EPS_TIGHT = 1.0e-3


def _ic(ndev: int, seed: int = 20260830):
    rng = np.random.default_rng(seed)
    n = _PER_DEVICE * ndev
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(n,)).astype(np.float32)
    return pts, mass


def _run(mesh, pts, mass, **kwargs):
    config = DistributedFMMConfig(
        leaf_size=_LEAF,
        order=_ORDER,
        theta=_THETA,
        softening=_SOFTENING,
        **kwargs,
    )
    result = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=False
    )
    diagnostics = {k: np.asarray(v) for k, v in result.diagnostics.items()}
    return result, diagnostics


def _direct(pts, mass, softening):
    pos = np.asarray(pts, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = (diff**2).sum(-1) + softening**2
    inv = d2 ** (-1.5)
    return -(m[None, :, None] * diff * inv[..., None]).sum(axis=1)


def _far(diagnostics) -> int:
    return int(diagnostics["self_far_pairs"].sum())


def test_the_criterion_runs_on_the_mesh_and_reproduces_the_direct_sum():
    """End to end: the policy traces under ``shard_map`` and the forces are right.

    The accuracy bar is loose on purpose -- this asserts that carrying a pair policy
    across the mesh does not corrupt the result, not that the criterion is accurate.
    What the criterion buys is a separate, measured question.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    result, diagnostics = _run(
        mesh, pts, mass, mac_type="dehnen_error", adaptive_eps=_EPS
    )

    assert not result.overflow, "traversal buffers overflowed -- grow the caps"
    assert _far(diagnostics) > 0, (
        "the criterion accepted no far pairs at all, so this run degenerated to "
        "direct summation and proves nothing about the far field"
    )

    direct = _direct(pts, mass, _SOFTENING)
    err = float(np.linalg.norm(result.accelerations - direct) / np.linalg.norm(direct))
    assert err < 1e-2, f"aggL2 err {err:.6f} exceeds 1%"


def test_the_criterion_decides_a_different_accept_mask_from_the_geometric_mac():
    """The far/near split must differ from ``theta``'s, or no policy is installed.

    This is the test that fails if ``dehnen_error`` is ever translated straight to
    the geometric ``"dehnen"`` with the pair policy dropped: same ``theta``, same
    tree, same IC, so an identical far-pair count means the criterion decided
    nothing.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    _, geometric = _run(mesh, pts, mass)
    _, criterion = _run(mesh, pts, mass, mac_type="dehnen_error", adaptive_eps=_EPS)

    assert _far(geometric) > 0, "the geometric arm has no far field to compare against"
    assert _far(criterion) != _far(geometric), (
        f"the criterion accepted the same {_far(criterion)} far pairs as the "
        f"geometric MAC at theta={_THETA} -- the pair policy is not deciding"
    )


def test_the_installed_force_scale_is_neither_missing_nor_the_unit_fallback():
    """``min_b f_b`` must vary across nodes, and must not be a constant 1.

    ``build_adaptive_policy_state`` substitutes ``jnp.ones(...)`` for a missing
    force scale, and ``AdaptivePolicyState`` keeps only
    ``target_accept_threshold``, so the fallback's signature is a *constant*
    threshold rather than an obviously wrong value. A range with min == max is that
    fallback; a range around 1.0 is it too.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    _, criterion = _run(mesh, pts, mass, mac_type="dehnen_error", adaptive_eps=_EPS)
    lo = float(criterion["force_scale_min"].min())
    hi = float(criterion["force_scale_max"].max())

    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo > 0.0, "a zero force scale refuses every far pair on its whole chain"
    assert hi > lo, (
        f"the force scale is constant at {lo:.6e} across every node -- that is the "
        "unit-scale fallback's signature, not a per-node min_b f_b"
    )

    _, geometric = _run(mesh, pts, mass)
    assert float(geometric["force_scale_max"].max()) == 0.0, (
        "the geometric arm reported a force scale, so this diagnostic is not "
        "actually keyed to the criterion"
    )


def test_tightening_eps_refuses_far_pairs_rather_than_accepting_more():
    """The response to ``eps`` must have the sign eq (16a) says it has.

    ``eps`` multiplies the acceptance threshold, so a smaller one can only accept
    fewer pairs. The inverted sign is what a scale wired in upside down looks like,
    and it would show up as "the criterion is faster when you ask for accuracy".
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    _, loose = _run(mesh, pts, mass, mac_type="dehnen_error", adaptive_eps=_EPS)
    _, tight = _run(mesh, pts, mass, mac_type="dehnen_error", adaptive_eps=_EPS_TIGHT)

    assert _far(loose) > 0, "the loose arm accepted nothing; there is no trend here"
    assert _far(tight) < _far(loose), (
        f"tightening eps from {_EPS:g} to {_EPS_TIGHT:g} did not reduce the far "
        f"field ({_far(loose)} -> {_far(tight)})"
    )


# --------------------------------------------------------------------------- #
# refusals -- each one is a configuration that would otherwise run the WRONG
# criterion quietly, and be faster for it
# --------------------------------------------------------------------------- #


def _expect(match: str, **kwargs):
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)
    with pytest.raises(ValueError, match=match):
        _run(mesh, pts, mass, **kwargs)


def test_the_refuted_folded_angle_mode_is_rejected_not_translated():
    """``dehnen_theta`` folds the criterion into ``geometry.radius``.

    The cross walk does not carry that folding, so translating it here would leave
    the two walks on different criteria with nothing to say so -- on top of the mode
    being refuted on the single-GPU lane at 12-9300x worse error.
    """

    _expect("not available on the distributed lane", mac_type="dehnen_theta")


def test_the_criterion_without_a_tolerance_is_refused():
    """``adaptive_eps`` replaces ``theta`` as the accuracy knob; it has no default.

    A default would be a criterion nobody chose, and choosing wrong costs *less*
    work, so nothing downstream would flag it.
    """

    _expect("requires adaptive_eps", mac_type="dehnen_error")


def test_the_treecode_walk_cannot_carry_the_criterion():
    """The fast-lane local walk takes no pair policy, so it must not be offered."""

    _expect(
        "needs local_walk='dual_tree'",
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        local_walk="treecode",
    )


def test_eq_16a_force_scale_modes_are_refused():
    """``min_b |a_b|`` is an acceleration: a second distributed evaluation."""

    _expect(
        "mac_force_scale_mode must be one of",
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_force_scale_mode="paper",
    )


def test_the_host_loop_geometry_modes_are_refused_before_the_trace():
    """A numpy host loop over nodes cannot run inside ``shard_map``.

    Rejected up front rather than at trace time, so the message names the knob
    instead of surfacing as a ``ConcretizationTypeError`` from three frames down.
    """

    _expect(
        "dehnen_geometry_mode must be one of",
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        dehnen_geometry_mode="exact",
    )
