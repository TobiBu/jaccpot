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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import DistributedFMMConfig, distributed_fmm_accelerations
from jaccpot.distributed.fmm import (
    cross_walk_accepts_a_pair_policy,
    make_force_evaluator,
    partition_for_devices,
)

#: The cross-domain half needs yggdrax PR #54. jaccpot depends on yggdrax by version
#: RANGE, so an older one is an ordinary install rather than a broken one -- but a
#: skip here must stay loud, because "the cross criterion is never exercised" and "the
#: cross criterion works" look identical in a green run. The self-walk tests above do
#: not need it and never skip.
_needs_cross_hook = pytest.mark.skipif(
    not cross_walk_accepts_a_pair_policy(),
    reason="yggdrax's cross walk takes no pair_policy (needs yggdrax PR #54)",
)

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


#: Memoised runs, keyed by the config knobs. A criterion evaluation on this tier is
#: two self walks plus a policy graph, which is minutes per call on a forced-CPU
#: mesh; the tests below ask for the same three arms repeatedly, and without this
#: the file re-runs each one up to three times for no extra coverage.
_RUNS: dict = {}


def _run(mesh, pts, mass, **kwargs):
    # Every test here pins `mac_cross_criterion` rather than inheriting the default.
    # A test named for the SELF walk must not change what it exercises because the
    # cross-walk default flipped -- and against a yggdrax without the cross hook the
    # default would make it raise, which says nothing about the self walk. The
    # cross-domain tests below pass True explicitly and skip when the hook is absent.
    kwargs.setdefault("mac_cross_criterion", False)
    key = tuple(sorted(kwargs.items()))
    if key not in _RUNS:
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
        _RUNS[key] = (
            result,
            {k: np.asarray(v) for k, v in result.diagnostics.items()},
        )
    return _RUNS[key]


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
    """Assert a configuration is refused, and refused for the stated reason.

    Deliberately not routed through the memo: a refused call produces no run to
    cache, and going through ``_run`` would make a passing test depend on the memo
    never having been poisoned by one.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)
    # Same reason as `_run`: a refusal test is about ONE rejected field, so it must
    # not also depend on the cross-walk default and the yggdrax that default needs.
    kwargs.setdefault("mac_cross_criterion", False)
    config = DistributedFMMConfig(
        leaf_size=_LEAF,
        order=_ORDER,
        theta=_THETA,
        softening=_SOFTENING,
        **kwargs,
    )
    with pytest.raises(ValueError, match=match):
        distributed_fmm_accelerations(pts, mass, config=config, mesh=mesh, jit=False)


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


# --------------------------------------------------------------------------- #
# the seam -- the criterion decides topology, so it must carry no cotangent
# --------------------------------------------------------------------------- #


def _evaluators(ndev, mesh, part, **kwargs):
    config = DistributedFMMConfig(
        leaf_size=_LEAF,
        order=_ORDER,
        theta=_THETA,
        softening=_SOFTENING,
        nearfield_backend="baseline",
        **kwargs,
    )
    return (
        make_force_evaluator(
            config, ndev, part["cap"], mesh, jit=True, differentiable=False
        ),
        make_force_evaluator(
            config, ndev, part["cap"], mesh, jit=True, differentiable=True
        ),
    )


def test_the_criterion_does_not_break_the_fixed_topology_seam():
    """``jax.grad`` must still work, and the forward must not move.

    The distributed lane's differentiable mode rests on a seam: the tree, its
    geometry and every walk are built from ``stop_gradient``-ed inputs, so no
    cotangent reaches the discrete topology -- which is what keeps the reverse pass
    off the traversal's ``lax.while_loop``, which JAX cannot transpose.

    The criterion decides the far/near split, so it is *on* the topology side of
    that seam, and everything feeding it has to be frozen. The first version of
    this port fed it ``lp``/``lm`` -- which in differentiable mode are the LIVE
    re-gather, not the frozen sorted arrays -- and this is the test that says so.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)
    part = partition_for_devices(pts, mass, ndev, leaf_size=_LEAF)
    forward, grad_path = _evaluators(
        ndev,
        mesh,
        part,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=False,
    )

    args = (
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        jnp.asarray(part["counts"]),
    )
    shipped = np.asarray(forward(*args)[0])
    live = np.asarray(grad_path(*args)[0])
    assert np.array_equal(shipped, live), (
        "differentiable=True changed the forward force under the criterion "
        f"(max abs diff {np.max(np.abs(shipped - live)):.3e})"
    )

    def loss(positions, masses):
        return jnp.sum(grad_path(positions, masses, args[2], args[3])[0] ** 2)

    g_pos, g_mass = jax.grad(loss, argnums=(0, 1))(args[0], args[1])
    for name, g in (("positions", g_pos), ("masses", g_mass)):
        g = np.asarray(g)
        assert np.all(np.isfinite(g)), f"the {name} gradient has non-finite entries"
        assert np.abs(g).max() > 0.0, (
            f"the {name} gradient is identically zero, so the reverse pass is not "
            "reaching the force at all"
        )


# --------------------------------------------------------------------------- #
# the cross-domain half -- the larger half of the near field on a mesh
# --------------------------------------------------------------------------- #


def _cross_far(diagnostics) -> int:
    return int(diagnostics["cross_far_pairs"].sum())


def _cross_near(diagnostics) -> int:
    return int(diagnostics["cross_near_pairs"].sum())


@_needs_cross_hook
def test_the_criterion_decides_the_cross_walk_too():
    """The cross-domain accept mask must move, and only when asked to.

    Both halves matter. If the criterion never reached the cross walk, the cross
    counts would equal the geometric arm's and every accuracy result would be
    reporting a half-ported criterion. If it reached the cross walk unconditionally,
    ``mac_cross_criterion=False`` would stop being an ablation and the measurement
    that separates the two contributions would be gone.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    _, geometric = _run(mesh, pts, mass)
    _, self_only = _run(
        mesh,
        pts,
        mass,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=False,
    )
    _, both = _run(
        mesh,
        pts,
        mass,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=True,
    )

    assert _cross_far(geometric) > 0, "no cross far field on this IC; nothing measured"
    assert (_cross_far(self_only), _cross_near(self_only)) == (
        _cross_far(geometric),
        _cross_near(geometric),
    ), (
        "mac_cross_criterion=False changed the cross walk, so the self-only ablation "
        "is not actually an ablation"
    )
    assert (_cross_far(both), _cross_near(both)) != (
        _cross_far(geometric),
        _cross_near(geometric),
    ), (
        "the criterion left the cross walk on the geometric MAC -- the larger half "
        "of the near field at ndev>=2 is not being decided"
    )


@_needs_cross_hook
def test_the_self_walk_is_untouched_by_the_cross_ablation():
    """Turning the cross criterion off must not perturb the local walk.

    The two walks are decided by the same force scale but by separate policy states.
    If flipping ``mac_cross_criterion`` moved the self counts, the states would be
    coupled somewhere they should not be -- and the ablation would no longer isolate
    the cross-domain contribution.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    _, self_only = _run(
        mesh,
        pts,
        mass,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=False,
    )
    _, both = _run(
        mesh,
        pts,
        mass,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=True,
    )

    assert _far(self_only) == _far(both), (
        f"self far moved from {_far(self_only)} to {_far(both)} when only the CROSS "
        "walk's criterion was switched on"
    )
    assert int(self_only["self_near_pairs"].sum()) == int(both["self_near_pairs"].sum())


@_needs_cross_hook
def test_the_cross_criterion_still_reproduces_the_direct_sum():
    """Carrying the policy across domains must not corrupt the force.

    The cross walk drives the halo import, so a criterion applied in the wrong place
    in the sequence would fetch one set of remote leaves and evaluate against
    another. That shows up as missing sources -- a wrong force with every buffer flag
    clear.
    """

    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    pts, mass = _ic(ndev)

    result, diagnostics = _run(
        mesh,
        pts,
        mass,
        mac_type="dehnen_error",
        adaptive_eps=_EPS,
        mac_cross_criterion=True,
    )
    assert not result.overflow
    assert _cross_far(diagnostics) > 0

    direct = _direct(pts, mass, _SOFTENING)
    err = float(np.linalg.norm(result.accelerations - direct) / np.linalg.norm(direct))
    assert err < 1e-2, f"aggL2 err {err:.6f} exceeds 1% with the cross criterion on"
