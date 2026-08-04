"""The interaction cache key must see the acceptance criterion, not just geometry.

``_interaction_cache_key`` hashes topology, ``theta``, the base ``mac_type``,
``dehnen_radius_scale``, the basis, the centre mode, the traversal caps and the
refinement knobs. None of that describes the Dehnen mass-dependent criterion:
``mac_type="dehnen_error"`` reports the *geometric* base MAC ``"dehnen"`` through
:meth:`_base_mac_type`, and paper mode pins ``theta`` at 1.0 on the grounds that
it does not gate acceptance. So every knob that actually decides acceptance --
``adaptive_eps``, the per-node force scale, the force-scale mode, the geometry
mode -- is invisible to the key.

That is the "faster and wronger" failure mode this branch keeps running into: a
reused interaction list makes the solver *cheaper* while silently answering a
different criterion, so no cost measurement can detect it (see trap 6 in
``docs/dehnen_mass_mac_status_and_plan.md``).

Measured before the fix, at N=2048 / leaf 8 / eps=1e-3 with
``mac_type="dehnen_theta"``: injecting per-node force scales of 1e-3, 1.0 and
1e+3 -- six orders of magnitude, the entire right-hand side of eq (16a) -- all
produced **17520** far pairs with a cache *hit* on the second prepare. The
injected scale had no effect whatsoever; the second call was served the first
call's accept mask.

``dehnen_error`` escapes that today only by an accident of control flow:
``_prepare_state_dual_and_downward`` computes ``cache_key`` in the ``else``
branch of the policy test, so a request that installs a solver-owned
``pair_policy`` gets ``cache_key=None`` and the cache is bypassed entirely. That
is not an invariant anybody stated, and relaxing the fast-lane policy gates
(Step 3') is exactly the change that would turn it into a live bug. These tests
pin it as a contract instead.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FMMAdvancedConfig
from jaccpot.runtime._interaction_cache import (
    _interaction_cache_key,
    pair_policy_cache_identity,
)
from jaccpot.solver import FastMultipoleMethod

N = 2048
LEAF = 8
ORDER = 4
EPS = 1e-3


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(N, 3))
    mass = rng.uniform(0.5, 1.5, size=N)
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


def _solver(mac_type: str, *, eps: float | None = EPS):
    advanced = FMMAdvancedConfig()
    advanced = replace(
        advanced,
        mac_type=mac_type,
        runtime=replace(
            advanced.runtime,
            retain_traversal_result=True,
            retain_interactions=True,
        ),
    )
    kwargs: dict = dict(theta=0.6, advanced=advanced)
    if eps is not None:
        kwargs["adaptive_eps"] = eps
    return FastMultipoleMethod(**kwargs)


def _engine(fmm):
    return getattr(fmm, "_impl", fmm)


def _far_pairs(state) -> int:
    assert state.interactions is not None, "test requires a retained interaction list"
    return int(jnp.sum(jnp.asarray(state.interactions.sources) >= 0))


def _accept_mask(state) -> tuple[tuple[int, int], ...]:
    """The accepted (target node, source node) far pairs, as a comparable set."""

    src = np.asarray(state.interactions.sources).ravel()
    tgt = np.asarray(state.interactions.targets).ravel()
    keep = (src >= 0) & (tgt >= 0)
    return tuple(sorted(zip(tgt[keep].tolist(), src[keep].tolist())))


# --------------------------------------------------------------------------
# behaviour: a reused interaction list must not change the answer
# --------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_an_injected_force_scale_is_not_ignored_by_a_cached_interaction_list():
    """Two force scales, one solver: the cache must not serve the first mask.

    ``force_scale_nodes`` is the right-hand side of eq (16a). Injecting two
    scales three orders of magnitude apart must give two different accept masks.
    The reference for each is a *fresh* solver, which has nothing to reuse -- so
    the assertion is "reuse does not change the answer", and the guard that keeps
    it from passing vacuously is that the two fresh answers genuinely differ.
    """

    positions, masses = _problem()

    fresh_masks = []
    for multiplier in (1e-3, 1e3):
        fmm = _solver("dehnen_theta")
        probe = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
        node_count = int(probe.tree.parent.shape[0])
        scale = jnp.full((node_count,), multiplier, dtype=jnp.float64)
        fresh = _solver("dehnen_theta")
        state = fresh.prepare_state(
            positions,
            masses,
            leaf_size=LEAF,
            max_order=ORDER,
            force_scale_nodes=scale,
        )
        fresh_masks.append(_accept_mask(state))

    assert fresh_masks[0] != fresh_masks[1], (
        "the two injected force scales produce the same accept mask, so this "
        "test cannot see a stale reuse -- pick scales that actually disagree"
    )

    # Same two scales, but on ONE solver, so the second call meets a populated
    # interaction cache whose key was computed for the first scale.
    reused = _solver("dehnen_theta")
    probe = reused.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    node_count = int(probe.tree.parent.shape[0])
    reused_masks = []
    for multiplier in (1e-3, 1e3):
        state = reused.prepare_state(
            positions,
            masses,
            leaf_size=LEAF,
            max_order=ORDER,
            force_scale_nodes=jnp.full((node_count,), multiplier, dtype=jnp.float64),
        )
        reused_masks.append(_accept_mask(state))

    assert reused_masks[0] == fresh_masks[0], (
        "the first cached-solver prepare disagrees with a fresh solver at the "
        "same force scale"
    )
    assert reused_masks[1] == fresh_masks[1], (
        f"a cached interaction list built at force scale 1e-3 "
        f"({len(reused_masks[1])} far pairs) was served to a request at force "
        f"scale 1e+3 ({len(fresh_masks[1])} far pairs): the cache key cannot "
        "see the criterion's right-hand side"
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_two_tolerances_do_not_share_a_cache_entry():
    """Two solvers identical except ``adaptive_eps`` must not share an entry.

    Stated as the prompt's own bar. It holds for ``dehnen_error`` today because
    a solver-owned pair policy leaves ``cache_key`` at ``None``; the point of
    pinning it is that the fast-lane relaxation must not quietly remove that.
    """

    positions, masses = _problem()
    keys = []
    far = []
    for eps in (1e-3, 1e-5):
        fmm = _solver("dehnen_error", eps=eps)
        state = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
        entry = _engine(fmm)._interaction_cache
        keys.append(None if entry is None else entry.key)
        far.append(_far_pairs(state))

    assert far[0] != far[1], (
        "the two tolerances accept the same number of far pairs, so this "
        "configuration cannot see a shared entry"
    )
    assert keys[0] != keys[1] or keys[0] is None, (
        "two dehnen_error solvers differing only in adaptive_eps produced the "
        f"same interaction cache key ({keys[0]})"
    )


def test_a_geometric_solver_still_reuses_its_interaction_list():
    """Control: the fix must not be "stop caching".

    Without this, every assertion above passes vacuously by returning ``None``
    from :func:`_interaction_cache_key` unconditionally.
    """

    positions, masses = _problem()
    fmm = _solver("dehnen", eps=None)
    fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    assert _engine(fmm)._interaction_cache_hits >= 1, (
        "a plain geometric solver no longer reuses its interaction list, so the "
        "cache has been disabled rather than keyed"
    )


# --------------------------------------------------------------------------
# the key itself
# --------------------------------------------------------------------------


def _key(tree, *, identity):
    return _interaction_cache_key(
        tree,
        topology_key="fixed-topology",
        tree_mode="lbvh",
        leaf_parameter=LEAF,
        theta=1.0,
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
        expansion_basis="solidfmm",
        center_mode="com",
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=None,
        refine_local=False,
        max_refine_levels=0,
        aspect_threshold=16.0,
        pair_policy_identity=identity,
    )


@pytest.fixture(scope="module")
def _tree():
    positions, masses = _problem()
    fmm = _solver("dehnen", eps=None)
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    return state.tree


def _identity(**overrides):
    kwargs: dict = dict(
        pair_policy=None,
        policy_state=None,
        eps=EPS,
        force_scale_mode="paper_cached",
        geometry_mode="com",
        theta_max=1.0,
        error_model_code=2,
        force_scale_nodes=None,
        mac_geometry_radius=None,
    )
    kwargs.update(overrides)
    return pair_policy_cache_identity(**kwargs)


def test_the_key_separates_two_tolerances(_tree):
    """The literal bar: differing ``eps`` must give differing keys."""

    tight = _key(_tree, identity=_identity(eps=1e-7))
    loose = _key(_tree, identity=_identity(eps=1e-5))
    assert tight is not None and loose is not None
    assert tight != loose


def test_the_key_separates_two_force_scales(_tree):
    """The force scale *is* part of the criterion, so it belongs in the key."""

    node_count = int(np.asarray(_tree.parent).shape[0])
    small = _key(
        _tree,
        identity=_identity(force_scale_nodes=jnp.full((node_count,), 1e-3)),
    )
    large = _key(
        _tree,
        identity=_identity(force_scale_nodes=jnp.full((node_count,), 1e3)),
    )
    assert small is not None and large is not None
    assert small != large


def test_the_key_separates_folded_per_node_angles(_tree):
    """``dehnen_theta`` carries the criterion in ``geometry.radius``.

    Nothing else in the key can see it: the base MAC is still ``"dehnen"`` and
    ``theta`` still cancels out of acceptance.
    """

    node_count = int(np.asarray(_tree.parent).shape[0])
    tight = _key(
        _tree,
        identity=_identity(mac_geometry_radius=jnp.full((node_count,), 0.1)),
    )
    loose = _key(
        _tree,
        identity=_identity(mac_geometry_radius=jnp.full((node_count,), 0.2)),
    )
    assert tight is not None and loose is not None
    assert tight != loose


def test_a_solver_owned_pair_policy_refuses_the_cache(_tree):
    """A pair-policy request must not be cacheable at all.

    The policy is evaluated against ``policy_state``, which is built from the
    multipole power (so, the *masses*) and the per-particle positions. The
    geometric key hashes neither, so no entry can honestly be shown to match --
    and caching is a perf optimisation, so it yields.
    """

    assert _key(_tree, identity=_identity(pair_policy=object())) is None
    assert _key(_tree, identity=_identity(policy_state=object())) is None


def test_an_identical_policy_keys_identically(_tree):
    """Control: the identity must be a hash, not a nonce."""

    node_count = int(np.asarray(_tree.parent).shape[0])
    scale = jnp.full((node_count,), 3.0)
    first = _key(_tree, identity=_identity(force_scale_nodes=scale))
    second = _key(_tree, identity=_identity(force_scale_nodes=scale))
    assert first is not None
    assert first == second


def test_no_criterion_in_play_leaves_the_geometric_key_alone(_tree):
    """A geometric request keys exactly as it did before this parameter existed."""

    assert _identity() != ""  # eps/mode are set: something criterion-shaped
    bare = pair_policy_cache_identity(
        pair_policy=None,
        policy_state=None,
        eps=None,
        force_scale_mode=None,
        geometry_mode=None,
        theta_max=None,
        error_model_code=None,
        force_scale_nodes=None,
        mac_geometry_radius=None,
    )
    assert bare == ""
    assert _key(_tree, identity=bare) == _key(_tree, identity="")
