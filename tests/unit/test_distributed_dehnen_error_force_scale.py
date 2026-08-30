"""Dehnen eq (16b)'s force scale, assembled from a decomposed domain's own lists.

On one device the ``f_b`` estimator reads the interaction lists the traversal
already built. On a mesh those lists cover only the device's own particles, so
:mod:`jaccpot.distributed._force_scale` adds two remote terms from the cross walk.
These tests pin the properties that make that assembly safe, each of which has a
way to be silently wrong:

* it stays a **lower bound** on the exact global ``f_b``. eq (16a)'s threshold is
  ``eps * s``, so an over-large scale loosens acceptance -- the solver gets faster
  *and* wronger, which no cost measurement can detect (trap 6 in
  ``docs/dehnen_mass_mac_status_and_plan.md``);
* the **remote terms are load-bearing**. Dropping them leaves a scale that is still
  a valid lower bound and still looks like a plausible force scale, while being
  wrong by roughly the domain fraction -- so there is a test that fails if they go;
* the **remote near term inflates by both radii**, without which monopole-at-COM is
  a bound in neither direction and the lower-bound property is lost.

The "coarse tree" here is a real tree over the remote particles rather than a leaf
frontier. The cross walk's contract is target-tree-against-source-tree either way,
and using real particles is what makes the exact ``f_b`` computable to compare
against.

No mesh is needed: every function under test is an ordinary array function.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl
from yggdrax.interactions import build_interactions_and_neighbors
from yggdrax.tree import Tree

from jaccpot.distributed._force_scale import (
    cross_force_scale_own,
    distributed_force_scale_nodes,
    flatten_neighbor_csr,
)
from jaccpot.runtime._adaptive_policy import (
    _far_field_force_scale_by_node,
    accumulate_own_down_parent_chain,
    compute_node_force_scale_from_sorted_magnitudes,
)
from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled

LEAF = 8
THETA = 0.5
SOFTENING = 1.0e-3
G = 1.0

#: Per-domain particle count. NOT arbitrary: at 256 particles / leaf 8 / theta 0.5
#: the self walk accepts **zero** far pairs, so the far term under test is
#: identically zero and every assertion here passes without measuring it (trap 11 in
#: ``docs/dehnen_mass_mac_status_and_plan.md``, at small N). Measured on this IC:
#: 256 -> 0 far pairs, 1024 -> 1256, 2048 -> 11942. The vacuity guards below fail
#: rather than pass if this is ever lowered back.
N_PER_DOMAIN = 1024


def _two_domains(
    n_local: int = N_PER_DOMAIN,
    n_remote: int = N_PER_DOMAIN,
    *,
    seed: int = 20260830,
):
    """A local clump and a remote one, far enough apart to be separate domains."""

    rng = np.random.default_rng(seed)
    local = rng.normal(scale=1.0, size=(n_local, 3))
    remote = rng.normal(scale=1.0, size=(n_remote, 3)) + np.array([8.0, 0.0, 0.0])
    m_local = rng.uniform(0.5, 2.0, size=(n_local,))
    m_remote = rng.uniform(0.5, 2.0, size=(n_remote,))
    return (
        local.astype(np.float64),
        m_local.astype(np.float64),
        remote.astype(np.float64),
        m_remote.astype(np.float64),
    )


def _build(points, masses):
    """Tree, geometry, sorted payload and the self walk's far/near lists."""

    tree = Tree.from_particles(
        jnp.asarray(points),
        jnp.asarray(masses),
        tree_type="radix",
        return_reordered=True,
        leaf_size=LEAF,
    )
    order = np.asarray(tree.particle_indices)
    pos_sorted = jnp.asarray(points)[order]
    mass_sorted = jnp.asarray(masses)[order]
    geom = compute_tree_geometry_compiled(tree, pos_sorted, max_leaf_size=LEAF)
    inter, nbr = build_interactions_and_neighbors(
        tree, geom, theta=THETA, mac_type="dehnen"
    )
    return tree, geom, pos_sorted, mass_sorted, inter, nbr


def _exact_fb(target_pos, source_pos, source_mass, *, exclude_self):
    """``sum_a G m_a / |x_a - x_b|^2`` over the given sources, per target."""

    diff = np.asarray(source_pos)[None, :, :] - np.asarray(target_pos)[:, None, :]
    dist_sq = np.sum(diff * diff, axis=-1) + SOFTENING * SOFTENING
    contrib = np.asarray(source_mass)[None, :] / dist_sq
    if exclude_self:
        np.fill_diagonal(contrib, 0.0)
    return G * contrib.sum(axis=1)


def _scale(local, remote, *, with_remote: bool):
    """Run the assembly, optionally with the remote terms zeroed out."""

    lp, lm = local
    rp, rm = remote
    tree, geom, pos_s, mass_s, inter, nbr = _build(lp, lm)
    rtree, rgeom, rpos_s, rmass_s, _, _ = _build(rp, rm)
    cross = dual_tree_walk_cross_impl(
        tree,
        geom,
        rtree,
        rgeom,
        THETA,
        mac_type="dehnen",
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
        max_pair_queue=262144,
    )
    empty_pairs = jnp.zeros((0,), dtype=jnp.int32)
    empty_counts = jnp.zeros_like(jnp.asarray(cross.neighbor_counts))
    return (
        tree,
        pos_s,
        np.asarray(
            distributed_force_scale_nodes(
                tree=tree,
                positions_sorted=pos_s,
                masses_sorted=mass_s,
                node_centers=geom.center,
                node_radii=geom.radius,
                self_far_sources=inter.sources,
                self_far_targets=inter.targets,
                self_near_offsets=nbr.offsets,
                self_near_counts=nbr.counts,
                self_near_indices=nbr.neighbors,
                self_near_leaf_indices=nbr.leaf_indices,
                coarse_tree=rtree,
                coarse_masses_sorted=rmass_s,
                coarse_centers=rgeom.center,
                coarse_radii=rgeom.radius,
                cross_far_sources=(
                    cross.interaction_sources if with_remote else empty_pairs
                ),
                cross_far_targets=(
                    cross.interaction_targets if with_remote else empty_pairs
                ),
                cross_near_counts=(
                    cross.neighbor_counts if with_remote else empty_counts
                ),
                cross_near_indices=cross.neighbor_indices,
                cross_near_leaf_indices=cross.leaf_indices,
                max_leaf_size=LEAF,
                softening=SOFTENING,
                gravitational_constant=G,
            )
        ),
    )


def _assert_the_self_walk_has_a_far_field(points, masses) -> None:
    """Fail if this IC accepts no far pairs, so nothing here can pass vacuously.

    The monopole far term is the piece these tests exist to check. With no far
    pairs it is identically zero, every bound holds trivially and the assertions
    say nothing -- which is exactly what a 256-particle IC did here.

    Parameters
    ----------
    points : Any
        Particle positions for the local domain.
    masses : Any
        Particle masses for the local domain.
    """

    _, _, _, _, inter, _ = _build(points, masses)
    sources = np.asarray(inter.sources)
    targets = np.asarray(inter.targets)
    far = int(((sources >= 0) & (targets >= 0)).sum())
    assert far > 0, (
        f"the self walk accepted {far} far pairs at N={len(masses)} / leaf {LEAF} / "
        f"theta {THETA}, so the far term under test is identically zero"
    )


# --------------------------------------------------------------------------- #
# the CSR expansion
# --------------------------------------------------------------------------- #


def test_flatten_neighbor_csr_recovers_every_pair_with_its_own_target():
    """Each flat entry must come back paired with the leaf whose list holds it.

    Getting this wrong attributes remote mass to the wrong local node, which
    perturbs the criterion without changing any count -- so no diagnostic sees it.
    """

    counts = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)
    leaf_indices = jnp.asarray([10, 11, 12, 13], dtype=jnp.int32)
    indices = jnp.asarray([100, 101, 200, 201, 202, 300, -1, -1], dtype=jnp.int32)

    src, tgt, valid = flatten_neighbor_csr(
        counts=counts, indices=indices, leaf_indices=leaf_indices
    )
    src, tgt, valid = np.asarray(src), np.asarray(tgt), np.asarray(valid)

    got = sorted(zip(src[valid].tolist(), tgt[valid].tolist()))
    expected = sorted(
        [(100, 10), (101, 10), (200, 12), (201, 12), (202, 12), (300, 13)]
    )
    assert got == expected
    # The padded tail is dropped, not attributed to the last leaf.
    assert int(valid.sum()) == 6


# --------------------------------------------------------------------------- #
# the property that makes it safe
# --------------------------------------------------------------------------- #


def test_the_estimate_is_a_lower_bound_on_the_exact_global_fb():
    """Every node's scale must not exceed the exact ``min_b f_b`` over its span.

    The direction is the whole point. ``eps * s`` is the acceptance threshold, so
    an over-large ``s`` accepts pairs the criterion should have refused: cheaper,
    and wrong, with nothing in the cost to show for it.
    """

    lp, lm, rp, rm = _two_domains()
    _assert_the_self_walk_has_a_far_field(lp, lm)
    tree, pos_s, scale = _scale((lp, lm), (rp, rm), with_remote=True)

    # Exact global f_b for every local particle: local sources (self excluded)
    # plus every remote source.
    order = np.asarray(tree.particle_indices)
    mass_s = np.asarray(lm)[order]
    fb = _exact_fb(np.asarray(pos_s), np.asarray(pos_s), mass_s, exclude_self=True)
    fb = fb + _exact_fb(np.asarray(pos_s), rp, rm, exclude_self=False)

    ranges = np.asarray(tree.node_ranges)
    checked = 0
    for node in range(ranges.shape[0]):
        lo, hi = int(ranges[node, 0]), int(ranges[node, 1])
        if hi < lo:
            continue
        exact_min = float(fb[lo : hi + 1].min())
        assert scale[node] <= exact_min * (1.0 + 1e-9), (
            f"node {node} scale {scale[node]:.6e} exceeds the exact min_b f_b "
            f"{exact_min:.6e} -- the criterion would over-accept"
        )
        checked += 1
    assert checked > 1, "the tree collapsed to a single node; this proves nothing"


def test_the_remote_terms_are_load_bearing():
    """Dropping the cross-domain terms must move the scale, and downward.

    A local-only estimate is still a lower bound and still looks like a force
    scale, so nothing about its *shape* announces that half the system is missing.
    This is the test that fails if the cross walk's contribution is ever dropped.
    """

    lp, lm, rp, rm = _two_domains()
    _assert_the_self_walk_has_a_far_field(lp, lm)
    _, _, with_remote = _scale((lp, lm), (rp, rm), with_remote=True)
    _, _, local_only = _scale((lp, lm), (rp, rm), with_remote=False)

    assert np.all(
        local_only <= with_remote * (1.0 + 1e-12)
    ), "adding remote sources must only raise f_b: it is a sum of positive terms"
    live = with_remote > 0
    gain = float(np.median(with_remote[live] / np.maximum(local_only[live], 1e-300)))
    assert gain > 1.01, (
        f"remote terms changed the median scale by only {gain:.4f}x -- either the "
        "cross lists are empty or their contribution is being discarded"
    )


def test_the_remote_near_term_inflates_by_both_radii():
    """The source radius must enter the reach, and it must lower the estimate.

    A near pair's source is not compact by construction -- that is what makes it
    near -- so monopole-at-COM is a bound in neither direction. Adding the source
    radius restores the under-estimate. If this ever stops mattering, the reach
    formula has drifted.
    """

    lp, lm, rp, rm = _two_domains()
    tree, geom, pos_s, mass_s, _, _ = _build(lp, lm)
    rtree, rgeom, _, rmass_s, _, _ = _build(rp, rm)
    cross = dual_tree_walk_cross_impl(
        tree,
        geom,
        rtree,
        rgeom,
        THETA,
        mac_type="dehnen",
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
        max_pair_queue=262144,
    )
    src, tgt, valid = flatten_neighbor_csr(
        counts=cross.neighbor_counts,
        indices=cross.neighbor_indices,
        leaf_indices=cross.leaf_indices,
    )
    assert int(np.asarray(valid).sum()) > 0, "no cross near pairs; nothing measured"

    from jaccpot.runtime._adaptive_policy import node_span_mass

    common = dict(
        source_masses=node_span_mass(tree=rtree, masses_sorted=rmass_s),
        source_centers=rgeom.center,
        target_centers=geom.center,
        target_radii=geom.radius,
        pair_sources=src,
        pair_targets=tgt,
        pair_valid=valid,
        num_target_nodes=int(np.asarray(tree.node_ranges).shape[0]),
        g=jnp.asarray(G),
        eps_sq=jnp.asarray(SOFTENING**2),
        inflation=jnp.asarray(1.0),
    )
    with_radii = np.asarray(cross_force_scale_own(source_radii=rgeom.radius, **common))
    without = np.asarray(cross_force_scale_own(source_radii=None, **common))

    assert np.all(with_radii <= without + 1e-12)
    assert float(with_radii.sum()) < float(
        without.sum()
    ), "the source radius made no difference, so it is not entering the reach"


# --------------------------------------------------------------------------- #
# the refusals and the shared helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["paper", "paper_cached", "paper_fb_cached"])
def test_eq_16a_force_scale_modes_are_refused_rather_than_approximated(mode):
    """``min_b |a_b|`` is an acceleration, so this lane must say so, not guess.

    Substituting ``f_b`` for it silently would answer a different criterion at a
    different tolerance while reporting the mode the caller asked for.
    """

    lp, lm, rp, rm = _two_domains(64, 64)
    with pytest.raises(ValueError, match="eq \\(16b\\) only"):
        _scale_with_mode(lp, lm, rp, rm, mode)


def _scale_with_mode(lp, lm, rp, rm, mode):
    tree, geom, pos_s, mass_s, inter, nbr = _build(lp, lm)
    rtree, rgeom, _, rmass_s, _, _ = _build(rp, rm)
    empty = jnp.zeros((0,), dtype=jnp.int32)
    return distributed_force_scale_nodes(
        tree=tree,
        positions_sorted=pos_s,
        masses_sorted=mass_s,
        node_centers=geom.center,
        node_radii=geom.radius,
        self_far_sources=inter.sources,
        self_far_targets=inter.targets,
        self_near_offsets=nbr.offsets,
        self_near_counts=nbr.counts,
        self_near_indices=nbr.neighbors,
        self_near_leaf_indices=nbr.leaf_indices,
        coarse_tree=rtree,
        coarse_masses_sorted=rmass_s,
        coarse_centers=rgeom.center,
        coarse_radii=rgeom.radius,
        cross_far_sources=empty,
        cross_far_targets=empty,
        cross_near_counts=jnp.zeros((1,), dtype=jnp.int32),
        cross_near_indices=empty,
        cross_near_leaf_indices=jnp.zeros((1,), dtype=jnp.int32),
        max_leaf_size=LEAF,
        softening=SOFTENING,
        gravitational_constant=G,
        force_scale_mode=mode,
    )


def test_a_static_leaf_width_reduces_onto_nodes_exactly_as_the_host_read_does():
    """The traced path must not be a different reduction from the untraced one.

    ``max_leaf_size`` exists only because the exact per-leaf maximum is a host read
    of a traced value. It is a *bound*, not the maximum, so the gathered block is
    wider and the extra columns are masked -- the result has to be identical, or
    the distributed lane is silently running a different reduction.
    """

    lp, lm, _, _ = _two_domains()
    tree, _, _, _, _, _ = _build(lp, lm)
    order = np.asarray(tree.particle_indices)
    values = jnp.asarray(np.asarray(lm)[order] + 0.25)

    host_read = compute_node_force_scale_from_sorted_magnitudes(
        tree=tree, magnitudes_sorted=values, reduction="min"
    )
    static = compute_node_force_scale_from_sorted_magnitudes(
        tree=tree, magnitudes_sorted=values, reduction="min", max_leaf_size=LEAF
    )
    np.testing.assert_array_equal(np.asarray(host_read), np.asarray(static))

    # A generous bound must give the same answer too -- the mask, not the width,
    # is what selects a leaf's own particles.
    wide = compute_node_force_scale_from_sorted_magnitudes(
        tree=tree, magnitudes_sorted=values, reduction="min", max_leaf_size=4 * LEAF
    )
    np.testing.assert_array_equal(np.asarray(host_read), np.asarray(wide))


def test_the_extracted_parent_chain_accumulation_matches_the_far_field_helper():
    """The shared push-down must reproduce what it was extracted from.

    Parents have to be accumulated before their children and radix-tree internal
    indices are not in postorder, which is the ordering trap that once dropped 97%
    of the system mass in the M2M pass. Extracting the loop is only safe if this
    holds.
    """

    lp, lm, _, _ = _two_domains()
    tree, geom, pos_s, mass_s, inter, _ = _build(lp, lm)

    combined = _far_field_force_scale_by_node(
        tree=tree,
        masses=mass_s,
        node_centers=geom.center,
        node_radii=geom.radius,
        interaction_sources=jnp.asarray(inter.sources, dtype=jnp.int32),
        interaction_targets=jnp.asarray(inter.targets, dtype=jnp.int32),
        g=jnp.asarray(G),
        eps_sq=jnp.asarray(SOFTENING**2),
        inflation=jnp.asarray(1.0),
    )
    from jaccpot.runtime._adaptive_policy import node_span_mass

    own = cross_force_scale_own(
        source_masses=node_span_mass(tree=tree, masses_sorted=mass_s),
        source_centers=geom.center,
        source_radii=None,
        target_centers=geom.center,
        target_radii=geom.radius,
        pair_sources=inter.sources,
        pair_targets=inter.targets,
        pair_valid=None,
        num_target_nodes=int(np.asarray(tree.node_ranges).shape[0]),
        g=jnp.asarray(G),
        eps_sq=jnp.asarray(SOFTENING**2),
        inflation=jnp.asarray(1.0),
    )
    rebuilt = accumulate_own_down_parent_chain(tree=tree, own=own)

    assert float(np.abs(np.asarray(combined)).max()) > 0.0, "no far field to compare"
    np.testing.assert_allclose(
        np.asarray(rebuilt), np.asarray(combined), rtol=1e-12, atol=0.0
    )


# --------------------------------------------------------------------------- #
# cap sizing -- theta gates the cross walk, and nothing on the self walk
# --------------------------------------------------------------------------- #


def test_the_criterion_floors_the_self_queue_instead_of_scaling_it_by_theta():
    """A loose theta must not shrink the self queue when a pair policy decides.

    ``_derive_walk_caps`` scales every wavefront queue as ``(0.4 / theta) ** 1.5``,
    which is right whenever ``theta`` is what the walk accepts on. Under
    ``mac_type="dehnen_error"`` it is not: ``adaptive_pair_policy`` deletes
    ``mac_ok`` outright in paper mode, so the geometric verdict decides nothing and
    ``adaptive_eps`` is the accuracy knob. At a loose ``theta`` and a tight ``eps``
    the unmodified rule under-provisions the self queue -- and the rule's own
    docstring says what that costs: the walk truncates SILENTLY, reading *faster*
    with only ``self_near_pairs`` as the witness.

    The cross walk really is geometric here, so its queue must keep tracking
    ``theta``. Both halves are asserted: a criterion config at theta 0.8 gets the
    self queue of theta 0.3 and the cross queue of theta 0.8.
    """

    from jaccpot.distributed.fmm import (
        _SELF_QUEUE_CRITERION_THETA,
        DistributedFMMConfig,
    )

    common = dict(leaf_size=64, theta=0.8)
    geometric = DistributedFMMConfig(**common).resolved_for(262144, 4)
    criterion = DistributedFMMConfig(
        **common, mac_type="dehnen_error", adaptive_eps=1e-4
    ).resolved_for(262144, 4)
    floored = DistributedFMMConfig(
        **{**common, "theta": _SELF_QUEUE_CRITERION_THETA}
    ).resolved_for(262144, 4)

    assert criterion.max_pair_queue > geometric.max_pair_queue, (
        "the criterion's self queue was sized by a theta that gates nothing "
        f"({criterion.max_pair_queue} vs geometric {geometric.max_pair_queue})"
    )
    assert criterion.max_pair_queue == floored.max_pair_queue, (
        "the criterion's self queue is not the theta-0.3 floor "
        f"({criterion.max_pair_queue} vs {floored.max_pair_queue})"
    )
    assert (
        criterion.cross_max_pair_queue == geometric.cross_max_pair_queue
    ), "the CROSS walk is still geometric, so its queue must keep tracking theta"


def test_a_theta_tighter_than_the_floor_still_raises_the_self_queue():
    """The floor is a floor, not a pin.

    A configured ``theta`` below ``_SELF_QUEUE_CRITERION_THETA`` means the prepass
    walk underneath the criterion is itself tighter, so the queue requirement is
    larger, not capped.
    """

    from jaccpot.distributed.fmm import DistributedFMMConfig

    floored = DistributedFMMConfig(
        leaf_size=64, theta=0.3, mac_type="dehnen_error", adaptive_eps=1e-4
    ).resolved_for(262144, 4)
    tighter = DistributedFMMConfig(
        leaf_size=64, theta=0.15, mac_type="dehnen_error", adaptive_eps=1e-4
    ).resolved_for(262144, 4)

    assert tighter.max_pair_queue > floored.max_pair_queue
