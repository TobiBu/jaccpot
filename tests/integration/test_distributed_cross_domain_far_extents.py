"""The distributed cross-domain FAR field, pinned against the direct sum.

WHY THIS FILE IS NOT IN ``tests/distributed/``. Every file there skips below two
devices, which is exactly how the cross-domain far field went unmeasured until
2026-08-21 (audit row F34). At ``ndev=2`` no collective is needed to reproduce a
device's view: ``build_remote_coarse_tree`` all-gathers both frontiers and keeps
``domain != me``, so what device 0 sees *is* the coarse tree over device 1's frontier,
built from the same array in the same order. Replaying that on one device makes the
cross-domain far field testable in CPU CI for the first time. The replay is verified
faithful by its pair counts: it reproduces the driver's ``cross_far_pairs`` /
``cross_near_pairs`` diagnostics (12 far, 51 near for device 0 at the default
``theta_cross=0.1``) exactly.

WHAT IT PINS. Two properties of the cross far field, each measured in both domain
geometries:

1. **Well-separatedness.** Every pair the cross walk accepts as far must satisfy
   ``r_source + r_target < d(centres)`` on the *true* particle extents. Violating it
   means the M2L is evaluated inside the region it is expanding, where the series does
   not converge.
2. **Accuracy against its own exact reference.** The far field must match the direct
   sum over exactly the (target node, source node) pairs the walk accepted -- the cross
   term isolated, with same-domain pairs excluded by construction.

Both hold when the Morton domains are spatially separated and both FAIL when they
interpenetrate, which is the state ``tests/distributed/test_distributed_fmm_driver.py``
is red in. The cause is that ``build_coarse_frontier`` reduces each remote leaf to its
centre of mass and ``build_remote_coarse_tree`` computes the coarse geometry -- hence
the MAC extent -- over those points, so it bounds the centres of mass and not the
particles. See ``docs/distributed_cross_domain_far_diagnosis.md`` for the measurements
and ``bench/diagnose_cross_domain_far.py`` to reproduce them.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yggdrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402
from yggdrax.distributed.let import build_coarse_frontier  # noqa: E402
from yggdrax.dtypes import INDEX_DTYPE  # noqa: E402
from yggdrax.geometry import compute_tree_geometry  # noqa: E402
from yggdrax.tree import (  # noqa: E402
    Tree,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
)
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed import DistributedFMMConfig  # noqa: E402
from jaccpot.distributed.fmm import partition_for_devices  # noqa: E402
from jaccpot.downward.local_expansions import LocalExpansionData  # noqa: E402
from jaccpot.operators.real_harmonics import sh_size  # noqa: E402
from jaccpot.runtime.kernels.core import (  # noqa: E402
    _apply_real_m2l,
    _evaluate_local_expansions_for_particles,
    _propagate_solidfmm_locals_by_level,
)
from jaccpot.upward.real_tree_expansions import (  # noqa: E402
    aggregate_m2m_real_by_level,
    prepare_real_upward_sweep,
)
from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled  # noqa: E402

# The failing driver tests' default config, spelled out so a default change cannot
# silently move what this file measures.
_CONFIG = DistributedFMMConfig()
_ORDER = _CONFIG.order
_LEAF = _CONFIG.leaf_size
_THETA_CROSS = _CONFIG.theta_cross
_MAC = _CONFIG.mac_type
_ROTATION = _CONFIG.rotation
_G = _CONFIG.G
_SOFTENING = _CONFIG.softening
_PER = 64

# Same 1% bar the driver tests hold the total field to, applied to the cross far term
# in isolation. That is strictly harder, because the term is not diluted here by the
# local field or by the near pairs: MEASURED 0.001535 with separated domains (47
# accepted far pairs) against 1.042014 with interpenetrating ones (12 pairs). The
# interpenetrating far term is off by more than 100% of its own reference -- it is not
# an inaccurate approximation of the field, it is uncorrelated with it -- while the
# driver's aggregate sees the same defect as a comparatively mild 1.8e-2 once the local
# field dilutes it.
#
# 0.001535 is an order 3 expansion's truncation error at the separations the MAC
# accepts, so the bar sits ~6.5x above the passing case and ~104x below the failing one
# (which are themselves ~680x apart): nothing about it is finely tuned. The float32 M2L
# floor is ~1e-6 at leaf 8, four orders below, so this threshold tests the scheme and
# not precision.
_CROSS_FAR_RTOL = 1e-2

_INTERPENETRATING_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect in the cross-domain far field, root cause in yggdrax: "
        "build_coarse_frontier reduces each remote leaf to its centre of mass, and "
        "build_remote_coarse_tree computes the coarse tree's geometry -- the MAC "
        "extent the cross walk tests -- over those points. The extent therefore "
        "bounds the centres of mass, not the particles behind them, so a coarse leaf "
        "presents an extent of ~0 however large the remote leaf actually is. Where "
        "the Morton domains interpenetrate (measured: ndev=2 and ndev=3 on the "
        "separated-cluster IC) the MAC accepts pairs whose true separation is smaller "
        "than the source's own radius and the M2L is evaluated inside the region it "
        "expands -- worst accepted pair (r_src + r_tgt)/d = 1.193 against the 0.104 "
        "the MAC computed, an 11x understatement, and the resulting far term is off by "
        "104% of its own direct sum (1.042014 against 0.001535 when the same code runs "
        "on separated domains). The fix is for CoarseFrontier to "
        "carry each leaf's radius and build_remote_coarse_tree to inflate the coarse "
        "extents by it; that takes the cross-field error from 6.4e-1 to 8e-6 at "
        "theta_cross=1.0 while still accepting 10 far pairs. Remove this marker once "
        "yggdrax bounds the true extents. See "
        "docs/distributed_cross_domain_far_diagnosis.md."
    ),
)

_GEOMETRIES = [
    pytest.param(False, id="separated"),
    pytest.param(True, id="interpenetrating", marks=_INTERPENETRATING_XFAIL),
]


def _direct_sum(targets, sources, source_masses):
    """Exact float64 acceleration on ``targets`` from ``(sources, source_masses)``.

    Parameters
    ----------
    targets : np.ndarray
        Target positions ``[T, 3]``.
    sources : np.ndarray
        Source positions ``[S, 3]``.
    source_masses : np.ndarray
        Source masses ``[S]``.

    Returns
    -------
    np.ndarray
        Accelerations ``[T, 3]``, float64 so the reference is never the noisy side.
    """
    diff = (
        np.asarray(targets, np.float64)[:, None, :]
        - np.asarray(sources, np.float64)[None, :, :]
    )
    d2 = (diff**2).sum(-1) + np.float64(_SOFTENING) ** 2
    inv = d2 ** (-1.5)
    masses = np.asarray(source_masses, np.float64)
    return -np.float64(_G) * (masses[None, :, None] * diff * inv[..., None]).sum(axis=1)


def _separated_clusters(ndev, per, seed=4):
    """The failing driver tests' IC: balls of radius 0.5 spaced 6 apart.

    Parameters
    ----------
    ndev : int
        Number of clusters.
    per : int
        Particles per cluster.
    seed : int
        RNG seed; 4 is what the driver tests use.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions ``[ndev * per, 3]`` and masses ``[ndev * per]``, float32.
    """
    rng = np.random.default_rng(seed)
    cluster_centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    pts = np.concatenate(
        [cluster_centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return pts, mass


class _Domain:
    """One device's local tree, real upward sweep, MAC geometry and frontier."""

    def __init__(self, positions, masses, bounds):
        self.tree = Tree.from_particles(
            jnp.asarray(positions),
            jnp.asarray(masses),
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=_LEAF,
        )
        self.positions = self.tree.positions_sorted
        self.masses = self.tree.masses_sorted
        self.upward = prepare_real_upward_sweep(
            self.tree,
            self.positions,
            self.masses,
            max_order=_ORDER,
            max_leaf_size=_LEAF,
        )
        self.geometry = compute_tree_geometry_compiled(
            self.tree, self.positions, max_leaf_size=_LEAF
        )
        moments = compute_tree_mass_moments(self.tree, self.positions, self.masses)
        self.frontier = build_coarse_frontier(
            self.tree, moments.mass, moments.center_of_mass
        )
        self.node_ranges = np.asarray(self.tree.node_ranges)

    def particle_rows(self, node):
        """Particle rows held by ``node``, in this domain's sorted order.

        Parameters
        ----------
        node : int
            Local tree node index.

        Returns
        -------
        np.ndarray
            Row indices into ``positions``/``masses``.
        """
        lo, hi = self.node_ranges[int(node)]
        return np.arange(lo, hi + 1)


class _CrossView:
    """Device 0's view of device 1: the remote coarse tree and its far field.

    Mirrors ``distributed/fmm.py``'s cross-domain stage for ``ndev=2``: the coarse
    tree over the remote frontier, its leaves seeded with the remote leaves' own
    multipoles and aggregated by M2M, the real cross walk, and the M2L -> L2L -> L2P
    evaluation of whatever the walk accepted as far.
    """

    def __init__(self, interpenetrating):
        pts, mass = _separated_clusters(2, _PER)
        part = partition_for_devices(pts, mass, 2, leaf_size=_LEAF)
        bounds = part["bounds"]
        if interpenetrating:
            # What the driver actually does: contiguous ranges of the global Morton
            # order. At ndev=2 on this IC the domains interpenetrate.
            cap = part["cap"]
            pos_d = part["pos_flat"].reshape(2, cap, 3)
            mass_d = part["mass_flat"].reshape(2, cap)
        else:
            # One cluster per domain, which is what the driver tests' docstrings
            # assume and what the Morton split delivers only at ndev=4.
            pos_d = pts.reshape(2, _PER, 3)
            mass_d = mass.reshape(2, _PER)

        # Both parametrisations are self-validating, because neither is a property of
        # this file: the interpenetrating one comes out of partition_for_devices, and a
        # future partitioner that made Morton domains spatially compact would quietly
        # turn it into the separated case -- flipping the strict xfails and reading as
        # "yggdrax fixed the extents" when nothing of the kind happened. Assert the
        # geometry each case is named for, so that change fails loudly here instead.
        # The clusters sit at x=0 and x=6, so x=3 separates them.
        spans_both = [
            bool((pos_d[d][:, 0] < 3.0).any() and (pos_d[d][:, 0] > 3.0).any())
            for d in range(2)
        ]
        if interpenetrating:
            assert all(spans_both), (
                "the interpenetrating case is no longer interpenetrating: "
                f"per-domain 'spans both clusters' = {spans_both}. The Morton "
                "partition changed, so this parametrisation no longer exercises what "
                "it is named for -- re-derive it before trusting any verdict here."
            )
        else:
            assert not any(spans_both), (
                "the separated case is not separated: per-domain 'spans both "
                f"clusters' = {spans_both}"
            )

        self.target = _Domain(pos_d[0], mass_d[0], bounds)
        self.source = _Domain(pos_d[1], mass_d[1], bounds)

        frontier = self.source.frontier
        self.coarse_tree = Tree.from_particles(
            frontier.com,
            frontier.mass,
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=1,
        )
        self.coarse_geometry = compute_tree_geometry(
            self.coarse_tree, self.coarse_tree.positions_sorted, max_leaf_size=1
        )
        perm = jnp.asarray(self.coarse_tree.particle_indices, INDEX_DTYPE)
        self.tag_range = np.asarray(frontier.node_range[perm])
        tag_node_id = np.asarray(frontier.node_id[perm])
        self.coarse_ranges = np.asarray(self.coarse_tree.node_ranges)

        coarse_upward = prepare_real_upward_sweep(
            self.coarse_tree,
            self.coarse_tree.positions_sorted,
            self.coarse_tree.masses_sorted,
            max_order=_ORDER,
            max_leaf_size=1,
        )
        self.coarse_centers = coarse_upward.multipoles.centers
        c_total = self.coarse_centers.shape[0]
        c_internal = int(self.coarse_tree.left_child.shape[0])
        c_leaves = jnp.arange(c_internal, c_total, dtype=INDEX_DTYPE)

        start = jnp.asarray(self.coarse_ranges[np.asarray(c_leaves), 0], INDEX_DTYPE)
        node_id = jnp.asarray(tag_node_id)[start]
        present = node_id >= 0
        seeded = self.source.upward.multipoles.packed[jnp.where(present, node_id, 0)]
        seeded = jnp.where(present[:, None], seeded, 0.0)
        seed = (
            jnp.zeros((c_total, sh_size(_ORDER)), dtype=self.target.positions.dtype)
            .at[c_leaves]
            .set(seeded)
        )
        self.coarse_packed = aggregate_m2m_real_by_level(
            seed,
            self.coarse_centers,
            jnp.asarray(self.coarse_tree.left_child, INDEX_DTYPE),
            jnp.asarray(self.coarse_tree.right_child, INDEX_DTYPE),
            jnp.asarray(get_nodes_by_level(self.coarse_tree), INDEX_DTYPE),
            jnp.asarray(get_level_offsets(self.coarse_tree), INDEX_DTYPE),
            order=_ORDER,
            num_internal=c_internal,
            num_levels=int(get_level_offsets(self.coarse_tree).shape[0] - 1),
            level_batch_width=max(c_internal, 1),
        )

        self.walk = dual_tree_walk_cross_impl(
            self.target.tree,
            self.target.geometry,
            self.coarse_tree,
            self.coarse_geometry,
            _THETA_CROSS,
            mac_type=_MAC,
            max_interactions_per_node=_CONFIG.cross_max_interactions_per_node,
            max_neighbors_per_leaf=_CONFIG.cross_max_neighbors_per_leaf,
            max_pair_queue=_CONFIG.cross_max_pair_queue,
        )
        targets = np.asarray(self.walk.interaction_targets)
        sources = np.asarray(self.walk.interaction_sources)
        live = targets >= 0
        self.far_targets = targets[live]
        self.far_sources = sources[live]

    def source_rows(self, coarse_node):
        """Remote particle rows behind a coarse node.

        Parameters
        ----------
        coarse_node : int
            Coarse tree node index.

        Returns
        -------
        np.ndarray
            Row indices into the remote domain's sorted particles -- the particles the
            halo import would ship for a near pair, and the ones the coarse node's
            multipole stands for in a far pair.
        """
        lo, hi = self.coarse_ranges[int(coarse_node)]
        rows = []
        for particle in range(lo, hi + 1):
            a, b = self.tag_range[particle]
            rows.extend(range(a, b + 1))
        return np.asarray(rows, dtype=np.int64)

    def far_field(self):
        """Evaluate the accepted cross far pairs: M2L, L2L cascade, then L2P.

        Returns
        -------
        np.ndarray
            Accelerations ``[cap, 3]`` on the local particles from the far list alone.
        """
        target = self.target
        total_nodes = int(target.node_ranges.shape[0])
        num_internal = int(target.tree.left_child.shape[0])
        centers = target.upward.multipoles.centers
        deltas = (
            centers[jnp.asarray(self.far_targets, INDEX_DTYPE)]
            - self.coarse_centers[jnp.asarray(self.far_sources, INDEX_DTYPE)]
        )
        contributions = _apply_real_m2l(
            self.coarse_packed[jnp.asarray(self.far_sources, INDEX_DTYPE)],
            deltas,
            order=_ORDER,
            m2l_impl="rot_scale",
        )
        locals_ = jax.ops.segment_sum(
            contributions, jnp.asarray(self.far_targets, INDEX_DTYPE), total_nodes
        )
        locals_ = _propagate_solidfmm_locals_by_level(
            locals_,
            centers,
            jnp.asarray(target.tree.left_child, INDEX_DTYPE),
            jnp.asarray(target.tree.right_child, INDEX_DTYPE),
            get_node_levels(target.tree),
            order=_ORDER,
            rotation=_ROTATION,
            total_nodes=total_nodes,
            basis_mode="real",
            num_levels=None,
        )
        expansion = LocalExpansionData(
            order=_ORDER, centers=centers, coefficients=locals_
        )
        potential_gradient = _evaluate_local_expansions_for_particles(
            expansion,
            target.positions,
            leaf_nodes=jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE),
            node_ranges=jnp.asarray(target.tree.node_ranges, INDEX_DTYPE),
            max_leaf_size=_LEAF,
            order=_ORDER,
            expansion_basis="solidfmm",
            return_potential=False,
        )[0]
        return np.asarray(-_G * potential_gradient)

    def far_field_reference(self):
        """The exact direct sum over exactly the pairs the walk accepted as far.

        Returns
        -------
        np.ndarray
            Accelerations ``[cap, 3]``, float64. Same-domain pairs are excluded by
            construction: every source is a node of the *remote* coarse tree.
        """
        reference = np.zeros_like(np.asarray(self.target.positions, np.float64))
        source_positions = np.asarray(self.source.positions)
        source_masses = np.asarray(self.source.masses)
        target_positions = np.asarray(self.target.positions)
        for source, target in zip(self.far_sources, self.far_targets):
            target_rows = self.target.particle_rows(target)
            source_rows = self.source_rows(source)
            reference[target_rows] += _direct_sum(
                target_positions[target_rows],
                source_positions[source_rows],
                source_masses[source_rows],
            )
        return reference


@pytest.fixture(scope="module")
def cross_views():
    """Both domain geometries, built once (the trees and sweeps are not cheap).

    Returns
    -------
    dict[bool, _CrossView]
        Keyed by ``interpenetrating``.
    """
    return {flag: _CrossView(flag) for flag in (False, True)}


@pytest.mark.parametrize("interpenetrating", _GEOMETRIES)
def test_accepted_cross_far_pairs_are_well_separated(cross_views, interpenetrating):
    """Every accepted cross far pair must be separated on its TRUE extents.

    ``r_source + r_target < d`` is the condition for the M2L series to converge at
    all. The MAC is supposed to enforce it; it tests the coarse tree's COM-only
    extents instead, so this is the assertion that fails when domains interpenetrate.
    """
    view = cross_views[interpenetrating]
    assert view.far_targets.size > 0, "no cross far pairs: the far path is not engaged"

    centers = np.asarray(view.target.upward.multipoles.centers)
    coarse_centers = np.asarray(view.coarse_centers)
    target_positions = np.asarray(view.target.positions)
    source_positions = np.asarray(view.source.positions)

    worst_ratio = 0.0
    worst_pair = None
    for source, target in zip(view.far_sources, view.far_targets):
        target_rows = view.target.particle_rows(target)
        source_rows = view.source_rows(source)
        center_source = coarse_centers[int(source)]
        center_target = centers[int(target)]
        distance = float(np.linalg.norm(center_target - center_source))
        radius_source = float(
            np.linalg.norm(source_positions[source_rows] - center_source, axis=1).max()
        )
        radius_target = float(
            np.linalg.norm(target_positions[target_rows] - center_target, axis=1).max()
        )
        ratio = (radius_source + radius_target) / distance
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_pair = (
                int(target),
                int(source),
                radius_source,
                radius_target,
                distance,
            )

    print(
        f"worst accepted far pair (r_src + r_tgt)/d = {worst_ratio:.3f} over "
        f"{view.far_targets.size} pairs "
        f"({'interpenetrating' if interpenetrating else 'separated'} domains)"
    )
    assert worst_ratio < 1.0, (
        f"cross far pair overlaps its own source region: worst "
        f"(r_src + r_tgt)/d = {worst_ratio:.3f} for (target node, coarse source, "
        f"r_src, r_tgt, d) = {worst_pair}"
    )


@pytest.mark.parametrize("interpenetrating", _GEOMETRIES)
def test_cross_far_field_matches_its_direct_sum(cross_views, interpenetrating):
    """The cross far field must match the direct sum over the pairs it approximates.

    The cross term isolated: the reference sums only over the (target node, coarse
    source) pairs the walk accepted, and every source belongs to the remote coarse
    tree, so same-domain pairs are excluded by construction rather than by masking.
    """
    view = cross_views[interpenetrating]
    assert view.far_targets.size > 0, "no cross far pairs: the far path is not engaged"

    got = view.far_field()
    reference = view.far_field_reference()
    error = float(np.linalg.norm(got - reference) / (np.linalg.norm(reference) + 1e-30))
    print(
        f"cross far aggL2 vs its own direct sum = {error:.6f} over "
        f"{view.far_targets.size} accepted far pairs "
        f"({'interpenetrating' if interpenetrating else 'separated'} domains)"
    )
    assert error < _CROSS_FAR_RTOL, (
        f"cross far field aggL2 {error:.6f} exceeds {_CROSS_FAR_RTOL:g} over "
        f"{view.far_targets.size} accepted far pairs "
        f"(||far|| = {np.linalg.norm(got):.4f}, "
        f"||exact|| = {np.linalg.norm(reference):.4f})"
    )
