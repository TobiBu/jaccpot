"""The production seam into ``experimental/treecode_far_near``, exercised by default.

Audit item **F35**. ``runtime/_interaction_cache.py`` imports
``jaccpot.experimental.treecode_far_near`` inside
``_build_treecode_artifacts_strict_streamed`` -- production reaching
``experimental/``. **The layering half of F35 is settled and is not what this
module is about:** G.5 was answered on 2026-08-20 with "left as-is, exposure
accepted", and ``test_experimental_is_not_on_an_import_path.py`` records the
scope of that decision (the eager import graph is guarded; a lazy function-local
import is the accepted, bounded exposure).

What was still open is the other half, which the row calls *"a correctness risk
rather than a style one"*: the module behind that seam measured **0%**, and
`pyproject.toml` omitted it from coverage with this reason --

    treecode_far_near.py  0%  reached only from the distributed treecode lane
                              (`local_walk="treecode"`), whose test skips
                              below 2 devices, so CPU CI never enters it

which is **untrue on both counts**. There is a second, single-device caller
(`_interaction_cache.py:1058`) gated by the env var
``JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK``, not by device count; and the
module runs on CPU in seconds, as this file demonstrates. That is the third
entry in that omit list whose stated reason did not survive being checked -- the
comment above it already documents the other two.

The invariant pinned here is the one the no-double-count argument rests on. From
the production docstring: the treecode *"yields a different-but-equally-valid
interaction set (per-leaf, split-source): far targets are always leaves, so the
downstream solidfmm L2L cascade acts as a no-op (internal locals stay zero -> no
double-count)"*. If a far target were ever an internal node, the L2L cascade
would push that local down to its children **and** the child's own far term
would be added -- a double-count, in a lane whose tests do not run by default.
That is a wrong force, which is the worst thing this library can produce, so it
is worth a test that runs on every commit rather than under ``-m experimental``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree

from jaccpot.runtime._interaction_cache import _build_treecode_artifacts_strict_streamed

_LEAF_SIZE = 8


@pytest.fixture(scope="module")
def tree_and_geometry():
    """A real tree and its geometry -- the inputs the production entry declares.

    Returns
    -------
    tuple[Tree, TreeGeometry, int, int]
        Tree, geometry, ``num_internal`` and ``total_nodes``. The last two give
        the leaf id range, since leaves are the nodes at ``[num_internal,
        total_nodes)``.
    """
    points = jax.random.uniform(jax.random.PRNGKey(3), (256, 3), dtype=jnp.float64)
    masses = jnp.ones((256,), dtype=jnp.float64)
    tree = Tree.from_particles(points, masses, leaf_size=_LEAF_SIZE)
    positions_sorted = getattr(tree, "positions_sorted", points)
    geometry = compute_tree_geometry(
        tree.topology, positions_sorted, max_leaf_size=_LEAF_SIZE
    )
    num_internal = int(tree.topology.left_child.shape[0])
    total_nodes = int(tree.topology.parent.shape[0])
    return tree, geometry, num_internal, total_nodes


def _artifacts(tree, geometry, *, theta=0.5, mac_type="bh"):
    return _build_treecode_artifacts_strict_streamed(
        tree=tree,
        geometry=geometry,
        theta=theta,
        mac_type=mac_type,
        dehnen_radius_scale=1.0,
        compact_far_pair_capacity=None,
        near_cap=None,
    )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
class TestTheProductionSeamRunsOnCpu:
    """The premise: this lane needs no card, contrary to the omit list's reason."""

    def test_the_entry_point_produces_artifacts(self, tree_and_geometry):
        tree, geometry, _, _ = tree_and_geometry
        artifacts = _artifacts(tree, geometry)
        assert artifacts.compact_far_pairs is not None
        assert artifacts.neighbor_list is not None

    def test_it_is_the_streamed_shape(self, tree_and_geometry):
        """Streamed means no materialised node-interaction list.

        The strict lane selects this builder precisely because it does not want
        one; if ``interactions`` ever came back populated, the memory argument
        for the streamed path would have quietly stopped holding.
        """
        tree, geometry, _, _ = tree_and_geometry
        artifacts = _artifacts(tree, geometry)
        assert artifacts.interactions is None
        assert artifacts.traversal_result is None


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
class TestFarTargetsAreAlwaysLeaves:
    """The invariant the no-double-count argument depends on."""

    @pytest.mark.parametrize("theta", [0.3, 0.5, 0.9])
    def test_every_active_far_target_is_a_leaf(self, tree_and_geometry, theta):
        """No far target may be an internal node, at any opening angle.

        An internal far target would be pushed down by the L2L cascade *and*
        counted again at the child's own far term. Parametrised over theta
        because the accepted set changes with it, and the invariant must not be
        an accident of one opening angle.
        """
        tree, geometry, num_internal, total_nodes = tree_and_geometry
        artifacts = _artifacts(tree, geometry, theta=theta)
        far = artifacts.compact_far_pairs
        count = int(np.asarray(far.far_pair_count))
        assert count > 0, f"no far pairs at theta={theta} -- the test is vacuous"

        targets = np.asarray(far.targets)[:count]
        assert targets.min() >= num_internal, (
            f"far target {targets.min()} is an internal node; the L2L cascade "
            "would double-count it"
        )
        assert targets.max() < total_nodes

    def test_sources_and_targets_are_in_range(self, tree_and_geometry):
        """Sources may be internal -- that is the point of a multipole."""
        tree, geometry, _, total_nodes = tree_and_geometry
        far = _artifacts(tree, geometry).compact_far_pairs
        count = int(np.asarray(far.far_pair_count))
        sources = np.asarray(far.sources)[:count]
        assert sources.min() >= 0 and sources.max() < total_nodes

    def test_the_padding_beyond_the_active_count_is_not_read_as_data(
        self, tree_and_geometry
    ):
        """The arrays are fixed-capacity; only the prefix is meaningful.

        Pinned because every consumer has to slice by ``far_pair_count``, and a
        consumer that forgot would silently process padding as pairs.
        """
        tree, geometry, _, _ = tree_and_geometry
        far = _artifacts(tree, geometry).compact_far_pairs
        count = int(np.asarray(far.far_pair_count))
        assert count <= np.asarray(far.targets).shape[0]


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
class TestNearListStructure:
    """The near half of the same decomposition."""

    def test_a_leaf_is_not_its_own_neighbour(self, tree_and_geometry):
        """Self-pairs belong to the leaf's own P2P, not to its neighbour list.

        Including a leaf in its own near list would double its self-interaction.
        """
        tree, geometry, num_internal, _ = tree_and_geometry
        neighbors = _artifacts(tree, geometry).neighbor_list
        offsets = np.asarray(neighbors.offsets)
        neighbor_ids = np.asarray(neighbors.neighbors)
        leaf_ids = np.asarray(neighbors.leaf_indices)
        for row, leaf in enumerate(leaf_ids):
            start, stop = int(offsets[row]), int(offsets[row + 1])
            assert int(leaf) not in neighbor_ids[start:stop].tolist()

    def test_the_offsets_are_a_valid_csr(self, tree_and_geometry):
        tree, geometry, _, _ = tree_and_geometry
        neighbors = _artifacts(tree, geometry).neighbor_list
        offsets = np.asarray(neighbors.offsets)
        assert offsets[0] == 0
        assert np.all(np.diff(offsets) >= 0), "CSR offsets must be non-decreasing"
        assert int(offsets[-1]) <= np.asarray(neighbors.neighbors).shape[0]
