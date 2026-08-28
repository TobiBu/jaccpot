"""The cell-level rung the distributed far field splits on, checked against an oracle.

``jaccpot.mutual.distributed._node_rungs`` is the distributed lane's cell rung: each
node takes the rung of its most active (finest) massive particle, which is falcON
activity-gating (strategy B2) and the same predicate
``jaccpot.mutual.force._cell_rungs`` implements for the single-device lane.

It is worth its own unit test for two reasons. It gets there by a different route --
a range maximum expressed as one prefix count per level, because this module has only
the inclusive node ranges and not the level schedule ``_cell_rungs`` walks -- and a
wrong cell rung is invisible downstream. A pair whose level is wrong but *consistently*
wrong is still a single symmetric scalar per pair, so it still conserves momentum
exactly, the levels still partition the total, and the weighting is still linear. Every
criterion in ``tests/integration/test_mutual_distributed.py`` passes. Only comparing
the assignment itself catches it.

No devices needed: this is pure array arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from jaccpot.mutual.distributed import _node_rungs


def _oracle(node_ranges, rung, masses):
    """Max rung over each inclusive range, counting massive particles only."""
    out = np.full(len(node_ranges), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(node_ranges):
        sel = masses[lo : hi + 1] > 0
        if sel.any():
            out[i] = rung[lo : hi + 1][sel].max()
    return out


def _complete_ranges(n_leaves, leaf):
    """Inclusive ranges for a complete binary tree, internal nodes first."""
    total = 2 * n_leaves - 1
    left = np.full(total, -1, dtype=np.int64)
    right = np.full(total, -1, dtype=np.int64)
    for i in range(n_leaves - 1):
        left[i], right[i] = 2 * i + 1, 2 * i + 2
    ranges = np.zeros((total, 2), dtype=np.int32)
    for li in range(n_leaves):
        ranges[n_leaves - 1 + li] = (li * leaf, (li + 1) * leaf - 1)
    for i in range(n_leaves - 2, -1, -1):
        ranges[i] = (ranges[left[i]][0], ranges[right[i]][1])
    return ranges


@pytest.mark.parametrize("k_max", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 5])
def test_a_node_takes_the_rung_of_its_finest_massive_particle(k_max, seed):
    """Against a loop-written oracle, over internal nodes as well as leaves.

    Internal nodes matter here specifically: a cross-domain FAR pair's remote endpoint
    must be a coarse leaf (it has to be addressable), but its LOCAL endpoint may be an
    internal node, so the whole array is consulted and not just the leaf rows.
    """
    rng = np.random.default_rng(seed)
    n_leaves, leaf = 8, 4
    n = n_leaves * leaf
    ranges = _complete_ranges(n_leaves, leaf)
    rung = rng.integers(0, k_max + 1, size=n).astype(np.int32)
    # Pin the first leaf to rung 0 so the node array is guaranteed non-constant when
    # there is more than one level. A free draw is not: with leaf 4 and k_max 1 almost
    # every leaf contains a 1, so every node maximum is 1 and the guard below would
    # fire on a correct answer.
    rung[:leaf] = 0
    masses = rng.uniform(0.5, 1.5, size=n)

    got = np.asarray(
        _node_rungs(jnp.asarray(ranges), jnp.asarray(rung), jnp.asarray(masses), k_max)
    )
    want = _oracle(ranges, rung, masses)
    np.testing.assert_array_equal(got, want)
    # Vacuity guard: a single-rung draw would make the max trivial.
    if k_max > 0:
        assert len(np.unique(want)) > 1, "the oracle is constant -- test is vacuous"


def test_a_massless_particle_cannot_decide_a_cells_level():
    """Padding must not raise a cell's rung, or the two lanes disagree on a padded domain.

    ``partition_for_devices`` pads every domain to a common capacity with mass-0 rows.
    Those rows exert no force, so letting one set a cell's level would over-refine the
    far field for a particle that is not there -- and would do it differently from the
    single-device lane, which has no padding at all.
    """
    ranges = np.array([[0, 3], [0, 1], [2, 3]], dtype=np.int32)
    rung = np.array([0, 1, 3, 3], dtype=np.int32)
    masses = np.array([1.0, 1.0, 0.0, 0.0])
    got = np.asarray(
        _node_rungs(jnp.asarray(ranges), jnp.asarray(rung), jnp.asarray(masses), 3)
    )
    # Node 2 holds only massless rows, so it has no level at all.
    np.testing.assert_array_equal(got, np.array([1, 1, -1]))


def test_an_all_massless_node_reports_the_minus_one_sentinel():
    """``-1`` means "no particle here", the same sentinel ``_cell_rungs`` uses.

    Every consumer clamps into ``[0, k_max]`` afterwards, so ``-1`` and ``0`` give the
    same weight -- and such a node carries no mass, hence no force. The sentinel is
    kept anyway so the two lanes' intermediate values agree and can be compared.
    """
    ranges = np.array([[0, 1]], dtype=np.int32)
    got = np.asarray(
        _node_rungs(
            jnp.asarray(ranges),
            jnp.asarray(np.array([2, 2], dtype=np.int32)),
            jnp.asarray(np.zeros(2)),
            2,
        )
    )
    assert int(got[0]) == -1


def test_the_range_maximum_agrees_with_the_single_device_propagation():
    """Same answer as ``force._cell_rungs``, which gets it by a level-schedule scan.

    The two lanes must split the far field the same way, or a system's levels stop
    partitioning it the moment it is spread across devices. ``_cell_rungs`` propagates
    a leaf maximum up the tree; this is the range maximum of the same particles, so
    they agree by construction -- which is exactly the kind of "by construction" worth
    checking once, on a real tree rather than the hand-built ones above.

    Built through the DEVICE topology because that is the one that takes node ranges as
    an input, and is what the distributed driver builds. ``MutualTreeArrays`` does not
    carry them.
    """
    pytest.importorskip("yggdrax")
    from yggdrax.tree import Tree

    from jaccpot.mutual.device_topology import build_mutual_state_device
    from jaccpot.mutual.distributed import _default_caps
    from jaccpot.mutual.force import _cell_rungs

    rng = np.random.default_rng(4)
    n, k_max, leaf = 96, 2, 8
    pos = jnp.asarray(rng.normal(size=(n, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    tree = Tree.from_particles(
        pos, mass, tree_type="radix", return_reordered=True, leaf_size=leaf
    )
    parent = jnp.asarray(tree.parent, dtype=jnp.int32)
    node_ranges = jnp.asarray(tree.node_ranges, dtype=jnp.int32)
    state = build_mutual_state_device(
        tree.positions_sorted,
        tree.masses_sorted,
        parent=parent,
        left_child=jnp.asarray(tree.left_child, dtype=jnp.int32),
        right_child=jnp.asarray(tree.right_child, dtype=jnp.int32),
        node_ranges=node_ranges,
        inverse_permutation=jnp.arange(n, dtype=jnp.int32),
        root=jnp.argmin(parent).astype(jnp.int32),
        theta=0.5,
        order=2,
        leaf_size=leaf,
        caps=_default_caps(n, leaf),
        softening=1e-2,
    )

    # Per-LEAF blocks rather than a free draw, plus two rung-mixed leaves. A free
    # draw is vacuous here: with leaf 8 and three levels every leaf contains a 2, so
    # every cell rung is 2 and the comparison holds for a function that ignored its
    # input. Blocks make the answer vary, and the two flips keep the mixed-cell case
    # -- the one the cell-level split over-refines -- in the comparison.
    r = ((np.arange(n) // leaf) % (k_max + 1)).astype(np.int32)
    r[3] = k_max
    r[leaf + 1] = 0
    rung_sorted = jnp.asarray(r)
    want = np.asarray(_cell_rungs(state, rung_sorted))
    got = np.asarray(_node_rungs(node_ranges, rung_sorted, tree.masses_sorted, k_max))
    # `_cell_rungs` reduces padded LEAF BLOCKS with a -1 fill and never sees a
    # massless particle, so the two differ only on nodes holding nothing at all,
    # where it reports -1 as well. Compare where either says something.
    real = want >= 0
    assert real.sum() > 0, "no populated nodes -- test would be vacuous"
    assert len(np.unique(want[real])) > 1, "constant cell rungs -- test is vacuous"
    np.testing.assert_array_equal(got[real], want[real])
