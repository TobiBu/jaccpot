"""The far/near partition must cover every source particle exactly once.

An FMM is only correct if, for each target leaf, the accepted far-field source
nodes plus the near-field source leaves partition the particle set: every source
particle must contribute exactly once, either through an M2L into the target (or
one of its ancestors, whose local expansion is passed down by L2L) or through a
direct P2P.

There is no such invariant elsewhere in the suite, and it is the only test that
can see a dropped pair. Dropped pairs arise when a solver-owned ``pair_policy``
is asymmetric in target<->source: the traversal evaluates the policy in both
orientations and only accepts on ``accept_both`` / marks near on ``near_both``,
so a disagreeing leaf-leaf pair falls through to REFINE -- and a leaf-leaf pair
cannot be refined, so it is silently dropped. Dehnen's eq (16a) is genuinely
asymmetric (source mass and power against the sink's force scale), which is why
the ``dehnen_paper`` model needs an explicit symmetrization.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FMMAdvancedConfig
from jaccpot.solver import FastMultipoleMethod

N = 512
LEAF = 16
ORDER = 4


def _problem(seed: int = 0, clustered: bool = False):
    rng = np.random.default_rng(seed)
    if clustered:
        # two clumps with a 1:100 mass ratio -- the regime a mass-dependent MAC
        # is supposed to exploit, and where an asymmetric policy is most likely
        # to disagree between orientations.
        half = N // 2
        pos = np.concatenate(
            [
                rng.normal(scale=0.15, size=(half, 3)),
                rng.normal(scale=0.15, size=(N - half, 3)) + np.array([3.0, 0.0, 0.0]),
            ]
        )
        mass = np.concatenate([np.full(half, 1.0), np.full(N - half, 0.01)])
    else:
        pos = rng.uniform(-1.0, 1.0, size=(N, 3))
        mass = rng.uniform(0.5, 1.5, size=N)
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


def _ancestors(parent: np.ndarray, node: int) -> list[int]:
    """Return ``node`` and every ancestor up to the root."""

    chain = [node]
    cur = node
    while True:
        nxt = int(parent[cur])
        if nxt < 0 or nxt == cur:
            break
        chain.append(nxt)
        cur = nxt
    return chain


def _coverage_counts(state) -> np.ndarray:
    """Return, per (target leaf, source particle), how many times it contributes."""

    tree = state.tree
    parent = np.asarray(tree.parent)
    node_ranges = np.asarray(tree.node_ranges)
    num_internal = int(tree.num_internal_nodes)
    num_nodes = int(parent.shape[0])
    leaf_nodes = list(range(num_internal, num_nodes))

    # Far field: an accepted (target, source) pair deposits a local expansion on
    # `target`, which L2L then pushes down to every leaf below it. So a leaf is
    # served by the accepted sources of itself and of all its ancestors.
    far_by_node: dict[int, list[int]] = {}
    interactions = state.interactions
    assert interactions is not None, "test requires a retained interaction list"
    src = np.asarray(interactions.sources)
    tgt = np.asarray(interactions.targets)
    for s, t in zip(src.tolist(), tgt.tolist()):
        if s < 0 or t < 0:
            continue
        far_by_node.setdefault(int(t), []).append(int(s))

    neighbor_list = state.neighbor_list
    nb_offsets = np.asarray(neighbor_list.offsets)
    nb_counts = np.asarray(neighbor_list.counts)
    nb_neighbors = np.asarray(neighbor_list.neighbors)
    nb_leaf_indices = np.asarray(neighbor_list.leaf_indices)

    counts = np.zeros((len(leaf_nodes), N), dtype=np.int32)
    for row, leaf in enumerate(leaf_nodes):
        # far contributions, inherited down the ancestor chain
        for node in _ancestors(parent, leaf):
            for source_node in far_by_node.get(node, ()):
                lo, hi = int(node_ranges[source_node, 0]), int(
                    node_ranges[source_node, 1]
                )
                if hi >= lo:
                    counts[row, lo : hi + 1] += 1
        # near contributions
        slot = int(np.nonzero(nb_leaf_indices == leaf)[0][0])
        start = int(nb_offsets[slot])
        for k in range(int(nb_counts[slot])):
            source_leaf = int(nb_neighbors[start + k])
            if source_leaf < 0:
                continue
            lo, hi = int(node_ranges[source_leaf, 0]), int(node_ranges[source_leaf, 1])
            if hi >= lo:
                counts[row, lo : hi + 1] += 1
        # a leaf always evaluates itself directly
        lo, hi = int(node_ranges[leaf, 0]), int(node_ranges[leaf, 1])
        if hi >= lo:
            counts[row, lo : hi + 1] += 1
    return counts


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize(
    "mac_type,adaptive_eps",
    [("dehnen", None), ("dehnen_error", 1e-3), ("dehnen_error", 1e-4)],
)
def test_far_near_partition_is_complete(mac_type, adaptive_eps, clustered):
    positions, masses = _problem(clustered=clustered)
    advanced = FMMAdvancedConfig()
    advanced = replace(
        advanced,
        mac_type=mac_type,
        runtime=replace(
            advanced.runtime, retain_traversal_result=True, retain_interactions=True
        ),
    )
    kwargs = dict(theta=0.6, advanced=advanced)
    if adaptive_eps is not None:
        kwargs["adaptive_eps"] = adaptive_eps

    fmm = FastMultipoleMethod(**kwargs)
    state = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)

    counts = _coverage_counts(state)
    missing = int(np.sum(counts == 0))
    doubled = int(np.sum(counts > 1))

    assert missing == 0, (
        f"{mac_type}: {missing} (target leaf, source particle) incidences receive "
        f"neither an M2L nor a P2P contribution"
    )
    assert doubled == 0, f"{mac_type}: {doubled} incidences are counted more than once"
