"""Every node that owns particles must carry their mass in its monopole.

The far field of an FMM works by substituting a node's multipole expansion for
its particles. If a node spans particles but its expansion is zero, every M2L
that uses it as a source silently contributes nothing -- the mass of those
particles simply vanishes from those targets' accelerations. Nothing else in the
suite can see this:

- the far/near partition invariant in ``test_far_near_partition.py`` checks index
  bookkeeping. It counts a source node as covering the particles in its
  ``node_ranges``, so a zero-multipole node reads as perfectly covered.
- error benchmarks see a distribution whose median sits at machine precision --
  most targets never use the affected node -- with a heavy tail on the targets
  that do. That reads as ordinary truncation error.

Discovered while measuring Dehnen eq (16b): the mass MAC's error tail was
independent of ``adaptive_eps`` across five decades, because eq (15)'s truncation
estimate is built from the node's own multipole power, so a zero-multipole node
is scored as having exactly zero truncation error and is accepted at *any*
tolerance. The criterion was not at fault -- it was faithfully reporting that
these nodes have no multipole content.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, FMMPreset

# Compile-bound: every test here builds a solver and runs at least one full FMM
# solve, measured at 26-95 s each on CPU. `ci.yml` runs the version-compatibility
# matrix (`test-smoke`) with `-m "not slow and not experimental"` on a 30 minute
# budget and reserves the compile-heavy tests for `test-full` on 3.13. Leaving
# these unmarked put 94 such cases into that matrix and timed it out.
#
# `test_dehnen_mac_reference.py` is deliberately NOT marked: it checks eqs (12),
# (13), (15) and (16a) against independent numpy references at 1-10 s per test, so
# the criterion's correctness is still verified on every supported Python.
pytestmark = pytest.mark.slow


def _uniform_problem(n: int, *, seed: int = 0):
    key_pos = jax.random.PRNGKey(seed)
    positions = jax.random.normal(key_pos, (n, 3), dtype=jnp.float64)
    # Unit masses make the monopole of a node exactly its particle count, so a
    # discrepancy is readable directly as "how many particles went missing".
    masses = jnp.ones((n,), dtype=jnp.float64)
    return positions, masses


def _solver() -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="real",
        theta=0.5,
        softening=1.0e-6,
        advanced=FMMAdvancedConfig(mac_type="dehnen"),
    )


def _reachable_from_root(tree) -> set[int]:
    left = np.asarray(tree.left_child)
    right = np.asarray(tree.right_child)
    num_internal = int(tree.num_internal_nodes)
    seen: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node < 0 or node in seen:
            continue
        seen.add(node)
        if node < num_internal:
            stack.append(int(left[node]))
            stack.append(int(right[node]))
    return seen


@pytest.mark.parametrize(
    "n,leaf_size,order",
    [(1024, 16, 4), (2048, 16, 8), (4096, 16, 8), (4096, 32, 8)],
)
def test_every_reachable_node_with_particles_has_a_nonzero_expansion(
    n, leaf_size, order
):
    """A node reachable from the root that spans particles must not be empty.

    Measured before the fix: 19 of 255 internal nodes at N=4096/leaf=16 spanned 32
    real particles each, had valid parents and two real leaf children, and had all
    81 packed coefficients exactly zero.
    """

    positions, masses = _uniform_problem(n)
    state = _solver().prepare_state(
        positions, masses, leaf_size=leaf_size, max_order=order
    )

    packed = np.asarray(state.upward.multipoles.packed)
    node_ranges = np.asarray(state.tree.node_ranges)
    spans = np.maximum(node_ranges[:, 1] - node_ranges[:, 0] + 1, 0)
    reachable = _reachable_from_root(state.tree)

    empty_expansion = np.abs(packed).sum(axis=1) == 0.0
    offenders = [
        node
        for node in range(packed.shape[0])
        if node in reachable and spans[node] > 0 and empty_expansion[node]
    ]

    assert not offenders, (
        f"{len(offenders)} reachable nodes span particles but have an all-zero "
        f"multipole expansion, so their mass is dropped from every M2L that uses "
        f"them as a source; first few: "
        f"{[(node, int(spans[node])) for node in offenders[:5]]}"
    )


@pytest.mark.parametrize("n,leaf_size,order", [(1024, 16, 4), (4096, 16, 8)])
def test_node_monopole_equals_the_mass_it_spans(n, leaf_size, order):
    """Stronger form: the monopole must actually equal the enclosed mass.

    Catches partial M2M failures that leave a node with *some* coefficients but
    the wrong total, which the all-zero check above would pass.
    """

    positions, masses = _uniform_problem(n)
    state = _solver().prepare_state(
        positions, masses, leaf_size=leaf_size, max_order=order
    )

    packed = np.asarray(state.upward.multipoles.packed)
    monopole = packed[:, 0].real if np.iscomplexobj(packed) else packed[:, 0]
    masses_sorted = np.asarray(state.masses_sorted)
    node_ranges = np.asarray(state.tree.node_ranges)
    reachable = _reachable_from_root(state.tree)

    worst_node, worst_error = -1, 0.0
    for node in sorted(reachable):
        lo, hi = int(node_ranges[node, 0]), int(node_ranges[node, 1])
        if hi < lo:
            continue
        enclosed = float(masses_sorted[lo : hi + 1].sum())
        error = abs(float(monopole[node]) - enclosed) / max(enclosed, 1e-300)
        if error > worst_error:
            worst_node, worst_error = node, error

    assert worst_error < 1e-10, (
        f"node {worst_node} monopole differs from its enclosed mass by a relative "
        f"{worst_error:.3e}"
    )
