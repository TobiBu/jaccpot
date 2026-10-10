"""Tests for the expansion-centre mode check the upward sweeps share.

The Cartesian upward sweep this file tested (``compute_node_multipoles`` and the
module-level ``prepare_upward_sweep``) was removed with the Cartesian basis in the
2026-10 cleanup (docs/cleanup_2026-10.md, K). Its centre-mode checks are kept
here, retargeted at the complex upward sweep, the one sweep left that takes a
``center_mode`` (the real sweep is COM only).
"""

import jax.numpy as jnp
import pytest
from yggdrax.tree import build_tree

from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)

DEFAULT_TEST_LEAF_SIZE = 1


def _build_sample_tree():
    positions = jnp.array(
        [
            [-0.5, -0.5, -0.5],
            [0.1, 0.0, 0.3],
            [0.6, 0.4, -0.2],
        ]
    )
    masses = jnp.array([1.0, 2.0, 3.0])
    bounds = (
        jnp.array([-1.0, -1.0, -1.0]),
        jnp.array([1.0, 1.0, 1.0]),
    )

    tree, pos_sorted, mass_sorted, _ = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=DEFAULT_TEST_LEAF_SIZE,
    )
    return tree, pos_sorted, mass_sorted


def test_explicit_centre_mode_requires_centres():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError, match="explicit_centers must be provided"):
        prepare_solidfmm_complex_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            center_mode="explicit",
        )

    centers = jnp.zeros((tree.parent.shape[0], 3), dtype=pos_sorted.dtype)
    result = prepare_solidfmm_complex_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        center_mode="explicit",
        explicit_centers=centers,
    )
    assert jnp.allclose(result.multipoles.centers, centers)


def test_an_unknown_centre_mode_is_rejected():
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError, match="Unknown center_mode"):
        prepare_solidfmm_complex_upward_sweep(
            tree,
            pos_sorted,
            mass_sorted,
            center_mode="nope",
        )


@pytest.mark.parametrize("mode", ["aabb", "AABB", "geometric"])
def test_removed_geometric_centres_raise_a_removal_error(mode):
    """AABB expansion centres went in the 2026-10 cleanup (X3); naming them must say so.

    The upward sweeps share the check, so an old caller gets the removal message,
    not an "unknown mode". Asserted on the complex sweep; the two Cartesian sweeps
    this also asserted went in K.
    """
    tree, pos_sorted, mass_sorted = _build_sample_tree()

    with pytest.raises(ValueError, match="expansion centres were removed"):
        prepare_solidfmm_complex_upward_sweep(
            tree, pos_sorted, mass_sorted, center_mode=mode
        )
