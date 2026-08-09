"""The per-pair class id used by the grouped M2L must match the pair it labels.

``GroupedInteractionBuffers`` stores ``class_sources`` / ``class_targets`` sorted
by displacement class, but stores ``class_ids`` under the *inverse* permutation,
in the original pair order. The two are therefore not co-indexed, and the
``pair_grouped`` far-field mode used to gather its rotation blocks with
``class_ids`` -- handing most pairs another class's rotation (G.11 in
``docs/refactor_audit_2026-08.md``).

These tests pin the corrected assignment: it is derived from the CSR
``class_offsets``, which is the same table the class-major scan reads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

from yggdrax.geometry import compute_tree_geometry  # noqa: E402
from yggdrax.grouped_interactions import build_grouped_interactions  # noqa: E402
from yggdrax.interactions import (  # noqa: E402
    DualTreeTraversalConfig,
    build_well_separated_interactions,
)
from yggdrax.tree import build_tree  # noqa: E402

from jaccpot.runtime.kernels.core import (  # noqa: E402
    _pair_class_ids_from_offsets,
)

# The class representative is the lattice displacement of the class, so it is not
# exactly the pair's own centre-to-centre vector; it is parallel to within the
# AABB-centre jitter inside a cell. Measured max over the tree below is ~2 deg,
# while a misaligned gather reaches 150 deg.
MAX_REPRESENTATIVE_ANGLE_DEG = 10.0


def _grouped_buffers():
    """Grouped far-field buffers for a small uniform system, plus node centres."""
    key = jax.random.PRNGKey(0xC0FFEE)
    k_pos, k_mass = jax.random.split(key)
    positions = jax.random.uniform(
        k_pos, (256, 3), dtype=jnp.float64, minval=-1.0, maxval=1.0
    )
    masses = jnp.abs(jax.random.normal(k_mass, (256,), dtype=jnp.float64)) + 0.5
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=jnp.float64),
        jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    )
    tree, positions_sorted, _, _ = build_tree(
        positions, masses, bounds, leaf_size=8, return_reordered=True
    )
    geometry = compute_tree_geometry(tree, positions_sorted)
    interactions = build_well_separated_interactions(
        tree,
        geometry,
        theta=0.5,
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=1 << 18,
            process_block=512,
            max_interactions_per_node=1 << 16,
            max_neighbors_per_leaf=1 << 16,
        ),
    )
    grouped = build_grouped_interactions(tree, geometry, interactions)
    return grouped, np.asarray(geometry.center, dtype=np.float64)


def _representative_angles_deg(grouped, centers, class_ids) -> np.ndarray:
    """Angle between each pair's class representative and its own displacement."""
    sources = np.asarray(grouped.class_sources)
    targets = np.asarray(grouped.class_targets)
    displacements = np.asarray(grouped.class_displacements)
    pair_delta = centers[targets] - centers[sources]
    representative = displacements[np.asarray(class_ids)]
    numerator = np.sum(representative * pair_delta, axis=1)
    denominator = np.linalg.norm(representative, axis=1) * np.linalg.norm(
        pair_delta, axis=1
    )
    cosine = np.clip(numerator / np.maximum(denominator, 1e-300), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="requires float64 (JAX_ENABLE_X64=1)",
)
def test_pair_class_ids_reproduce_the_csr_offsets():
    """The derived ids are exactly the CSR expansion of ``class_offsets``."""
    grouped, _ = _grouped_buffers()
    offsets = np.asarray(grouped.class_offsets, dtype=np.int64)
    pair_count = int(grouped.class_sources.shape[0])
    assert pair_count > 0, "vacuous: no far pairs to classify"

    expected = np.zeros(pair_count, dtype=np.int64)
    for class_idx in range(offsets.shape[0] - 1):
        expected[offsets[class_idx] : offsets[class_idx + 1]] = class_idx

    derived = np.asarray(
        _pair_class_ids_from_offsets(
            jnp.asarray(grouped.class_offsets),
            jnp.arange(pair_count),
        )
    )
    np.testing.assert_array_equal(derived, expected)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="requires float64 (JAX_ENABLE_X64=1)",
)
def test_derived_class_representative_is_parallel_to_its_pair():
    """Every pair gets a class whose representative points the way the pair does.

    This is the invariant the ``pair_grouped`` M2L relies on: the cached rotation
    aligns the class representative with the z axis, so it is only the right
    rotation for a pair whose own displacement is parallel to it.
    """
    grouped, centers = _grouped_buffers()
    pair_count = int(grouped.class_sources.shape[0])
    derived = _pair_class_ids_from_offsets(
        jnp.asarray(grouped.class_offsets), jnp.arange(pair_count)
    )
    angles = _representative_angles_deg(grouped, centers, derived)
    assert angles.max() < MAX_REPRESENTATIVE_ANGLE_DEG, (
        f"class representative misaligned with its pair: max {angles.max():.1f} deg "
        f"(mean {angles.mean():.1f})"
    )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="requires float64 (JAX_ENABLE_X64=1)",
)
def test_shipped_class_ids_are_not_co_indexed_with_the_sorted_pairs():
    """Guard the reason the derivation exists, so it cannot be "simplified" back.

    If yggdrax ever stores ``class_ids`` in the sorted pair order this test goes
    red, and :func:`_pair_class_ids_from_offsets` can be replaced by a direct
    read. Until then, using ``class_ids`` per pair is the G.11 defect.
    """
    grouped, centers = _grouped_buffers()
    shipped = np.asarray(grouped.class_ids)
    pair_count = int(grouped.class_sources.shape[0])
    derived = np.asarray(
        _pair_class_ids_from_offsets(
            jnp.asarray(grouped.class_offsets), jnp.arange(pair_count)
        )
    )
    assert not np.array_equal(shipped, derived), (
        "grouped.class_ids now agrees with the CSR order -- re-check whether the "
        "derivation in _pair_class_ids_from_offsets is still needed"
    )
    shipped_angles = _representative_angles_deg(grouped, centers, shipped)
    assert shipped_angles.max() > MAX_REPRESENTATIVE_ANGLE_DEG
