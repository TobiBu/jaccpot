"""The real leaf P2M's result does not depend on its scan batch width."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMPreset
from jaccpot.upward.real_tree_expansions import _p2m_leaves_real

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="round-off comparison needs float64"
)

ORDER = 4


@pytest.mark.parametrize("leaf_batch_size", [1, 5, 24, 64])
def test_leaf_p2m_does_not_depend_on_the_batch_width(leaf_batch_size):
    """A batch wider than the leaf count must not drop the first leaf.

    The scan pads its last batch, and padded slots used to write the (stale)
    current value back to node ``num_internal`` -- the first leaf -- in the SAME
    scatter as that leaf's real result whenever the first leaf and the padding
    shared a batch (``leaf_batch_size > num_leaves``). With duplicate indices in a
    scatter-set either write may win; on CPU the stale zero did, and the first
    leaf's multipole vanished. The default batch (``min(num_leaves, 2048)``)
    never pads the first batch, but ``upward_leaf_batch_size`` is a user knob.
    """
    rng = np.random.default_rng(11)
    positions = jnp.asarray(rng.normal(scale=0.6, size=(96, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=96))
    fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    state = fmm.prepare_state(positions, masses, max_order=ORDER, leaf_size=4)
    tree = state.tree
    total = int(jnp.asarray(tree.parent).shape[0])
    internal = int(jnp.asarray(tree.left_child).shape[0])
    assert total - internal == 24  # the widths above straddle it
    common = dict(
        order=ORDER,
        max_leaf_size=int(state.max_leaf_size),
        num_internal=internal,
        total_nodes=total,
    )
    ranges = jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE)
    x = jnp.asarray(state.positions_sorted)
    m = jnp.asarray(state.masses_sorted)
    centers = jnp.asarray(state.upward.multipoles.centers)
    reference = _p2m_leaves_real(ranges, x, m, centers, leaf_batch_size=24, **common)
    got = _p2m_leaves_real(
        ranges, x, m, centers, leaf_batch_size=leaf_batch_size, **common
    )
    # Round-off only: a different batch width vectorises the per-leaf sum
    # differently (measured 2e-17 relative at width 1). The defect zeroed a leaf.
    scale = float(jnp.max(jnp.abs(reference)))
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-13 * scale)
