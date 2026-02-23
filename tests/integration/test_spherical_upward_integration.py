import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod
from jaccpot.runtime.fmm import _infer_bounds
from yggdrax.tree import build_tree


def test_prepare_state_spherical_upward_produces_sh_packed_shape():
    key = jax.random.PRNGKey(0)
    n = 64
    positions = jax.random.normal(key, shape=(n, 3), dtype=jnp.float64)
    masses = jnp.abs(jax.random.normal(jax.random.fold_in(key, 1), shape=(n,)))

    # Note: The spherical basis currently provides an *upward-only* path.
    # Calling `prepare_state()` triggers the cartesian/STF downward sweep,
    # which intentionally isn't compatible with spherical multipole packing.
    fmm = FastMultipoleMethod(expansion_basis="spherical")
    # `prepare_upward_sweep` expects inputs already sorted by a pre-built tree.
    bounds = _infer_bounds(positions)
    tree, pos_sorted, mass_sorted, _inverse = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=8,
    )
    upward = fmm.prepare_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=3,
    )

    num_nodes = int(tree.parent.shape[0])
    expected_coeffs = (3 + 1) ** 2
    assert upward.multipoles.packed.shape == (num_nodes, expected_coeffs)
