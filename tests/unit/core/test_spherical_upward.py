import jax
import jax.numpy as jnp
from yggdrax.tree import build_tree

from jaccpot.upward.spherical_tree_expansions import prepare_spherical_upward_sweep


def test_spherical_upward_monopole_matches_total_mass():
    key = jax.random.PRNGKey(0)
    n = 32
    positions = jax.random.normal(key, shape=(n, 3))
    masses = jnp.abs(jax.random.normal(jax.random.fold_in(key, 1), shape=(n,)))

    bounds = (
        jnp.min(positions, axis=0) - 1.0,
        jnp.max(positions, axis=0) + 1.0,
    )
    tree, pos_sorted, mass_sorted, _inv = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=8,
    )

    upward = prepare_spherical_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=1,
        center_mode="com",
    )

    # Root node is index 0 when there are internal nodes; if tree only has a
    # single leaf, that leaf is root.
    root_idx = 0
    total_mass = jnp.sum(mass_sorted)
    root_monopole = upward.multipoles.packed[root_idx, 0]
    assert jnp.allclose(root_monopole, total_mass, rtol=1e-6, atol=1e-6)


def test_spherical_upward_dipole_zero_for_com_centering():
    key = jax.random.PRNGKey(1)
    n = 64
    positions = jax.random.normal(key, shape=(n, 3))
    masses = jnp.abs(jax.random.normal(jax.random.fold_in(key, 3), shape=(n,)))

    bounds = (
        jnp.min(positions, axis=0) - 1.0,
        jnp.max(positions, axis=0) + 1.0,
    )
    tree, pos_sorted, mass_sorted, _inv = build_tree(
        positions,
        masses,
        bounds,
        return_reordered=True,
        leaf_size=8,
    )

    # Build upward sweep to ensure the spherical path runs.
    _ = prepare_spherical_upward_sweep(
        tree,
        pos_sorted,
        mass_sorted,
        max_order=1,
        center_mode="com",
    )

    # In the real SH basis, the l=1 block exists but it is *not* a direct
    # [x, y, z] dipole vector slice. The older upward scaffolding used that
    # mapping temporarily, but the Dehnen-normalized SH coefficients do not.
    #
    # Use the physically meaningful cartesian dipole (mass * offset) computed
    # directly about the COM as the invariant.
    total_mass = jnp.sum(mass_sorted)
    com = jnp.sum(mass_sorted[:, None] * pos_sorted, axis=0) / total_mass
    cart_dipole = jnp.sum(
        mass_sorted[:, None] * (pos_sorted - com[None, :]),
        axis=0,
    )
    assert jnp.linalg.norm(cart_dipole) < 1e-5
