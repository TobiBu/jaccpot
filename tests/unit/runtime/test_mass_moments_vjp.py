"""The tree's mass moments with the gather-only reverse (plan S8, the large-N seam).

``jaccpot.upward._mass_moments_vjp.tree_mass_moments`` is yggdrax's
``compute_tree_mass_moments`` under a ``custom_vjp`` whose reverse sums the node
cotangents along each particle's leaf-to-root path (pointer doubling, gathers only)
instead of transposing the prefix sums (two colliding float64 scatters). Checked on a
padded cell-leaf static radix tree (empty leaf slots, single-particle leaves) in
float64:

* off by default: outside the context it IS yggdrax's call (no ``custom_vjp``);
* the forward is yggdrax's bitwise inside the context too;
* the reverse equals ``jax.vjp`` of yggdrax's to round-off, for the masses and the
  centres of mass, in the positions and the masses;
* the reverse scatters no floating-point value (the leaf of each particle is an
  integer scatter of the leaf starts, the same in any order).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.bounds import infer_bounds
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.runtime.fmm_state import TreeBuilderConfig, _build_tree_with_config
from jaccpot.upward._mass_moments_vjp import (
    gather_reverse_mass_moments,
    tree_mass_moments,
)

jax.config.update("jax_enable_x64", True)

_N = 1500


@pytest.fixture(scope="module")
def tree_and_inputs():
    """A float64 cell-leaf radix tree with padded leaf slots, and its inputs."""
    rng = np.random.default_rng(5)
    positions = jnp.asarray(rng.normal(size=(_N, 3)) ** 3)  # clustered: tiny leaves
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=_N))
    config = TreeBuilderConfig(
        mode="static_radix",
        target_leaf_particles=8,
        refine_local=False,
        max_refine_levels=0,
        aspect_threshold=8.0,
    )
    art = _build_tree_with_config(
        positions,
        masses,
        infer_bounds(positions),
        tree_type="radix",
        tree_config=config,
        leaf_size=8,
        workspace=None,
        jit_tree=False,
        refine_local=False,
        max_refine_levels=0,
        aspect_threshold=8.0,
        leaf_partition="cells",
        leaf_capacity=1024,
    )
    tree = art.tree
    order = np.asarray(tree.particle_indices)
    return tree, positions[order], masses[order]


def _vjp(fn, tree, positions, masses, seed=0):
    nodes = int(np.asarray(tree.parent).shape[0])
    rng = np.random.default_rng(seed)
    g_mass = jnp.asarray(rng.normal(size=nodes))
    g_com = jnp.asarray(rng.normal(size=(nodes, 3)))
    out, vjp = jax.vjp(lambda p, m: fn(tree, p, m), positions, masses)
    return out, vjp(type(out)(mass=g_mass, center_of_mass=g_com))


def test_the_tree_has_what_the_reverse_must_handle(tree_and_inputs):
    """Non-vacuous: empty leaf slots and single-particle leaves."""
    tree, *_ = tree_and_inputs
    num_internal = int(np.asarray(tree.left_child).shape[0])
    ranges = np.asarray(tree.node_ranges)[num_internal:]
    counts = ranges[:, 1] - ranges[:, 0] + 1
    assert np.any(counts <= 0) and np.any(counts == 1)


def test_off_by_default_it_is_the_yggdrax_call(tree_and_inputs):
    tree, positions, masses = tree_and_inputs
    jaxpr = jax.make_jaxpr(lambda p, m: tree_mass_moments(tree, p, m))(
        positions, masses
    )
    assert all("custom_vjp" not in eqn.primitive.name for eqn in jaxpr.jaxpr.eqns)


def test_the_gather_reverse_is_the_prefix_sums_reverse(tree_and_inputs):
    tree, positions, masses = tree_and_inputs
    out_r, (gp_r, gm_r) = _vjp(compute_tree_mass_moments, tree, positions, masses)
    with gather_reverse_mass_moments():
        out_g, (gp_g, gm_g) = _vjp(tree_mass_moments, tree, positions, masses)
    for a, b in zip(out_g, out_r):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for got, want in ((gp_g, gp_r), (gm_g, gm_r)):
        got, want = np.asarray(got), np.asarray(want)
        assert np.linalg.norm(want) > 0
        assert np.linalg.norm(got - want) <= 1e-13 * np.linalg.norm(want)


def _float_scatters(fn, *args) -> list:
    """The floating-point scatters of ``fn``'s jaxpr, sub-jaxprs included."""
    found = []

    def collect(j) -> None:
        for eqn in j.eqns:
            if eqn.primitive.name.startswith("scatter") and jnp.issubdtype(
                eqn.outvars[0].aval.dtype, jnp.floating
            ):
                found.append(eqn.outvars[0].aval)
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        collect(inner)

    collect(jax.make_jaxpr(fn)(*args).jaxpr)
    return found


def test_the_gather_reverse_scatters_no_float(tree_and_inputs):
    tree, positions, masses = tree_and_inputs

    def reverse(p, m):
        with gather_reverse_mass_moments():
            return _vjp(tree_mass_moments, tree, p, m)[1]

    def prefix_reverse(p, m):
        return _vjp(compute_tree_mass_moments, tree, p, m)[1]

    # non-vacuous: the prefix sums' reverse is float64 scatters
    assert _float_scatters(prefix_reverse, positions, masses)
    assert _float_scatters(reverse, positions, masses) == []
