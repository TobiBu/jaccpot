"""The native real upward sweep (P2M + M2M) against a direct per-node P2M.

Every node's multipole about its centre of mass is the sum of the single-particle
P2M of the particles it holds, so a native-real P2M + M2M tree upward must reproduce
that sum: an exact identity, not a truncation. This pins the real upward operators
(used by the general path, the large-N lane and the distributed FMM's per-device
upward + coarse M2M).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import Tree, build_tree

from jaccpot.operators.real_harmonics import p2m_real_direct
from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep


def _tree(n, leaf, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float64)
    mass = rng.uniform(0.5, 2.0, (n,)).astype(np.float64)
    lo = jnp.asarray(pos.min(0) - 0.05)
    hi = jnp.asarray(pos.max(0) + 0.05)
    tree = Tree.from_particles(
        jnp.asarray(pos),
        jnp.asarray(mass),
        tree_type="radix",
        bounds=(lo, hi),
        return_reordered=True,
        leaf_size=leaf,
    )
    return tree


def test_real_upward_matches_direct_per_node_p2m():
    """Each node's multipole == the sum of its particles' P2M about its centre.

    Replaces ``test_real_upward_matches_complex_convert`` (2026-10 cleanup, C),
    which compared the sweep with the complex sweep's multipoles converted to the
    real basis. This oracle needs no second basis: for every node (leaves and
    internal alike) it sums ``p2m_real_direct`` over the node's particle range about
    the node's centre, and checks the centres are the nodes' centres of mass.
    Measured (fp64, N = 300, leaf 8, 75 nodes): rel-L2 1.0e-16 (p = 2) and 1.2e-16
    (p = 4), worst node 3.7e-16; the complex sweep + conversion was 1.3e-16 and
    1.4e-16 from the same oracle. The 1e-9 bound is the old test's.
    """
    for p in (2, 4):
        tree = _tree(n=300, leaf=8, seed=p)
        lp, lm = tree.positions_sorted, tree.masses_sorted
        up_r = prepare_real_upward_sweep(tree, lp, lm, max_order=p, max_leaf_size=8)
        centers = up_r.multipoles.centers
        ranges = np.asarray(tree.node_ranges)
        p2m = jax.vmap(lambda d, m, p=p: p2m_real_direct(d, m, order=p))

        reference = []
        com_err = 0.0
        for node in range(ranges.shape[0]):
            start, stop = int(ranges[node, 0]), int(ranges[node, 1]) + 1
            center = centers[node]
            reference.append(jnp.sum(p2m(lp[start:stop] - center, lm[start:stop]), 0))
            com = jnp.sum(lm[start:stop, None] * lp[start:stop], axis=0) / jnp.sum(
                lm[start:stop]
            )
            com_err = max(com_err, float(jnp.linalg.norm(com - center)))
        reference = jnp.stack(reference)

        rel = float(
            jnp.linalg.norm(up_r.multipoles.packed - reference)
            / (jnp.linalg.norm(reference) + 1e-30)
        )
        print(
            f"p={p}: real upward vs direct per-node P2M rel={rel:.3e} com={com_err:.3e}"
        )
        assert com_err < 1e-9, f"centres are not the nodes' COMs (p={p}): {com_err:.3e}"
        assert rel < 1e-9, f"real upward != direct per-node P2M (p={p}): {rel:.3e}"


def test_real_static_num_levels_bit_identical_to_padded():
    """``static_num_levels`` must be a pure performance knob, bit-for-bit.

    The complex sweep's identical parameter carried this claim and was pinned by
    ``test_solidfmm_complex_tree_expansions.py::test_static_num_levels_bit_identical_to_padded``
    (removed with the complex basis in the 2026-10 cleanup, C); the real sweep grew
    the same parameter with neither, until this test.

    Why it has to be *bit*-identical rather than close: the knob only skips
    level-loop iterations over levels that are empty padding, so it removes
    additions of exact zeros. Anything other than exact equality would mean the
    padded levels were contributing, which is the bug the parameter could
    plausibly introduce -- the level loop's ``dynamic_slice_in_dim`` clamps rather
    than erroring, so a wrong level count silently shifts the window instead of
    failing loudly (see the note in
    :func:`~jaccpot.upward.real_tree_expansions.aggregate_m2m_real_by_level`).
    """
    from yggdrax.tree import get_level_offsets, get_num_levels

    for p in (2, 4):
        tree = _tree(n=300, leaf=8, seed=p)
        lp, lm = tree.positions_sorted, tree.masses_sorted
        concrete_num_levels = int(get_num_levels(tree))

        # Non-vacuity gate. If the padded shape already equalled the concrete depth
        # the two calls below would be the same call and this test would assert
        # nothing while still passing. Measured here: 64 padded vs 7 real, i.e. 57
        # empty levels skipped.
        padded_num_levels = int(get_level_offsets(tree).shape[0] - 1)
        assert padded_num_levels > concrete_num_levels, (
            f"nothing to skip at p={p}: padded depth {padded_num_levels} is not "
            f"greater than the concrete depth {concrete_num_levels}, so this test "
            "would compare a call against itself"
        )

        padded = prepare_real_upward_sweep(tree, lp, lm, max_order=p, max_leaf_size=8)
        optimized = prepare_real_upward_sweep(
            tree,
            lp,
            lm,
            max_order=p,
            max_leaf_size=8,
            static_num_levels=concrete_num_levels,
        )

        assert jnp.array_equal(
            padded.multipoles.packed, optimized.multipoles.packed
        ), f"static_num_levels changed the multipoles at p={p}"
        assert jnp.array_equal(
            padded.multipoles.centers, optimized.multipoles.centers
        ), f"static_num_levels changed the centers at p={p}"


def _build_multilevel_tree(n=600, leaf_size=8, seed=0):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(np.clip(rng.normal(0.0, 0.3, (n, 3)), -0.99, 0.99))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, n))
    bounds = (
        jnp.array([-1.0, -1.0, -1.0], dtype=pos.dtype),
        jnp.array([1.0, 1.0, 1.0], dtype=pos.dtype),
    )
    tree, ps, msorted, _ = build_tree(
        pos, mass, bounds, return_reordered=True, leaf_size=leaf_size
    )
    return tree, ps, msorted


def test_prepare_upward_sweep_stashes_and_reuses_concrete_depth():
    """The runtime method stashes the concrete depth and a traced (jitted) call
    reuses it, staying bit-identical to the concrete result.

    An engine test (``FMMEngine(expansion_basis="solidfmm")`` runs the real upward
    since the 2026-10 cleanup, D1); moved here from the complex sweep's module when
    that module was removed (C).
    """
    import jax
    from yggdrax.tree import get_num_levels

    from jaccpot.runtime._fmm_impl import FMMEngine

    tree, ps, ms = _build_multilevel_tree()
    actual = int(get_num_levels(tree))
    fmm = FMMEngine(expansion_basis="solidfmm")

    concrete = fmm.prepare_upward_sweep(
        tree, ps, ms, max_order=4, center_mode="com", max_leaf_size=8
    )
    assert fmm._static_upward_num_levels == actual

    def _run(t, p, m):
        return fmm.prepare_upward_sweep(
            t, p, m, max_order=4, center_mode="com", max_leaf_size=8
        ).multipoles.packed

    packed_jit = jax.jit(_run)(tree, ps, ms)
    assert jnp.array_equal(packed_jit, concrete.multipoles.packed)


def test_the_real_upward_refuses_every_centre_mode_but_com():
    """The engine's (real) upward sweep is COM-only and says so for anything else.

    Takes over the rejection half of ``test_tree_expansions.py`` (the removed-mode
    and unknown-mode checks of ``_resolve_center_mode``), which ran on the complex
    upward sweep and went with it in the 2026-10 cleanup (C). The AABB / geometric
    centres went in X3; ``"explicit"`` was the complex sweep's test seam.
    """
    import pytest

    from jaccpot.runtime._fmm_impl import FMMEngine

    tree, ps, ms = _build_multilevel_tree(n=200)
    fmm = FMMEngine(expansion_basis="solidfmm")
    for mode in ("explicit", "aabb", "geometric", "Bogus"):
        with pytest.raises(ValueError, match="center_mode='com' only"):
            fmm.prepare_upward_sweep(
                tree, ps, ms, max_order=2, center_mode=mode, max_leaf_size=8
            )
    packed = fmm.prepare_upward_sweep(
        tree, ps, ms, max_order=2, center_mode=" COM ", max_leaf_size=8
    ).multipoles.packed
    assert packed.shape[1] == 9
