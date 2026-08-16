"""Native real upward sweep == complex upward sweep + exact real conversion.

The Dehnen real operators are consistent with ``complex_to_dehnen_real_coeffs``
to machine precision, so a native-real P2M+M2M tree upward must reproduce the
complex sweep's multipoles converted to real. This pins the real upward operators
(used by the distributed FMM's per-device upward + coarse M2M).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from yggdrax.tree import Tree

from jaccpot.operators.real_harmonics import complex_to_dehnen_real_coeffs
from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)


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


def test_real_upward_matches_complex_convert():
    # Exact machine-precision identity (not a convergence trend), so it holds at
    # every order; two representative orders (low + high coeff counts) suffice.
    for p in (2, 4):
        tree = _tree(n=300, leaf=8, seed=p)
        lp, lm = tree.positions_sorted, tree.masses_sorted
        up_c = prepare_solidfmm_complex_upward_sweep(
            tree, lp, lm, max_order=p, max_leaf_size=8, rotation="solidfmm"
        )
        ref = complex_to_dehnen_real_coeffs(up_c.multipoles.packed, order=p)
        up_r = prepare_real_upward_sweep(tree, lp, lm, max_order=p, max_leaf_size=8)

        num = float(jnp.linalg.norm(up_r.multipoles.packed - ref))
        den = float(jnp.linalg.norm(ref)) + 1e-30
        rel = num / den
        cerr = float(jnp.linalg.norm(up_r.multipoles.centers - up_c.multipoles.centers))
        print(
            f"p={p}: real-upward vs complex+convert rel={rel:.3e}  center_err={cerr:.3e}"
        )
        assert cerr < 1e-9, f"centers differ (p={p}): {cerr:.3e}"
        assert rel < 1e-9, f"real upward != complex+convert (p={p}): {rel:.3e}"


def test_real_static_num_levels_bit_identical_to_padded():
    """``static_num_levels`` must be a pure performance knob, bit-for-bit.

    The complex sweep's identical parameter has carried this claim in its docstring
    and been pinned by
    ``tests/unit/core/test_solidfmm_complex_tree_expansions.py::test_static_num_levels_bit_identical_to_padded``
    for some time; the real sweep grew the same parameter with neither. This closes
    that asymmetry.

    Why it has to be *bit*-identical rather than close: the knob only skips
    level-loop iterations over levels that are empty padding, so it removes
    additions of exact zeros. Anything other than exact equality would mean the
    padded levels were contributing, which is the bug the parameter could
    plausibly introduce -- the level loop's ``dynamic_slice_in_dim`` clamps rather
    than erroring, so a wrong level count silently shifts the window instead of
    failing loudly (see the note in
    :func:`~jaccpot.upward.solidfmm_complex_tree_expansions.prepare_solidfmm_complex_upward_sweep`).
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
