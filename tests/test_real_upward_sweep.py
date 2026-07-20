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
