"""``m2l_real_csr_lanes_pallas(n_targets=...)``: a target range shorter than the sources.

The one-sided distributed lane hands the M2L a concatenated multipole array
``[local ; imported]`` in which only the local prefix receives. Without ``n_targets``
the grid, the ``out_shape`` and ``csr_by_target``'s ``total_nodes`` all come from
``multipoles.shape[0]``, so the kernel emits ``n_local + n_remote`` rows: twice the
launches and twice the output buffer for rows that are then discarded, and a shape
mismatch when the result meets the local-only accumulator.

Interpret mode, so this runs anywhere.

    JAX_PLATFORMS=cpu pytest tests/unit/operators/test_m2l_real_csr_lanes_n_targets.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr_lanes import (
    m2l_real_csr_lanes_pallas,
    m2l_real_csr_lanes_pallas_cvjp,
)

_ORDER = 3
_K = 8


def _case(seed=0, n_local=12, n_remote=7, pairs=40):
    """Local targets, sources drawn from the whole concatenated array."""
    rng = np.random.default_rng(seed)
    n = n_local + n_remote
    C = (_ORDER + 1) ** 2
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float32)
    cent = jnp.asarray(rng.standard_normal((n, 3)) * 3.0, jnp.float32)
    tgt = rng.integers(0, n_local, size=pairs)
    src = rng.integers(0, n, size=pairs)
    src = np.where(src == tgt, (src + 1) % n, src)
    return (
        mult,
        cent,
        jnp.asarray(src, jnp.int32),
        jnp.asarray(tgt, jnp.int32),
        n_local,
        n,
    )


def _run(mult, cent, src, tgt, **kw):
    return m2l_real_csr_lanes_pallas(
        mult, cent, src, tgt, order=_ORDER, k_lanes=_K, interpret=True, **kw
    )


def test_the_cut_rows_are_bit_identical_and_the_dropped_rows_were_empty():
    mult, cent, src, tgt, n_local, n = _case()
    full = _run(mult, cent, src, tgt)
    cut = _run(mult, cent, src, tgt, n_targets=n_local)
    assert full.shape == (n, mult.shape[1])
    assert cut.shape == (n_local, mult.shape[1])
    # exact, not close: the same programs ran on the same rows
    assert np.array_equal(np.asarray(full[:n_local]), np.asarray(cut))
    # and what the full call computed beyond the target range was all zero, so
    # those programs and that buffer were pure waste
    assert not np.any(np.asarray(full[n_local:]))


def test_none_is_the_old_behaviour():
    mult, cent, src, tgt, _n_local, _n = _case(seed=1)
    assert np.array_equal(
        np.asarray(_run(mult, cent, src, tgt)),
        np.asarray(_run(mult, cent, src, tgt, n_targets=None)),
    )


def test_sources_may_still_reach_beyond_the_target_range():
    """The point of the argument: imported sources must still contribute.

    Zeroing the imported multipoles must change the local rows, or the kernel is
    quietly ignoring the half of the source array the import provides.
    """
    mult, cent, src, tgt, n_local, _n = _case(seed=2)
    with_remote = _run(mult, cent, src, tgt, n_targets=n_local)
    local_only = _run(mult.at[n_local:].set(0.0), cent, src, tgt, n_targets=n_local)
    assert not np.allclose(np.asarray(with_remote), np.asarray(local_only))


@pytest.mark.parametrize("bad", [-1, 20])
def test_out_of_range_is_rejected(bad):
    mult, cent, src, tgt, _n_local, _n = _case(seed=3)
    with pytest.raises(ValueError, match="n_targets must lie in"):
        _run(mult, cent, src, tgt, n_targets=bad)


def test_cvjp_forward_honours_it_and_the_reverse_refuses():
    """Forward-only by decision, and it says so instead of returning a wrong grad."""
    mult, cent, src, tgt, n_local, _n = _case(seed=4)
    args = (mult, cent, src, tgt, None, _ORDER, _K, True, "triton", 1)
    assert np.array_equal(
        np.asarray(m2l_real_csr_lanes_pallas_cvjp(*args, n_local)),
        np.asarray(_run(mult, cent, src, tgt, n_targets=n_local)),
    )
    # the single-tree gradient still works
    g = jax.grad(lambda m: jnp.sum(m2l_real_csr_lanes_pallas_cvjp(m, *args[1:], None)))(
        mult
    )
    assert np.all(np.isfinite(np.asarray(g)))
    with pytest.raises(NotImplementedError, match="forward-only"):
        jax.grad(
            lambda m: jnp.sum(m2l_real_csr_lanes_pallas_cvjp(m, *args[1:], n_local))
        )(mult)
