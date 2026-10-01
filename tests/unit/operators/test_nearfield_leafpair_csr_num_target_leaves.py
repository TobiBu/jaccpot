"""``nearfield_leafpair_csr_pallas(num_target_leaves=...)``: a halo source pool.

The one-sided distributed lane gives the near field a leaf pool ``[local ; halo]`` in
which only the local prefix receives, so ``L_source = L_local + L_halo`` while
``L_target = L_local``. The kernel already supports that -- ``leaf_positions`` is
"targets and the source gather table alike" and the grid comes from the chunk table,
which is built over the target rows -- and these tests pin that claim rather than
assume it.

What the argument adds is only the tail: without it the ``segment_sum`` allocates
``L + 1`` segments and returns ``L`` rows whose halo entries are zero and discarded.

Interpret mode, so this runs anywhere.

    JAX_PLATFORMS=cpu pytest tests/unit/operators/test_nearfield_leafpair_csr_num_target_leaves.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_pallas,
)

_W = 8
_CHUNK = 2


def _case(seed=0, n_local=6, n_halo=4, per_row=3):
    """A CSR over TARGET rows only; sources drawn from the whole pool."""
    rng = np.random.default_rng(seed)
    n = n_local + n_halo
    pos = jnp.asarray(rng.standard_normal((n, _W, 3)), jnp.float32)
    mass = jnp.asarray(np.abs(rng.standard_normal((n, _W))), jnp.float32)
    mask = jnp.asarray(np.ones((n, _W), bool))
    rows = [
        rng.choice([i for i in range(n) if i != t], size=per_row, replace=False)
        for t in range(n_local)
    ]
    nbr = jnp.asarray(np.concatenate(rows).astype(np.int32))
    counts = np.full(n_local, per_row, np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    cap = leafpair_chunk_capacity(int(nbr.size), n_local, _CHUNK)
    table = build_leafpair_chunk_table(
        jnp.asarray(offsets), jnp.asarray(counts), chunk=_CHUNK, capacity=cap
    )
    return pos, mass, mask, nbr, table, n_local, n


def _run(pos, mass, mask, nbr, table, **kw):
    return nearfield_leafpair_csr_pallas(
        pos,
        mass,
        mask,
        nbr,
        table,
        softening_sq=jnp.asarray(0.01, jnp.float32),
        G=jnp.asarray(1.0, jnp.float32),
        chunk=_CHUNK,
        interpret=True,
        **kw,
    )


def test_the_cut_rows_are_bit_identical_and_the_halo_rows_were_empty():
    pos, mass, mask, nbr, table, n_local, n = _case()
    full = _run(pos, mass, mask, nbr, table)
    cut = _run(pos, mass, mask, nbr, table, num_target_leaves=n_local)
    assert full.shape == (n, _W, 4)
    assert cut.shape == (n_local, _W, 4)
    assert np.array_equal(np.asarray(full[:n_local]), np.asarray(cut))
    # no chunk names a halo row, so the rows the full call returned for them are
    # exactly zero -- the buffer was pure waste, not an approximation
    assert not np.any(np.asarray(full[n_local:]))


def test_none_is_the_old_behaviour():
    pos, mass, mask, nbr, table, _n_local, _n = _case(seed=1)
    assert np.array_equal(
        np.asarray(_run(pos, mass, mask, nbr, table)),
        np.asarray(_run(pos, mass, mask, nbr, table, num_target_leaves=None)),
    )


def test_halo_sources_contribute():
    """The claim under test: a pool longer than the target range really is read.

    Zeroing the halo masses must change the local rows. Without this the kernel
    could be ignoring exactly the half of the pool the import provides, and every
    shape assertion above would still pass.
    """
    pos, mass, mask, nbr, table, n_local, _n = _case(seed=2)
    with_halo = _run(pos, mass, mask, nbr, table, num_target_leaves=n_local)
    without = _run(
        pos, mass.at[n_local:].set(0.0), mask, nbr, table, num_target_leaves=n_local
    )
    assert not np.allclose(np.asarray(with_halo), np.asarray(without))


@pytest.mark.parametrize("bad", [-1, 11])
def test_out_of_range_is_rejected(bad):
    pos, mass, mask, nbr, table, _n_local, _n = _case(seed=3)
    with pytest.raises(ValueError, match="num_target_leaves must lie in"):
        _run(pos, mass, mask, nbr, table, num_target_leaves=bad)
