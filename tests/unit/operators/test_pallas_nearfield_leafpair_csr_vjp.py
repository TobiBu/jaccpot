"""Reverse of the CSR leaf-pair near field vs ``jax.vjp`` of its dense twin (fp64, interpret).

All four cotangents: positions, masses, ``softening_sq`` and ``G``. The CSR is
built SYMMETRIC (every pair listed from both ends), which is what the one-sided
forward needs to be a correct force and what the target-centric reverse relies on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported
from jaccpot.pallas.nearfield_leafpair_csr import (
    build_leafpair_chunk_table,
    leafpair_chunk_capacity,
    nearfield_leafpair_csr_jax,
    nearfield_leafpair_csr_pallas,
    nearfield_leafpair_csr_pallas_cvjp,
)

jax.config.update("jax_enable_x64", True)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _native_or_skip(interpret: bool) -> None:
    """Native Triton lowering needs an Ampere+ GPU; interpret mode runs anywhere."""
    if not interpret and not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas GPU lowering not available here")


def _symmetric_csr(seed, L, W, *, p_edge, empty_rows=()):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((L, W, 3))
    mass = np.abs(rng.standard_normal((L, W))) + 0.1
    mask = rng.random((L, W)) > 0.25
    mask[:, 0] = True
    adj = rng.random((L, L)) < p_edge
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    for r in empty_rows:
        adj[r, :] = False
        adj[:, r] = False
    rows = [np.flatnonzero(adj[i]) for i in range(L)]
    counts = np.array([r.size for r in rows])
    neighbors = np.concatenate(rows) if counts.sum() else np.zeros(0, int)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    cap = int(max(1, neighbors.size)) + 5
    nbr = np.zeros(cap, np.int32)
    nbr[: neighbors.size] = neighbors
    i32 = jnp.int32
    return (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(mask),
        jnp.asarray(nbr, i32),
        jnp.asarray(offsets, i32),
        jnp.asarray(counts, i32),
    )


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("chunk,accum", [(3, "input"), (8, "input"), (4, "wide")])
def test_csr_reverse_matches_vjp_of_the_dense_twin(chunk, accum, interpret):
    _native_or_skip(interpret)
    L, W = 9, 8
    pos, mass, mask, nbr, offsets, counts = _symmetric_csr(
        1, L, W, p_edge=0.5, empty_rows=(4,)
    )
    cap = leafpair_chunk_capacity(int(nbr.shape[0]), L, chunk)
    table = build_leafpair_chunk_table(offsets, counts, chunk=chunk, capacity=cap)
    soft = jnp.asarray(0.01)
    G = jnp.asarray(1.7)
    rng = np.random.default_rng(5)
    cot = (
        jnp.asarray(rng.standard_normal((L, W, 4))).at[..., 3].set(0.0)
    )  # acceleration only

    def ref(p, m, s, g):
        return nearfield_leafpair_csr_jax(
            p, m, mask, nbr, offsets, counts, softening_sq=s, G=g
        )

    def got(p, m, s, g):
        return nearfield_leafpair_csr_pallas_cvjp(
            p, m, mask, nbr, table, s, g, chunk, None, 1, None, interpret, accum, True
        )

    out_r, vjp_r = jax.vjp(ref, pos, mass, soft, G)
    out_g, vjp_g = jax.vjp(got, pos, mass, soft, G)
    assert _rel(out_g[..., :3], out_r[..., :3]) < 1e-10
    pb_r, mb_r, sb_r, gb_r = vjp_r(cot)
    pb_g, mb_g, sb_g, gb_g = vjp_g(cot)
    for name, g in (("positions", pb_g), ("masses", mb_g), ("soft", sb_g), ("G", gb_g)):
        assert np.all(np.isfinite(np.asarray(g))), name
    assert (
        np.linalg.norm(np.asarray(pb_r)) > 0
        and abs(float(sb_r)) > 0
        and abs(float(gb_r)) > 0
    )
    assert _rel(pb_g, pb_r) < 1e-9, f"positions rel-L2 {_rel(pb_g, pb_r):.3e}"
    assert _rel(mb_g, mb_r) < 1e-9, f"masses rel-L2 {_rel(mb_g, mb_r):.3e}"
    assert abs(float(sb_g) - float(sb_r)) < 1e-9 * abs(float(sb_r)), (
        float(sb_g),
        float(sb_r),
    )
    assert abs(float(gb_g) - float(gb_r)) < 1e-9 * abs(float(gb_r)), (
        float(gb_g),
        float(gb_r),
    )
    # masked slots get no cotangent
    assert np.all(np.asarray(pb_g)[~np.asarray(mask)] == 0)
    plain = nearfield_leafpair_csr_pallas(
        pos,
        mass,
        mask,
        nbr,
        table,
        softening_sq=soft,
        G=G,
        chunk=chunk,
        interpret=interpret,
        accum=accum,
    )
    assert np.array_equal(np.asarray(plain), np.asarray(out_g))
