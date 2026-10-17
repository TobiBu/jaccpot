"""The CSR M2L lanes kernel's force-scale lane (S3b), interpret mode.

``mac_type='dehnen_error'`` needs eq (16b)'s far term per node every step: each far
pair ``(A -> B)`` adds ``G M_A / ((|c_A - c_B| + rho_B)^2 + eps^2)`` to B. With
``force_scale`` the lanes kernel sums it per target row in the launch it makes for
the M2L anyway. Pinned against
:func:`~jaccpot.runtime._force_scale_levels.far_force_scale_own` (the per-step
segment sum it replaces): on ragged rows (empty, one tile, a partial tile), on a
live prefix with padding, on a presorted CSR, with a zero-reach pair masked, with
the walk centres given separately and with them shared with the expansion centres;
the local increments are the plain kernel's, bit for bit, and the custom VJP's
gradient is the plain one's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.m2l_real_csr_lanes import (
    m2l_real_csr_lanes_fb_pallas_cvjp,
    m2l_real_csr_lanes_pallas,
    m2l_real_csr_lanes_pallas_cvjp,
)
from jaccpot.runtime._force_scale_levels import (
    FarForceScaleInputs,
    far_force_scale_own,
)

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the float64 case needs x64"
)

ORDER, N, G, EPS2 = 4, 40, 1.3, 1e-3


def _case(seed, dtype, pad=0):
    rng = np.random.default_rng(seed)
    C = (ORDER + 1) ** 2
    rows = rng.integers(0, 70, N)
    rows[3] = 0  # an empty row
    rows[5] = 32  # exactly one tile
    rows[7] = 65  # two tiles and one lane
    tgt = np.repeat(np.arange(N), rows)
    src = rng.integers(0, N, tgt.size)
    src = np.where(src == tgt, (src + 1) % N, src)
    mult = jnp.asarray(rng.standard_normal((N, C)), dtype)
    cent = rng.uniform(-1.0, 1.0, (N, 3)) * 4.0
    radii = rng.uniform(0.0, 0.5, N)
    radii[12] = 0.0
    src[np.nonzero(tgt == 12)[0][:1]] = 11
    mass = rng.uniform(0.1, 1.0, N)
    src_p = np.concatenate([src, -np.ones(pad, int)])
    tgt_p = np.concatenate([tgt, -np.ones(pad, int)])
    return dict(
        mult=mult,
        cent=jnp.asarray(cent, dtype),
        src=jnp.asarray(src_p, jnp.int32),
        tgt=jnp.asarray(tgt_p, jnp.int32),
        live=src.size,
        mass=jnp.asarray(mass, dtype),
        radii=jnp.asarray(radii, dtype),
        eps2=jnp.asarray(EPS2, dtype),
    )


def _reference(c, centres=None):
    live = jnp.arange(c["src"].shape[0]) < c["live"]
    return far_force_scale_own(
        sources=c["src"],
        targets=c["tgt"],
        live=live,
        node_mass=c["mass"],
        node_centers=c["cent"] if centres is None else centres,
        node_radii=c["radii"],
        gravitational_constant=G,
        softening_sq=c["eps2"],
        num_nodes=N,
    )


def _inputs(c, *, walk_centres=False):
    return FarForceScaleInputs(
        node_mass=c["mass"],
        node_radii=c["radii"],
        softening_sq=c["eps2"],
        gravitational_constant=G,
        node_centers=c["cent"] if walk_centres else None,
    )


@pytest.mark.parametrize(
    "dtype, rtol", [(jnp.float64, 1e-13), (jnp.float32, 2e-6)], ids=["f64", "f32"]
)
@pytest.mark.parametrize("pad", [0, 9])
def test_the_force_scale_lane_equals_the_segment_sum(dtype, rtol, pad):
    c = _case(1, dtype, pad=pad)
    want = np.asarray(_reference(c), np.float64)
    kw = dict(order=ORDER, interpret=True, active_pair_count=jnp.asarray(c["live"]))
    plain = m2l_real_csr_lanes_pallas(c["mult"], c["cent"], c["src"], c["tgt"], **kw)
    inc, fb = m2l_real_csr_lanes_pallas(
        c["mult"], c["cent"], c["src"], c["tgt"], force_scale=_inputs(c), **kw
    )
    assert fb.shape == (N,) and fb.dtype == dtype
    assert float(want[3]) == 0.0 and float(fb[3]) == 0.0
    assert int(np.sum(want > 0)) >= N - 3, "vacuous: most rows must carry a term"
    np.testing.assert_allclose(np.asarray(fb, np.float64), want, rtol=rtol, atol=0.0)
    # the local increments are the plain kernel's, bit for bit
    assert bool(jnp.all(inc == plain))


def test_walk_centres_given_separately():
    """The expansion centres again: the same floats; other centres (the box walk
    geometry): the reference on those, with a zero-reach pair masked."""
    c = _case(2, jnp.float64)
    kw = dict(order=ORDER, interpret=True)
    _, shared = m2l_real_csr_lanes_pallas(
        c["mult"], c["cent"], c["src"], c["tgt"], force_scale=_inputs(c), **kw
    )
    _, given = m2l_real_csr_lanes_pallas(
        c["mult"],
        c["cent"],
        c["src"],
        c["tgt"],
        force_scale=_inputs(c, walk_centres=True),
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(given), np.asarray(shared))
    rng = np.random.default_rng(7)
    walk = np.asarray(c["cent"]) + rng.uniform(-0.3, 0.3, (N, 3))
    walk[11] = walk[12]  # target 12 (radius 0) has a pair from 11: reach 0
    walk = jnp.asarray(walk)
    want = np.asarray(_reference(c, centres=walk))
    _, got = m2l_real_csr_lanes_pallas(
        c["mult"],
        c["cent"],
        c["src"],
        c["tgt"],
        force_scale=_inputs(c)._replace(node_centers=walk),
        **kw,
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=0.0)
    assert np.all(np.isfinite(np.asarray(got)))
    assert not np.allclose(np.asarray(got), np.asarray(shared))


def test_a_presorted_csr_gives_the_coo_lists_term():
    c = _case(3, jnp.float64)
    tgt = np.asarray(c["tgt"])
    offsets = np.searchsorted(tgt, np.arange(N + 1), side="left")
    kw = dict(order=ORDER, interpret=True)
    _, coo = m2l_real_csr_lanes_pallas(
        c["mult"], c["cent"], c["src"], c["tgt"], force_scale=_inputs(c), **kw
    )
    _, csr = m2l_real_csr_lanes_pallas(
        c["mult"],
        c["cent"],
        jnp.concatenate([c["src"], jnp.full((5,), -1, jnp.int32)]),
        jnp.asarray(offsets, jnp.int32),
        presorted=True,
        force_scale=_inputs(c),
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(csr), np.asarray(coo))


def test_the_custom_vjp_differentiates_the_locals_only():
    c = _case(4, jnp.float64)
    src, tgt = c["src"], c["tgt"]

    def plain(mult, cent):
        out = m2l_real_csr_lanes_pallas_cvjp(
            mult, cent, src, tgt, None, ORDER, 32, True, "triton", 1, None, False
        )
        return jnp.sum(out**2)

    def with_scale(mult, cent):
        out, fb = m2l_real_csr_lanes_fb_pallas_cvjp(
            mult,
            cent,
            src,
            tgt,
            None,
            c["mass"],
            c["radii"],
            c["eps2"],
            None,
            ORDER,
            32,
            True,
            "triton",
            1,
            None,
            False,
            G,
        )
        # the far term is a topology quantity: it carries no cotangent back
        return jnp.sum(out**2) + jnp.sum(fb)

    want = jax.grad(plain, argnums=(0, 1))(c["mult"], c["cent"])
    got = jax.grad(with_scale, argnums=(0, 1))(c["mult"], c["cent"])
    for g, w in zip(got, want):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_inputs_in_another_dtype_are_refused():
    c = _case(5, jnp.float32)
    bad = _inputs(c)._replace(node_mass=c["mass"].astype(jnp.float64))
    with pytest.raises(ValueError, match="dtype"):
        m2l_real_csr_lanes_pallas(
            c["mult"],
            c["cent"],
            c["src"],
            c["tgt"],
            order=ORDER,
            interpret=True,
            force_scale=bad,
        )
