"""The near-field kernel's force-scale lane: eq (16b)'s ``f_b`` near term, per particle.

``nearfield_leafpair_csr_sorted_direct_pallas(..., with_force_scale=True)`` puts
``sum_a G m_a / (|x_a - x_b|^2 + eps^2)`` over the kernel's own pair set (the
neighbour leaves and the own leaf, self excluded) in the fourth output lane, with
``eps`` the Plummer-equivalent softening whatever the pair kernel -- the eager
estimator's form, so the per-step scale and the first step's prepass agree. Pinned:

* the lane equals a numpy loop over the same pairs, for every kernel and every
  source-tile mode (the lean, generic and compact pair bodies each have one);
* the accelerations are unchanged by the switch;
* the switch and ``with_potential`` are exclusive;
* on Triton (float32) the lane multiplies by the reciprocal inside Triton's
  division instead of dividing, with the division's bits (S3b).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.nearfield_leafpair_csr import (
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from tests.unit.operators.test_pallas_nearfield_leafpair_csr_sorted import _case

_EPS = 0.35
_G = 1.3
_W = 8


def _native_or_skip(interpret: bool) -> None:
    if interpret:
        return
    from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

    if not pallas_m2l_real_csr_supported():
        pytest.skip("native Pallas needs an sm_80+ GPU")


def _reference(c) -> np.ndarray:
    pos = np.asarray(c["pos"], np.float64)
    mass = np.asarray(c["mass"], np.float64)
    starts, counts = np.asarray(c["starts"]), np.asarray(c["counts"])
    nbr, offsets = np.asarray(c["nbr"]), np.asarray(c["offsets"])
    row_counts = np.asarray(c["row_counts"])
    fb = np.zeros(pos.shape[0])
    for leaf in range(c["L"]):
        if counts[leaf] == 0:
            continue
        sources = [leaf] + [
            int(x) for x in nbr[offsets[leaf] : offsets[leaf] + row_counts[leaf]]
        ]
        src = np.concatenate(
            [np.arange(starts[s], starts[s] + counts[s]) for s in sources]
        )
        for t in range(starts[leaf], starts[leaf] + counts[leaf]):
            other = src[src != t]
            d2 = np.sum((pos[other] - pos[t]) ** 2, axis=1)
            fb[t] = _G * np.sum(mass[other] / (d2 + _EPS**2))
    return fb


def _run(c, kernel, interpret, source_tile, flags, **kw):
    return nearfield_leafpair_csr_sorted_direct_pallas(
        jnp.asarray(c["pos"], jnp.float64),
        jnp.asarray(c["mass"], jnp.float64),
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
        leaf_width=_W,
        softening_sq=jnp.asarray(_EPS**2),
        G=jnp.asarray(_G),
        chunk=1,
        interpret=interpret,
        source_tile=source_tile,
        source_flags=flags,
        softening_kernel=kernel,
        **kw,
    )


@pytest.mark.parametrize("interpret", [True, False])
@pytest.mark.parametrize("kernel", ["plummer", "ferrers3", "wendland_c2"])
@pytest.mark.parametrize(
    "source_tile, flags", [(0, None), (4, ""), (4, "l"), (8, "alr")]
)
def test_force_scale_lane_equals_the_near_pair_sum(
    kernel, source_tile, flags, interpret
):
    _native_or_skip(interpret)
    c = _case(7, num_live=9, num_pad=3, W=_W, n_dead=2, max_row=6, empty_rows=(2,))
    acc_fb, fb = _run(c, kernel, interpret, source_tile, flags, with_force_scale=True)
    acc, none = _run(c, kernel, interpret, source_tile, flags)
    assert none is None
    want = _reference(c)
    live = np.asarray(want) > 0
    assert live.sum() > 20, "vacuous: no near pairs"
    tol = 1e-12 if interpret else 2e-5
    np.testing.assert_allclose(np.asarray(fb), want, rtol=tol, atol=0.0)
    # the switch changes the fourth lane only
    np.testing.assert_allclose(
        np.asarray(acc_fb), np.asarray(acc), rtol=tol, atol=1e-14
    )


def test_force_scale_and_potential_are_exclusive():
    c = _case(7, num_live=9, num_pad=3, W=_W, n_dead=2, max_row=6, empty_rows=(2,))
    with pytest.raises(ValueError, match="fourth lane"):
        _run(c, "plummer", True, 0, None, with_potential=True, with_force_scale=True)


def _kernel_primitives(interpret, with_force_scale, source_tile, flags):
    """Primitive counts of the float32 kernels, traced (not lowered: runs on CPU)."""
    import collections

    import jax

    def _eqns(jaxpr):
        for eqn in jaxpr.eqns:
            yield eqn
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    inner = getattr(sub, "jaxpr", sub)  # a ClosedJaxpr's Jaxpr
                    if hasattr(inner, "eqns"):
                        yield from _eqns(inner)

    c = _case(7, num_live=9, num_pad=3, W=_W, n_dead=2, max_row=6, empty_rows=(2,))
    traced = jax.make_jaxpr(
        lambda: nearfield_leafpair_csr_sorted_direct_pallas(
            jnp.asarray(c["pos"], jnp.float32),
            jnp.asarray(c["mass"], jnp.float32),
            c["starts"],
            c["counts"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
            leaf_width=_W,
            softening_sq=jnp.asarray(_EPS**2, jnp.float32),
            G=jnp.asarray(_G, jnp.float32),
            chunk=1,
            interpret=interpret,
            source_tile=source_tile,
            source_flags=flags,
            softening_kernel="ferrers3",
            with_force_scale=with_force_scale,
        )
    )()
    counts = collections.Counter()
    for call in (e for e in _eqns(traced.jaxpr) if e.primitive.name == "pallas_call"):
        counts.update(e.primitive.name for e in _eqns(call.params["jaxpr"]))
    return counts


@pytest.mark.parametrize("source_tile, flags", [(0, None), (8, "alr")])
def test_force_scale_lane_takes_the_division_reciprocal_on_triton(source_tile, flags):
    """The float32 Triton lane divides nowhere: ``_triton_f32_reciprocal`` (the
    division's own ``MUFU.RCP``) replaces its division -- 8 SASS instructions per pair
    instead of 16. Interpret mode (CPU semantics) keeps the division."""
    triton = _kernel_primitives(False, True, source_tile, flags)
    triton_off = _kernel_primitives(False, False, source_tile, flags)
    assert triton["div"] == triton_off["div"], (triton, triton_off)
    assert triton["elementwise_inline_asm_p"] > 0
    cpu = _kernel_primitives(True, True, source_tile, flags)
    cpu_off = _kernel_primitives(True, False, source_tile, flags)
    assert cpu["div"] > cpu_off["div"]
    assert cpu["elementwise_inline_asm_p"] == 0


def test_triton_f32_reciprocal_has_the_division_bits():
    """``a * _triton_f32_reciprocal(b)`` is Triton's float32 ``a / b`` bit for bit for
    every divisor in ``[2^-126, 2^126]`` (ptxas's ``div.full.f32`` rescales only
    outside it): what keeps the lane's ``f_b`` -- and the next step's eq (16a)
    lists -- unchanged."""
    _native_or_skip(False)
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    from jaccpot.pallas.nearfield_leafpair_csr import _triton_f32_reciprocal

    rng = np.random.default_rng(11)
    n, block = 1 << 18, 1024
    b = np.exp2(rng.uniform(-126.0, 126.0, n)).astype(np.float32)
    b[:4] = [
        2.0**-126,
        np.nextafter(np.float32(2.0**-126), np.float32(1)),
        2.0**126,
        1.0,
    ]
    b = np.clip(b, np.float32(2.0**-126), np.float32(2.0**126))
    a = (np.exp2(rng.uniform(-40.0, 40.0, n)) * rng.uniform(1.0, 2.0, n)).astype(
        np.float32
    )

    def kernel(a_ref, b_ref, q_ref, m_ref):
        x, y = a_ref[...], b_ref[...]
        q_ref[...] = x / y
        m_ref[...] = x * _triton_f32_reciprocal(y)

    spec = pl.BlockSpec((block,), lambda i: (i,))
    shape = jax.ShapeDtypeStruct((n,), jnp.float32)
    out = pl.pallas_call(
        kernel,
        out_shape=[shape, shape],
        grid=(n // block,),
        in_specs=[spec, spec],
        out_specs=[spec, spec],
        compiler_params=plgpu.CompilerParams(num_warps=4),
    )(jnp.asarray(a), jnp.asarray(b))
    q, m = (np.asarray(x).view(np.uint32) for x in out)
    assert np.array_equal(q, m), f"{int(np.sum(q != m))} of {n} differ"


@pytest.mark.parametrize("source_tile, flags", [(0, None), (8, "alr")])
def test_force_scale_lane_float32_on_triton_equals_the_near_pair_sum(
    source_tile, flags
):
    """The float32 Triton lane (the reciprocal path) against the float64 pair loop."""
    _native_or_skip(False)
    c = _case(7, num_live=9, num_pad=3, W=_W, n_dead=2, max_row=6, empty_rows=(2,))
    _, fb = nearfield_leafpair_csr_sorted_direct_pallas(
        jnp.asarray(c["pos"], jnp.float32),
        jnp.asarray(c["mass"], jnp.float32),
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
        leaf_width=_W,
        softening_sq=jnp.asarray(_EPS**2, jnp.float32),
        G=jnp.asarray(_G, jnp.float32),
        chunk=1,
        source_tile=source_tile,
        source_flags=flags,
        softening_kernel="ferrers3",
        with_force_scale=True,
    )
    want = _reference(c)
    assert (want > 0).sum() > 20, "vacuous: no near pairs"
    np.testing.assert_allclose(np.asarray(fb, np.float64), want, rtol=2e-5, atol=0.0)
