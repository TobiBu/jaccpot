"""The near-field kernel's force-scale lane: eq (16b)'s ``f_b`` near term, per particle.

``nearfield_leafpair_csr_sorted_direct_pallas(..., with_force_scale=True)`` puts
``sum_a G m_a / (|x_a - x_b|^2 + eps^2)`` over the kernel's own pair set (the
neighbour leaves and the own leaf, self excluded) in the fourth output lane, with
``eps`` the Plummer-equivalent softening whatever the pair kernel -- the eager
estimator's form, so the per-step scale and the first step's prepass agree. Pinned:

* the lane equals a numpy loop over the same pairs, for every kernel and every
  source-tile mode (the lean, generic and compact pair bodies each have one);
* the accelerations are unchanged by the switch;
* the switch and ``with_potential`` are exclusive.
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
