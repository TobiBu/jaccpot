"""Tests for the fused leaf-major near-field Pallas kernel.

The Pallas kernel is exercised in ``interpret=True`` mode so these run on CPU
(and therefore in CI where no Ampere+ GPU is available). On a supported GPU the
same kernel is validated end-to-end by the runtime tests in
``tests/unit/core/test_near_field.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_fused_leaf_backend,
    nearfield_fused_leaf_jax,
    nearfield_fused_leaf_pallas,
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
    pallas_nearfield_fused_supported,
)


def _manual_reference(tp, tmask, sp, sm, smask, *, softening_sq, G):
    """Direct leaf-major (accel, potential) reference in float64."""
    tp = np.asarray(tp, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    sm = np.asarray(sm, dtype=np.float64)
    tmask = np.asarray(tmask, dtype=bool)
    smask = np.asarray(smask, dtype=bool)
    L, Wt, _ = tp.shape
    K = sp.shape[1]
    out = np.zeros((L, Wt, 4), dtype=np.float64)
    for leaf in range(L):
        for i in range(Wt):
            if not tmask[leaf, i]:
                continue
            acc = np.zeros(3)
            pot = 0.0
            for j in range(K):
                if not smask[leaf, j]:
                    continue
                diff = tp[leaf, i] - sp[leaf, j]
                r2 = float(np.dot(diff, diff)) + softening_sq
                inv = 1.0 / np.sqrt(r2)
                acc += -G * (inv**3) * sm[leaf, j] * diff
                pot += -G * inv * sm[leaf, j]
            out[leaf, i, :3] = acc
            out[leaf, i, 3] = pot
    return out


def _random_inputs(seed=0, L=4, Wt=8, K=20):
    rng = np.random.default_rng(seed)
    tp = jnp.asarray(rng.standard_normal((L, Wt, 3)), jnp.float32)
    sp = jnp.asarray(rng.standard_normal((L, K, 3)), jnp.float32)
    sm = jnp.asarray(np.abs(rng.standard_normal((L, K))) + 0.1, jnp.float32)
    tmask = jnp.asarray(rng.random((L, Wt)) > 0.25)
    smask = jnp.asarray(rng.random((L, K)) > 0.30)
    return tp, tmask, sp, sm, smask


def test_backend_returns_known_value():
    assert nearfield_fused_leaf_backend(prefer_pallas=True) in {"jax", "pallas"}
    assert nearfield_fused_leaf_backend(prefer_pallas=False) == "jax"


def test_supported_returns_bool():
    assert isinstance(pallas_nearfield_fused_supported(), bool)


def test_jax_reference_matches_manual():
    tp, tmask, sp, sm, smask = _random_inputs(seed=1)
    soft = 0.05**2
    G = 1.25
    ref = nearfield_fused_leaf_jax(
        tp, tmask, sp, sm, smask, softening_sq=jnp.float32(soft), G=jnp.float32(G)
    )
    manual = _manual_reference(tp, tmask, sp, sm, smask, softening_sq=soft, G=G)
    assert np.allclose(np.asarray(ref), manual, rtol=1e-4, atol=1e-5)


def test_interpret_matches_jax_reference():
    tp, tmask, sp, sm, smask = _random_inputs(seed=2)
    soft = jnp.float32(0.05**2)
    G = jnp.float32(1.25)
    ref = nearfield_fused_leaf_jax(tp, tmask, sp, sm, smask, softening_sq=soft, G=G)
    got = nearfield_fused_leaf_pallas(
        tp, tmask, sp, sm, smask, softening_sq=soft, G=G, interpret=True
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


def test_interpret_fully_masked_target_lane():
    tp, tmask, sp, sm, smask = _random_inputs(seed=3)
    tmask = tmask.at[:, 0].set(False)  # force some invalid target lanes
    soft = jnp.float32(1e-3**2)
    G = jnp.float32(1.0)
    got = nearfield_fused_leaf_pallas(
        tp, tmask, sp, sm, smask, softening_sq=soft, G=G, interpret=True
    )
    # invalid target lanes must be exactly zero
    assert np.all(np.asarray(got)[:, 0, :] == 0.0)


def test_interpret_all_sources_invalid_is_zero():
    tp, tmask, sp, sm, smask = _random_inputs(seed=4)
    smask = jnp.zeros_like(smask)  # no valid sources
    soft = jnp.float32(1e-3**2)
    G = jnp.float32(1.0)
    got = nearfield_fused_leaf_pallas(
        tp, tmask, sp, sm, smask, softening_sq=soft, G=G, interpret=True
    )
    assert np.allclose(np.asarray(got), 0.0, atol=1e-7)


def _leafpair_inputs(seed=0, L=6, W=8, S=5):
    rng = np.random.default_rng(seed)
    lp = jnp.asarray(rng.standard_normal((L, W, 3)), jnp.float32)
    lm = jnp.asarray(np.abs(rng.standard_normal((L, W))) + 0.1, jnp.float32)
    lmask = jnp.asarray(rng.random((L, W)) > 0.2)
    sids = np.zeros((L, S), np.int32)
    svalid = np.zeros((L, S), bool)
    for i in range(L):
        cand = [x for x in range(L) if x != i]
        pick = rng.choice(
            cand, size=min(int(rng.integers(0, S + 1)), len(cand)), replace=False
        )
        for k, p in enumerate(pick):
            sids[i, k] = p
            svalid[i, k] = True
    return lp, lm, lmask, jnp.asarray(sids), jnp.asarray(svalid)


def test_leafpair_interpret_matches_jax_reference():
    lp, lm, lmask, sids, svalid = _leafpair_inputs(seed=3)
    soft = jnp.float32(0.05**2)
    G = jnp.float32(1.3)
    ref = nearfield_leafpair_jax(lp, lm, lmask, sids, svalid, softening_sq=soft, G=G)
    got = nearfield_leafpair_pallas(
        lp, lm, lmask, sids, svalid, softening_sq=soft, G=G, interpret=True
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


def test_leafpair_interpret_subtile_matches_jax_reference():
    lp, lm, lmask, sids, svalid = _leafpair_inputs(seed=4, W=8)
    soft = jnp.float32(0.01**2)
    G = jnp.float32(0.9)
    ref = nearfield_leafpair_jax(lp, lm, lmask, sids, svalid, softening_sq=soft, G=G)
    got = nearfield_leafpair_pallas(
        lp,
        lm,
        lmask,
        sids,
        svalid,
        softening_sq=soft,
        G=G,
        target_subtile=4,
        interpret=True,
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-6)


def test_leafpair_interpret_no_valid_sources_is_zero():
    lp, lm, lmask, sids, svalid = _leafpair_inputs(seed=5)
    svalid = jnp.zeros_like(svalid)
    got = nearfield_leafpair_pallas(
        lp,
        lm,
        lmask,
        sids,
        svalid,
        softening_sq=jnp.float32(1e-3**2),
        G=jnp.float32(1.0),
        interpret=True,
    )
    assert np.allclose(np.asarray(got), 0.0, atol=1e-7)


@pytest.mark.skipif(
    not pallas_nearfield_fused_supported(),
    reason="leaf-pair near-field Pallas kernel requires an Ampere+ (sm_80+) GPU",
)
def test_leafpair_gpu_matches_jax_reference():
    lp, lm, lmask, sids, svalid = _leafpair_inputs(seed=6, L=10, W=16, S=6)
    soft = jnp.float32(0.02**2)
    G = jnp.float32(1.1)
    ref = nearfield_leafpair_jax(lp, lm, lmask, sids, svalid, softening_sq=soft, G=G)
    got = nearfield_leafpair_pallas(
        lp,
        lm,
        lmask,
        sids,
        svalid,
        softening_sq=soft,
        G=G,
        target_subtile=8,
        interpret=False,
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not pallas_nearfield_fused_supported(),
    reason="fused near-field Pallas kernel requires an Ampere+ (sm_80+) GPU",
)
def test_gpu_matches_jax_reference():
    tp, tmask, sp, sm, smask = _random_inputs(seed=5, L=6, Wt=16, K=48)
    soft = jnp.float32(0.01**2)
    G = jnp.float32(0.9)
    ref = nearfield_fused_leaf_jax(tp, tmask, sp, sm, smask, softening_sq=soft, G=G)
    got = nearfield_fused_leaf_pallas(
        tp, tmask, sp, sm, smask, softening_sq=soft, G=G, interpret=False
    )
    assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)
