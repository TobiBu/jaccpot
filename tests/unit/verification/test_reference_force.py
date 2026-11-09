"""The A4 reference force (``jaccpot.verification.reference``) on one CPU device.

* it equals ``jaccpot.runtime.reference.direct_sum`` to fp64 round-off for every
  softening kernel (and Newton), and an extended-precision sum to ~1 ulp;
* its potential equals a host fp64 direct sum;
* it is the same bits for ANY row order, padding and split of the sources into
  groups (the integer-limb accumulation is associative): what makes it the same
  number on any device or process count (the cross-device gates are
  ``tests/distributed/test_reference_force_devices.py`` and
  ``test_multiprocess_reference.py``);
* the targets depend on ``(N, K, seed)`` only; a missing or duplicated target is
  refused; ``score_force`` and the single-card adapter score against it.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.reference import direct_sum
from jaccpot.softening import pair_factors, softening_params_np
from jaccpot.verification import (
    ForceLaw,
    gather_by_id,
    reference_force,
    score_force,
    score_single_result,
    select_targets,
)

N = 400
K = 48


def _particles(seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 3))
    x[:20] *= 0.01  # a dense clump: terms spanning many decades
    m = rng.uniform(0.5, 1.5, N) / N
    return x, m


def _host_reference(
    x: np.ndarray, m: np.ndarray, t: np.ndarray, law: ForceLaw
) -> tuple:
    """fp64 and extended-precision host sums (self pair excluded)."""
    params = softening_params_np(law.kernel, law.softening)
    acc64, pot64, accld = [], [], []
    for i in t:
        keep = np.arange(N) != i
        d = x[keep] - x[i]
        r2 = np.sum(d * d, axis=1)
        g, psi, _ = pair_factors(r2, params, law.kernel, potential=True, xp=np)
        acc64.append(law.G * np.sum((m[keep] * g)[:, None] * d, axis=0))
        pot64.append(-law.G * np.sum(m[keep] * psi))
        terms = ((m[keep] * g)[:, None] * d).astype(np.longdouble)
        accld.append(law.G * np.sum(np.sort(terms, axis=0), axis=0))
    return np.array(acc64), np.array(pot64), np.array(accld, dtype=np.longdouble)


LAWS = [
    ForceLaw(1.0, 0.02, "plummer"),
    ForceLaw(0.7, 0.02, "ferrers3"),
    ForceLaw(1.0, 0.02, "wendland_c2"),
    ForceLaw(1.0, 0.0, "plummer"),
]


@pytest.mark.parametrize("law", LAWS, ids=lambda w: f"{w.kernel}-{w.softening}")
def test_the_reference_is_the_fp64_direct_sum(law: ForceLaw) -> None:
    x, m = _particles()
    t = select_targets(N, K, seed=3)
    ref = reference_force(jnp.asarray(x), jnp.asarray(m), jnp.arange(N), t, law)
    assert ref.num_sources == N
    np.testing.assert_array_equal(ref.positions, x[t])
    want = np.stack(
        [
            np.asarray(
                direct_sum(
                    jnp.asarray(np.delete(x, i, 0)),
                    jnp.asarray(np.delete(m, i, 0)),
                    jnp.asarray(x[i]),
                    G=law.G,
                    softening=law.softening,
                    softening_kernel=law.kernel,
                )
            )
            for i in t
        ]
    )
    rel = np.linalg.norm(ref.acceleration - want, axis=1) / np.linalg.norm(want, axis=1)
    assert rel.max() < 1e-14, rel.max()
    acc64, pot64, accld = _host_reference(x, m, t, law)
    rel_pot = np.abs(ref.potential - pot64) / np.abs(pot64)
    assert rel_pot.max() < 1e-14, rel_pot.max()
    # against an extended-precision sum of numpy's fp64 terms: a few ulp. XLA:CPU may
    # contract the pair term's products into FMAs depending on the host's codegen, so
    # its terms can differ from numpy's by an ulp: 1.9-2.8e-16 on an AMD EPYC 7452,
    # 4.7e-16 / 6.5e-16 on a GitHub runner. A naive fp64 sum of the same terms is at
    # 1.5-1.7e-15, so the bound still separates the exact summation from it; the
    # summation's order independence is gated bitwise in the tests below.
    err_ld = np.abs(ref.acceleration.astype(np.longdouble) - accld)
    scale = np.max(np.abs(accld), axis=1, keepdims=True)
    err = float(np.max(err_ld / scale))
    assert err < 1e-15, err


def _layout(x: np.ndarray, m: np.ndarray, seed: int, pad: int) -> tuple:
    """The particles in a random row order with ``pad`` padding rows spread in."""
    rng = np.random.default_rng(seed)
    rows = N + pad
    order = rng.permutation(rows)
    xs = np.zeros((rows, 3))
    ms = np.zeros(rows)
    ids = -np.ones(rows, np.int32)
    live = order[:N]
    xs[live], ms[live], ids[live] = x, m, np.arange(N)
    xs[order[N:]] = x[0]  # padding sitting on a live particle, as the runner's
    return xs, ms, ids


def test_any_row_order_padding_or_split_is_the_same_bits() -> None:
    x, m = _particles(2)
    t = select_targets(N, K, seed=5)
    law = ForceLaw(1.0, 0.02, "ferrers3")
    base = reference_force(jnp.asarray(x), jnp.asarray(m), jnp.arange(N), t, law)
    for seed, pad in ((0, 0), (1, 37), (2, 301)):
        xs, ms, ids = _layout(x, m, seed, pad)
        got = reference_force(
            jnp.asarray(xs), jnp.asarray(ms), jnp.asarray(ids), t, law
        )
        np.testing.assert_array_equal(got.acceleration, base.acceleration)
        np.testing.assert_array_equal(got.potential, base.potential)
        np.testing.assert_array_equal(got.positions, base.positions)
    # two source groups (e.g. an escaper tail): limbs added exactly
    cut = 123
    split = reference_force(
        jnp.asarray(x[:cut]),
        jnp.asarray(m[:cut]),
        jnp.arange(cut),
        t,
        law,
        extra_sources=(
            (jnp.asarray(x[cut:]), jnp.asarray(m[cut:]), jnp.arange(cut, N)),
        ),
    )
    np.testing.assert_array_equal(split.acceleration, base.acceleration)
    np.testing.assert_array_equal(split.potential, base.potential)
    assert split.num_sources == N
    # float32 positions are read as stored
    x32 = x.astype(np.float32)
    r32 = reference_force(jnp.asarray(x32), jnp.asarray(m), jnp.arange(N), t, law)
    np.testing.assert_array_equal(r32.positions, x32[t].astype(np.float64))


def test_targets_are_a_function_of_n_k_seed_only() -> None:
    a = select_targets(10_000, 64, seed=7)
    np.testing.assert_array_equal(a, select_targets(10_000, 64, seed=7))
    assert a.dtype == np.int64 and np.all(np.diff(a) > 0) and a[-1] < 10_000
    assert not np.array_equal(a, select_targets(10_000, 64, seed=8))
    assert select_targets(10, 64).shape == (10,)
    with pytest.raises(ValueError):
        select_targets(10, 0)


def test_a_missing_or_duplicated_target_is_refused() -> None:
    x, m = _particles()
    t = select_targets(N, 8, seed=0)
    ids = np.arange(N)
    ids[t[0]] = -1
    with pytest.raises(ValueError, match="not found exactly once"):
        reference_force(jnp.asarray(x), jnp.asarray(m), jnp.asarray(ids), t, ForceLaw())
    ids = np.arange(N)
    ids[(t[1] + 1) % N] = t[1]
    with pytest.raises(ValueError, match="not found exactly once"):
        reference_force(jnp.asarray(x), jnp.asarray(m), jnp.asarray(ids), t, ForceLaw())
    with pytest.raises(ValueError, match="ascending"):
        reference_force(
            jnp.asarray(x), jnp.asarray(m), jnp.arange(N), t[::-1], ForceLaw()
        )


def test_gather_by_id_is_exact() -> None:
    rng = np.random.default_rng(4)
    vals = rng.normal(size=(N, 3)).astype(np.float32)
    perm = rng.permutation(N)
    t = select_targets(N, K, seed=1)
    rows, found = gather_by_id(jnp.asarray(vals[perm]), jnp.asarray(perm), t)
    np.testing.assert_array_equal(rows, vals[t])
    assert rows.dtype == np.float32 and np.all(found == 1)
    flat, _ = gather_by_id(jnp.asarray(vals[perm, 0]), jnp.asarray(perm), t)
    np.testing.assert_array_equal(flat, vals[t, 0])


def test_score_force_and_the_single_card_adapter() -> None:
    x, m = _particles()
    law = ForceLaw(1.0, 0.02, "ferrers3")
    t = select_targets(N, K, seed=3)
    ref = reference_force(jnp.asarray(x), jnp.asarray(m), jnp.arange(N), t, law)
    exact = score_force(ref.acceleration, ref, potential=ref.potential)
    assert exact["rel_l2"] == 0.0 and exact["max"] == 0.0 and exact["num_targets"] == K
    assert exact["potential_rel_l2"] == 0.0
    noisy = ref.acceleration * (1.0 + 1e-3)
    s = score_force(noisy, ref)
    assert abs(s["rel_l2"] - 1e-3) < 1e-9 and abs(s["p50"] - 1e-3) < 1e-9
    assert s["p50"] <= s["p90"] <= s["p99"] <= s["max"]
    bad = ref.acceleration.copy()
    bad[0] = np.nan
    assert score_force(bad, ref)["nonfinite"] == 1

    # the single-card adapter on a stand-in RunResult whose "production" force is
    # the host fp64 direct sum at every particle
    acc_all = np.zeros((N, 3))
    params = softening_params_np(law.kernel, law.softening)
    for i in range(N):
        d = x - x[i]
        r2 = np.sum(d * d, axis=1)
        g = pair_factors(r2, params, law.kernel, xp=np)[0]
        g[i] = 0.0
        acc_all[i] = np.sum((m * g)[:, None] * d, axis=0)
    state = jnp.asarray(np.stack([x, np.zeros_like(x)], axis=1))
    result = SimpleNamespace(
        mode="single",
        escapers=None,
        state=state,
        masses=jnp.asarray(m),
        handle=SimpleNamespace(self_acceleration=jnp.asarray(acc_all), force_key=None),
    )
    solver = SimpleNamespace(
        _impl=SimpleNamespace(G=1.0, softening=0.02, softening_kernel="ferrers3")
    )
    out = score_single_result(result, solver, num_targets=K, seed=3)
    assert out["rel_l2"] < 1e-14 and out["num_targets"] == K
    assert out["law"] == law.as_dict()
    # force_key wins over the solver's parameters
    result.handle.force_key = (1.0, 0.02**2)
    assert score_single_result(result, solver, num_targets=K, seed=3)["rel_l2"] < 1e-14
    with pytest.raises(NotImplementedError):
        score_single_result(SimpleNamespace(mode="mesh"), solver)
    assert jax.device_count() >= 1
