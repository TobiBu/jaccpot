"""The two host-side autotune helpers -- audit item **F33**.

``fmm_autotune.py`` measured **11%** (101 of 119 statements missing), unchanged
by the A100 run #193 recorded. Nothing reached it, on either backend.

The file is three methods. Two are pure host-side logic with precise documented
contracts and are covered here: ``_select_autotune_m2l_candidates`` (a table
lookup) and ``_sample_and_remap_far_pairs_for_autotune`` (a strided sample plus a
node-id remap). The third, ``_autotune_runtime_m2l_chunk_size``, is a timing loop
that needs a built tree and real M2L work, and is its own increment.

Neither method under test touches engine state, but both are called on a real
engine rather than a stand-in ``self``: they are mixin methods, and a test that
passes a dummy object would keep passing if one of them started reading
``self.`` and would then be testing something the engine never does.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime.fmm_autotune import (
    _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES,
    _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES,
    _GPU_M2L_AUTOTUNE_PAIR_BINS,
    _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES,
    _GPU_M2L_AUTOTUNE_XL_CANDIDATES,
)


@pytest.fixture(scope="module")
def engine():
    """A default engine; these helpers are host-side and need no prepared state.

    ``FMMEngine`` rather than the public ``FastMultipoleMethod``: the autotune
    methods live on ``AutotuneMixin``, which the engine inherits and the solver
    wraps, so they are not reachable through the public class.
    """
    return FMMEngine(theta=0.6, working_dtype=jnp.float32)


# --------------------------------------------------------------------------
# _select_autotune_m2l_candidates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pair_count,expected",
    [
        (0, _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[0] - 1, _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[0], _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[1] - 1, _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[1], _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[2] - 1, _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES),
        (_GPU_M2L_AUTOTUNE_PAIR_BINS[2], _GPU_M2L_AUTOTUNE_XL_CANDIDATES),
        (10 * _GPU_M2L_AUTOTUNE_PAIR_BINS[-1], _GPU_M2L_AUTOTUNE_XL_CANDIDATES),
    ],
)
def test_candidate_bins_are_half_open_at_every_edge(engine, pair_count, expected):
    """Each bin is ``[lo, hi)`` -- checked *at* the edges, not near them.

    Off-by-one on a bin edge is the failure this cannot catch by sampling the
    middle of each regime, so every boundary is tested at ``hi - 1`` and ``hi``.
    """
    assert engine._select_autotune_m2l_candidates(pair_count=pair_count) == expected


@pytest.mark.parametrize(
    "pair_count", [0, 1, 1000, 65_536, 262_144, 1_048_576, 4_194_304, 1 << 40]
)
def test_candidates_are_never_empty(engine, pair_count):
    """The docstring promises a caller always has something to time."""
    candidates = engine._select_autotune_m2l_candidates(pair_count=pair_count)
    assert candidates, f"empty candidate list for pair_count={pair_count}"
    assert all(int(c) > 0 for c in candidates)


def test_candidate_regimes_grow_with_pair_count(engine):
    """ "Smallest regime first" -- the regimes must actually be ordered.

    Asserted across regimes rather than within one, because the point of the
    table is that more pairs get larger chunks.
    """
    regimes = [
        engine._select_autotune_m2l_candidates(pair_count=0),
        engine._select_autotune_m2l_candidates(
            pair_count=_GPU_M2L_AUTOTUNE_PAIR_BINS[0]
        ),
        engine._select_autotune_m2l_candidates(
            pair_count=_GPU_M2L_AUTOTUNE_PAIR_BINS[1]
        ),
        engine._select_autotune_m2l_candidates(
            pair_count=_GPU_M2L_AUTOTUNE_PAIR_BINS[2]
        ),
    ]
    maxima = [max(r) for r in regimes]
    assert maxima == sorted(maxima), f"regime maxima are not increasing: {maxima}"


# --------------------------------------------------------------------------
# _sample_and_remap_far_pairs_for_autotune
# --------------------------------------------------------------------------


def test_empty_pair_list_returns_empty_arrays_of_the_documented_dtypes(engine):
    """No pairs is "do not autotune", not an error."""
    src_local, tgt_local, local_to_global = (
        engine._sample_and_remap_far_pairs_for_autotune(
            src=jnp.zeros((0,), dtype=jnp.int32), tgt=jnp.zeros((0,), dtype=jnp.int32)
        )
    )
    assert src_local.shape == (0,) and tgt_local.shape == (0,)
    assert local_to_global.shape == (0,)
    assert src_local.dtype == np.int32
    assert tgt_local.dtype == np.int32
    assert local_to_global.dtype == np.int64


def test_remap_round_trips_through_local_to_global(engine):
    """The whole point of the remap: local ids must decode to the originals."""
    src = jnp.asarray([70, 70, 91, 12], dtype=jnp.int32)
    tgt = jnp.asarray([91, 12, 70, 70], dtype=jnp.int32)
    src_local, tgt_local, local_to_global = (
        engine._sample_and_remap_far_pairs_for_autotune(src=src, tgt=tgt)
    )
    np.testing.assert_array_equal(local_to_global[src_local], np.asarray(src))
    np.testing.assert_array_equal(local_to_global[tgt_local], np.asarray(tgt))


def test_local_ids_are_dense_and_contiguous(engine):
    """ "Renumbered into a dense range" -- so the gather can use a compact array."""
    rng = np.random.default_rng(0)
    globals_ = rng.choice(10_000, size=64, replace=False)
    src = jnp.asarray(globals_[:32], dtype=jnp.int32)
    tgt = jnp.asarray(globals_[32:], dtype=jnp.int32)
    src_local, tgt_local, local_to_global = (
        engine._sample_and_remap_far_pairs_for_autotune(src=src, tgt=tgt)
    )
    used = set(src_local.tolist()) | set(tgt_local.tolist())
    assert used == set(range(len(local_to_global)))


def test_the_sample_is_strided_not_random(engine):
    """Documented as strided, which makes it deterministic and reproducible."""
    n = 1000
    src = jnp.arange(n, dtype=jnp.int32)
    tgt = jnp.arange(n, dtype=jnp.int32) + n
    src_local, _, local_to_global = engine._sample_and_remap_far_pairs_for_autotune(
        src=src, tgt=tgt, max_pairs=10, max_nodes=10_000
    )
    stride = max(1, n // 10)
    expected_first_global = int(np.arange(n)[::stride][0])
    assert int(local_to_global[src_local[0]]) == expected_first_global
    # every kept source is a stride multiple, which a random sample would not be
    kept = local_to_global[src_local]
    assert np.all(kept % stride == 0), f"sample is not strided: {kept[:8]}"


def test_max_pairs_bounds_the_sample(engine):
    """The cap is a cap, including after the stride rounds down."""
    n = 4097
    src = jnp.arange(n, dtype=jnp.int32)
    tgt = jnp.arange(n, dtype=jnp.int32) + n
    src_local, tgt_local, _ = engine._sample_and_remap_far_pairs_for_autotune(
        src=src, tgt=tgt, max_pairs=16, max_nodes=10_000
    )
    assert src_local.shape[0] <= 16
    assert src_local.shape == tgt_local.shape


def test_node_admission_stops_at_max_nodes_and_drops_those_pairs(engine):
    """Admission is capped, and a pair whose endpoint missed the cap is dropped.

    Also pins an edge the implementation has and the docstring does not spell
    out: a node can be admitted while its own pair is still dropped, when the
    source is taken and the target then hits the cap. So ``local_to_global`` may
    contain a node that no surviving pair references, and a test asserting
    "every admitted node is used" would fail against correct code.
    """
    n = 64
    src = jnp.arange(n, dtype=jnp.int32)
    tgt = jnp.arange(n, dtype=jnp.int32) + n
    src_local, tgt_local, local_to_global = (
        engine._sample_and_remap_far_pairs_for_autotune(
            src=src, tgt=tgt, max_pairs=n, max_nodes=7
        )
    )
    assert len(local_to_global) <= 7
    # every surviving pair indexes an admitted node
    assert src_local.max(initial=-1) < len(local_to_global)
    assert tgt_local.max(initial=-1) < len(local_to_global)
    # and the pair list really was truncated by the node cap
    assert src_local.shape[0] < n
