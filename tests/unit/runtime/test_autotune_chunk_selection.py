"""``_autotune_runtime_m2l_chunk_size`` -- the last F33 block.

Lines 243-397 of ``fmm_autotune.py``: the timing loop that picks an M2L chunk
size by measuring candidates and keeping the fastest.

**The measured work is stubbed, deliberately.** The docstring is explicit that
this is a wall-clock measurement and "only as good as the machine is quiet: on a
shared or loaded GPU the winner is noise". Asserting *which* candidate wins would
therefore be asserting noise. What is worth pinning is the surrounding contract:
which conditions produce "no opinion", that a winner is always one of the offered
candidates, that the result is memoized by pair **bin** rather than pair count,
and that a cache hit skips the measurement entirely.

Driving the real kernel here is not an option and the reason is itself worth
recording. With ``jax.default_backend`` faked to ``"gpu"`` on this CPU box the
loop *does* run, but every candidate raises inside XLA ("Buffer has been deleted
or donated"), the ``except Exception: continue`` at line 389 swallows each one,
and the method returns ``None``. So a systematically broken kernel is reported
exactly like "autotuning is off" -- which the last test here pins, because it is
the one failure mode this function cannot distinguish.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime.fmm_autotune as autotune_mod
from jaccpot.config import RuntimePolicyConfig
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime.fmm_autotune import _GPU_M2L_AUTOTUNE_PAIR_BINS

N_PARTICLES = 2048
LEAF_SIZE = 32
ORDER = 2


class _StubResult:
    """Stands in for the kernel's output; only ``block_until_ready`` is called."""

    def block_until_ready(self):
        return self


@pytest.fixture(scope="module")
def problem():
    """A real upward sweep and far-pair list -- the loop reads both."""
    fmm = FMMEngine(
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        theta=0.6,
        working_dtype=jnp.float32,
        runtime_policy=RuntimePolicyConfig(autotune_m2l_chunk=True),
    )
    key = jax.random.PRNGKey(2)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    prepared = fmm.prepare_state(
        positions, masses, leaf_size=LEAF_SIZE, max_order=ORDER
    )
    interactions = prepared.interactions
    return fmm, prepared.upward, interactions.sources, interactions.targets


@pytest.fixture
def on_gpu(monkeypatch):
    """The backend gate reads ``jax.default_backend()`` at call time."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")


def _pairs(src) -> int:
    return int(np.asarray(src).size)


def test_returns_none_when_autotuning_is_off(problem, on_gpu):
    """``None`` means "no opinion, leave the chunk size alone"."""
    _, upward, src, tgt = problem
    off = FMMEngine(
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        theta=0.6,
        working_dtype=jnp.float32,
        runtime_policy=RuntimePolicyConfig(autotune_m2l_chunk=False),
    )
    assert (
        off._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=_pairs(src)
        )
        is None
    )


def test_returns_none_for_a_non_solidfmm_basis(problem, on_gpu):
    """The timed kernel is the solidfmm M2L; other bases get no opinion."""
    _, upward, src, tgt = problem
    other = FMMEngine(
        expansion_basis="cartesian",
        theta=0.6,
        working_dtype=jnp.float32,
        runtime_policy=RuntimePolicyConfig(autotune_m2l_chunk=True),
    )
    assert (
        other._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=_pairs(src)
        )
        is None
    )


def test_returns_none_off_gpu(problem, monkeypatch):
    """No ``on_gpu`` fixture: the real CPU backend must short-circuit."""
    fmm, upward, src, tgt = problem
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert (
        fmm._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=_pairs(src)
        )
        is None
    )


@pytest.mark.parametrize("pair_count", [0, -1])
def test_returns_none_for_a_non_positive_pair_count(problem, on_gpu, pair_count):
    """Documented as short-circuiting rather than sampling an empty problem."""
    fmm, upward, src, tgt = problem
    assert (
        fmm._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=pair_count
        )
        is None
    )


def test_returns_none_when_the_sample_comes_back_empty(problem, on_gpu):
    """A positive pair count with nothing to sample is still "no opinion"."""
    fmm, upward, _, _ = problem
    empty = jnp.zeros((0,), dtype=jnp.int32)
    assert (
        fmm._autotune_runtime_m2l_chunk_size(
            upward=upward, src=empty, tgt=empty, order=ORDER, pair_count=1024
        )
        is None
    )


def test_the_winner_is_one_of_the_offered_candidates(problem, on_gpu, monkeypatch):
    """Pins the selection contract without asserting a timing outcome.

    The kernel is stubbed so every candidate "succeeds" instantly; which one wins
    is then arbitrary and is *not* asserted. What must hold is that the winner
    came from the candidate list for this pair count.
    """
    fmm, upward, src, tgt = problem
    monkeypatch.setattr(
        autotune_mod, "_accumulate_m2l_chunked_scan", lambda *a, **k: _StubResult()
    )
    monkeypatch.setattr(autotune_mod, "_m2l_autotune_lookup", lambda key: None)
    stored: list[tuple] = []
    monkeypatch.setattr(
        autotune_mod,
        "_m2l_autotune_store",
        lambda key, value: stored.append((key, value)),
    )

    pair_count = _pairs(src)
    chosen = fmm._autotune_runtime_m2l_chunk_size(
        upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=pair_count
    )
    candidates = fmm._select_autotune_m2l_candidates(pair_count=pair_count)
    assert chosen in candidates, f"{chosen} is not among {candidates}"
    assert stored and stored[-1][1] == chosen, "the winner was not memoized"


def test_a_cache_hit_skips_the_measurement(problem, on_gpu, monkeypatch):
    """A hit must return without timing anything -- that is the point of it."""
    fmm, upward, src, tgt = problem
    calls: list[int] = []
    monkeypatch.setattr(
        autotune_mod,
        "_accumulate_m2l_chunked_scan",
        lambda *a, **k: (calls.append(1), _StubResult())[1],
    )
    monkeypatch.setattr(autotune_mod, "_m2l_autotune_lookup", lambda key: 4242)

    chosen = fmm._autotune_runtime_m2l_chunk_size(
        upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=_pairs(src)
    )
    assert chosen == 4242
    assert calls == [], "the kernel was timed despite a cache hit"


def test_the_cache_key_bins_the_pair_count(problem, on_gpu, monkeypatch):
    """Memoized by pair *bin*, so nearby sizes share one measurement.

    Two counts inside the same bin must produce the same key; the docstring says
    this is what lets a second run at a nearby problem size reuse the result
    rather than re-time it.
    """
    fmm, upward, src, tgt = problem
    keys: list[tuple] = []
    monkeypatch.setattr(
        autotune_mod, "_accumulate_m2l_chunked_scan", lambda *a, **k: _StubResult()
    )
    monkeypatch.setattr(
        autotune_mod, "_m2l_autotune_lookup", lambda key: (keys.append(key), None)[1]
    )
    monkeypatch.setattr(autotune_mod, "_m2l_autotune_store", lambda key, value: None)

    low = 1
    high = int(_GPU_M2L_AUTOTUNE_PAIR_BINS[0]) - 1  # same (first) bin
    other = int(_GPU_M2L_AUTOTUNE_PAIR_BINS[0])  # the next bin
    for count in (low, high, other):
        fmm._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=count
        )
    assert keys[0] == keys[1], "counts within one bin produced different keys"
    assert keys[2] != keys[0], "counts across a bin edge shared a key"


def test_every_candidate_failing_is_reported_as_no_opinion(
    problem, on_gpu, monkeypatch
):
    """The one failure mode this function cannot distinguish from "off".

    ``except Exception: continue`` swallows each candidate's failure, so a kernel
    that is broken for *every* candidate returns ``None`` -- the same answer as a
    disabled autotuner. Observed for real here: with the backend faked on a CPU
    box, every candidate raises inside XLA and this is exactly what happens.
    Pinned so the behaviour is a decision rather than a surprise.
    """
    fmm, upward, src, tgt = problem

    def _always_raises(*args, **kwargs):
        raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(autotune_mod, "_accumulate_m2l_chunked_scan", _always_raises)
    monkeypatch.setattr(autotune_mod, "_m2l_autotune_lookup", lambda key: None)
    monkeypatch.setattr(autotune_mod, "_m2l_autotune_store", lambda key, value: None)

    assert (
        fmm._autotune_runtime_m2l_chunk_size(
            upward=upward, src=src, tgt=tgt, order=ORDER, pair_count=_pairs(src)
        )
        is None
    )
