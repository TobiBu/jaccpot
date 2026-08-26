"""The strict lane's cap-profile catalogue: what it selects, and what it swallows.

``fmm_strict_cap_profile.py`` measured **64%** (160 statements, 49 missed) and is
the remainder of audit item **F33** now that ``fmm_strict_run.py`` is at 79% and
``fmm_autotune.py`` at 92%. Like `cap_presets`, it is host-side: a JSON catalogue
of traversal capacities plus a selection policy, no device anywhere in it.

Two things make it worth pinning beyond line coverage.

**The selection policy fails toward a larger cap, and silently.** On an exact
context-key miss it keeps the same ``tree_mode`` and ``leaf`` and takes the
*largest* ``max_pair_queue`` under that prefix -- an over-estimate, because a cap
that is too small costs a retry and a recompile on the per-step hot path. Nothing
reports that a fallback happened; the only trace is
``_strict_profiled_context_key``, which is why these tests assert on it.

**The loader is fail-open by construction.** ``_maybe_load_strict_cap_profile``
wraps its whole body in ``except Exception: return``, so a corrupt profile, an
unreadable file and a well-formed file are indistinguishable to the caller -- all
three leave the run on default caps. That is a deliberate choice (a bad cache must
not break a science run) and it is exactly the kind of choice that should have a
test saying so out loud, because it also means a typo'd profile is a silent
performance regression rather than an error.

Every test here pins ``JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH`` to ``tmp_path``.
The default is ``/tmp/jaccpot_static_strict_caps.json``, a real shared file on a
shared box -- reading it would make these tests depend on whatever a benchmark run
left behind.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest
from yggdrax.interactions import DualTreeRetryEvent

from jaccpot.runtime._fmm_impl import FMMEngine

_ENV = "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH"


@pytest.fixture
def engine():
    """A real engine, per the house pattern -- the mixin is only ever mixed in.

    Returns
    -------
    FMMEngine
        A fresh engine; constructing one builds no tree, so this is cheap.
    """
    return FMMEngine(theta=0.6, working_dtype=jnp.float32)


@pytest.fixture
def profile_path(tmp_path, monkeypatch):
    """Point the loader at a private file.

    Returns
    -------
    pathlib.Path
        The path the engine will read; it does not exist until a test writes it.
    """
    path = tmp_path / "strict_caps.json"
    monkeypatch.setenv(_ENV, str(path))
    return path


def _write(path, profiles):
    path.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")


def _key(tree_mode="dehnen", leaf=32, n=1000):
    return f"tree_mode={tree_mode}|leaf={leaf}|n={n}"


class TestContextKey:
    """The key is the identity of a profile, so its normalisation is load-bearing."""

    def test_tree_mode_is_normalised(self, engine):
        """Case and surrounding whitespace must not fork the catalogue."""
        assert (
            engine._strict_cap_profile_context_key(
                tree_mode="  Dehnen  ", leaf_parameter=32, particle_count=1000
            )
            == "tree_mode=dehnen|leaf=32|n=1000"
        )

    def test_numeric_fields_are_coerced(self, engine):
        """A float leaf or count must not produce a key like ``leaf=32.0``."""
        assert (
            engine._strict_cap_profile_context_key(
                tree_mode="dehnen", leaf_parameter=32.0, particle_count=1000.0
            )
            == "tree_mode=dehnen|leaf=32|n=1000"
        )


class TestLoading:
    """What reaches the catalogue, and what is swallowed on the way."""

    def test_a_missing_file_leaves_defaults_untouched(self, engine, profile_path):
        """No profile is the normal case, not an error."""
        engine._maybe_load_strict_cap_profile(context_key=_key())
        assert engine._strict_profile_catalog == {}
        assert engine._strict_profiled_max_pair_queue == 0

    def test_a_catalogue_is_loaded_and_applied(self, engine, profile_path):
        _write(
            profile_path,
            {_key(): {"max_pair_queue": 4096, "pair_process_block": 256}},
        )
        engine._maybe_load_strict_cap_profile(context_key=_key())
        assert engine._strict_profiled_max_pair_queue == 4096
        assert engine._strict_profiled_pair_process_block == 256
        assert engine._strict_profiled_context_key == _key()

    def test_the_legacy_single_profile_payload_still_loads(self, engine, profile_path):
        """The original format had no ``profiles`` map; it becomes one entry."""
        profile_path.write_text(
            json.dumps({"max_pair_queue": 2048, "pair_process_block": 128}),
            encoding="utf-8",
        )
        engine._maybe_load_strict_cap_profile()
        assert engine._strict_profile_catalog == {
            "legacy_default": {"max_pair_queue": 2048, "pair_process_block": 128}
        }
        assert engine._strict_profiled_max_pair_queue == 2048

    def test_a_corrupt_profile_is_swallowed(self, engine, profile_path):
        """Fail-open, deliberately -- and worth stating, because it is silent.

        A bad cache must not break a science run, so the loader cannot raise. The
        cost is that a typo'd profile is indistinguishable from no profile: the
        run continues on default caps and nothing says why.
        """
        profile_path.write_text("{not json at all", encoding="utf-8")
        engine._maybe_load_strict_cap_profile(context_key=_key())
        assert engine._strict_profile_catalog == {}
        assert engine._strict_profiled_max_pair_queue == 0

    def test_the_file_is_read_once_but_reapplied_per_key(self, engine, profile_path):
        """The latch guards the *read*, not the selection.

        A run visits several contexts; each must get its own caps without paying
        for a re-read, so deleting the file after the first call must not stop the
        second context from resolving.
        """
        _write(
            profile_path,
            {
                _key(n=1000): {"max_pair_queue": 1024, "pair_process_block": 64},
                _key(n=2000): {"max_pair_queue": 8192, "pair_process_block": 512},
            },
        )
        engine._maybe_load_strict_cap_profile(context_key=_key(n=1000))
        assert engine._strict_profiled_max_pair_queue == 1024
        profile_path.unlink()
        engine._maybe_load_strict_cap_profile(context_key=_key(n=2000))
        assert engine._strict_profiled_max_pair_queue == 8192


class TestSelectionPolicy:
    """On a miss the catalogue guesses, and the direction of the guess matters."""

    def test_an_exact_key_is_preferred(self, engine, profile_path):
        _write(
            profile_path,
            {
                _key(n=1000): {"max_pair_queue": 1024, "pair_process_block": 64},
                _key(n=9999): {"max_pair_queue": 99999, "pair_process_block": 999},
            },
        )
        engine._maybe_load_strict_cap_profile(context_key=_key(n=1000))
        assert engine._strict_profiled_max_pair_queue == 1024

    def test_a_miss_takes_the_largest_queue_at_the_same_tree_mode_and_leaf(
        self, engine, profile_path
    ):
        """Over-estimate on purpose: too small costs a retry and a recompile."""
        _write(
            profile_path,
            {
                _key(n=500): {"max_pair_queue": 512, "pair_process_block": 32},
                _key(n=2000): {"max_pair_queue": 4096, "pair_process_block": 256},
            },
        )
        engine._maybe_load_strict_cap_profile(context_key=_key(n=1234))
        assert engine._strict_profiled_max_pair_queue == 4096
        assert engine._strict_profiled_context_key == _key(n=2000)

    def test_a_different_leaf_is_not_borrowed_from(self, engine, profile_path):
        """The prefix is ``tree_mode`` *and* ``leaf`` -- both change the traversal.

        Without the leaf in the prefix, a leaf-8 profile would supply caps for a
        leaf-64 run, which is a different pair count entirely.
        """
        _write(
            profile_path,
            {_key(leaf=8, n=1000): {"max_pair_queue": 4096, "pair_process_block": 256}},
        )
        engine._maybe_load_strict_cap_profile(context_key=_key(leaf=64, n=1000))
        assert engine._strict_profiled_max_pair_queue == 0
        assert engine._strict_profiled_context_key == ""

    def test_legacy_default_is_the_last_resort(self, engine, profile_path):
        _write(
            profile_path,
            {
                "legacy_default": {"max_pair_queue": 777, "pair_process_block": 77},
                _key(tree_mode="bh", leaf=8): {
                    "max_pair_queue": 4096,
                    "pair_process_block": 256,
                },
            },
        )
        engine._maybe_load_strict_cap_profile(context_key=_key(leaf=64))
        assert engine._strict_profiled_max_pair_queue == 777
        assert engine._strict_profiled_context_key == "legacy_default"

    def test_a_zero_cap_does_not_clobber_a_live_one(self, engine, profile_path):
        """``if q > 0`` -- a profile that omits a field must not zero it.

        A zeroed ``max_pair_queue`` would size the traversal buffer to nothing,
        so "unset" and "set to zero" have to stay distinguishable.
        """
        engine._strict_profiled_max_pair_queue = 4096
        engine._strict_profiled_pair_process_block = 256
        _write(profile_path, {_key(): {"max_pair_queue": 0, "pair_process_block": 0}})
        engine._maybe_load_strict_cap_profile(context_key=_key())
        assert engine._strict_profiled_max_pair_queue == 4096
        assert engine._strict_profiled_pair_process_block == 256


class TestCapacityCompatibility:
    """``_compiled_profile_capacity_compatible`` decides whether to recompile."""

    BASE = {
        "max_nodes": 100,
        "max_leaves": 50,
        "max_nearfield_blocks": 20,
        "max_nearfield_target_block_slots": 10,
        "max_leaf_particle_slots": 5,
    }

    def test_a_candidate_needing_less_fits(self, engine):
        smaller = {k: v - 1 for k, v in self.BASE.items()}
        assert engine._compiled_profile_capacity_compatible(self.BASE, smaller)

    def test_an_equal_candidate_fits(self, engine):
        assert engine._compiled_profile_capacity_compatible(self.BASE, dict(self.BASE))

    @pytest.mark.parametrize("field", sorted(BASE))
    def test_exceeding_any_single_capacity_forces_a_recompile(self, engine, field):
        """One field over is enough -- the padded shape no longer holds it."""
        bigger = dict(self.BASE)
        bigger[field] += 1
        assert not engine._compiled_profile_capacity_compatible(self.BASE, bigger)

    def test_the_relation_is_asymmetric(self, engine):
        """ "Fits inside", not "equals" -- so swapping the arguments differs."""
        smaller = {k: v - 1 for k, v in self.BASE.items()}
        assert engine._compiled_profile_capacity_compatible(self.BASE, smaller)
        assert not engine._compiled_profile_capacity_compatible(smaller, self.BASE)

    def test_an_unrecognised_profile_reads_as_compatible(self, engine):
        """The documented sharp edge: missing keys read as 0 on both sides.

        So a profile carrying none of the five fields "needs nothing" and is
        judged reusable. That is fail-open in the direction of *not* recompiling,
        which is the direction that can hand a too-small buffer to a real run --
        pinned here so the behaviour is a decision on record rather than a
        side effect of ``.get(name, 0)``.
        """
        assert engine._compiled_profile_capacity_compatible(self.BASE, {"unrelated": 1})


class TestTransitionCounting:
    """Fingerprint transitions are the recompile signal the strict lane reports."""

    def test_the_first_fingerprint_is_not_a_transition(self, engine):
        before = engine._compiled_profile_transitions
        engine._compiled_profile_record_transition("aaa")
        assert engine._compiled_profile_transitions == before

    def test_a_change_counts_and_a_repeat_does_not(self, engine):
        engine._compiled_profile_record_transition("aaa")
        before = engine._compiled_profile_transitions
        engine._compiled_profile_record_transition("bbb")
        assert engine._compiled_profile_transitions == before + 1
        engine._compiled_profile_record_transition("bbb")
        assert engine._compiled_profile_transitions == before + 1

    def test_the_fingerprint_is_stable_and_order_independent(self, engine):
        """Same profile, different key order -- one fingerprint, or every step
        would look like a recompile."""
        a = engine._compiled_profile_fingerprint({"x": 1, "y": 2})
        b = engine._compiled_profile_fingerprint({"y": 2, "x": 1})
        assert a == b
        assert a != engine._compiled_profile_fingerprint({"x": 1, "y": 3})


class TestFusedProfileNGate:
    """``_strict_fused_profile_allows_n`` reads an operator-supplied N list."""

    def test_an_unset_list_allows_everything(self, engine):
        assert engine._strict_fused_profile_allows_n(1234)

    def test_membership_is_exact(self, engine):
        engine._strict_fused_profile_set_raw = "1000,2000"
        assert engine._strict_fused_profile_allows_n(1000)
        assert not engine._strict_fused_profile_allows_n(1500)

    def test_whitespace_and_empty_tokens_are_tolerated(self, engine):
        engine._strict_fused_profile_set_raw = " 1000 , , 2000 ,"
        assert engine._strict_fused_profile_allows_n(2000)

    def test_unparseable_tokens_are_skipped_not_fatal(self, engine):
        engine._strict_fused_profile_set_raw = "1000,not-a-number,2000"
        assert engine._strict_fused_profile_allows_n(2000)
        assert not engine._strict_fused_profile_allows_n(3000)

    def test_an_entirely_unparseable_list_allows_everything(self, engine):
        """Fail-open: a typo must not silently disable the fused lane for all N.

        The alternative -- an empty allow-set meaning "allow nothing" -- would
        turn one bad character in an env var into a whole-run performance change
        with no error.
        """
        engine._strict_fused_profile_set_raw = "nonsense,,also-nonsense"
        assert engine._strict_fused_profile_allows_n(1234)


def _retry(queue_capacity):
    """A real ``DualTreeRetryEvent``, not a stand-in.

    Parameters
    ----------
    queue_capacity : int
        The capacity the retry settled on -- the only field this path reads.

    Returns
    -------
    DualTreeRetryEvent
        Populated positionally-by-name so a field reorder upstream fails loudly.
    """
    return DualTreeRetryEvent(
        attempt=1,
        queue_capacity=queue_capacity,
        interaction_capacity=0,
        status="grown",
        far_pair_count=0,
        near_pair_count=0,
    )


class TestRecording:
    """The write half. Its invariant is a ratchet: caps grow, never shrink."""

    def test_no_retries_records_nothing(self, engine, profile_path):
        """A clean run has nothing to learn from, and must not write a file."""
        engine._record_strict_cap_profile_from_retries((), context_key=_key())
        assert not profile_path.exists()

    def test_the_largest_retry_capacity_wins(self, engine, profile_path):
        engine._record_strict_cap_profile_from_retries(
            (_retry(1024), _retry(8192), _retry(2048)), context_key=_key()
        )
        assert engine._strict_profiled_max_pair_queue == 8192

    def test_caps_ratchet_upward_only(self, engine, profile_path):
        """A later, smaller run must not forget what an earlier one needed.

        Lowering the cap here would reintroduce the retry -- and each retry is a
        recompile on the per-step hot path, which is the cost this whole
        catalogue exists to avoid.
        """
        engine._record_strict_cap_profile_from_retries(
            (_retry(8192),), context_key=_key()
        )
        engine._record_strict_cap_profile_from_retries(
            (_retry(512),), context_key=_key()
        )
        assert engine._strict_profiled_max_pair_queue == 8192
        entry = engine._strict_profile_catalog[_key()]
        assert entry["max_pair_queue"] == 8192

    def test_an_unreadable_capacity_counts_as_zero(self, engine, profile_path):
        """One malformed event must not lose the whole recording."""

        class _Exploding(tuple):
            @property
            def queue_capacity(self):
                raise RuntimeError("no capacity here")

        engine._record_strict_cap_profile_from_retries(
            (_Exploding(), _retry(4096)), context_key=_key()
        )
        assert engine._strict_profiled_max_pair_queue == 4096

    def test_nothing_positive_to_record_writes_no_file(self, engine, profile_path):
        engine._record_strict_cap_profile_from_retries((_retry(0),), context_key=_key())
        assert not profile_path.exists()

    def test_the_written_payload_reloads_into_a_fresh_engine(
        self, engine, profile_path
    ):
        """The round trip is the point: this catalogue is a cross-run cache."""
        engine._record_strict_cap_profile_from_retries(
            (_retry(4096),), context_key=_key()
        )
        assert profile_path.exists()
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        assert payload["version"] == 2
        assert payload["active_context_key"] == _key()

        fresh = FMMEngine(theta=0.6, working_dtype=jnp.float32)
        fresh._maybe_load_strict_cap_profile(context_key=_key())
        assert fresh._strict_profiled_max_pair_queue == 4096

    def test_recording_can_be_disabled_without_losing_the_in_memory_cap(
        self, engine, profile_path
    ):
        """The gate stops the *write*, not the learning.

        The distinction matters: a run with recording off still benefits from the
        cap it discovered, it just does not publish it to a shared file.
        """
        engine._strict_cap_record_enabled = False
        engine._record_strict_cap_profile_from_retries(
            (_retry(4096),), context_key=_key()
        )
        assert engine._strict_profiled_max_pair_queue == 4096
        assert engine._strict_profile_catalog[_key()]["max_pair_queue"] == 4096
        assert not profile_path.exists()

    def test_an_unwritable_path_is_swallowed_and_the_cap_survives(
        self, engine, tmp_path, monkeypatch
    ):
        """The write is fail-open too, matching the read.

        Pointing the profile at a directory makes ``open(..., "w")`` raise. The
        run must continue on the cap it just learned -- persisting the catalogue
        is an optimisation, and losing it is not worth failing a science step.
        """
        blocked = tmp_path / "not_a_file"
        blocked.mkdir()
        monkeypatch.setenv(_ENV, str(blocked))
        engine._record_strict_cap_profile_from_retries(
            (_retry(4096),), context_key=_key()
        )
        assert engine._strict_profiled_max_pair_queue == 4096
