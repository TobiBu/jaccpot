"""The strict lane's compiled-profile bookkeeping: what forces a recompile.

``fmm_strict_cap_profile.py`` measured **64%** (160 statements, 49 missed) and was
the remainder of audit item **F33** once ``fmm_strict_run.py`` was at 79%. Like
`cap_presets`, it is host-side, no device anywhere in it.

What is left after the 2026-10 cleanup (X4) are the compiled-profile fingerprints
(capacity compatibility and transition counting) and the fused lane's
``PROFILE_SET`` gate. The on-disk cap-profile catalogue -- the JSON file at
``JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH`` (default
``/tmp/jaccpot_static_strict_caps.json``), its context key, loader, selection
policy and recorder -- went in that phase, and with it the four classes that
pinned it (``TestContextKey``, ``TestLoading``, ``TestSelectionPolicy``,
``TestRecording``).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaccpot.runtime._fmm_impl import FMMEngine


@pytest.fixture
def engine():
    """A real engine, per the house pattern -- the mixin is only ever mixed in.

    Returns
    -------
    FMMEngine
        A fresh engine; constructing one builds no tree, so this is cheap.
    """
    return FMMEngine(theta=0.6, working_dtype=jnp.float32)


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
