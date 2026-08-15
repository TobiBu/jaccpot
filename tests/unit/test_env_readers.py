"""What ``jaccpot._env`` promises, including what a malformed value means.

``_env`` is the single implementation behind every ``JACCPOT_*`` switch, and it had
no tests. Audit item 2.2 is about folding four hand-rolled readers into it, and that
could not be done safely while the answer to "what does a typo mean?" differed
between them:

- ``_fused_m2l_vjp_enabled`` (twice, byte-identical) read anything outside
  ``{0,false,no,off}`` as ON -- a *denylist*, so a typo left the default alone;
- ``env_flag`` read anything outside ``{1,true,yes,on}`` as OFF -- an *allowlist*, so
  a typo silently flipped a default-on knob;
- ``env_int``/``env_float`` already documented "unset **or unparseable** -> default".

So ``_env`` disagreed with itself, and the two conventions were both live: five
``env_flag(..., True)`` call sites in ``grad_options`` sit on the allowlist side, and
they select reverse-mode VJP kernels.

**The rule these tests pin: a malformed value means the default, whatever the default
is.** That makes a typo a no-op rather than a silent behaviour change, and makes
``env_flag`` agree with ``env_int``/``env_float``. It is also the only choice under
which the fused-VJP readers can be deduped without changing which reverse-mode kernel
runs (audit A.4).
"""

from __future__ import annotations

import warnings

import pytest

from jaccpot import _env


@pytest.fixture(autouse=True)
def _clear_warning_cache():
    """Reset the warn-once cache so each test sees a fresh module.

    The cache is what stops a hot path spamming, so it is deliberately process-wide;
    without this fixture the second test to read a given variable would silently
    assert nothing.
    """
    _env._reset_malformed_warning_cache()
    yield
    _env._reset_malformed_warning_cache()


class TestEnvFlag:
    """``env_flag`` truthiness, and the malformed-value rule."""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "  On  "])
    def test_truthy_spellings_are_true(self, monkeypatch, raw):
        """Case and surrounding whitespace must not matter."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", raw)
        assert _env.env_flag("JACCPOT_TEST_FLAG", False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off "])
    def test_falsey_spellings_are_false(self, monkeypatch, raw):
        """An explicit off must turn a default-on knob off -- the point of the knob."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", raw)
        assert _env.env_flag("JACCPOT_TEST_FLAG", True) is False

    @pytest.mark.parametrize("default", [True, False])
    def test_unset_gives_the_default(self, monkeypatch, default):
        """Unset is the uncontroversial half."""
        monkeypatch.delenv("JACCPOT_TEST_FLAG", raising=False)
        assert _env.env_flag("JACCPOT_TEST_FLAG", default) is default

    @pytest.mark.parametrize("default", [True, False])
    @pytest.mark.parametrize("raw", ["ture", "", "2", "maybe", "on!", "yess"])
    def test_malformed_gives_the_default(self, monkeypatch, default, raw):
        """**The decision.** A typo is a no-op, not a silent flip.

        ``default=True`` is the case that changed: it used to return False, so
        ``JACCPOT_ANALYTIC_P2P_VJP=ture`` silently disabled the analytic VJP. It now
        returns the default, matching ``env_int``/``env_float`` and matching the
        hand-rolled fused-VJP readers this consolidates.
        """
        monkeypatch.setenv("JACCPOT_TEST_FLAG", raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _env.env_flag("JACCPOT_TEST_FLAG", default) is default

    def test_the_fused_vjp_default_on_case_specifically(self, monkeypatch):
        """The regression audit A.4 warns about, asserted directly.

        ``JACCPOT_FUSED_M2L_VJP=garbage`` must leave the fused reverse ON. Under the
        old allowlist it would have turned it OFF, changing which reverse-mode M2L
        kernel runs -- a numerics-adjacent switch, silently, from a typo.
        """
        monkeypatch.setenv("JACCPOT_FUSED_M2L_VJP", "yez")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _env.env_flag("JACCPOT_FUSED_M2L_VJP", True) is True


class TestEnvIntAndFloat:
    """The numeric readers, whose malformed rule was already 'the default'."""

    def test_int_parses_and_falls_back(self, monkeypatch):
        """Parse when possible, default when not."""
        monkeypatch.setenv("JACCPOT_TEST_INT", " 42 ")
        assert _env.env_int("JACCPOT_TEST_INT", 7) == 42
        monkeypatch.setenv("JACCPOT_TEST_INT", "4.5")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _env.env_int("JACCPOT_TEST_INT", 7) == 7

    def test_int_minimum_clamps_but_only_when_given(self, monkeypatch):
        """0 stays meaningful where no minimum is passed -- several knobs use it."""
        monkeypatch.setenv("JACCPOT_TEST_INT", "0")
        assert _env.env_int("JACCPOT_TEST_INT", 7) == 0
        assert _env.env_int("JACCPOT_TEST_INT", 7, minimum=1) == 1

    def test_float_parses_and_falls_back(self, monkeypatch):
        """Same contract as the int reader."""
        monkeypatch.setenv("JACCPOT_TEST_FLOAT", "1.5")
        assert _env.env_float("JACCPOT_TEST_FLOAT", 0.25) == 1.5
        monkeypatch.setenv("JACCPOT_TEST_FLOAT", "wide")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _env.env_float("JACCPOT_TEST_FLOAT", 0.25) == 0.25


class TestEnvChoice:
    """The enum reader, which ``_env`` was missing entirely."""

    MODES = ("full", "m2l_only", "off")

    def test_known_values_pass_through(self, monkeypatch):
        """Case and whitespace normalised, as the hand-rolled reader did."""
        monkeypatch.setenv("JACCPOT_TEST_MODE", "  M2L_Only ")
        assert _env.env_choice("JACCPOT_TEST_MODE", "full", self.MODES) == "m2l_only"

    def test_unknown_value_falls_back_to_the_default(self, monkeypatch):
        """An unrecognised mode is malformed, so it means the default."""
        monkeypatch.setenv("JACCPOT_TEST_MODE", "verbose")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _env.env_choice("JACCPOT_TEST_MODE", "full", self.MODES) == "full"

    def test_unset_gives_the_default(self, monkeypatch):
        """Unset is the default, without a warning -- nothing was malformed."""
        monkeypatch.delenv("JACCPOT_TEST_MODE", raising=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _env.env_choice("JACCPOT_TEST_MODE", "full", self.MODES) == "full"
        assert not caught

    def test_the_default_must_be_one_of_the_choices(self):
        """A default outside the choice set is a programming error, not user input.

        Raising here does not violate the module's "never raise" rule: that rule is
        about *environment* values, which are user input. This is a call-site bug and
        should surface immediately.
        """
        with pytest.raises(ValueError):
            _env.env_choice("JACCPOT_TEST_MODE", "nope", self.MODES)


class TestMalformedWarning:
    """Falling back silently is what lets a typo survive a whole session."""

    def test_a_malformed_value_warns_and_names_what_it_ignored(self, monkeypatch):
        """The message has to carry the variable and the value, or it cannot help."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", "ture")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _env.env_flag("JACCPOT_TEST_FLAG", True)
        assert len(caught) == 1
        text = str(caught[0].message)
        assert "JACCPOT_TEST_FLAG" in text
        assert "ture" in text

    def test_it_warns_once_per_variable(self, monkeypatch):
        """A knob read inside a hot path must not warn on every call."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", "ture")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                _env.env_flag("JACCPOT_TEST_FLAG", True)
        assert len(caught) == 1

    def test_a_different_variable_warns_separately(self, monkeypatch):
        """Per variable, not global -- otherwise the second typo is invisible."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", "ture")
        monkeypatch.setenv("JACCPOT_TEST_INT", "lots")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _env.env_flag("JACCPOT_TEST_FLAG", True)
            _env.env_int("JACCPOT_TEST_INT", 3)
        assert len(caught) == 2

    def test_well_formed_values_never_warn(self, monkeypatch):
        """The common path must stay silent, or the warning becomes noise."""
        monkeypatch.setenv("JACCPOT_TEST_FLAG", "0")
        monkeypatch.setenv("JACCPOT_TEST_INT", "8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _env.env_flag("JACCPOT_TEST_FLAG", True)
            _env.env_int("JACCPOT_TEST_INT", 3)
        assert not caught


def test_values_are_read_at_call_time(monkeypatch):
    """`_env`'s other stated rule, which nothing pinned either.

    A knob captured into a module-level constant cannot be changed by a user who sets
    the variable after importing ``jaccpot`` -- it silently does nothing, which is
    worse than not having the knob at all.
    """
    monkeypatch.delenv("JACCPOT_TEST_FLAG", raising=False)
    assert _env.env_flag("JACCPOT_TEST_FLAG", False) is False
    monkeypatch.setenv("JACCPOT_TEST_FLAG", "1")
    assert _env.env_flag("JACCPOT_TEST_FLAG", False) is True
