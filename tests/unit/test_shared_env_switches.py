"""Two env switches that were duplicated or unreachable (audit F13, F38).

**F13** -- ``JACCPOT_FUSED_M2L_VJP`` selects the reverse-mode kernel for *both*
fused M2L lanes, and had two readers: byte-identical bodies in
``pallas/m2l_real_fused`` and ``pallas/m2l_complex_fused``. They had already
begun to drift -- the docstrings disagreed about what the switch does while the
code still agreed -- which is the cheap half. The expensive half is a fix applied
to one copy and not the other, on a knob that decides which VJP kernel runs.
There is now one definition in ``pallas/_flags``; the test below pins that the
two names are the *same object*, so they cannot drift again.

**F38** -- ``JACCPOT_M2L_DEGREE_BATCHED`` was read into a module-level constant
at import:

    _DEGREE_BATCHED = os.environ.get("JACCPOT_M2L_DEGREE_BATCHED", "0")...

which is the defect ``jaccpot._env`` exists to prevent, and says so in its own
docstring: a knob captured at import cannot be changed by anyone who sets the
variable after ``import jaccpot``, so it *silently does nothing* -- worse than
not having the knob. It is now read at call time through the sanctioned reader.

Both switches keep their defaults. Nothing here changes what the library
computes with no environment set.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.operators.m2l_real_rot_scale as rot_scale
from jaccpot.pallas import m2l_complex_fused, m2l_real_fused
from jaccpot.pallas._flags import fused_m2l_vjp_enabled


class TestTheFusedVjpSwitchHasOneReader:
    """F13: one shared switch, one definition."""

    def test_both_lanes_resolve_to_the_same_object(self):
        """Not merely equal behaviour -- the same function.

        Equality of results would still permit the two to drift apart later,
        which is what this row was about. Identity cannot.
        """
        assert (
            m2l_real_fused._fused_m2l_vjp_enabled
            is m2l_complex_fused._fused_m2l_vjp_enabled
        )
        assert m2l_real_fused._fused_m2l_vjp_enabled is fused_m2l_vjp_enabled

    def test_the_package_defines_it_once(self):
        """A second definition anywhere reopens the drift."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "jaccpot"
        hits = [
            path
            for path in root.rglob("*.py")
            if "def fused_m2l_vjp_enabled" in path.read_text()
            or "def _fused_m2l_vjp_enabled" in path.read_text()
        ]
        assert [p.name for p in hits] == ["_flags.py"], hits

    def test_it_is_on_by_default_and_switchable(self, monkeypatch):
        monkeypatch.delenv("JACCPOT_FUSED_M2L_VJP", raising=False)
        assert fused_m2l_vjp_enabled() is True
        monkeypatch.setenv("JACCPOT_FUSED_M2L_VJP", "0")
        assert fused_m2l_vjp_enabled() is False

    def test_a_typo_leaves_the_default_alone(self, monkeypatch):
        """`_env`'s house rule (audit 2.2): malformed means *the* default."""
        monkeypatch.setenv("JACCPOT_FUSED_M2L_VJP", "ture")
        assert fused_m2l_vjp_enabled() is True


class TestTheDegreeBatchedSwitchIsReadAtCallTime:
    """F38: the knob has to be reachable after import, or it is not a knob."""

    def test_it_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv("JACCPOT_M2L_DEGREE_BATCHED", raising=False)
        assert rot_scale._degree_batched() is False

    def test_setting_it_after_import_is_honoured(self, monkeypatch):
        """The defect itself.

        Before this change the value was captured into a module-level constant
        at import, so this assertion failed for every process that did not set
        the variable *before* ``import jaccpot`` -- and failed silently, by
        running the unrolled path while the operator believed the knob was on.
        """
        monkeypatch.setenv("JACCPOT_M2L_DEGREE_BATCHED", "1")
        assert rot_scale._degree_batched() is True

    def test_the_module_captures_no_env_value_at_import(self):
        """No module-level constant may hold this switch again.

        Pinned structurally rather than behaviourally, because the behavioural
        symptom only shows up in a process that sets the variable late -- which
        is exactly why it survived unnoticed.
        """
        import pathlib

        source = pathlib.Path(rot_scale.__file__).read_text()
        assert "os.environ" not in source
        assert "_DEGREE_BATCHED =" not in source


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
class TestTheTwoRotationPathsAgree:
    """What the now-reachable switch actually selects.

    The batched path is a different reduction order, so it is not bit-identical
    and CLAUDE.md's rule about reduction order is why it stays off by default.
    Measured here so the size of the difference is on record rather than
    assumed: it is at the float64 noise floor, which is what makes the knob an
    A/B choice rather than a numerics change.
    """

    ORDER = 4

    def _inputs(self):
        multipole = jax.random.normal(
            jax.random.PRNGKey(0), (rot_scale.sh_size(self.ORDER),), dtype=jnp.float64
        )
        return multipole, jnp.asarray([0.7, -1.3, 2.1], dtype=jnp.float64)

    def test_batched_matches_unrolled_at_the_noise_floor(self, monkeypatch):
        multipole, delta = self._inputs()

        monkeypatch.delenv("JACCPOT_M2L_DEGREE_BATCHED", raising=False)
        unrolled = np.asarray(
            rot_scale._rotate_multipole_to_z_single(multipole, delta, order=self.ORDER)
        )
        monkeypatch.setenv("JACCPOT_M2L_DEGREE_BATCHED", "1")
        batched = np.asarray(
            rot_scale._rotate_multipole_to_z_single(multipole, delta, order=self.ORDER)
        )

        assert np.max(np.abs(unrolled - batched)) < 1e-15
        assert not np.array_equal(unrolled, batched), (
            "bit-identical would mean the switch selects nothing -- if the two "
            "paths ever converge exactly, this test has stopped testing the "
            "switch and should be re-pointed"
        )


class TestRawEnvReadersOutsideRuntimeStayAccountedFor:
    """The set of raw ``os.environ`` readers outside ``runtime/`` is closed.

    STYLE_GUIDE section 8 records audit G.2's decision -- ``_env`` is the
    sanctioned reader for any layer, and ``runtime/`` is the only place that
    resolves an ``"auto"`` policy into a concrete choice -- and it names the two
    raw readers that remain, measured 2026-08-20.

    That count did not stay true on its own. By 2026-08-27 there were **three**:
    ``operators/m2l_real_rot_scale.py`` had acquired a module-level
    ``os.environ`` read after the rule was decided, and nothing noticed, because
    a prose count in a guide is not a check. This test is the check.
    """

    SANCTIONED = {
        # Structural: reads its own import-hook flag before `jaccpot` is
        # importable enough to use `_env`, so it cannot route through it.
        "jaccpot/_typecheck.py",
        # A genuine violation of the narrowed rule -- it resolves
        # JACCPOT_MUTUAL_M2L="auto" outside `runtime/`. Left deliberately:
        # STYLE_GUIDE section 8 says *not* to convert it to `env_choice`, because
        # that would turn its documented `ValueError` into a quiet default, and
        # that moving the resolution into `runtime/` is the mutual lane's own
        # decision to make.
        "jaccpot/mutual/farfield.py",
    }

    def test_no_new_raw_reader_has_appeared(self):
        """Anything new here should route through ``jaccpot._env`` instead."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "jaccpot"
        found = set()
        for path in root.rglob("*.py"):
            relative = path.relative_to(root.parent).as_posix()
            if relative.startswith("jaccpot/runtime/") or path.name == "_env.py":
                continue
            text = path.read_text()
            if "os.environ" in text or "os.getenv" in text:
                found.add(relative)
        assert found == self.SANCTIONED, (
            f"raw env readers outside runtime/ changed.\n"
            f"  unexpected: {sorted(found - self.SANCTIONED)}\n"
            f"  gone:       {sorted(self.SANCTIONED - found)}\n"
            "Route new reads through `jaccpot._env`, which reads at call time "
            "and falls back to the default on a malformed value."
        )
