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
at import, which is the defect ``jaccpot._env`` exists to prevent: a knob captured
at import cannot be changed by anyone who sets the variable after
``import jaccpot``, so it *silently does nothing*. It was then read at call time
through the sanctioned reader. The batched rotation it selected was removed in the
2026-10 cleanup (X5); the switch is still read at call time, and setting it now
raises, which the class below pins.

``JACCPOT_FUSED_M2L_VJP`` keeps its default. Nothing here changes what the library
computes with no environment set.
"""

from __future__ import annotations

import jax.numpy as jnp
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


class TestTheDegreeBatchedSwitchWasRemoved:
    """F38's switch selected a batched rotation the 2026-10 cleanup (X5) removed."""

    def test_setting_it_after_import_raises(self, monkeypatch):
        """Read at call time, and refused by name rather than ignored.

        Ignoring it would run the unrolled rotation under a switch that asked for
        another one -- the silent no-op F38 was about, in a new form.
        """
        multipole = jnp.zeros((rot_scale.sh_size(2),), dtype=jnp.float32)
        delta = jnp.asarray([0.7, -1.3, 2.1], dtype=jnp.float32)
        monkeypatch.delenv("JACCPOT_M2L_DEGREE_BATCHED", raising=False)
        rot_scale._rotate_multipole_to_z_single(multipole, delta, order=2)
        monkeypatch.setenv("JACCPOT_M2L_DEGREE_BATCHED", "1")
        with pytest.raises(ValueError, match="removed in the 2026-10 cleanup"):
            rot_scale._rotate_multipole_to_z_single(multipole, delta, order=2)
        with pytest.raises(ValueError, match="JACCPOT_M2L_DEGREE_BATCHED"):
            rot_scale._rotate_local_from_z_single(multipole, delta, order=2)

    def test_the_module_captures_no_env_value_at_import(self):
        """No module-level constant may hold an env switch again (F38's defect)."""
        import pathlib

        source = pathlib.Path(rot_scale.__file__).read_text()
        assert "os.environ" not in source
        assert "_DEGREE_BATCHED =" not in source


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
