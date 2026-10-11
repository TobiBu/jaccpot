"""A comparison benchmark that faked its output.

``bench/bench_jaxfmm_paper_compare.py`` wrote ``status=error`` rows when jaxFMM
could not be imported, so a dead comparison arm produced a CSV that looked like
data. It stayed dead for months. An explicitly requested runner that cannot run
is now a failed invocation.

Moved here from ``tests/unit/test_basis_and_runner_hygiene.py`` in the 2026-10
cleanup (Z-a): it tests a benchmark script, not the library.
"""

from __future__ import annotations

import pytest


class TestComparisonBenchExitCode:
    """An explicitly requested runner that cannot run is a failed invocation.

    Tests ``resolve_runner`` rather than ``main``: the guard is the contract, and
    running ``main`` would run the whole benchmark.
    """

    @staticmethod
    def _resolve(requested, have_jaxfmm):
        from bench.bench_jaxfmm_paper_compare import resolve_runner

        return resolve_runner(requested, have_jaxfmm=have_jaxfmm)

    @pytest.mark.parametrize("runner", ("jaxfmm", "both"))
    def test_explicit_unavailable_runner_exits_non_zero(self, runner: str) -> None:
        with pytest.raises(SystemExit) as excinfo:
            self._resolve(runner, have_jaxfmm=False)
        message = str(excinfo.value)
        assert "not importable" in message
        assert "status=error" in message
        # A SystemExit carrying a string is a non-zero exit.
        assert excinfo.value.code not in (0, None)

    def test_the_unset_default_tolerates_a_missing_arm(self) -> None:
        """Not naming a runner keeps the old behaviour, so CI without jaxFMM runs."""

        assert self._resolve(None, have_jaxfmm=False) == "both"

    def test_jaccpot_only_never_needs_jaxfmm(self) -> None:
        assert self._resolve("jaccpot", have_jaxfmm=False) == "jaccpot"

    @pytest.mark.parametrize("requested", (None, "jaxfmm", "jaccpot", "both"))
    def test_available_jaxfmm_is_always_fine(self, requested) -> None:
        expected = "both" if requested is None else requested
        assert self._resolve(requested, have_jaxfmm=True) == expected

    def test_the_flag_default_is_the_tolerant_sentinel(self) -> None:
        """If the default went back to "both", the explicit check would misfire."""

        import bench.bench_jaxfmm_paper_compare as module

        parser_source = __import__("inspect").getsource(module._parse_args)
        assert '"--runner"' in parser_source
        assert "default=None" in parser_source

    def test_importing_the_bench_does_not_parse_sys_argv(self) -> None:
        """Importing a benchmark must not read the importer's command line."""

        import bench.bench_jaxfmm_paper_compare as module

        assert module.ARGS is None
