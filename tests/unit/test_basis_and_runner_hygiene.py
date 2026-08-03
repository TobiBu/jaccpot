"""Arguments that were silently inert, and a benchmark that faked its output.

Three separate ways the library or its benchmarks said something untrue without
failing:

* ``basis="complex"`` and ``basis="solidfmm"`` are the same path (measured
  bit-identical at N=2048/p=4/theta=0.5, max difference 0.0), so a user
  switching between them to cross-check was comparing a run against itself.
* ``basis="cartesian"`` is ~1.8e-1 rel-L2 *independent of order* -- a
  divergent-series signature -- against 8.1e-5 for solidfmm, and nothing at the
  call site said so.
* ``bench/bench_jaxfmm_paper_compare.py`` wrote ``status=error`` rows when
  jaxFMM could not be imported, so a dead comparison arm produced a CSV that
  looked like data. It stayed dead for months.
"""

from __future__ import annotations

import warnings

import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.solver import _resolve_basis_input


class TestBasisAliasing:
    def test_complex_and_solidfmm_resolve_identically(self) -> None:
        """Not "two bases that agree" -- literally one resolution."""

        a = _resolve_basis_input("complex")
        b = _resolve_basis_input("solidfmm")
        assert a.public_name == b.public_name
        assert a.runtime_basis == b.runtime_basis
        assert type(a.basis_impl) is type(b.basis_impl)

    def test_real_is_a_genuinely_different_implementation(self) -> None:
        """The real cross-check, so the alias test above is not vacuous."""

        real = _resolve_basis_input("real")
        solidfmm = _resolve_basis_input("solidfmm")
        assert real.runtime_basis == solidfmm.runtime_basis
        assert type(real.basis_impl) is not type(solidfmm.basis_impl)

    @pytest.mark.parametrize("basis", ("real", "solidfmm", "complex"))
    def test_the_supported_bases_do_not_warn(self, basis: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            FastMultipoleMethod(preset="fast", basis=basis)


class TestCartesianIsExperimental:
    def test_it_warns_by_name_and_says_why(self) -> None:
        with pytest.warns(UserWarning, match="EXPERIMENTAL"):
            FastMultipoleMethod(preset="fast", basis="cartesian")

    def test_the_warning_names_the_order_independence(self) -> None:
        """The specific fact a user needs: raising max_order will not help."""

        with pytest.warns(UserWarning) as record:
            FastMultipoleMethod(preset="fast", basis="cartesian")
        message = str(record[0].message)
        assert "independent of expansion order" in message
        assert "solidfmm" in message

    def test_it_can_be_silenced_deliberately(self, monkeypatch) -> None:
        monkeypatch.setenv("JACCPOT_ALLOW_CARTESIAN_BASIS", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            FastMultipoleMethod(preset="fast", basis="cartesian")

    def test_it_still_works(self) -> None:
        """Warned, not removed: it remains a usable cross-check of structure."""

        with pytest.warns(UserWarning):
            solver = FastMultipoleMethod(preset="fast", basis="cartesian")
        assert solver is not None


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
