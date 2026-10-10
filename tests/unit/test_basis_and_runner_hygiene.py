"""Arguments that were silently inert, and a benchmark that faked its output.

Three separate ways the library or its benchmarks said something untrue without
failing:

* ``basis="complex"`` and ``basis="solidfmm"`` are the same path (measured
  bit-identical at N=2048/p=4/theta=0.5, max difference 0.0), so a user
  switching between them to cross-check was comparing a run against itself.
* ``basis="cartesian"`` was ~1.8e-1 rel-L2 *independent of order* -- a
  divergent-series signature -- against 8.1e-5 for solidfmm, and nothing at the
  call site said so. It warned from then on, and was removed in the 2026-10
  cleanup (K): every spelling of it now raises one migration error.
* ``bench/bench_jaxfmm_paper_compare.py`` wrote ``status=error`` rows when
  jaxFMM could not be imported, so a dead comparison arm produced a CSV that
  looked like data. It stayed dead for months.
"""

from __future__ import annotations

import dataclasses
import re
import warnings

import pytest

from jaccpot import FastMultipoleMethod, RealSHBasis
from jaccpot.distributed import DistributedFMMConfig
from jaccpot.distributed.fmm import make_force_evaluator
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.solver import _resolve_basis_input

#: The migration message, matched literally (``re.escape``) by every spelling below.
_REMOVED = (
    "basis='cartesian' was removed in the 2026-10 cleanup (K); see "
    "docs/cleanup_2026-10.md. Use basis='real' (the default)."
)


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


class TestCartesianWasRemoved:
    """Every way to name the Cartesian basis raises the one migration error.

    Replaces the four ``TestCartesianIsExperimental`` tests (2026-10 cleanup, K):
    the basis they warned about is gone, so the warning, its wording, its silencing
    switch and "it still works" became this refusal, spelling by spelling.
    """

    @pytest.mark.parametrize("basis", ("cartesian", "Cartesian", " CARTESIAN "))
    def test_the_facade_basis_argument_raises(self, basis: str) -> None:
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FastMultipoleMethod(preset="fast", basis=basis)

    def test_the_legacy_expansion_basis_kwarg_raises(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match=re.escape(_REMOVED)):
                FastMultipoleMethod(expansion_basis="cartesian")

    def test_a_basis_object_that_names_it_raises(self) -> None:
        basis = dataclasses.replace(RealSHBasis(), runtime_expansion_basis="cartesian")
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FastMultipoleMethod(preset="fast", basis=basis)

    def test_the_engine_expansion_basis_raises(self) -> None:
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FMMEngine(expansion_basis="cartesian")

    def test_the_distributed_config_basis_raises(self) -> None:
        """Raised while the evaluator is built, before the mesh is touched.

        ``DistributedFMMConfig.basis`` was never validated: ``"cartesian"`` failed
        the ``== "real"`` test and silently ran the complex far field.
        """
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            make_force_evaluator(
                DistributedFMMConfig(basis="cartesian"), 1, 64, mesh=None
            )

    @pytest.mark.parametrize("basis", ("real", "solidfmm"))
    def test_the_distributed_config_still_takes_its_two_bases(self, basis) -> None:
        assert callable(
            make_force_evaluator(DistributedFMMConfig(basis=basis), 1, 64, mesh=None)
        )

    def test_the_old_silencing_switch_is_inert(self, monkeypatch) -> None:
        """``JACCPOT_ALLOW_CARTESIAN_BASIS=1`` silenced the warning; it opens nothing."""
        monkeypatch.setenv("JACCPOT_ALLOW_CARTESIAN_BASIS", "1")
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FastMultipoleMethod(preset="fast", basis="cartesian")
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FMMEngine(expansion_basis="cartesian")

    def test_an_unknown_basis_is_not_offered_cartesian(self) -> None:
        with pytest.raises(ValueError, match="basis must be one of") as excinfo:
            FastMultipoleMethod(preset="fast", basis="spherical")
        assert "cartesian" not in str(excinfo.value)


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
