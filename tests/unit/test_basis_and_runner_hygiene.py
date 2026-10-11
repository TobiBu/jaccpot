"""Arguments that were silently inert.

Two ways the library said something untrue without failing:

* ``basis="complex"`` and ``basis="solidfmm"`` were the same path (measured
  bit-identical at N=2048/p=4/theta=0.5, max difference 0.0), so a user
  switching between them to cross-check was comparing a run against itself.
  The complex basis was removed in the 2026-10 cleanup (C): ``"complex"`` raises
  one migration error, and ``"solidfmm"`` (the family name) is the real basis.
* ``basis="cartesian"`` was ~1.8e-1 rel-L2 *independent of order* -- a
  divergent-series signature -- against 8.1e-5 for solidfmm, and nothing at the
  call site said so. It warned from then on, and was removed in the 2026-10
  cleanup (K): every spelling of it now raises one migration error.

The third, a benchmark that faked its output, is tested with the benchmark, in
``bench/tests/test_comparison_bench_exit_code.py``.
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
_COMPLEX_REMOVED = (
    "basis='complex' was removed in the 2026-10 cleanup (C); see "
    "docs/cleanup_2026-10.md. Use basis='real' (the default) -- it agrees with the "
    "complex basis to round-off."
)


class TestBasisAliasing:
    @pytest.mark.parametrize("spelling", ("solidfmm", "SolidFMM", " solidfmm "))
    def test_solidfmm_resolves_to_the_real_basis(self, spelling: str) -> None:
        """Not "two bases that agree" -- literally one resolution (C).

        ``"solidfmm"`` meant the complex basis until the 2026-10 cleanup (C); it
        names the spherical-harmonic family, whose only basis is now the real one,
        and Odisseo's non-integration branches and nornax's smoke test pass it.
        """
        real = _resolve_basis_input("real")
        solidfmm = _resolve_basis_input(spelling)
        assert solidfmm.public_name == real.public_name == "real"
        assert solidfmm.runtime_basis == real.runtime_basis == "solidfmm"
        assert type(solidfmm.basis_impl) is type(real.basis_impl) is RealSHBasis
        assert FastMultipoleMethod(preset="fast", basis=spelling).basis == "real"

    def test_the_legacy_expansion_basis_kwarg_resolves_to_real(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            fmm = FastMultipoleMethod(expansion_basis="solidfmm")
        assert fmm.basis == "real"

    @pytest.mark.parametrize("basis", ("real", "solidfmm"))
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

    @pytest.mark.parametrize("basis", ("real", " Real "))
    def test_the_distributed_config_still_takes_the_real_basis(self, basis) -> None:
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


class TestComplexWasRemoved:
    """Every way to name the complex basis raises the one migration error (C)."""

    @pytest.mark.parametrize("basis", ("complex", "Complex", " COMPLEX "))
    def test_the_facade_basis_argument_raises(self, basis: str) -> None:
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            FastMultipoleMethod(preset="fast", basis=basis)

    def test_the_legacy_expansion_basis_kwarg_raises(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
                FastMultipoleMethod(expansion_basis="complex")

    @pytest.mark.parametrize(
        "fields",
        ({"name": "complex"}, {"runtime_expansion_basis": "complex"}),
        ids=("named-complex", "complex-runtime-family"),
    )
    def test_a_basis_object_that_is_not_real_raises(self, fields) -> None:
        """What a ``ComplexSHBasis`` was: the family ``"solidfmm"``, named complex."""
        basis = dataclasses.replace(RealSHBasis(), **fields)
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            FastMultipoleMethod(preset="fast", basis=basis)

    def test_the_engine_expansion_basis_raises(self) -> None:
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            FMMEngine(expansion_basis="complex")

    def test_the_engine_rejects_a_basis_object_not_named_real(self) -> None:
        basis = dataclasses.replace(RealSHBasis(), name="complex")
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            FMMEngine(expansion_basis="solidfmm", basis_impl=basis)

    def test_the_class_import_raises_with_the_message(self) -> None:
        with pytest.raises(ImportError, match=re.escape(_COMPLEX_REMOVED)):
            from jaccpot import ComplexSHBasis  # noqa: F401

    def test_an_unknown_basis_is_not_offered_complex(self) -> None:
        with pytest.raises(ValueError, match="basis must be one of") as excinfo:
            FastMultipoleMethod(preset="fast", basis="spherical")
        assert "complex" not in str(excinfo.value)

    @pytest.mark.parametrize("basis", ("complex", "solidfmm", " Complex ", "SOLIDFMM"))
    def test_the_distributed_config_basis_raises(self, basis: str) -> None:
        """Raised while the evaluator is built, before the mesh is touched.

        ``"solidfmm"`` was the distributed driver's name for its complex far field
        (``DistributedFMMConfig.basis`` ran complex for every string but ``"real"``).
        """
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            make_force_evaluator(DistributedFMMConfig(basis=basis), 1, 64, mesh=None)

    def test_an_unknown_distributed_basis_raises(self) -> None:
        """Before C, any string but ``"real"`` silently ran the complex far field."""
        with pytest.raises(ValueError, match="must be 'real'"):
            make_force_evaluator(
                DistributedFMMConfig(basis="spherical"), 1, 64, mesh=None
            )
