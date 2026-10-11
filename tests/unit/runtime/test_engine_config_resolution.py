"""What ``FMMEngine.__init__`` resolves for the lane and memory options.

Audit **F09**. ``__init__`` is 217 lines over 63 parameters, and the row rates it
numerics-sensitive *"(resolution order)"*. Almost all of it now delegates to
``_resolve_*`` helpers; one inline block was left, and this module characterises
it **before** it moves, so the extraction has something to be verified against.

The order-sensitivity this block had was one option reading another:
``autotune_m2l_chunk`` read ``self.fail_fast`` two lines after it was set, and
was forced off by it. The M2L chunk autotune went in the 2026-10 cleanup (X4),
and with it that interaction and the class that pinned it.

What is left is validate-and-assign: three lane strings, two positive-integer
guards, and the ``_explicit_*`` flags that record whether the caller named a
value or accepted the default -- a distinction the policy layer later reads, so
"auto" and an explicit "auto" are not the same state.

Also here since the 2026-10 cleanup (Z-a): which basis a name resolves to
(``TestBasisResolution``, from ``test_basis_and_runner_hygiene.py``) and how the
facade's configuration reaches the engine (from ``test_solver_api.py``).
"""

from __future__ import annotations

import dataclasses
import warnings

import jax.numpy as jnp
import pytest

from jaccpot import FastMultipoleMethod, RealSHBasis
from jaccpot.config import NearFieldConfig, RuntimePolicyConfig, TreeConfig
from jaccpot.distributed import DistributedFMMConfig
from jaccpot.distributed.fmm import make_force_evaluator
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.solver import _resolve_basis_input

_POLICY_FIELDS = {f.name for f in dataclasses.fields(RuntimePolicyConfig)}
# NearFieldConfig drops the redundant `nearfield_` prefix, so this group needs a
# mapping where the policy group needed none. Engine-parameter name -> field name.
_NEARFIELD_FIELDS = {
    "nearfield_mode": "mode",
    "nearfield_edge_chunk_size": "edge_chunk_size",
    "precompute_nearfield_scatter_schedules": "precompute_scatter_schedules",
}


def _engine(**kwargs):
    """Build an engine, routing execution-policy knobs through their group.

    The seventeen policy knobs moved into ``RuntimePolicyConfig`` (audit F09),
    so this splits them out of ``kwargs`` and passes them as one object. Tests
    below keep naming them individually, which is what makes them readable --
    the grouping is a constructor detail, not something each test should have to
    spell out.

    Parameters
    ----------
    **kwargs
        Engine keywords; any belonging to ``RuntimePolicyConfig`` are collected
        into a ``runtime_policy=`` argument.

    Returns
    -------
    FMMEngine
        A fresh engine.
    """
    policy = {k: kwargs.pop(k) for k in list(kwargs) if k in _POLICY_FIELDS}
    nearfield = {
        _NEARFIELD_FIELDS[k]: kwargs.pop(k)
        for k in list(kwargs)
        if k in _NEARFIELD_FIELDS
    }
    base = dict(theta=0.6, working_dtype=jnp.float32)
    base.update(kwargs)
    if policy:
        base["runtime_policy"] = RuntimePolicyConfig(**policy)
    if nearfield:
        base["nearfield"] = NearFieldConfig(**nearfield)
    return FMMEngine(**base)


class TestLaneModeNormalisation:
    """Three strings, normalised then validated -- exercised at the resolver.

    **Two contracts disagree here, and the tests are at the layer whose contract
    matches the code.** ``__init__`` annotates these as
    ``Literal["auto", "baseline", "bucketed"]`` and friends, but the body does
    ``str(x).strip().lower()`` and then raises on an unrecognised value. Under
    ``JACCPOT_RUNTIME_TYPECHECK=1`` beartype enforces the ``Literal``, so
    ``"  Bucketed  "`` and ``"nonsense"`` never reach the body -- which makes
    both the normalisation *and* the ``ValueError`` unreachable for any caller
    who honours the annotation.

    So these go through ``_resolve_lane_modes``, which takes ``str`` because that
    is what it actually accepts. The ``__init__``-level behaviour is covered by
    the exact-literal cases elsewhere in this module. Which of the two contracts
    should win is a real question and is recorded in the audit rather than
    decided here: narrowing the resolver to ``Literal`` would delete live
    validation, and widening ``__init__`` to ``str`` would drop a check the
    public solver currently gets for free.
    """

    @pytest.mark.parametrize(
        ("kwarg", "supplied", "expected"),
        [
            ("nearfield_mode", "  Bucketed  ", "bucketed"),
            ("runtime_path", "LARGE_N", "large_n"),
            ("execution_backend", " Radix ", "radix"),
        ],
    )
    def test_case_and_whitespace_are_normalised(self, kwarg, supplied, expected):
        engine = _engine()
        defaults = dict(
            nearfield_mode="auto",
            runtime_path="auto",
            execution_backend="auto",
            nearfield_edge_chunk_size=engine.nearfield_edge_chunk_size,
            precompute_nearfield_scatter_schedules=(
                engine.precompute_nearfield_scatter_schedules
            ),
        )
        defaults[kwarg] = supplied
        engine._resolve_lane_modes(**defaults)
        assert getattr(engine, kwarg) == expected

    @pytest.mark.parametrize(
        ("kwarg", "message"),
        [
            ("nearfield_mode", "nearfield_mode must be"),
            ("runtime_path", "runtime_path must be"),
            ("execution_backend", "execution_backend must be"),
        ],
    )
    def test_an_unknown_value_is_rejected(self, kwarg, message):
        engine = _engine()
        defaults = dict(
            nearfield_mode="auto",
            runtime_path="auto",
            execution_backend="auto",
            nearfield_edge_chunk_size=engine.nearfield_edge_chunk_size,
            precompute_nearfield_scatter_schedules=(
                engine.precompute_nearfield_scatter_schedules
            ),
        )
        defaults[kwarg] = "nonsense"
        with pytest.raises(ValueError, match=message):
            engine._resolve_lane_modes(**defaults)


class TestExplicitVersusDefault:
    """ "auto" and an explicitly-supplied "auto" are different states.

    The policy layer reads these flags to decide whether it may override a
    setting, so collapsing the two would let it silently overrule a caller who
    asked for the default by name.
    """

    def test_default_nearfield_mode_is_not_explicit(self):
        assert _engine().nearfield_mode == "auto"
        assert _engine()._explicit_nearfield_mode is False

    def test_a_named_mode_is_explicit(self):
        assert _engine(nearfield_mode="baseline")._explicit_nearfield_mode is True

    def test_an_explicit_auto_is_still_not_explicit(self):
        """Pinned as observed behaviour, not as an endorsement.

        The flag is computed as ``!= "auto"``, so naming "auto" is
        indistinguishable from omitting it. Recorded here so the extraction
        cannot change it by accident; whether it *should* differ is a separate
        question for the policy layer.
        """
        assert _engine(nearfield_mode="auto")._explicit_nearfield_mode is False

    def test_the_memory_objective_flag_works_the_same_way(self):
        assert _engine().memory_objective == "balanced"
        assert _engine()._explicit_memory_objective is False
        assert _engine(memory_objective="throughput")._explicit_memory_objective is True


class TestPositiveIntegerGuards:
    """Two capacities that must be positive, and one that may be None."""

    @pytest.mark.parametrize("value", [0, -1])
    def test_nearfield_edge_chunk_size_must_be_positive(self, value):
        with pytest.raises(ValueError, match="nearfield_edge_chunk_size must be"):
            _engine(nearfield_edge_chunk_size=value)

    @pytest.mark.parametrize("value", [0, -8])
    def test_a_supplied_memory_budget_must_be_positive(self, value):
        with pytest.raises(ValueError, match="memory_budget_bytes must be"):
            _engine(memory_budget_bytes=value)

    def test_no_memory_budget_is_allowed(self):
        assert _engine(memory_budget_bytes=None).memory_budget_bytes is None

    def test_a_valid_budget_is_coerced_to_int(self):
        engine = _engine(memory_budget_bytes=2048)
        assert engine.memory_budget_bytes == 2048
        assert isinstance(engine.memory_budget_bytes, int)


class TestBooleanCoercion:
    """The plain flags, pinned so the extraction cannot drop a ``bool()``."""

    def test_the_cache_and_retention_flags_are_coerced(self):
        engine = _engine(
            enable_interaction_cache=True,
            retain_traversal_result=False,
            retain_interactions=True,
        )
        assert engine.enable_interaction_cache is True
        assert engine.retain_traversal_result is False
        assert engine.retain_interactions is True

    def test_the_memory_split_flag_keeps_none_distinct_from_false(self):
        """``None`` means "let the policy decide"; ``False`` means "do not"."""
        assert _engine().prepare_stage_memory_split_enabled is None
        assert (
            _engine(
                prepare_stage_memory_split_enabled=False
            ).prepare_stage_memory_split_enabled
            is False
        )
        assert (
            _engine(
                prepare_stage_memory_split_enabled=True
            ).prepare_stage_memory_split_enabled
            is True
        )


@pytest.mark.parametrize("with_object", [False, True])
def test_the_spherical_harmonic_family_defaults_to_the_real_basis(with_object):
    """``"solidfmm"`` without a basis object is REAL (cleanup 2026-10, D1).

    It used to be complex, so every engine built as ``expansion_basis="solidfmm"``
    ran the basis the public API had already stopped defaulting to. Since C the
    real basis is the only one: ``"complex"`` and a basis object not named
    ``"real"`` raise (``TestComplexWasRemoved`` in
    ``tests/unit/test_removed_options.py``).
    """
    from jaccpot.basis import RealSHBasis

    kwargs = {"expansion_basis": "solidfmm"}
    if with_object:
        kwargs["basis_impl"] = RealSHBasis()
    engine = FMMEngine(**kwargs)
    assert engine.expansion_basis == "solidfmm"
    assert isinstance(engine.basis_impl, RealSHBasis)
    assert engine._solidfmm_basis_mode() == "real"
    assert engine.m2l_impl == "rot_scale"


class TestBasisResolution:
    """Which basis a name resolves to, and what an unknown one is told.

    From ``test_basis_and_runner_hygiene.py`` (2026-10 cleanup, Z-a). ``basis="complex"``
    and ``basis="solidfmm"`` were the same path (measured bit-identical at
    N=2048/p=4/theta=0.5), so a user switching between them to cross-check was
    comparing a run against itself; since C the real basis is the only one and
    ``"solidfmm"`` names it. The migration errors for ``"complex"`` and
    ``"cartesian"`` are in ``tests/unit/test_removed_options.py``.
    """

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

    @pytest.mark.parametrize("basis", ("real", " Real "))
    def test_the_distributed_config_still_takes_the_real_basis(self, basis) -> None:
        assert callable(
            make_force_evaluator(DistributedFMMConfig(basis=basis), 1, 64, mesh=None)
        )

    def test_an_unknown_basis_is_not_offered_cartesian(self) -> None:
        with pytest.raises(ValueError, match="basis must be one of") as excinfo:
            FastMultipoleMethod(preset="fast", basis="spherical")
        assert "cartesian" not in str(excinfo.value)

    def test_an_unknown_basis_is_not_offered_complex(self) -> None:
        with pytest.raises(ValueError, match="basis must be one of") as excinfo:
            FastMultipoleMethod(preset="fast", basis="spherical")
        assert "complex" not in str(excinfo.value)

    def test_an_unknown_distributed_basis_raises(self) -> None:
        """Before C, any string but ``"real"`` silently ran the complex far field."""
        with pytest.raises(ValueError, match="must be 'real'"):
            make_force_evaluator(
                DistributedFMMConfig(basis="spherical"), 1, 64, mesh=None
            )


def test_an_unknown_tree_build_mode_raises():
    """An unknown ``TreeConfig.mode`` is refused with the generic list of modes.

    The live half of ``test_fmm.py::test_capacity_fixed_depth_tree_mode_is_removed``
    (2026-10 cleanup, Z-a); its ``fixed_depth`` half, which names that removal, is in
    ``tests/unit/test_removed_options.py``.
    """
    with pytest.raises(ValueError, match="tree_build_mode"):
        FMMEngine(
            tree=TreeConfig(mode="capacity_fixed_depth"),
            softening_kernel="plummer",
        )
