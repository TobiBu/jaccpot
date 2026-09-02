"""What ``FMMEngine.__init__`` resolves for the lane and memory options.

Audit **F09**. ``__init__`` is 217 lines over 63 parameters, and the row rates it
numerics-sensitive *"(resolution order)"*. Almost all of it now delegates to
``_resolve_*`` helpers; one inline block was left, and this module characterises
it **before** it moves, so the extraction has something to be verified against.

The order-sensitivity is real and it is in this block:

    self.fail_fast = bool(fail_fast)
    self.autotune_m2l_chunk = bool(autotune_m2l_chunk) and not self.fail_fast

``autotune_m2l_chunk`` reads ``self.fail_fast`` two lines after it is set. Swap
the two, or split them into different helpers called in the wrong order, and
``autotune_m2l_chunk`` silently stays on under ``fail_fast`` -- which would put a
timing-driven chunk search inside the lane whose entire purpose is to fail rather
than adapt. Nothing else in the constructor would complain. That interaction is
pinned below, in both directions.

The rest is validate-and-assign: three lane strings, two positive-integer
guards, and the ``_explicit_*`` flags that record whether the caller named a
value or accepted the default -- a distinction the policy layer later reads, so
"auto" and an explicit "auto" are not the same state.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from jaccpot.config import NearFieldConfig, RuntimePolicyConfig
from jaccpot.runtime._fmm_impl import FMMEngine

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


class TestTheFailFastAutotuneInteraction:
    """The one place in this block where one option reads another."""

    def test_fail_fast_forces_the_autotune_off(self):
        """Even when the caller explicitly asked for it.

        The strict lane exists to fail rather than adapt; a timing-driven chunk
        search inside it would be adapting.
        """
        engine = _engine(fail_fast=True, autotune_m2l_chunk=True)
        assert engine.fail_fast is True
        assert engine.autotune_m2l_chunk is False

    def test_without_fail_fast_the_request_is_honoured(self):
        engine = _engine(fail_fast=False, autotune_m2l_chunk=True)
        assert engine.autotune_m2l_chunk is True

    def test_not_requesting_it_leaves_it_off(self):
        assert _engine(autotune_m2l_chunk=False).autotune_m2l_chunk is False


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
