"""A named traversal capacity must not move the ones the caller did not name.

Regression guard for a *measurement* bug rather than a crash. On ``large_n_gpu``
at N=65536, passing an explicit ``DualTreeTraversalConfig`` whose
``max_pair_queue`` equalled the value already in force took per-step time from
1085 ms to ~3200 ms, and the instrumented fraction from 76% to 27%. Nothing
failed; the override simply also replaced ``process_block`` and the
interaction/neighbour caps with whatever the caller had typed to satisfy the
dataclass's three required fields. That produced a published-looking conclusion
("raising the queue costs 3x") which was wrong.

So the contract these tests freeze is: name one capacity, and the other three
stay at the values the runtime resolved for this particle count.

Deliberately CPU-only and device-independent: they assert on the *resolved*
config from ``_resolve_runtime_execution_overrides``, which is closed-form in
the particle count, so they cost no kernel compilation and cannot be flaky on a
shared GPU.
"""

from __future__ import annotations

import pytest
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    RuntimePolicyConfig,
    TraversalOverrides,
)
from jaccpot.runtime.fmm_overrides import normalize_traversal_config_request

TRAVERSAL_FIELDS = (
    "max_pair_queue",
    "process_block",
    "max_interactions_per_node",
    "max_neighbors_per_leaf",
)

# Both presets that carry their own traversal tuning, at an N on each side of the
# _GPU_LARGE_PARTICLE_THRESHOLD (65536) so the merge is exercised against two
# different resolved baselines.
PRESETS = ("fast", "large_n_gpu")
PARTICLE_COUNTS = (16384, 131072)


def _resolved(solver: FastMultipoleMethod, n: int) -> DualTreeTraversalConfig:
    """Return the traversal config the runtime would use at ``n`` particles."""

    overrides = solver._impl._resolve_runtime_execution_overrides(num_particles=n)
    config = overrides.traversal_config
    assert config is not None, "these presets always resolve a traversal config"
    return config


def _as_dict(config: DualTreeTraversalConfig) -> dict[str, int]:
    return {name: int(getattr(config, name)) for name in TRAVERSAL_FIELDS}


def _solver(preset: str, traversal_config: object = None) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset=preset,
        basis="real",
        advanced=_advanced(preset, traversal_config),
    )


def _advanced(preset: str, traversal_config: object) -> FMMAdvancedConfig:
    from dataclasses import replace

    from jaccpot.config import FMMPreset
    from jaccpot.solver import _default_advanced_for_preset

    base = _default_advanced_for_preset(FMMPreset(preset))
    return replace(
        base, runtime=replace(base.runtime, traversal_config=traversal_config)
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("n", PARTICLE_COUNTS)
@pytest.mark.parametrize("field", TRAVERSAL_FIELDS)
def test_single_field_override_leaves_the_others_at_preset_values(
    preset: str, n: int, field: str
) -> None:
    """Overriding one capacity changes exactly that capacity."""

    baseline = _as_dict(_resolved(_solver(preset), n))
    # Double it, so the request is unambiguously different from the baseline and
    # cannot be confused with "the clamp happened to land here anyway".
    wanted = int(baseline[field]) * 2

    overridden = _as_dict(
        _resolved(_solver(preset, TraversalOverrides(**{field: wanted})), n)
    )

    assert overridden[field] == wanted, (
        f"{preset}/N={n}: {field} override was not honoured "
        f"(asked {wanted}, resolved {overridden[field]})"
    )
    untouched = {k: v for k, v in overridden.items() if k != field}
    expected = {k: v for k, v in baseline.items() if k != field}
    assert untouched == expected, (
        f"{preset}/N={n}: overriding {field} also moved "
        f"{ {k: (expected[k], untouched[k]) for k in expected if expected[k] != untouched[k]} }"
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("n", PARTICLE_COUNTS)
def test_override_to_the_current_value_is_a_no_op(preset: str, n: int) -> None:
    """The exact case that misled the tranche-1 measurement.

    Naming a capacity and giving it the value already in force must resolve to
    the same four numbers -- not to a config where the other three reverted.
    """

    baseline = _as_dict(_resolved(_solver(preset), n))
    for field, value in baseline.items():
        echoed = _as_dict(
            _resolved(_solver(preset, TraversalOverrides(**{field: value})), n)
        )
        assert echoed == baseline, (
            f"{preset}/N={n}: echoing {field}={value} back changed the resolved "
            f"config from {baseline} to {echoed}"
        )


@pytest.mark.parametrize("preset", PRESETS)
def test_a_mapping_is_accepted_and_means_the_same_thing(preset: str) -> None:
    n = 16384
    baseline = _as_dict(_resolved(_solver(preset), n))
    wanted = int(baseline["max_pair_queue"]) * 2
    via_dataclass = _as_dict(
        _resolved(_solver(preset, TraversalOverrides(max_pair_queue=wanted)), n)
    )
    via_mapping = _as_dict(_resolved(_solver(preset, {"max_pair_queue": wanted}), n))
    assert via_mapping == via_dataclass


@pytest.mark.parametrize("preset", PRESETS)
def test_multiple_fields_can_be_overridden_together(preset: str) -> None:
    n = 16384
    baseline = _as_dict(_resolved(_solver(preset), n))
    request = TraversalOverrides(
        max_pair_queue=int(baseline["max_pair_queue"]) * 2,
        max_neighbors_per_leaf=int(baseline["max_neighbors_per_leaf"]) * 4,
    )
    resolved = _as_dict(_resolved(_solver(preset, request), n))
    assert resolved["max_pair_queue"] == int(baseline["max_pair_queue"]) * 2
    assert resolved["max_neighbors_per_leaf"] == (
        int(baseline["max_neighbors_per_leaf"]) * 4
    )
    assert resolved["process_block"] == baseline["process_block"]
    assert resolved["max_interactions_per_node"] == (
        baseline["max_interactions_per_node"]
    )


def test_empty_overrides_are_indistinguishable_from_none() -> None:
    n = 16384
    assert _as_dict(_resolved(_solver("large_n_gpu", TraversalOverrides()), n)) == (
        _as_dict(_resolved(_solver("large_n_gpu"), n))
    )


def test_full_config_still_replaces_everything_but_says_so() -> None:
    """The legacy path keeps its behaviour, and stops being silent about it."""

    full = DualTreeTraversalConfig(
        max_pair_queue=131072,
        process_block=512,
        max_interactions_per_node=8192,
        max_neighbors_per_leaf=4096,
    )
    with pytest.warns(UserWarning, match="replaces ALL FOUR"):
        solver = _solver("large_n_gpu", full)
    # Unchanged semantics: an explicit full config suppresses the policy's own
    # sizing, which is precisely why it warns.
    assert _as_dict(_resolved(solver, 16384)) == _as_dict(full)


def test_a_partial_override_does_not_suppress_the_preset_sizing() -> None:
    """The mechanism behind the 3x: ``_explicit_traversal_config`` gating.

    A full config sets that flag, which switches off
    ``_minimum_memory_streamed_gpu_traversal_seed`` and the memory clamps. A
    field-by-field override must not.
    """

    merged = _solver("large_n_gpu", TraversalOverrides(max_pair_queue=1 << 19))
    assert merged._impl._explicit_traversal_config is False
    with pytest.warns(UserWarning):
        replaced = _solver(
            "large_n_gpu",
            DualTreeTraversalConfig(
                max_pair_queue=1 << 19,
                process_block=256,
                max_interactions_per_node=4096,
                max_neighbors_per_leaf=1024,
            ),
        )
    assert replaced._impl._explicit_traversal_config is True


class TestNormalizer:
    """The request splitter, tested directly -- it is the whole contract."""

    def test_none_is_neither(self) -> None:
        assert normalize_traversal_config_request(None) == (None, {})

    def test_full_config_is_a_replacement(self) -> None:
        full = DualTreeTraversalConfig(
            max_pair_queue=1024,
            process_block=64,
            max_interactions_per_node=128,
            max_neighbors_per_leaf=256,
        )
        assert normalize_traversal_config_request(full) == (full, {})

    def test_overrides_become_a_field_merge(self) -> None:
        assert normalize_traversal_config_request(
            TraversalOverrides(process_block=8)
        ) == (None, {"process_block": 8})

    def test_all_none_overrides_are_neither(self) -> None:
        assert normalize_traversal_config_request(TraversalOverrides()) == (None, {})

    def test_unknown_mapping_key_is_rejected_by_name(self) -> None:
        # A typo'd key must not be a silently ignored tuning request.
        with pytest.raises(ValueError, match="max_pairqueue"):
            normalize_traversal_config_request({"max_pairqueue": 1024})

    def test_non_positive_capacity_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            normalize_traversal_config_request({"process_block": 0})
        with pytest.raises(ValueError, match=">= 1"):
            TraversalOverrides(max_pair_queue=0)

    def test_wrong_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="traversal_config must be"):
            normalize_traversal_config_request(4096)


def test_runtime_policy_config_accepts_the_override_type() -> None:
    """The documented spelling from the class docstring actually constructs."""

    cfg = FMMAdvancedConfig(
        runtime=RuntimePolicyConfig(
            traversal_config=TraversalOverrides(max_pair_queue=1 << 19)
        )
    )
    solver = FastMultipoleMethod(preset="large_n_gpu", basis="real", advanced=cfg)
    assert solver._impl._traversal_field_overrides == {"max_pair_queue": 1 << 19}
