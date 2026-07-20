"""Guard the stable public API surface of :mod:`jaccpot`.

This test freezes the contract that downstream consumers (e.g. ODISSEO's
``jaccpot_coupling.py`` and yggdrax) depend on: the top-level ``__all__``, the
kind of each exported symbol, the :class:`FMMPreset` members, and the
user-facing :class:`FastMultipoleMethod` facade (its keyword-only constructor
surface + public method set).

It exists to make the ``_fmm_impl`` refactor observable: if an internal change
leaks into the public surface, this test goes red. Intentional public-API
changes (e.g. dropping ``legacy_kwargs`` in the constructor-redesign phase) are
made by updating the frozen sets below in the *same* change, which forces a
conscious review of the contract.
"""

from __future__ import annotations

import enum
import inspect

import jaccpot

# The frozen top-level export set. Changing this set is a public-API change.
EXPECTED_ALL = {
    "ComplexSHBasis",
    "FMMAdvancedConfig",
    "FMMPreset",
    "FarFieldConfig",
    "FastMultipoleMethod",
    "MemoryObjective",
    "NearFieldConfig",
    "OdisseoFMMCoupler",
    "RealSHBasis",
    "RuntimePolicyConfig",
    "TreeConfig",
    "differentiable_gravitational_acceleration",
}

# Frozen preset members (name -> value). ODISSEO selects presets by these.
EXPECTED_PRESETS = {
    "FAST": "fast",
    "BALANCED": "balanced",
    "ACCURATE": "accurate",
    "LARGE_N_GPU": "large_n_gpu",
}

# Frozen keyword-only constructor surface of the public facade.
EXPECTED_FMM_INIT_KWARGS = {
    "preset",
    "basis",
    "m2l_impl",
    "adaptive_order",
    "p_gears",
    "use_pallas",
    "reuse_topology",
    "rebuild_every",
    "mac_force_scale_mode",
    "adaptive_error_model",
    "adaptive_eps",
    "dehnen_geometry_mode",
    "theta",
    "G",
    "softening",
    "precision",
    "working_dtype",
    "advanced",
}

# Frozen public (non-underscore) method set of the facade.
EXPECTED_FMM_PUBLIC_METHODS = {
    "clear_prepared_state_cache",
    "clear_runtime_caches",
    "compute_accelerations",
    "compute_accelerations_and_jerk",
    "compute_accelerations_with_time_derivatives",
    "evaluate_prepared_state",
    "evaluate_prepared_state_with_jerk",
    "evaluate_prepared_state_with_time_derivatives",
    "export_m2l_autotune_cache",
    "get_runtime_diagnostics",
    "import_m2l_autotune_cache",
    "load_m2l_autotune_cache",
    "prepare_state",
    "prepare_upward_sweep",
    "rebuild_topology_in_place",
    "refresh_prepared_state",
    "save_m2l_autotune_cache",
    "strict_fused_prepared_eval_fn",
    "strict_prepare_refresh_and_evaluate",
    "strict_run_segmented",
    "strict_run_v2",
    "update_multipoles_only",
}


def test_all_exports_are_frozen() -> None:
    assert set(jaccpot.__all__) == EXPECTED_ALL


def test_every_export_is_importable_from_top_level() -> None:
    for name in EXPECTED_ALL:
        assert hasattr(jaccpot, name), f"jaccpot.{name} is not importable"


def test_export_kinds() -> None:
    # Classes / dataclasses.
    for name in (
        "ComplexSHBasis",
        "FMMAdvancedConfig",
        "FarFieldConfig",
        "FastMultipoleMethod",
        "NearFieldConfig",
        "OdisseoFMMCoupler",
        "RealSHBasis",
        "RuntimePolicyConfig",
        "TreeConfig",
    ):
        assert inspect.isclass(getattr(jaccpot, name)), f"{name} should be a class"
    # Enum.
    assert issubclass(jaccpot.FMMPreset, enum.Enum)
    # Callable (function).
    assert callable(jaccpot.differentiable_gravitational_acceleration)


def test_public_class_is_the_solver_facade() -> None:
    # The public FastMultipoleMethod must be the facade in jaccpot.solver,
    # never the runtime engine implementation.
    assert jaccpot.FastMultipoleMethod.__module__ == "jaccpot.solver"


def test_preset_members_are_frozen() -> None:
    actual = {m.name: m.value for m in jaccpot.FMMPreset}
    assert actual == EXPECTED_PRESETS


def test_facade_constructor_surface_is_frozen() -> None:
    sig = inspect.signature(jaccpot.FastMultipoleMethod.__init__)
    kwargs = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert kwargs == EXPECTED_FMM_INIT_KWARGS


def test_facade_public_method_set_is_frozen() -> None:
    methods = {
        name
        for name, _ in inspect.getmembers(
            jaccpot.FastMultipoleMethod, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert methods == EXPECTED_FMM_PUBLIC_METHODS
