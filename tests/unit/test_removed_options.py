"""Removed options, accepted-and-ignored shims, and the errors that name a removal.

The 2026-10 cleanup removed the complex and Cartesian bases, the octree execution
backend, the grouped / class-major far field, the dense interaction buffers, the
fixed-depth tree, the M2L autotune and the finite-difference jerk. It kept a door
open behind each: a migration error naming the removal, or a field or keyword that
is still accepted and does nothing (``runtime_path``, ``FarFieldConfig.mode``'s
``"auto"`` / ``"pair_grouped"``, ``grouped_interactions=False``,
``use_dense_interactions``, ``execution_backend``, ``autotune_m2l_chunk``,
``jerk_fd_dt``, ``complex_rotation``, the locals template).

These tests pin those doors, and only those. They were collected here in Z-a
from ``test_solver_api.py``, ``tests/integration/test_fmm.py``,
``test_basis_and_runner_hygiene.py`` and ``test_legacy_kwargs.py``, so that Z-b,
which removes the shims and the protective errors, deletes this file whole. Where a
test mixed a shim with live behaviour, the live half stayed behind:

* the basis aliasing, the distributed config taking the real basis and the
  unknown-basis messages: ``tests/unit/runtime/test_engine_config_resolution.py``;
* ``tree_type`` flowing from the advanced config: the same file;
* an unknown tree build mode raising: the same file;
* the large-N lane gated on ``preset="large_n_gpu"`` alone:
  ``test_large_n_fast_path_policy.py``;
* the live legacy keywords warning and working: ``test_legacy_kwargs.py``.
"""

from __future__ import annotations

import dataclasses
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime._fmm_impl as fmm_impl_private
from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    FMMPreset,
    RealSHBasis,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.distributed import DistributedFMMConfig
from jaccpot.distributed.fmm import make_force_evaluator
from jaccpot.runtime._fmm_impl import FMMEngine


def _sample_problem(n: int = 64):
    key = jax.random.PRNGKey(11)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos,
        (n, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        key_mass,
        (n,),
        minval=0.5,
        maxval=1.5,
        dtype=jnp.float32,
    )
    return positions, masses


def _sample_velocities(n: int = 64):
    key = jax.random.PRNGKey(17)
    return jax.random.uniform(
        key,
        (n, 3),
        minval=-0.2,
        maxval=0.2,
        dtype=jnp.float32,
    )


#: The migration messages, matched literally (``re.escape``) by every spelling below.
_REMOVED = (
    "basis='cartesian' was removed in the 2026-10 cleanup (K); see "
    "docs/cleanup_2026-10.md. Use basis='real' (the default)."
)
_COMPLEX_REMOVED = (
    "basis='complex' was removed in the 2026-10 cleanup (C); see "
    "docs/cleanup_2026-10.md. Use basis='real' (the default) -- it agrees with the "
    "complex basis to round-off."
)


# ---------------------------------------------------------------------------
# The bases (K, C)
# ---------------------------------------------------------------------------


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

    def test_the_old_silencing_switch_is_inert(self, monkeypatch) -> None:
        """``JACCPOT_ALLOW_CARTESIAN_BASIS=1`` silenced the warning; it opens nothing."""
        monkeypatch.setenv("JACCPOT_ALLOW_CARTESIAN_BASIS", "1")
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FastMultipoleMethod(preset="fast", basis="cartesian")
        with pytest.raises(ValueError, match=re.escape(_REMOVED)):
            FMMEngine(expansion_basis="cartesian")


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

    @pytest.mark.parametrize("basis", ("complex", "solidfmm", " Complex ", "SOLIDFMM"))
    def test_the_distributed_config_basis_raises(self, basis: str) -> None:
        """Raised while the evaluator is built, before the mesh is touched.

        ``"solidfmm"`` was the distributed driver's name for its complex far field
        (``DistributedFMMConfig.basis`` ran complex for every string but ``"real"``).
        """
        with pytest.raises(ValueError, match=re.escape(_COMPLEX_REMOVED)):
            make_force_evaluator(DistributedFMMConfig(basis=basis), 1, 64, mesh=None)


# ---------------------------------------------------------------------------
# The octree execution backend and the execution_backend field (X2)
# ---------------------------------------------------------------------------


def test_execution_backend_flows_from_advanced_config():
    # The default (from test_tree_type_flows_from_advanced_config, whose live half
    # stays in tests/unit/runtime/test_engine_config_resolution.py).
    default = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="radix")),
    )
    assert default._impl.execution_backend == "auto"
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(execution_backend="radix"),
        ),
    )
    assert fmm._impl.tree_type == "radix"
    assert fmm._impl.execution_backend == "radix"


def test_prepare_state_records_resolved_execution_backend():
    positions, masses = _sample_problem(n=16)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(execution_backend="auto"),
        ),
    )

    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=4,
        max_order=2,
    )

    assert state.execution_backend == "radix"


def test_explicit_octree_execution_backend_raises_removal_error():
    """The octree execution backend is gone (cleanup 2026-10); naming it must raise.

    The config itself still constructs, so an old caller gets the removal message
    from the solver rather than a silent radix run.
    """
    runtime = RuntimePolicyConfig(execution_backend="octree")
    with pytest.raises(ValueError, match="octree execution backend was removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(runtime=runtime),
        )


def test_octree_tree_type_raises_removal_error():
    """``tree_type="octree"`` went with the octree execution backend; it must raise."""
    with pytest.raises(ValueError, match="octree execution backend was removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(tree=TreeConfig(tree_type="octree")),
        )


# ---------------------------------------------------------------------------
# The grouped / class-major far field and FarFieldConfig.mode (X3)
# ---------------------------------------------------------------------------


def test_grouped_interactions_true_raises_removal_error():
    """The grouped far field is gone (cleanup 2026-10, X3); asking for it must raise.

    ``FarFieldConfig`` still constructs with the field set, so an old caller gets the
    removal message from the solver rather than a silent flat far field.
    """
    farfield = FarFieldConfig(grouped_interactions=True)
    with pytest.raises(ValueError, match="grouped interactions were removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(farfield=farfield),
        )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_grouped_interactions_kwarg_true_raises_removal_error():
    """The legacy ``grouped_interactions=`` kwarg reaches the same refusal."""
    with pytest.raises(ValueError, match="grouped interactions were removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST, basis="solidfmm", grouped_interactions=True
        )


def test_grouped_interactions_setter_refuses_true_and_accepts_false():
    """The property reads ``False``; setting ``True`` raises, ``False`` is a no-op."""
    fmm = FastMultipoleMethod(preset=FMMPreset.FAST, basis="solidfmm")
    assert fmm.grouped_interactions is False
    with pytest.raises(ValueError, match="grouped interactions were removed"):
        fmm.grouped_interactions = True
    fmm.grouped_interactions = False
    assert fmm.grouped_interactions is False
    assert fmm.advanced.farfield.grouped_interactions is False


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_class_major_farfield_mode_raises_removal_error():
    """``mode="class_major"`` went with the grouped far field; both spellings raise."""
    farfield = FarFieldConfig(mode="class_major")
    with pytest.raises(ValueError, match="class-major far field was removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(farfield=farfield),
        )
    with pytest.raises(ValueError, match="class-major far field was removed"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST, basis="solidfmm", farfield_mode="class_major"
        )


@pytest.mark.parametrize("mode", ["auto", "pair_grouped"])
def test_surviving_farfield_modes_still_construct(mode):
    """``"auto"`` and ``"pair_grouped"`` stay accepted; both run the flat pair list."""
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode=mode, grouped_interactions=False)
        ),
    )
    assert fmm.farfield_mode == mode
    assert fmm.grouped_interactions is False


def test_balanced_preset_still_constructs_with_its_pair_grouped_mode():
    """The block-step lane builds ``preset="balanced"``, whose far field names
    ``mode="pair_grouped"`` (``nornax_adapter.py``, ``mutual/topology.py``)."""
    fmm = FastMultipoleMethod(preset="balanced")
    assert fmm.farfield_mode == "pair_grouped"


def test_removed_grouped_runtime_policy_fields_are_inert():
    """The two grouped-schedule policy fields still construct, and do nothing.

    They go in a later phase; until then an old config naming them must not raise,
    including the non-positive budget the engine used to reject.
    """
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                precompute_grouped_class_segments=True,
                grouped_schedule_budget_bytes=0,
            )
        ),
    )
    assert not hasattr(fmm._impl, "precompute_grouped_class_segments")
    assert not hasattr(fmm._impl, "grouped_schedule_budget_bytes")


# ---------------------------------------------------------------------------
# fixed_depth, autotune and runtime_path (X4)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fixed_depth_tree_mode_raises_removal_error():
    """The fixed-depth builder is gone (cleanup 2026-10, X4); naming it must raise.

    ``TreeConfig`` still constructs with the value, so an old config gets the
    removal message from the solver rather than the generic list of valid modes.
    Both spellings reach it: the config field and the legacy ``tree_build_mode=``.
    """
    with pytest.raises(ValueError, match="fixed_depth.*removed.*X4"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST,
            basis="solidfmm",
            advanced=FMMAdvancedConfig(tree=TreeConfig(mode="fixed_depth")),
        )
    with pytest.raises(ValueError, match="fixed_depth.*removed.*X4"):
        FastMultipoleMethod(
            preset=FMMPreset.FAST, basis="solidfmm", tree_build_mode="fixed_depth"
        )


def test_the_engine_refuses_fixed_depth_by_name():
    """``fixed_depth`` went in the 2026-10 cleanup (X4), with a message that names
    the removal rather than the generic list of valid modes.

    The ``fixed_depth`` half of ``test_fmm.py::test_capacity_fixed_depth_tree_mode_is_removed``;
    its generic-mode half is ``test_an_unknown_tree_build_mode_raises`` in
    ``tests/unit/runtime/test_engine_config_resolution.py``.
    """
    with pytest.raises(ValueError, match="fixed_depth.*removed.*X4"):
        FMMEngine(
            tree=TreeConfig(mode="fixed_depth"),
            softening_kernel="plummer",
        )


def _engine_state(engine):
    """Every engine attribute as text, as the constructor-state golden compares it."""
    return {name: repr(value) for name, value in vars(engine).items()}


def test_removed_autotune_m2l_chunk_is_inert():
    """``autotune_m2l_chunk=True`` still constructs, and changes nothing.

    The M2L chunk autotune went in the 2026-10 cleanup (X4); the policy field and
    the legacy kwarg stay so old configs, examples and notebooks construct. The
    whole resolved engine state must equal the default's, on the facade (both
    spellings) and on the engine.
    """
    reference = _engine_state(
        FastMultipoleMethod(preset=FMMPreset.FAST, basis="solidfmm")._impl
    )
    via_config = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(autotune_m2l_chunk=True),
        ),
    )
    via_kwarg = FastMultipoleMethod(
        preset=FMMPreset.FAST, basis="solidfmm", autotune_m2l_chunk=True
    )
    assert _engine_state(via_config._impl) == reference
    assert _engine_state(via_kwarg._impl) == reference
    assert not hasattr(via_config._impl, "autotune_m2l_chunk")
    assert _engine_state(
        fmm_impl_private.FMMEngine(
            runtime_policy=RuntimePolicyConfig(autotune_m2l_chunk=True)
        )
    ) == _engine_state(fmm_impl_private.FMMEngine())


@pytest.mark.parametrize("runtime_path", ["auto", "large_n"])
def test_runtime_path_is_accepted_and_no_longer_selects_the_lane(
    monkeypatch, runtime_path
):
    """``runtime_path`` is a recorded value since the 2026-10 cleanup (X4).

    Both values still construct, through the facade's legacy kwarg and the
    engine, and are stored. The large-N lane is gated on ``preset="large_n_gpu"``
    alone: an explicit ``"large_n"`` under another preset used to open it, and no
    longer does, while the preset opens it whatever was passed (its contract pins
    ``"large_n"``). The GPU predicate is stubbed, as in the lane's own tests.
    """
    from jaccpot.runtime import _large_n_pipeline

    monkeypatch.setattr(_large_n_pipeline.jax, "default_backend", lambda: "gpu")
    positions, masses = _sample_problem(n=64)

    fast = FastMultipoleMethod(
        preset=FMMPreset.FAST, basis="solidfmm", runtime_path=runtime_path
    )
    assert fast._impl.runtime_path == runtime_path
    assert not _large_n_pipeline.can_use_large_n_prepare_path(
        fast._impl,
        positions_arr=positions,
        masses_arr=masses,
        allow_stateful_cache=True,
    )

    large_n = FastMultipoleMethod(
        preset=FMMPreset.LARGE_N_GPU, basis="solidfmm", runtime_path=runtime_path
    )
    assert large_n._impl.runtime_path == "large_n"
    assert _large_n_pipeline.can_use_large_n_prepare_path(
        large_n._impl,
        positions_arr=positions,
        masses_arr=masses,
        allow_stateful_cache=True,
    )

    engine = fmm_impl_private.FMMEngine(runtime_path=runtime_path)
    assert engine.runtime_path == runtime_path


# ---------------------------------------------------------------------------
# The dense interaction buffers, jerk_fd_dt and the locals template (K)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_use_dense_interactions_true_raises_removal_error():
    """The dense interaction buffers went with the Cartesian basis (cleanup 2026-10, K).

    Both doors: the engine keyword, and the facade's legacy kwarg that reaches it.
    """
    with pytest.raises(ValueError, match="dense interaction buffers were removed"):
        fmm_impl_private.FMMEngine(use_dense_interactions=True)
    with pytest.raises(ValueError, match="dense interaction buffers were removed"):
        FastMultipoleMethod(preset=FMMPreset.FAST, use_dense_interactions=True)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("value", [None, False])
def test_use_dense_interactions_unset_or_false_is_accepted_and_ignored(value):
    """``None``/``False`` construct, as before; the resolved field is always ``False``.

    The engine attribute and the ``refresh_dual_split_dense_buffers_seconds``
    diagnostics key went with the dense build (Odisseo reads the key with ``.get``).
    """
    engine = fmm_impl_private.FMMEngine(use_dense_interactions=value)
    assert engine.config.traversal.use_dense_interactions is False
    assert not hasattr(engine, "use_dense_interactions")
    fmm = FastMultipoleMethod(preset=FMMPreset.FAST, use_dense_interactions=value)
    assert fmm._impl.config.traversal.use_dense_interactions is False
    diagnostics = fmm.get_runtime_diagnostics()
    assert "refresh_dual_split_dense_buffers_seconds" not in diagnostics


@pytest.mark.parametrize("jerk_mode", ["accurate", "fast_approx"])
def test_jerk_fd_dt_is_accepted_silently_and_unused(jerk_mode):
    """``jerk_fd_dt`` stays on all four jerk signatures, accepted and unused.

    It stepped the finite-difference fallback of the Cartesian basis, removed in the
    2026-10 cleanup (K). nornax's jaccpot adapter passes ``jerk_fd_dt=1e-3`` on every
    call and ``bench/ci_benchmark_guard.py`` passes it too, so it must neither warn
    nor be range-checked (a non-positive step used to raise on the fallback): any
    value gives the result of not passing it, bitwise.
    """
    import warnings

    positions, masses = _sample_problem(n=24)
    velocities = _sample_velocities(n=24)
    fmm = FastMultipoleMethod(preset=FMMPreset.FAST, basis="real")
    engine = fmm._impl
    kwargs = dict(leaf_size=8, max_order=3, jerk_mode=jerk_mode)
    state = fmm.prepare_state(positions, masses, leaf_size=8, max_order=3)

    calls = {
        "facade compute": lambda **kw: fmm.compute_accelerations_and_jerk(
            positions, masses, velocities, **kwargs, **kw
        ),
        "engine compute": lambda **kw: engine.compute_accelerations_and_jerk(
            positions, masses, velocities, **kwargs, **kw
        ),
        "facade evaluate": lambda **kw: fmm.evaluate_prepared_state_with_jerk(
            state, velocities, jerk_mode=jerk_mode, **kw
        ),
        "engine evaluate": lambda **kw: engine.evaluate_prepared_state_with_jerk(
            state, velocities, jerk_mode=jerk_mode, **kw
        ),
    }
    for name, call in calls.items():
        reference = call()
        for step in (1e-3, 0.0, -1.0):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # jax 0.11's own notice, raised when a GPU lowers a Pallas kernel
                # (the upward P2M here); not one of jaccpot's warnings
                warnings.filterwarnings(
                    "ignore",
                    message="The Pallas Triton backend is deprecated",
                    category=DeprecationWarning,
                )
                got = call(jerk_fd_dt=step)
            for want_arr, got_arr in zip(reference, got):
                assert np.array_equal(
                    np.asarray(want_arr), np.asarray(got_arr)
                ), f"{name}: jerk_fd_dt={step} changed the result"


def test_prepare_state_solidfmm_skips_prebuilt_locals_template():
    positions, masses = _sample_problem(n=64)
    fmm = FastMultipoleMethod(
        preset=FMMPreset.FAST,
        basis="solidfmm",
    )

    seen_templates = []
    original = fmm._impl._prepare_downward_with_artifacts

    def spy_prepare_downward(*args, **kwargs):
        seen_templates.append(kwargs.get("locals_template"))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fmm._impl, "_prepare_downward_with_artifacts", spy_prepare_downward)
        state = fmm.prepare_state(positions, masses, leaf_size=16, max_order=2)

    assert seen_templates == [None]
    acc = fmm.evaluate_prepared_state(state)
    assert acc.shape == positions.shape


# ---------------------------------------------------------------------------
# complex_rotation and the legacy shim keywords (C)
# ---------------------------------------------------------------------------


def test_solidfmm_basis_rejects_non_solidfmm_rotation():
    with pytest.raises(
        ValueError,
        match="complex_rotation must be 'solidfmm'",
    ):
        FMMEngine(
            expansion_basis="solidfmm",
            farfield=FarFieldConfig(rotation="cached"),
            softening_kernel="plummer",
        )


def test_the_legacy_shim_keywords_still_parse_with_a_deprecation_warning():
    """``complex_rotation``, ``grouped_interactions=False`` and ``farfield_mode``.

    The shim half of ``test_legacy_kwargs.py::test_accepts_legacy_expanse_kwargs_with_deprecation_warning``,
    which keeps the live legacy keywords. ``True`` for ``grouped_interactions`` raises
    (pinned above); ``False`` still parses.
    """
    positions, masses = _sample_problem(n=32)
    with pytest.warns(DeprecationWarning):
        fmm = FastMultipoleMethod(
            preset="fast",
            basis="real",
            complex_rotation="solidfmm",
            grouped_interactions=False,
            farfield_mode="pair_grouped",
            theta=0.6,
            softening=1e-3,
            working_dtype=jnp.float32,
        )
    acc = fmm.compute_accelerations(
        positions,
        masses,
        leaf_size=16,
        max_order=4,
    )
    assert acc.shape == positions.shape
