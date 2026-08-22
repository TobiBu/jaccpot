"""Preset-first FMM solver facade for Jaccpot."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from ._env import env_flag
from .basis import BasisInterface, ComplexSHBasis, RealSHBasis
from .config import (
    Basis,
    FMMAdvancedConfig,
    FMMPreset,
    GradConfig,
)
from .runtime._large_n_types import LargeNPreparedState
from .runtime.fmm import FMMEngine as _RuntimeFMM
from .runtime.fmm import FMMPreparedState
from .runtime.fmm_constants import _LARGE_N_GPU_UPWARD_LEAF_BATCH_SIZE


def _default_advanced_for_preset(preset: FMMPreset) -> FMMAdvancedConfig:
    """Return default advanced overrides for a high-level preset.

    Parameters
    ----------
    preset : FMMPreset
        Normalized preset. Any value other than the three named below falls
        through to ``ACCURATE``, which is therefore the default rather than an
        error case.

    Returns
    -------
    FMMAdvancedConfig
        A fresh config; the caller may pass its own ``advanced=`` instead, in
        which case this is never called.
    """
    if preset is FMMPreset.FAST:
        return FMMAdvancedConfig()
    if preset is FMMPreset.LARGE_N_GPU:
        cfg = FMMAdvancedConfig()
        return replace(
            cfg,
            tree=replace(
                cfg.tree,
                tree_type="radix",
                mode="lbvh",
                leaf_target=64,
                refine_local=False,
                max_refine_levels=0,
                aspect_threshold=16.0,
            ),
            farfield=replace(
                cfg.farfield,
                mode="auto",
                grouped_interactions=False,
                rotation="solidfmm",
                m2l_chunk_size=None,
                l2l_chunk_size=None,
                streamed_far_pairs=True,
                mixed_order=False,
                mixed_order_min_order=None,
            ),
            nearfield=replace(
                cfg.nearfield,
                mode="bucketed",
                edge_chunk_size=256,
                precompute_scatter_schedules=False,
            ),
            runtime=replace(
                cfg.runtime,
                host_refine_mode="off",
                jit_tree=True,
                jit_traversal=True,
                memory_objective="minimum_memory",
                traversal_config=None,
                max_pair_queue=None,
                pair_process_block=None,
                enable_interaction_cache=False,
                retain_traversal_result=False,
                retain_interactions=False,
                autotune_m2l_chunk=True,
                precompute_grouped_class_segments=False,
                grouped_schedule_budget_bytes=8 * 1024 * 1024,
                upward_leaf_batch_size=_LARGE_N_GPU_UPWARD_LEAF_BATCH_SIZE,
            ),
            mac_type="dehnen",
            dehnen_radius_scale=1.0,
        )
    if preset is FMMPreset.BALANCED:
        cfg = FMMAdvancedConfig()
        return replace(
            cfg,
            farfield=replace(cfg.farfield, mode="pair_grouped"),
            nearfield=replace(cfg.nearfield, mode="bucketed", edge_chunk_size=512),
        )
    # ACCURATE
    cfg = FMMAdvancedConfig()
    return replace(
        cfg,
        tree=replace(cfg.tree, mode="lbvh", refine_local=True, max_refine_levels=1),
        farfield=replace(cfg.farfield, mode="pair_grouped"),
        # "auto", not "baseline". This preset's accuracy comes from the tree
        # (refine_local + an extra refinement level) and from the expansion, not
        # from how the near field is traversed: baseline and bucketed visit the
        # same leaf pairs and agree to one ulp at fp64 (4.2e-16 rel-L2 at
        # N=8192). Pinning "baseline" only forced the per-pair reference scan,
        # which on an A100 cost 507x at N=4096 and made this preset 139x slower
        # than large_n_gpu. On a CPU "auto" still resolves to baseline here, so
        # this changes the GPU answer only.
        nearfield=replace(cfg.nearfield, mode="auto"),
        runtime=replace(cfg.runtime, host_refine_mode="on"),
    )


def _normalize_preset(preset: Union[FMMPreset, str]) -> FMMPreset:
    """Normalize user preset input to :class:`FMMPreset`.

    Parameters
    ----------
    preset : Union[FMMPreset, str]
        An enum member, or its value as a string. Strings are stripped and
        lowercased, so ``" Fast "`` is accepted.

    Returns
    -------
    FMMPreset
        The matching member. An unrecognized string raises ``ValueError`` from
        the enum lookup rather than from an explicit check here.
    """
    if isinstance(preset, FMMPreset):
        return preset
    return FMMPreset(str(preset).strip().lower())


class _BasisResolution(NamedTuple):
    """Resolved basis routing for public API and runtime backend.

    Attributes
    ----------
    public_name : str
        What ``FastMultipoleMethod.basis`` reports back to the user. Not always
        the string they passed: ``"solidfmm"`` and ``"complex"`` both report
        ``"complex"``.
    runtime_basis : Basis
        What the engine is constructed with. This is the name that selects the
        expansion algebra.
    basis_impl : Optional[BasisInterface]
        The basis object, when one exists. ``None`` for bases the runtime
        implements internally rather than through the interface.
    """

    public_name: str
    runtime_basis: Basis
    basis_impl: Optional[BasisInterface]


def _warn_cartesian_basis_is_experimental() -> None:
    """Say that ``basis="cartesian"`` is not fit for quantitative work.

    Its relative L2 force error is ~1.8e-1 *independent of expansion order*,
    which is a divergent-series signature rather than truncation: raising the
    order does not improve it. solidfmm is 8.1e-5 on the same configuration,
    ~2000x better, and the characterization anchor carries a 0.35 tolerance for
    cartesian alone to accommodate this.

    Warned rather than removed: it is the only Cartesian-multipole implementation
    here and is useful for cross-checking the harmonic paths' *structure*. But an
    error that does not fall with order is not something anyone should select by
    accident, and until this tranche nothing said so at the call site.
    """

    if env_flag("JACCPOT_ALLOW_CARTESIAN_BASIS", False):
        return
    warnings.warn(
        "basis='cartesian' is EXPERIMENTAL and unsuitable for quantitative work: "
        "its relative L2 force error is ~1.8e-1 independent of expansion order "
        "(a divergent-series signature, not truncation), against 8.1e-5 for "
        "basis='solidfmm' on the same configuration. Raising max_order will not "
        "improve it. Use basis='real' (default) or 'solidfmm'. Set "
        "JACCPOT_ALLOW_CARTESIAN_BASIS=1 to silence this.",
        UserWarning,
        stacklevel=4,
    )


def _resolve_basis_input(basis: Union[Basis, BasisInterface, str]) -> _BasisResolution:
    """Normalize basis string/object to runtime expansion basis + metadata.

    ``"complex"`` is an **alias** for ``"solidfmm"``, not a third basis: both
    return the same runtime basis and the same ``ComplexSHBasis`` implementation,
    so they produce bit-identical forces (measured at N=2048/p=4/theta=0.5, max
    difference exactly 0.0). Bit-equality here is structural rather than a
    numerical coincidence, and the test that pins it is structural too:
    ``tests/unit/test_basis_and_runner_hygiene.py`` asserts the two resolve to the
    same object, which is the only way to pin an alias.

    The genuine independent-basis cross-check is ``real`` against ``solidfmm``.
    The 4.5e-13 figure is a **float64** result at that same N=2048/p=4/theta=0.5
    configuration; it is not what the suite enforces, and it does not carry to
    fp32. What the suite asserts is relative L2 below 1e-6 for the fp64
    acceleration-and-derivative-tower parity
    (``tests/integration/test_real_basis_runtime.py::test_real_basis_acceleration_derivatives_match_complex``,
    N=128) and below 3e-2 for the fp32 tracking test at N=96 -- the latter being
    considerably slacker than fp32 basis-change round-off should require, which is
    a tolerance worth re-measuring rather than a difference worth expecting.

    Parameters
    ----------
    basis : Union[Basis, BasisInterface, str]
        A basis name (stripped and lowercased) or an already-constructed
        :class:`~jaccpot.BasisInterface`. Passing an object routes it straight
        through as ``basis_impl``.

    Returns
    -------
    _BasisResolution
        Public name, runtime basis and implementation object.

    Raises
    ------
    ValueError
        If ``basis`` is a string that names no known basis.
    TypeError
        If ``basis`` is neither a string nor a :class:`BasisInterface`.
    """
    if isinstance(basis, str):
        basis_norm = basis.strip().lower()
        if basis_norm in ("solidfmm", "complex"):
            # Deliberately one branch for both spellings -- see the docstring.
            return _BasisResolution(
                public_name="complex",
                runtime_basis="solidfmm",
                basis_impl=ComplexSHBasis(),
            )
        if basis_norm == "real":
            return _BasisResolution(
                public_name="real",
                runtime_basis="solidfmm",
                basis_impl=RealSHBasis(),
            )
        if basis_norm == "cartesian":
            _warn_cartesian_basis_is_experimental()
            return _BasisResolution(
                public_name="cartesian",
                runtime_basis="cartesian",
                basis_impl=None,
            )
        raise ValueError(
            "basis must be one of 'cartesian', 'solidfmm', 'complex', or 'real', "
            f"got '{basis}'"
        )

    if isinstance(basis, BasisInterface):
        runtime_basis = str(basis.runtime_expansion_basis).strip().lower()
        if runtime_basis == "complex":
            runtime_basis = "solidfmm"
        if runtime_basis not in ("cartesian", "solidfmm"):
            raise ValueError(
                "basis.runtime_expansion_basis must be 'cartesian', "
                "'solidfmm', or 'complex'"
            )
        return _BasisResolution(
            public_name=str(basis.name),
            runtime_basis=runtime_basis,  # type: ignore[arg-type]
            basis_impl=basis,
        )

    raise TypeError("basis must be a string or BasisInterface implementation")


def _pop_legacy_common_overrides(
    *,
    basis: Union[Basis, BasisInterface, str],
    theta: float,
    G: float,
    softening: float,
    working_dtype: Optional[DTypeLike],
    legacy_kwargs: dict[str, Any],
) -> tuple[
    Union[Basis, BasisInterface, str], float, float, float, Optional[DTypeLike], bool
]:
    """Consume legacy constructor kwargs and map them to modern arguments.

    Each legacy spelling, when present, *wins* over the modern argument -- a
    caller who passes both gets the legacy value. That is deliberate: the old
    names were positional-in-spirit and explicit, whereas the modern ones carry
    defaults that a mixed call site would not have meant to select.

    Parameters
    ----------
    basis : Union[Basis, BasisInterface, str]
        Modern value, overridden by legacy ``expansion_basis``.
    theta : float
        Modern value, overridden by legacy ``theta``.
    G : float
        Modern value, overridden by legacy ``G``.
    softening : float
        Modern value, overridden by legacy ``softening``.
    working_dtype : Optional[DTypeLike]
        Modern value, overridden by legacy ``working_dtype``.
    legacy_kwargs : dict[str, Any]
        Mutated in place: every key consumed here is popped, so what remains
        after the two poppers have run is by construction unrecognized.

    Returns
    -------
    tuple[Union[Basis, BasisInterface, str], float, float, float, Optional[DTypeLike], bool]
        ``(basis, theta, G, softening, working_dtype, legacy_used)``. The final
        flag is what triggers the single ``DeprecationWarning``, so it must be
        threaded through the second popper rather than tested per argument.
    """
    used = False
    legacy_basis = legacy_kwargs.pop("expansion_basis", None)
    if legacy_basis is not None:
        basis = legacy_basis
        used = True
    legacy_theta = legacy_kwargs.pop("theta", None)
    if legacy_theta is not None:
        theta = float(legacy_theta)
        used = True
    legacy_softening = legacy_kwargs.pop("softening", None)
    if legacy_softening is not None:
        softening = float(legacy_softening)
        used = True
    legacy_g = legacy_kwargs.pop("G", None)
    if legacy_g is not None:
        G = float(legacy_g)
        used = True
    legacy_dtype = legacy_kwargs.pop("working_dtype", None)
    if legacy_dtype is not None:
        working_dtype = legacy_dtype
        used = True
    return basis, theta, G, softening, working_dtype, used


class _LegacyRuntimeOverrides(NamedTuple):
    """Runtime-facing settings after legacy kwargs have been folded in.

    Every field except the last two starts from ``advanced``/the preset and is
    then overridden if the matching legacy kwarg was supplied.

    Attributes
    ----------
    complex_rotation : str
        Rotation backend; falls back to ``"solidfmm"`` when the config leaves it
        ``None``. Legacy name: ``complex_rotation``.
    tree_type : Optional[str]
        Tree family. ``None`` here becomes ``"radix"`` at the engine call.
        Legacy name: ``tree_type``.
    execution_backend : str
        Engine execution backend. Legacy name: ``execution_backend``.
    tree_mode : Optional[str]
        Tree build mode. Legacy name: ``tree_build_mode``.
    target_leaf_particles : Optional[int]
        Leaf occupancy target. Legacy name: ``target_leaf_particles``.
    expanse_preset : Optional[str]
        The engine's own preset string, derived from :class:`FMMPreset`;
        ``None`` for presets the engine has no name for. Legacy name:
        ``preset``.
    mac_type : str
        Acceptance criterion; defaults to ``"dehnen"`` for the solidfmm basis
        and ``"bh"`` otherwise. Legacy name: ``mac_type``.
    grouped_interactions : Optional[bool]
        Grouped traversal toggle. Legacy name: ``grouped_interactions``.
    farfield_mode : str
        Far-field interaction mode. Legacy name: ``farfield_mode``.
    nearfield_mode : str
        Near-field interaction mode. Legacy name: ``nearfield_mode``.
    nearfield_edge_chunk_size : int
        Bucketed near-field chunk size. Legacy name:
        ``nearfield_edge_chunk_size``.
    fixed_order : Optional[int]
        Legacy-only: there is no modern spelling, so this is ``None`` unless
        ``fixed_order`` was passed.
    fixed_max_leaf_size : Optional[int]
        Legacy-only, as above, for ``fixed_max_leaf_size``.
    legacy_used : bool
        Threaded in and out so both poppers contribute to one warning decision.
    """

    complex_rotation: str
    tree_type: Optional[str]
    execution_backend: str
    tree_mode: Optional[str]
    target_leaf_particles: Optional[int]
    expanse_preset: Optional[str]
    mac_type: str
    grouped_interactions: Optional[bool]
    farfield_mode: str
    nearfield_mode: str
    nearfield_edge_chunk_size: int
    fixed_order: Optional[int]
    fixed_max_leaf_size: Optional[int]
    legacy_used: bool


def _pop_legacy_runtime_overrides(
    *,
    preset_norm: FMMPreset,
    basis: Basis,
    advanced_cfg: FMMAdvancedConfig,
    legacy_kwargs: dict[str, Any],
    legacy_used: bool,
) -> _LegacyRuntimeOverrides:
    """Resolve runtime-facing legacy kwargs while preserving old behavior.

    Parameters
    ----------
    preset_norm : FMMPreset
        Normalized preset, used only to derive ``expanse_preset``.
    basis : Basis
        Resolved *runtime* basis, used only to pick the default ``mac_type``.
    advanced_cfg : FMMAdvancedConfig
        Supplies the starting value of every field a legacy kwarg can override.
    legacy_kwargs : dict[str, Any]
        Mutated in place; consumed keys are popped.
    legacy_used : bool
        Carried in from :func:`_pop_legacy_common_overrides` and returned
        possibly-set, so one warning covers both poppers.

    Returns
    -------
    _LegacyRuntimeOverrides
        Resolved settings, with ``legacy_used`` folded in.
    """
    complex_rotation = advanced_cfg.farfield.rotation
    if complex_rotation is None:
        complex_rotation = "solidfmm"
    legacy_rotation = legacy_kwargs.pop("complex_rotation", None)
    if legacy_rotation is not None:
        complex_rotation = str(legacy_rotation)
        legacy_used = True

    tree_type = advanced_cfg.tree.tree_type
    legacy_tree_type = legacy_kwargs.pop("tree_type", None)
    if legacy_tree_type is not None:
        tree_type = str(legacy_tree_type)
        legacy_used = True

    execution_backend = advanced_cfg.runtime.execution_backend
    legacy_execution_backend = legacy_kwargs.pop("execution_backend", None)
    if legacy_execution_backend is not None:
        execution_backend = str(legacy_execution_backend)
        legacy_used = True

    tree_mode = advanced_cfg.tree.mode
    legacy_tree_mode = legacy_kwargs.pop("tree_build_mode", None)
    if legacy_tree_mode is not None:
        tree_mode = str(legacy_tree_mode)
        legacy_used = True

    target_leaf_particles = advanced_cfg.tree.leaf_target
    legacy_leaf_target = legacy_kwargs.pop("target_leaf_particles", None)
    if legacy_leaf_target is not None:
        target_leaf_particles = int(legacy_leaf_target)
        legacy_used = True

    if preset_norm is FMMPreset.FAST:
        expanse_preset = "fast"
    elif preset_norm is FMMPreset.LARGE_N_GPU:
        expanse_preset = "large_n_gpu"
    else:
        expanse_preset = None
    legacy_preset = legacy_kwargs.pop("preset", None)
    if legacy_preset is not None:
        if hasattr(legacy_preset, "value"):
            expanse_preset = str(legacy_preset.value)
        else:
            expanse_preset = str(legacy_preset)
        legacy_used = True

    mac_type = (
        str(advanced_cfg.mac_type)
        if advanced_cfg.mac_type is not None
        else ("dehnen" if basis == "solidfmm" else "bh")
    )
    legacy_mac_type = legacy_kwargs.pop("mac_type", None)
    if legacy_mac_type is not None:
        mac_type = str(legacy_mac_type)
        legacy_used = True

    grouped_interactions = advanced_cfg.farfield.grouped_interactions
    legacy_grouped = legacy_kwargs.pop("grouped_interactions", None)
    if legacy_grouped is not None:
        grouped_interactions = bool(legacy_grouped)
        legacy_used = True

    farfield_mode = advanced_cfg.farfield.mode
    legacy_farfield_mode = legacy_kwargs.pop("farfield_mode", None)
    if legacy_farfield_mode is not None:
        farfield_mode = str(legacy_farfield_mode)
        legacy_used = True

    nearfield_mode = advanced_cfg.nearfield.mode
    legacy_nearfield_mode = legacy_kwargs.pop("nearfield_mode", None)
    if legacy_nearfield_mode is not None:
        nearfield_mode = str(legacy_nearfield_mode)
        legacy_used = True

    nearfield_edge_chunk_size = advanced_cfg.nearfield.edge_chunk_size
    legacy_nf_chunk = legacy_kwargs.pop("nearfield_edge_chunk_size", None)
    if legacy_nf_chunk is not None:
        nearfield_edge_chunk_size = int(legacy_nf_chunk)
        legacy_used = True

    fixed_order = legacy_kwargs.pop("fixed_order", None)
    if fixed_order is not None:
        fixed_order = int(fixed_order)
        legacy_used = True

    fixed_max_leaf_size = legacy_kwargs.pop("fixed_max_leaf_size", None)
    if fixed_max_leaf_size is not None:
        fixed_max_leaf_size = int(fixed_max_leaf_size)
        legacy_used = True

    return _LegacyRuntimeOverrides(
        complex_rotation=complex_rotation,
        tree_type=tree_type,
        execution_backend=execution_backend,
        tree_mode=tree_mode,
        target_leaf_particles=target_leaf_particles,
        expanse_preset=expanse_preset,
        mac_type=mac_type,
        grouped_interactions=grouped_interactions,
        farfield_mode=farfield_mode,
        nearfield_mode=nearfield_mode,
        nearfield_edge_chunk_size=nearfield_edge_chunk_size,
        fixed_order=fixed_order,
        fixed_max_leaf_size=fixed_max_leaf_size,
        legacy_used=legacy_used,
    )


class FastMultipoleMethod:
    """Preset-first facade over the FMM engine. The only public class.

    Construction is deliberately shallow: pick a :class:`~jaccpot.FMMPreset`, then
    override individual things through ``advanced=``
    (:class:`~jaccpot.FMMAdvancedConfig`). The keyword-only constructor arguments
    here resolve, together with the preset, into the engine's much wider argument
    set -- so this signature is the supported surface, and the exact set is frozen
    by ``test_facade_constructor_surface_is_frozen`` in
    ``tests/unit/test_public_api_surface.py``. That engine is
    ``runtime._fmm_impl.FMMEngine``, an implementation detail reached only through
    this facade; ``test_public_class_is_the_solver_facade`` pins which of the two
    ``jaccpot.FastMultipoleMethod`` resolves to.

    Every argument is keyword-only. The ones worth knowing before reading the
    per-argument docs:

    * ``basis`` defaults to ``"real"`` (Dehnen real harmonics), which is what the
      production radix large-N fast lane runs end to end with no complex<->real
      conversion. ``"solidfmm"``/``"complex"`` are the same basis under two names
      and stay selectable for cross-checking; ``"cartesian"`` is **experimental**
      and warns -- its relative L2 force error is ~1.8e-1 *independent of expansion
      order*, a divergent-series signature rather than truncation.
    * ``use_pallas=None`` resolves at construction: Pallas near field on where it
      can run (Ampere sm_80+), pure-JAX on sm_75 and CPU. An explicit ``True`` or
      ``False`` wins.
    * ``theta`` is the opening angle; smaller is more accurate and slower.

    Nothing is computed at construction: it only resolves configuration and builds
    the engine.

    Parameters
    ----------
    preset : Union[FMMPreset, str]
        Starting point for everything below. Determines the default ``advanced``
        config, and for ``"large_n_gpu"`` also forces fp32 when no dtype was
        requested, because that lane's kernels are fp32-only.
    basis : Union[Basis, BasisInterface, str]
        ``"real"`` (default), ``"solidfmm"``/``"complex"``, or the experimental
        ``"cartesian"``, as above. May also be a constructed
        :class:`~jaccpot.BasisInterface`.
    m2l_impl : Optional[str]
        M2L translation implementation. ``None`` means "let the basis decide",
        which selects ``"rot_scale"`` for the real basis.
    adaptive_order : bool
        Choose the expansion order per interaction from ``p_gears`` instead of
        using ``max_order`` everywhere.
    p_gears : Optional[Sequence[int]]
        Candidate orders for ``adaptive_order``, smallest passing one wins.
        Ignored when ``adaptive_order`` is ``False``.
    use_pallas : Optional[bool]
        Near-field kernel selection; ``None`` auto-detects, as above.
    reuse_topology : bool
        Reuse the tree across calls instead of rebuilding every time. Trades
        accuracy for speed as the particles drift from the tree they were binned
        into.
    rebuild_every : int
        With ``reuse_topology``, rebuild after this many calls. Must be positive.
    mac_force_scale_mode : str
        How the per-node force scale entering Dehnen's eq (16a) is obtained:
        ``"prev"`` (default) reuses the previous full evaluation's accelerations,
        ``"paper"``/``"paper_fb"`` run a prepass, and the ``*_cached`` variants
        require a cached scale rather than computing one.
    mac_force_scale_prepass_theta : Optional[float]
        Opening angle for the force-scale prepass's own traversal. ``None`` takes
        the default. Only consulted by the prepass modes.
    mac_force_scale_fb_inflation : float
        Inflates the far-field source-to-target distance so the eq (16b) scale
        stays a strict lower bound.
    adaptive_error_model : str
        Error estimator for adaptive acceptance: ``"tail_proxy"`` (default),
        ``"dehnen_degree"``, or ``"dehnen_paper"`` -- the last being the eq (15)
        estimator, which also switches the node force-scale reduction from max to
        min.
    adaptive_eps : Optional[float]
        Relative force-accuracy target of eq (16a). ``None`` takes the default.
    dehnen_geometry_mode : str
        How node centres and radii entering the Dehnen MAC are measured: ``"com"``
        (default), ``"exact"``, ``"tree"``, ``"tree_approx"`` or ``"runtime"``.
        Some modes run a host loop over nodes and warn.
    mac_theta_max : float
        Upper clamp on the opening angle the adaptive MAC may choose.
    theta : float
        Opening angle. Smaller is more accurate and slower.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening length for the near-field kernel.
    precision : Optional[Literal['fp32', 'fp64']]
        Convenience spelling of ``working_dtype``. Passing both is an error unless
        they agree; ``"fp64"`` additionally requires ``jax_enable_x64``.
    working_dtype : Optional[DTypeLike]
        Explicit working dtype. ``None`` lets the preset decide.
    advanced : Optional[FMMAdvancedConfig]
        Replaces the preset's advanced defaults wholesale -- it is not merged into
        them. Build it from the preset's own config if you mean to override only a
        few fields.
    **legacy_kwargs : Any
        Transitional expanse-era spellings. Any recognized key overrides its modern
        equivalent and triggers one ``DeprecationWarning``; anything left unconsumed
        is a ``TypeError``.

    Raises
    ------
    ValueError
        If ``precision`` is not one of ``"fp32"``/``"fp64"``, conflicts with an
        explicit ``working_dtype``, or requests fp64 without ``jax_enable_x64``.
    TypeError
        If any keyword argument was neither in the signature nor recognized as a
        legacy spelling.

    Notes
    -----
    Accuracy and gradient behaviour are properties of the *methods*, not the
    constructor, and are documented there:
    :meth:`compute_accelerations` for the forward path and
    :meth:`differentiable_accelerations` for exact fixed-topology gradients w.r.t.
    positions and masses. ``prepare_state`` is not traceable, so a ``state`` for the
    differentiable path must be built once, concretely, before ``jax.grad``.

    The differentiable path is configured through :class:`~jaccpot.GradConfig`, not
    through this constructor.
    """

    def __init__(
        self,
        *,
        preset: Union[FMMPreset, str] = FMMPreset.FAST,
        # Real (Dehnen) harmonics is the production default everywhere: the radix
        # large-N fast lane runs pure-real end to end (no complex<->real
        # conversion). 'solidfmm'/'complex' remain selectable for cross-checking.
        basis: Union[Basis, BasisInterface, str] = "real",
        m2l_impl: Optional[str] = None,
        adaptive_order: bool = False,
        p_gears: Optional[Sequence[int]] = None,
        # None => resolve at construction: Pallas near-field ON where it can run
        # (Ampere sm_80+), pure-JAX only on sm_75/CPU. Explicit True/False wins.
        use_pallas: Optional[bool] = None,
        reuse_topology: bool = False,
        rebuild_every: int = 1,
        mac_force_scale_mode: str = "prev",
        mac_force_scale_prepass_theta: Optional[float] = None,
        mac_force_scale_fb_inflation: float = 1.0,
        adaptive_error_model: str = "tail_proxy",
        adaptive_eps: Optional[float] = None,
        dehnen_geometry_mode: str = "com",
        mac_theta_max: float = 1.0,
        theta: float = 0.6,
        G: float = 1.0,
        softening: float = 1e-3,
        precision: Optional[Literal["fp32", "fp64"]] = None,
        working_dtype: Optional[DTypeLike] = None,
        advanced: Optional[FMMAdvancedConfig] = None,
        **legacy_kwargs: Any,
    ):
        # Transitional compatibility: allow legacy expanse-style kwargs so
        # notebooks/scripts can migrate import paths incrementally.
        legacy_kwargs = dict(legacy_kwargs)
        basis, theta, G, softening, working_dtype, legacy_used = (
            _pop_legacy_common_overrides(
                basis=basis,
                theta=theta,
                G=G,
                softening=softening,
                working_dtype=working_dtype,
                legacy_kwargs=legacy_kwargs,
            )
        )
        if precision is not None:
            precision_norm = str(precision).strip().lower()
            if precision_norm not in ("fp32", "fp64"):
                raise ValueError("precision must be one of ('fp32', 'fp64')")
            precision_dtype = jnp.float32 if precision_norm == "fp32" else jnp.float64
            if working_dtype is not None and jnp.dtype(working_dtype) != jnp.dtype(
                precision_dtype
            ):
                raise ValueError(
                    "precision conflicts with explicit working_dtype; "
                    "use only one or ensure they match"
                )
            if precision_norm == "fp64" and not bool(jax.config.jax_enable_x64):
                raise ValueError("precision='fp64' requires jax_enable_x64=True")
            working_dtype = precision_dtype
        basis_resolution = _resolve_basis_input(basis)
        runtime_basis = basis_resolution.runtime_basis
        resolved_m2l_impl = m2l_impl
        if resolved_m2l_impl is None and basis_resolution.public_name == "real":
            resolved_m2l_impl = "rot_scale"

        preset_norm = _normalize_preset(preset)
        if preset_norm is FMMPreset.LARGE_N_GPU and working_dtype is None:
            # Large-N radix fast lane is currently specialized for fp32 kernels.
            working_dtype = jnp.float32
        advanced_cfg = (
            _default_advanced_for_preset(preset_norm) if advanced is None else advanced
        )

        runtime_overrides = _pop_legacy_runtime_overrides(
            preset_norm=preset_norm,
            basis=runtime_basis,
            advanced_cfg=advanced_cfg,
            legacy_kwargs=legacy_kwargs,
            legacy_used=legacy_used,
        )
        legacy_used = runtime_overrides.legacy_used

        traversal_config = advanced_cfg.runtime.traversal_config
        self._impl = _RuntimeFMM(
            preset=runtime_overrides.expanse_preset,
            theta=float(theta),
            G=float(G),
            softening=float(softening),
            working_dtype=working_dtype,
            expansion_basis=runtime_basis,
            basis_impl=basis_resolution.basis_impl,
            m2l_impl=resolved_m2l_impl,
            runtime_path=str(legacy_kwargs.pop("runtime_path", "auto")),
            adaptive_order=adaptive_order,
            p_gears=p_gears,
            use_pallas=use_pallas,
            reuse_topology=reuse_topology,
            rebuild_every=rebuild_every,
            mac_force_scale_mode=mac_force_scale_mode,
            mac_force_scale_prepass_theta=mac_force_scale_prepass_theta,
            mac_force_scale_fb_inflation=mac_force_scale_fb_inflation,
            adaptive_error_model=adaptive_error_model,
            adaptive_eps=adaptive_eps,
            dehnen_geometry_mode=dehnen_geometry_mode,
            mac_theta_max=mac_theta_max,
            complex_rotation=runtime_overrides.complex_rotation,
            tree_type=runtime_overrides.tree_type or "radix",
            execution_backend=runtime_overrides.execution_backend,
            tree_build_mode=runtime_overrides.tree_mode,
            target_leaf_particles=runtime_overrides.target_leaf_particles,
            refine_local=legacy_kwargs.pop(
                "refine_local", advanced_cfg.tree.refine_local
            ),
            max_refine_levels=legacy_kwargs.pop(
                "max_refine_levels",
                advanced_cfg.tree.max_refine_levels,
            ),
            aspect_threshold=legacy_kwargs.pop(
                "aspect_threshold",
                advanced_cfg.tree.aspect_threshold,
            ),
            grouped_interactions=runtime_overrides.grouped_interactions,
            retain_far_pairs_for_grad=legacy_kwargs.pop(
                "retain_far_pairs_for_grad",
                advanced_cfg.farfield.retain_far_pairs_for_grad,
            ),
            farfield_mode=runtime_overrides.farfield_mode,
            streamed_far_pairs=legacy_kwargs.pop(
                "streamed_far_pairs",
                advanced_cfg.farfield.streamed_far_pairs,
            ),
            mixed_order_farfield=legacy_kwargs.pop(
                "mixed_order_farfield",
                advanced_cfg.farfield.mixed_order,
            ),
            mixed_order_min_order=legacy_kwargs.pop(
                "mixed_order_min_order",
                advanced_cfg.farfield.mixed_order_min_order,
            ),
            m2l_chunk_size=legacy_kwargs.pop(
                "m2l_chunk_size",
                advanced_cfg.farfield.m2l_chunk_size,
            ),
            l2l_chunk_size=legacy_kwargs.pop(
                "l2l_chunk_size",
                advanced_cfg.farfield.l2l_chunk_size,
            ),
            nearfield_mode=runtime_overrides.nearfield_mode,
            nearfield_edge_chunk_size=runtime_overrides.nearfield_edge_chunk_size,
            precompute_nearfield_scatter_schedules=bool(
                legacy_kwargs.pop(
                    "precompute_nearfield_scatter_schedules",
                    advanced_cfg.nearfield.precompute_scatter_schedules,
                )
            ),
            host_refine_mode=legacy_kwargs.pop(
                "host_refine_mode",
                advanced_cfg.runtime.host_refine_mode,
            ),
            fail_fast=bool(
                legacy_kwargs.pop(
                    "fail_fast",
                    advanced_cfg.runtime.fail_fast,
                )
            ),
            memory_objective=str(
                legacy_kwargs.pop(
                    "memory_objective",
                    advanced_cfg.runtime.memory_objective,
                )
            ),
            memory_budget_bytes=legacy_kwargs.pop(
                "memory_budget_bytes",
                advanced_cfg.runtime.memory_budget_bytes,
            ),
            max_pair_queue=legacy_kwargs.pop(
                "max_pair_queue",
                advanced_cfg.runtime.max_pair_queue,
            ),
            pair_process_block=legacy_kwargs.pop(
                "pair_process_block",
                advanced_cfg.runtime.pair_process_block,
            ),
            dehnen_radius_scale=float(
                legacy_kwargs.pop(
                    "dehnen_radius_scale", advanced_cfg.dehnen_radius_scale
                )
            ),
            use_dense_interactions=legacy_kwargs.pop("use_dense_interactions", None),
            traversal_config=legacy_kwargs.pop("traversal_config", traversal_config),
            enable_interaction_cache=bool(
                legacy_kwargs.pop(
                    "enable_interaction_cache",
                    advanced_cfg.runtime.enable_interaction_cache,
                )
            ),
            retain_traversal_result=bool(
                legacy_kwargs.pop(
                    "retain_traversal_result",
                    advanced_cfg.runtime.retain_traversal_result,
                )
            ),
            retain_interactions=bool(
                legacy_kwargs.pop(
                    "retain_interactions",
                    advanced_cfg.runtime.retain_interactions,
                )
            ),
            prepare_stage_memory_split_enabled=legacy_kwargs.pop(
                "prepare_stage_memory_split_enabled",
                advanced_cfg.runtime.prepare_stage_memory_split_enabled,
            ),
            autotune_m2l_chunk=bool(
                legacy_kwargs.pop(
                    "autotune_m2l_chunk",
                    advanced_cfg.runtime.autotune_m2l_chunk,
                )
            ),
            precompute_grouped_class_segments=legacy_kwargs.pop(
                "precompute_grouped_class_segments",
                advanced_cfg.runtime.precompute_grouped_class_segments,
            ),
            grouped_schedule_budget_bytes=legacy_kwargs.pop(
                "grouped_schedule_budget_bytes",
                advanced_cfg.runtime.grouped_schedule_budget_bytes,
            ),
            nearfield_schedule_item_cap=legacy_kwargs.pop(
                "nearfield_schedule_item_cap",
                advanced_cfg.runtime.nearfield_schedule_item_cap,
            ),
            upward_leaf_batch_size=legacy_kwargs.pop(
                "upward_leaf_batch_size",
                advanced_cfg.runtime.upward_leaf_batch_size,
            ),
            fixed_order=runtime_overrides.fixed_order,
            fixed_max_leaf_size=runtime_overrides.fixed_max_leaf_size,
            mac_type=runtime_overrides.mac_type,
        )
        if legacy_used:
            warnings.warn(
                "Legacy expanse-style kwargs are deprecated in jaccpot.FastMultipoleMethod. "
                "Use preset/basis/advanced config objects instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(k) for k in legacy_kwargs.keys()))
            raise TypeError(f"Unknown jaccpot.FastMultipoleMethod kwargs: {unknown}")
        self.preset = preset_norm
        self.basis = basis_resolution.public_name
        self.basis_impl = basis_resolution.basis_impl
        self.advanced = advanced_cfg

    # Overloads for the two shapes callers actually ask for by name. The return
    # is keyed on `return_potential` and on whether `max_acc_derivative_order` is
    # zero, so a caller writing `return_potential=True` can be handed
    # `Tuple[Array, Array]` instead of a four-member union it has to narrow.
    #
    # The third overload is the honest fallback and is not optional: when either
    # flag is a runtime value, no literal overload can be selected, and a
    # `Literal[0]` overload would then declare a derivative tuple for a call that
    # returns a bare `Array`. That is why the *internal* multi-flag producers were
    # left alone (audit E.4) -- every one of their call sites passes runtime
    # values. This surface is different: it is what downstream code calls, and
    # downstream calls with literals.
    @overload
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = ...,
        leaf_size: int = ...,
        max_order: int = ...,
        return_potential: Literal[False] = ...,
        theta: Optional[float] = ...,
        reuse_prepared_state: bool = ...,
        max_acc_derivative_order: Literal[0] = ...,
    ) -> Array: ...

    @overload
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = ...,
        leaf_size: int = ...,
        max_order: int = ...,
        return_potential: Literal[True],
        theta: Optional[float] = ...,
        reuse_prepared_state: bool = ...,
        max_acc_derivative_order: Literal[0] = ...,
    ) -> Tuple[Array, Array]: ...

    @overload
    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = ...,
        leaf_size: int = ...,
        max_order: int = ...,
        return_potential: bool = ...,
        theta: Optional[float] = ...,
        reuse_prepared_state: bool = ...,
        max_acc_derivative_order: int = ...,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, tuple[Array, ...]],
        Tuple[Array, Array, tuple[Array, ...]],
    ]: ...

    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        return_potential: bool = False,
        theta: Optional[float] = None,
        reuse_prepared_state: bool = False,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, tuple[Array, ...]],
        Tuple[Array, Array, tuple[Array, ...]],
    ]:
        """Compute accelerations (and optional potentials) for particle data.

        When ``target_indices`` is provided, all particles remain source masses
        but outputs are returned only for the indexed target particles.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Source and target particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``, aligned with ``positions``. G=1 unless the
            solver was configured otherwise.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return. All particles remain
            sources.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds for tree construction.
            Defaults to the particle bounding box.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p`` for the upward and downward passes. Note this
            defaults to 4 here, while the underlying
            :class:`~jaccpot.runtime.fmm_evaluate.EvaluateMixin` method defaults
            to 2 -- the public default is the one that applies.
        return_potential : bool
            Also return the potential ``[N]``.
        theta : Optional[float]
            Per-call MAC opening-angle override. Tightening it lowers the
            far-field truncation error at the cost of more near-field work.
        reuse_prepared_state : bool
            Reuse the last prepared state when the array objects and preparation
            parameters are identical. An object-identity check, not a value
            comparison: mutating an array in place and passing it again reuses a
            stale tree.
        max_acc_derivative_order : int
            Request packed spatial derivatives of acceleration in addition to
            acceleration itself. ``0`` disables derivatives.
            ``1`` returns the acceleration Jacobian with shape ``(N, 3, 3)`` in
            packed-symmetric layout across the trailing axis. Non-zero requires
            ``expansion_basis="solidfmm"``.

        Returns
        -------
        Union[Array, Tuple[Array, Array], Tuple[Array, tuple[Array, ...]], Tuple[Array, Array, tuple[Array, ...]]]
            Accelerations ``[N, 3]``, or ``[len(target_indices), 3]`` when
            ``target_indices`` is given. ``return_potential`` inserts the
            potentials next and ``max_acc_derivative_order > 0`` appends the
            packed derivative tuple last, giving ``a``, ``(a, pot)``,
            ``(a, derivs)`` or ``(a, pot, derivs)``.

        Notes
        -----
        The forward-only entry point: it builds a tree per call. Use
        :meth:`prepare_state` plus :meth:`evaluate_prepared_state` to amortise
        the build, and :meth:`differentiable_accelerations` for gradients -- this
        method falls back to an O(N^2) direct sum under an outer trace.

        Accuracy is set by ``max_order`` and ``theta`` together; neither alone
        determines it, and a convergence check must vary both.
        """
        return self._impl.compute_accelerations(
            positions,
            masses,
            target_indices=target_indices,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            return_potential=return_potential,
            theta=theta,
            reuse_prepared_state=reuse_prepared_state,
            jit_tree=self.advanced.runtime.jit_tree,
            jit_traversal=self.advanced.runtime.jit_traversal,
            max_acc_derivative_order=max_acc_derivative_order,
        )

    def prepare_state(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
        cache_policy: str = "auto",
        force_scale_nodes: Optional[Array] = None,
        runtime_overrides_override: Optional[Any] = None,
        fused_device_mode: bool = False,
    ) -> Union[FMMPreparedState, LargeNPreparedState]:
        """Prepare and cache tree/interactions for repeated evaluations.

        The return type is the union because ``preset="large_n_gpu"`` prepares a
        ``LargeNPreparedState``, which is not an ``FMMPreparedState`` -- the two are
        unrelated classes. Everything else here already said so: the delegate this
        forwards to is annotated ``-> PreparedStateLike``, itself defined as this
        union, and ``compute_accelerations``/``compute_potential`` below already
        *accept* the union and document it. Only the producer's annotation disagreed,
        which is what made ``JACCPOT_RUNTIME_TYPECHECK=1`` reject the large-N path.

        ``force_scale_nodes`` overrides the per-node force scale used by the
        adaptive acceptance test for this call only, skipping the prepass and
        leaving the reuse cache untouched. Its length must match the node count of
        the tree this call builds.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Source and target particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``, aligned with ``positions``.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds for tree construction.
            Defaults to the particle bounding box.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        cache_policy : str
            How aggressively the prepared state may be cached and reused.
        force_scale_nodes : Optional[Array]
            Per-node force-scale override for this call only; see above.
        runtime_overrides_override : Optional[Any]
            Replacement runtime overrides for this preparation. ``Any`` because
            the override type is a runtime internal.
        fused_device_mode : bool
            Build the fused device-resident layout used by the strict lanes.

        Returns
        -------
        Union[FMMPreparedState, LargeNPreparedState]
            The prepared state. Not traceable -- build it concretely, before any
            ``jax.grad``.
        """
        return self._impl.prepare_state(
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            jit_tree=self.advanced.runtime.jit_tree,
            force_scale_nodes=force_scale_nodes,
            runtime_overrides_override=runtime_overrides_override,
            fused_device_mode=bool(fused_device_mode),
        )

    def compute_accelerations_and_jerk(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        velocities: Float[Array, "n 3"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
        reuse_prepared_state: bool = False,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Compute accelerations and jerk estimates for particle data.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``.
        velocities : Float[Array, 'n 3']
            Particle velocities ``[N, 3]``. Used only for the jerk; the
            acceleration does not depend on them.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds for tree construction.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        reuse_prepared_state : bool
            Reuse the last prepared state when the array objects and preparation
            parameters are identical.
        jerk_mode : str
            ``"fast_approx"`` uses exact near-field jerk plus far-field
            convective jerk from the acceleration Jacobian.
            ``"accurate"`` uses an analytic far-field source-motion term
            (`dM -> dL`) plus the convective and exact near-field terms.
        jerk_fd_dt : float
            Finite-difference step used only for non-``solidfmm`` fallback
            in ``jerk_mode="accurate"``.

        Returns
        -------
        tuple[Array, Array]
            ``(accelerations, jerk)``, each ``[N, 3]`` (or
            ``[len(target_indices), 3]``).

        Notes
        -----
        The two ``jerk_mode`` values are *not* equivalent: ``"fast_approx"``
        omits the far-field source-motion term that ``"accurate"`` includes, so
        they differ by a physical contribution rather than by round-off. Choose
        by accuracy requirement, not by speed alone.

        **The difference is small, and measured.** At N=512, leaf 16, ``p=4``,
        ``theta=0.6``, float32 the source-motion term contributes
        **3.826e-05 relative L2** on the jerk -- a correction, not a
        leading-order effect. The acceleration is unaffected (the same
        measurement bounds the two modes' acceleration drift below 1e-4, which is
        fp32 reassociation). Both figures come from
        :func:`bench.ci_benchmark_guard._validate_accurate_jerk_differs_from_fast`,
        which CI runs on every push; it asserts ``jerk_delta >= 1e-6``, ~40x below
        the measured value, so it is a tripwire for the term going *missing* and
        not an upper bound on the error.

        Note what is *not* measured: the two direct-sum accuracy tests
        (``tests/unit/test_solver_api.py``, ``5e-2`` for the default mode and
        ``2e-3`` for ``"accurate"``) both run at ``theta=1e-4``, where almost
        nothing is accepted as far-field. They therefore barely exercise the
        far-field term that distinguishes the modes, and the gap between those two
        tolerances should not be read as its size -- 3.826e-05 above is the figure
        to use.
        """
        return self._impl.compute_accelerations_and_jerk(
            positions,
            masses,
            velocities,
            target_indices=target_indices,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            reuse_prepared_state=reuse_prepared_state,
            jit_tree=self.advanced.runtime.jit_tree,
            jit_traversal=self.advanced.runtime.jit_traversal,
            jerk_mode=jerk_mode,
            jerk_fd_dt=jerk_fd_dt,
        )

    def compute_accelerations_with_time_derivatives(
        self: "FastMultipoleMethod",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        velocities: Float[Array, "n 3"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
        reuse_prepared_state: bool = False,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Compute accelerations and time derivatives up to order K.

        The generalisation of :meth:`compute_accelerations_and_jerk`: order 1 is
        the jerk, and the ``mode`` values mean what ``jerk_mode`` means there,
        including that they are not numerically equivalent.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Source and target particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``, aligned with ``positions``.
        velocities : Float[Array, 'n 3']
            Particle velocities ``[N, 3]``. Used only for the time derivatives;
            the acceleration does not depend on them.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds for tree construction.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        reuse_prepared_state : bool
            Reuse the last prepared state when the array objects and preparation
            parameters are identical.
        max_time_derivative_order : int
            Highest time-derivative order ``K`` to return. ``1`` is the jerk.
        mode : str
            ``"accurate"`` (default) or ``"fast_approx"``; see
            :meth:`compute_accelerations_and_jerk`.

        Returns
        -------
        tuple[Array, tuple[Array, ...]]
            ``(accelerations, derivatives)``, where ``derivatives`` holds ``K``
            arrays of shape ``[N, 3]`` in increasing order.
        """
        return self._impl.compute_accelerations_with_time_derivatives(
            positions,
            masses,
            velocities,
            target_indices=target_indices,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            reuse_prepared_state=reuse_prepared_state,
            jit_tree=self.advanced.runtime.jit_tree,
            jit_traversal=self.advanced.runtime.jit_traversal,
            max_time_derivative_order=max_time_derivative_order,
            mode=mode,
        )

    def prepare_upward_sweep(
        self: "FastMultipoleMethod",
        tree: Any,
        positions_sorted: Float[Array, "n 3"],
        masses_sorted: Float[Array, "n"],
        *,
        max_order: int = 4,
    ) -> Any:
        """Build upward multipole data for a prebuilt tree.

        The P2M/M2M half of :meth:`prepare_state`, for callers holding a tree
        built elsewhere. The particle arrays must already be in that tree's sort
        order; this method does not permute them.

        Parameters
        ----------
        tree : Any
            A built tree artifact. ``Any`` so this module need not import the
            ``yggdrax`` tree types.
        positions_sorted : Float[Array, 'n 3']
            Particle positions ``[N, 3]`` in ``tree``'s sort order.
        masses_sorted : Float[Array, 'n']
            Particle masses ``[N]`` in the same order.
        max_order : int
            Expansion order ``p``.

        Returns
        -------
        Any
            The upward-sweep result, likewise left as ``Any``.
        """
        return self._impl.prepare_upward_sweep(
            tree,
            positions_sorted,
            masses_sorted,
            max_order=max_order,
        )

    # Overloads for the two shapes callers actually ask for by name. The return
    # is keyed on `return_potential` and on whether `max_acc_derivative_order` is
    # zero, so a caller writing `return_potential=True` can be handed
    # `Tuple[Array, Array]` instead of a four-member union it has to narrow.
    #
    # The third overload is the honest fallback and is not optional: when either
    # flag is a runtime value, no literal overload can be selected, and a
    # `Literal[0]` overload would then declare a derivative tuple for a call that
    # returns a bare `Array`. That is why the *internal* multi-flag producers were
    # left alone (audit E.4) -- every one of their call sites passes runtime
    # values. This surface is different: it is what downstream code calls, and
    # downstream calls with literals.
    @overload
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        return_potential: Literal[False] = ...,
        jit_traversal: Optional[bool] = ...,
        max_acc_derivative_order: Literal[0] = ...,
    ) -> Array: ...

    @overload
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        return_potential: Literal[True],
        jit_traversal: Optional[bool] = ...,
        max_acc_derivative_order: Literal[0] = ...,
    ) -> Tuple[Array, Array]: ...

    @overload
    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Int[Array, "t"]] = ...,
        return_potential: bool = ...,
        jit_traversal: Optional[bool] = ...,
        max_acc_derivative_order: int = ...,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, tuple[Array, ...]],
        Tuple[Array, Array, tuple[Array, ...]],
    ]: ...

    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        # The union, for the same reason as `prepare_state` above: the delegate takes
        # `PreparedStateLike`, `differentiable_accelerations` just below already takes
        # the union, and `test_large_n_compiled_eval_uses_specialized_nearfield`
        # demonstrates at runtime that this method handles a `LargeNPreparedState`
        # correctly. Only the annotation said otherwise.
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        return_potential: bool = False,
        jit_traversal: Optional[bool] = None,
        max_acc_derivative_order: int = 0,
    ) -> Union[
        Array,
        Tuple[Array, Array],
        Tuple[Array, tuple[Array, ...]],
        Tuple[Array, Array, tuple[Array, ...]],
    ]:
        """Evaluate a previously prepared state for all particles or a subset.

        Evaluates the expansions ``state`` already carries. It takes no live
        positions or masses, so it must not be differentiated -- see
        :meth:`differentiable_accelerations` for that.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Tree and interaction lists from :meth:`prepare_state`.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        return_potential : bool
            Also return the potential ``[N]``.
        jit_traversal : Optional[bool]
            Per-call override; ``None`` takes the solver's configured value,
            itself defaulting to ``True``.
        max_acc_derivative_order : int
            Packed spatial derivatives of acceleration to return alongside it;
            ``0`` disables them. Non-zero requires the solidfmm basis.

        Returns
        -------
        Union[Array, Tuple[Array, Array], Tuple[Array, tuple[Array, ...]], Tuple[Array, Array, tuple[Array, ...]]]
            Accelerations, with potentials inserted next when
            ``return_potential`` and the packed derivative tuple appended last
            when ``max_acc_derivative_order > 0`` -- the same four shapes
            :meth:`compute_accelerations` returns.
        """
        jit_traversal = (
            (
                True
                if self.advanced.runtime.jit_traversal is None
                else bool(self.advanced.runtime.jit_traversal)
            )
            if jit_traversal is None
            else bool(jit_traversal)
        )
        return self._impl.evaluate_prepared_state(
            state,
            target_indices=target_indices,
            return_potential=return_potential,
            jit_traversal=jit_traversal,
            max_acc_derivative_order=max_acc_derivative_order,
        )

    def differentiable_accelerations(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        jit_traversal: bool = False,
        grad_plan: Optional[Any] = None,
        grad_config: Optional[GradConfig] = None,
    ) -> Array:
        """Exact fixed-topology gradients of the FMM force w.r.t. positions/masses.

        Differentiable single-GPU FMM acceleration: the tree topology carried by
        ``state`` is held constant while the numeric pipeline is re-evaluated on
        the live ``positions``/``masses``, so ``jax.grad``/``jax.vjp`` over this
        call yield exact gradients at fixed topology. Build ``state`` once with
        :meth:`prepare_state` (tree construction is not traceable), then
        differentiate this method::

            state = fmm.prepare_state(pos0, mass0, max_order=4, leaf_size=64)
            g = jax.grad(lambda p, m:
                (fmm.differentiable_accelerations(state, p, m) ** 2).sum()
            )(pos, mass)

        Works on a radix ``FMMPreparedState`` (solidfmm basis) and on a
        ``LargeNPreparedState`` from ``preset="large_n_gpu"``, which additionally
        needs ``retain_far_pairs_for_grad=True``. See ``docs/differentiable_fmm.md``.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Frozen topology from :meth:`prepare_state`, captured as a constant.
        positions : Float[Array, 'n 3']
            Differentiated positions ``[N, 3]``, in the original (unsorted)
            particle order -- the method applies ``state``'s permutation itself.
        masses : Float[Array, 'n']
            Differentiated masses ``[N]``, in the same original order.
        target_indices : Optional[Int[Array, 't']]
            Optional subset of targets to return; all particles remain sources.
            Not supported on the large-N path.
        jit_traversal : bool
            Kept ``False`` on the gradient path.
        grad_plan : Optional[Any]
            A ``LargeNGradPlan``; annotated ``Any`` here only to keep the
            public module from importing the large-N internals. Large-N only.
            Build once with
            :func:`~jaccpot.runtime._large_n_grad.prepare_large_n_grad_plan` and
            pass it here to hoist validation and pair-list setup out of an
            optimisation loop.
        grad_config : Optional[GradConfig]
            Gradient-path execution options -- chiefly ``nearfield_lane``, which
            defaults to ``"auto"`` and selects the leaf-major fast lane at
            N >= 100000 because the bucketed reverse OOMs at galaxy scale. Each
            field falls back to its ``JACCPOT_*`` environment variable when left
            ``None``, so existing env-configured scripts are unaffected.

        Returns
        -------
        Array
            ``(N, 3)`` accelerations in the original input order.
        """
        return self._impl.differentiable_accelerations(
            state,
            positions,
            masses,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            grad_plan=grad_plan,
            grad_config=grad_config,
        )

    def differentiable_step_fn(
        self: "FastMultipoleMethod",
        state: Union[FMMPreparedState, LargeNPreparedState],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        grad_config: Optional[GradConfig] = None,
        jit_traversal: bool = False,
        compile_now: Optional[Tuple[Array, Array]] = None,
    ) -> Callable[[Array, Array], Array]:
        """Return a compiled ``f(positions, masses) -> accelerations``.

        The step seam for a training or inference loop: :meth:`prepare_state`
        once, this once, then the returned function per step. The compile is paid
        once instead of a re-trace per call, and the result is differentiable, so
        ``jax.grad`` over it is compiled too::

            state = fmm.prepare_state(pos0, mass0, max_order=4, leaf_size=64)
            step = fmm.differentiable_step_fn(state, compile_now=(pos0, mass0))
            for _ in range(steps):
                g = jax.grad(lambda p: (step(p, mass) ** 2).sum())(pos)

        Measured on an idle A100 at N=4096, leaf 64, p=4, real basis:
        ``preset="accurate"`` goes from 6.681 s eager to 0.0110 s compiled, a
        factor of 607 after an 18.9 s one-time compile. On ``preset="large_n_gpu"``
        it is the other way round (2.843 s eager, 4.036 s compiled), because that
        path's fast-lane kernels are already compiled individually -- measure
        before adopting it there.

        Raises ``TypeError`` immediately if ``state`` holds tracers, rather than
        letting the failure surface deep inside the trace as a leaked-tracer
        error naming an internal cache.

        Parameters
        ----------
        state : Union[FMMPreparedState, LargeNPreparedState]
            Frozen topology from :meth:`prepare_state`, captured as a constant.
            Must hold concrete arrays, not tracers.
        target_indices : Optional[Int[Array, 't']]
            Optional subset of targets to return; all particles remain sources.
        grad_config : Optional[GradConfig]
            Gradient-path execution options; see
            :meth:`differentiable_accelerations`.
        jit_traversal : bool
            Kept ``False`` on the gradient path.
        compile_now : Optional[Tuple[Array, Array]]
            Representative ``(positions, masses)`` to compile against eagerly, so
            the one-time cost is paid here rather than on the first step.

        Returns
        -------
        Callable[[Array, Array], Array]
            ``f(positions, masses) -> accelerations``, differentiable and
            compiled.
        """

        return self._impl.differentiable_step_fn(
            state,
            target_indices=target_indices,
            grad_config=grad_config,
            jit_traversal=jit_traversal,
            compile_now=compile_now,
        )

    def evaluate_prepared_state_with_jerk(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Float[Array, "n 3"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        jerk_mode: str = "fast_approx",
        jerk_fd_dt: float = 1e-3,
    ) -> tuple[Array, Array]:
        """Evaluate accelerations and jerk for a prepared state.

        Parameters
        ----------
        state : FMMPreparedState
            Tree and interaction lists from :meth:`prepare_state`. Reused as-is;
            the topology is not rebuilt, so this is only valid while the
            positions it was built from remain appropriate.
        velocities : Float[Array, 'n 3']
            Particle velocities ``[N, 3]`` in the original (unsorted) particle
            order.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        jerk_mode : str
            ``"fast_approx"`` or ``"accurate"``; see
            :meth:`compute_accelerations_and_jerk`, including the note that the
            two are not numerically equivalent.
        jerk_fd_dt : float
            Finite-difference step for non-``solidfmm`` fallback in
            ``"accurate"`` mode.

        Returns
        -------
        tuple[Array, Array]
            ``(accelerations, jerk)``, each ``[N, 3]`` (or
            ``[len(target_indices), 3]``).

        Notes
        -----
        Unlike :meth:`compute_accelerations_and_jerk` this evaluates the
        *prebaked* expansions carried by ``state``. It takes no live positions or
        masses, so it must not be differentiated -- see
        :meth:`differentiable_accelerations` for that.
        """
        jit_traversal = (
            True
            if self.advanced.runtime.jit_traversal is None
            else bool(self.advanced.runtime.jit_traversal)
        )
        return self._impl.evaluate_prepared_state_with_jerk(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            jerk_mode=jerk_mode,
            jerk_fd_dt=jerk_fd_dt,
        )

    def evaluate_prepared_state_with_time_derivatives(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        velocities: Float[Array, "n 3"],
        *,
        target_indices: Optional[Int[Array, "t"]] = None,
        max_time_derivative_order: int = 1,
        mode: str = "accurate",
    ) -> tuple[Array, tuple[Array, ...]]:
        """Evaluate accelerations and time derivatives for a prepared state.

        The prepared-state counterpart of
        :meth:`compute_accelerations_with_time_derivatives`, standing to it as
        :meth:`evaluate_prepared_state_with_jerk` stands to
        :meth:`compute_accelerations_and_jerk` -- and carrying the same caveat:
        it evaluates prebaked expansions and must not be differentiated.

        Parameters
        ----------
        state : FMMPreparedState
            Tree and interaction lists from :meth:`prepare_state`.
        velocities : Float[Array, 'n 3']
            Particle velocities ``[N, 3]`` in the original (unsorted) particle
            order.
        target_indices : Optional[Int[Array, 't']]
            1-D indices selecting which targets to return; all particles remain
            sources.
        max_time_derivative_order : int
            Highest time-derivative order ``K`` to return. ``1`` is the jerk.
        mode : str
            ``"accurate"`` (default) or ``"fast_approx"``; see
            :meth:`compute_accelerations_and_jerk`.

        Returns
        -------
        tuple[Array, tuple[Array, ...]]
            ``(accelerations, derivatives)``, where ``derivatives`` holds ``K``
            arrays of shape ``[N, 3]`` in increasing order.
        """
        jit_traversal = (
            True
            if self.advanced.runtime.jit_traversal is None
            else bool(self.advanced.runtime.jit_traversal)
        )
        return self._impl.evaluate_prepared_state_with_time_derivatives(
            state,
            velocities,
            target_indices=target_indices,
            jit_traversal=jit_traversal,
            max_time_derivative_order=max_time_derivative_order,
            mode=mode,
        )

    def refresh_prepared_state(
        self: "FastMultipoleMethod",
        prepared_state: FMMPreparedState,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
        fused_device_mode: bool = False,
    ) -> FMMPreparedState:
        """Refresh prepared state under fixed-profile large-N runtime constraints.

        Rebinds an existing state to new particle data without paying a full
        :meth:`prepare_state`. Together with :meth:`update_multipoles_only` and
        :meth:`rebuild_topology_in_place` -- both of which delegate here, differing
        only in which diagnostic counter they bump and whether ``bounds`` is
        forwarded -- this is the refresh cadence the strict runners drive.

        Supported only on the large-N production profile
        (``preset="large_n_gpu"``, radix tree, solidfmm basis) and only for a
        ``LargeNPreparedState``; anything else raises from the engine.

        Parameters
        ----------
        prepared_state : FMMPreparedState
            State to refresh.
        positions : Float[Array, 'n 3']
            New particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            New particle masses ``[N]``.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.
        fused_device_mode : bool
            Refresh into the fused device-resident layout.

        Returns
        -------
        FMMPreparedState
            The refreshed state. A new object -- nothing is mutated in place,
            despite what :meth:`rebuild_topology_in_place` is called.
        """
        return self._impl.refresh_prepared_state(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            fused_device_mode=bool(fused_device_mode),
        )

    def strict_prepare_refresh_and_evaluate(
        self: "FastMultipoleMethod",
        prepared_state: Optional[FMMPreparedState],
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
    ) -> tuple[FMMPreparedState, Array]:
        """Strict static-radix helper: prepare/refresh and evaluate in one call.

        Prepares when ``prepared_state`` is ``None`` and refreshes otherwise, then
        evaluates -- so a loop can call this unconditionally and let the first
        iteration build. Returning the state alongside the accelerations is what
        makes that possible. Large-N production profile only.

        Parameters
        ----------
        prepared_state : Optional[FMMPreparedState]
            State to refresh, or ``None`` to prepare one.
        positions : Float[Array, 'n 3']
            Particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``. Defaults to 4 here against 2 in the engine;
            the public default is the one that applies.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.

        Returns
        -------
        tuple[FMMPreparedState, Array]
            ``(prepared_state, accelerations)``. Feed the state back in on the
            next call.
        """
        return self._impl.strict_prepare_refresh_and_evaluate(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_traversal=jit_traversal,
        )

    def strict_run_segmented(
        self: "FastMultipoleMethod",
        *,
        state: Any,
        masses: Float[Array, "n"],
        num_steps: int,
        refresh_every: int,
        segment_runner: Callable[[Any, Array, int], tuple[Any, Any]],
        positions_getter: Callable[[Any], Array],
        prepared_state: Optional[FMMPreparedState] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
        jit_traversal: Optional[bool] = True,
        rematerialize_fn: Optional[Callable[[Any], Any]] = None,
        collect_history: bool = False,
    ) -> tuple[Any, FMMPreparedState, Optional[list[Any]]]:
        """Run strict segmented refresh cadence with caller-provided segment kernel.

        The integrator-agnostic runner: it owns only the refresh cadence, and the
        caller supplies the stepping. ``num_steps`` is cut into segments of
        ``refresh_every`` (plus a tail), the prepared state is refreshed at each
        boundary, and ``segment_runner`` advances the caller's own state within a
        segment. Use :meth:`strict_run_v2` instead when the caller's integrator is
        velocity Verlet over a raw array. Large-N production profile only.

        Parameters
        ----------
        state : Any
            The caller's integrator state, opaque to this method: it is only
            passed to ``segment_runner`` and ``positions_getter``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``.
        num_steps : int
            Total steps. Must be positive.
        refresh_every : int
            Steps per segment. Must be positive.
        segment_runner : Callable[[Any, Array, int], tuple[Any, Any]]
            ``(state, accelerations, num_steps) -> (next_state, segment_output)``.
            The segment output is collected only under ``collect_history``.
        positions_getter : Callable[[Any], Array]
            Extracts ``[N, 3]`` positions from the caller's state, so the refresh
            knows where the particles are.
        prepared_state : Optional[FMMPreparedState]
            Existing state to refresh, or ``None`` to prepare on the first
            segment.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``. Defaults to 4 here against 2 in the engine.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.
        rematerialize_fn : Optional[Callable[[Any], Any]]
            Applied to the state between segments, for callers that must
            reconstruct device arrays across a refresh.
        collect_history : bool
            Accumulate each segment's output. Off by default because the history
            is retained on device.

        Returns
        -------
        tuple[Any, FMMPreparedState, Optional[list[Any]]]
            ``(final_state, prepared_state, history)``, where ``history`` is
            ``None`` unless ``collect_history``.
        """
        return self._impl.strict_run_segmented(
            state=state,
            masses=masses,
            num_steps=int(num_steps),
            refresh_every=int(refresh_every),
            segment_runner=segment_runner,
            positions_getter=positions_getter,
            prepared_state=prepared_state,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            jit_traversal=jit_traversal,
            rematerialize_fn=rematerialize_fn,
            collect_history=bool(collect_history),
        )

    def strict_run_v2(
        self: "FastMultipoleMethod",
        *,
        state: Float[Array, "n 2 3"],
        masses: Float[Array, "n"],
        dt: float,
        num_steps: int,
        refresh_every: int,
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
        prepared_state: Optional[FMMPreparedState] = None,
        initial_self_acceleration: Optional[Float[Array, "n 3"]] = None,
        jit_traversal: Optional[bool] = True,
        add_external: bool = False,
        external_acceleration_fn: Optional[Callable[[Array], Array]] = None,
        rematerialize_between_refresh: bool = True,
        return_history: bool = False,
        return_prepared_state: bool = True,
        step_callback: Optional[Callable[[Array, Array], None]] = None,
        step_callback_stride: int = 1,
    ) -> tuple[Array, Optional[FMMPreparedState], Optional[Array]]:
        """Strict V2 segmented runner with raw tensor API.

        ``step_callback``/``step_callback_stride`` add an optional fire-and-forget
        streaming hook inside the device-resident scan (see the impl docstring):
        ``step_callback(step_index, state)`` every ``step_callback_stride`` steps,
        for minimal-sync rendering via ``jax.debug.callback``.

        Endpoint-correct velocity Verlet, run device-resident. Unlike
        :meth:`strict_run_segmented` the integrator is fixed and the state is a
        raw array, which is what lets the whole loop live inside one scan.
        Large-N production profile only, and ``refresh_every`` must be 1 --
        endpoint correctness needs the self-gravity refreshed every step.

        Parameters
        ----------
        state : Float[Array, 'n 2 3']
            Packed integrator state ``[N, 2, 3]``: positions and velocities
            stacked on axis 1.
        masses : Float[Array, 'n']
            Particle masses ``[N]``.
        dt : float
            Timestep.
        num_steps : int
            Total steps. Must be positive.
        refresh_every : int
            Must be 1; see above.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.
        prepared_state : Optional[FMMPreparedState]
            Existing state to refresh, or ``None`` to prepare.
        initial_self_acceleration : Optional[Float[Array, 'n 3']]
            Self-gravity at step 0 ``[N, 3]``, if already known. ``None``
            evaluates it, costing one extra evaluation.
        jit_traversal : Optional[bool]
            Per-call override of jitted traversal.
        add_external : bool
            Add ``external_acceleration_fn`` to the self-gravity each step.
        external_acceleration_fn : Optional[Callable[[Array], Array]]
            ``positions -> accelerations``; traced into the scan, so it must be
            jittable.
        rematerialize_between_refresh : bool
            Rebuild device arrays at refresh boundaries.
        return_history : bool
            Return every step's state rather than only the last.
        return_prepared_state : bool
            Return the prepared state, so the next call can reuse it.
        step_callback : Optional[Callable[[Array, Array], None]]
            Fire-and-forget streaming hook; see above.
        step_callback_stride : int
            Steps between ``step_callback`` invocations.

        Returns
        -------
        tuple[Array, Optional[FMMPreparedState], Optional[Array]]
            ``(final_state, prepared_state, history)``. The second is ``None``
            unless ``return_prepared_state``, the third ``None`` unless
            ``return_history``.
        """
        return self._impl.strict_run_v2(
            state=state,
            masses=masses,
            dt=float(dt),
            num_steps=int(num_steps),
            refresh_every=int(refresh_every),
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
            prepared_state=prepared_state,
            initial_self_acceleration=initial_self_acceleration,
            jit_traversal=jit_traversal,
            add_external=bool(add_external),
            external_acceleration_fn=external_acceleration_fn,
            rematerialize_between_refresh=bool(rematerialize_between_refresh),
            return_history=bool(return_history),
            return_prepared_state=bool(return_prepared_state),
            step_callback=step_callback,
            step_callback_stride=int(step_callback_stride),
        )

    def strict_fused_prepared_eval_fn(
        self: "FastMultipoleMethod",
        *,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        leaf_size: int,
        max_order: int,
        theta: Optional[float] = None,
    ) -> tuple[FMMPreparedState, Callable[[FMMPreparedState], Array]]:
        """Fused-lane eval-only closure for apples-to-apples eval benchmarking.

        The prepared state is built with the fused device-mode layout, and
        ``eval_fn`` runs the fused self-force evaluation with no refresh and no
        velocity-Verlet update -- which is the point: it isolates evaluation cost
        from the refresh and integration that :meth:`strict_run_v2` bundles with
        it. A benchmarking seam, not a simulation entry point.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            Particle masses ``[N]``.
        leaf_size : int
            Target maximum particles per leaf.
        max_order : int
            Expansion order ``p``.
        theta : Optional[float]
            Per-call MAC opening-angle override.

        Returns
        -------
        tuple[FMMPreparedState, Callable[[FMMPreparedState], Array]]
            ``(prepared_state, eval_fn)``. Call ``eval_fn(prepared_state)``
            repeatedly to time evaluation alone.
        """
        return self._impl.strict_fused_prepared_eval_fn(
            positions=positions,
            masses=masses,
            leaf_size=int(leaf_size),
            max_order=int(max_order),
            theta=theta,
        )

    def update_multipoles_only(
        self: "FastMultipoleMethod",
        prepared_state: FMMPreparedState,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> FMMPreparedState:
        """Update multipole/local payloads when topology mapping is unchanged.

        A :meth:`refresh_prepared_state` under a different name and counter, for
        the case where the caller knows the tree mapping still holds. It does not
        take ``bounds``, since changing the domain would change the mapping.
        Large-N production profile only.

        Parameters
        ----------
        prepared_state : FMMPreparedState
            State whose payloads are refreshed.
        positions : Float[Array, 'n 3']
            New particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            New particle masses ``[N]``.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.

        Returns
        -------
        FMMPreparedState
            The refreshed state.
        """
        return self._impl.update_multipoles_only(
            prepared_state,
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )

    def rebuild_topology_in_place(
        self: "FastMultipoleMethod",
        prepared_state: FMMPreparedState,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        bounds: Optional[Tuple[Float[Array, "3"], Float[Array, "3"]]] = None,
        leaf_size: Optional[int] = None,
        max_order: Optional[int] = None,
        theta: Optional[float] = None,
    ) -> FMMPreparedState:
        """Rebuild topology while tracking profile compatibility diagnostics.

        The third face of :meth:`refresh_prepared_state`: same call, its own
        counter, and ``bounds`` forwarded because a rebuild may legitimately
        change the domain. "In place" names the intent to stay within the
        compiled profile's capacities, not mutation -- a new state is returned.
        Large-N production profile only.

        Parameters
        ----------
        prepared_state : FMMPreparedState
            State whose topology is rebuilt.
        positions : Float[Array, 'n 3']
            New particle positions ``[N, 3]``.
        masses : Float[Array, 'n']
            New particle masses ``[N]``.
        bounds : Optional[Tuple[Float[Array, '3'], Float[Array, '3']]]
            Explicit ``(lower, upper)`` domain bounds.
        leaf_size : Optional[int]
            Leaf target; ``None`` keeps the state's own.
        max_order : Optional[int]
            Expansion order; ``None`` keeps the state's own.
        theta : Optional[float]
            Opening angle; ``None`` keeps the state's own.

        Returns
        -------
        FMMPreparedState
            The rebuilt state.
        """
        return self._impl.rebuild_topology_in_place(
            prepared_state,
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
        )

    def get_runtime_diagnostics(self: "FastMultipoleMethod") -> dict[str, Any]:
        """Return runtime diagnostics for compile/profile reuse benchmarking.

        Returns
        -------
        dict[str, Any]
            Counters the engine has accumulated -- compile counts, profile-key
            hits and misses, per-entry-point call tallies. A snapshot for
            benchmarking and tests, not a stable schema.
        """
        return self._impl.get_runtime_diagnostics()

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        """Clear cached prepared states in the runtime backend."""
        self._impl.clear_prepared_state_cache()

    def clear_runtime_caches(
        self: "FastMultipoleMethod", *, clear_jax_compilation: bool = False
    ) -> None:
        """Clear solver/runtime caches; optionally clear JAX compilation caches.

        Parameters
        ----------
        clear_jax_compilation : bool
            Also call ``jax.clear_caches()``. Off by default because it discards
            compilation work belonging to the whole process, not just this
            solver, and does not shrink XLA's allocation arena.
        """
        self._impl.clear_runtime_caches(
            clear_jax_compilation=bool(clear_jax_compilation)
        )

    def export_m2l_autotune_cache(self: "FastMultipoleMethod") -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of M2L chunk autotune results.

        The cache is process-global, not per-solver, so this exports what every
        solver in the process has autotuned. Autotuning costs real time on first
        use, which is why it is worth persisting across runs.

        Returns
        -------
        list[dict[str, Any]]
            One entry per autotuned configuration.
        """

        return self._impl.export_m2l_autotune_cache()

    def import_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        payload: list[dict[str, Any]],
        *,
        merge: bool = True,
    ) -> int:
        """Restore M2L chunk autotune results from serialized payload.

        Parameters
        ----------
        payload : list[dict[str, Any]]
            Entries as produced by :meth:`export_m2l_autotune_cache`.
        merge : bool
            Merge into the process-global cache (default) rather than clearing it
            first.

        Returns
        -------
        int
            Number of entries restored. Malformed entries are skipped silently,
            so a result below ``len(payload)`` is the only signal that the
            payload was not fully understood.
        """

        return int(self._impl.import_m2l_autotune_cache(payload, merge=bool(merge)))

    def save_m2l_autotune_cache(self: "FastMultipoleMethod", path: str) -> int:
        """Write M2L chunk autotune results to a JSON file.

        Parameters
        ----------
        path : str
            Destination file, overwritten if it exists.

        Returns
        -------
        int
            Number of entries written.
        """

        return int(self._impl.save_m2l_autotune_cache(path))

    def load_m2l_autotune_cache(
        self: "FastMultipoleMethod",
        path: str,
        *,
        merge: bool = True,
    ) -> int:
        """Load M2L chunk autotune results from a JSON file.

        Parameters
        ----------
        path : str
            Source file, as written by :meth:`save_m2l_autotune_cache`. Its
            top-level JSON value must be a list.
        merge : bool
            Merge into the process-global cache (default) rather than clearing it
            first.

        Returns
        -------
        int
            Number of entries restored; see :meth:`import_m2l_autotune_cache`.
        """

        return int(self._impl.load_m2l_autotune_cache(path, merge=bool(merge)))

    @property
    def recent_topology_reused(self: "FastMultipoleMethod") -> bool:
        """Whether the latest prepare/evaluate path reused cached topology."""

        return bool(getattr(self._impl, "_recent_topology_reused", False))

    @property
    def complex_rotation(self: "FastMultipoleMethod") -> str:
        """Active complex-rotation backend identifier."""
        return str(self._impl.complex_rotation)

    @property
    def mac_type(self: "FastMultipoleMethod") -> str:
        """Active multipole-acceptance criterion policy."""
        return str(self._impl.mac_type)

    @property
    def farfield_mode(self: "FastMultipoleMethod") -> str:
        """Resolved far-field interaction mode."""
        return str(self._impl.farfield_mode)

    @property
    def nearfield_mode(self: "FastMultipoleMethod") -> str:
        """Resolved near-field interaction mode."""
        return str(self._impl.nearfield_mode)

    @property
    def nearfield_edge_chunk_size(self: "FastMultipoleMethod") -> int:
        """Chunk size used by bucketed near-field edge processing."""
        return int(self._impl.nearfield_edge_chunk_size)

    @property
    def grouped_interactions(self: "FastMultipoleMethod") -> bool:
        """Whether grouped interaction traversal is enabled."""
        return bool(self._impl.grouped_interactions)

    @grouped_interactions.setter
    def grouped_interactions(self: "FastMultipoleMethod", value: bool) -> None:
        """Set grouped-interaction mode and mirror it into advanced config.

        Writing this marks the choice explicit in the engine, so later automatic
        resolution will not override it, and mirrors it back into ``advanced`` so
        the config keeps agreeing with the engine.

        Parameters
        ----------
        value : bool
            Whether to group interactions during traversal.
        """
        next_value = bool(value)
        self._impl.grouped_interactions = next_value
        if hasattr(self._impl, "_explicit_grouped_interactions"):
            self._impl._explicit_grouped_interactions = True
        self.advanced = replace(
            self.advanced,
            farfield=replace(
                self.advanced.farfield,
                grouped_interactions=next_value,
            ),
        )


__all__ = [
    "FastMultipoleMethod",
]
