"""Preset-first FMM solver facade for Jaccpot."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

from jaxtyping import Array, DTypeLike

from .basis import BasisInterface, ComplexSHBasis, RealSHBasis
from .config import (
    Basis,
    FMMAdvancedConfig,
    FMMPreset,
)
from .runtime.fmm import FastMultipoleMethod as _RuntimeFMM
from .runtime.fmm import FMMPreparedState


def _default_advanced_for_preset(preset: FMMPreset) -> FMMAdvancedConfig:
    """Return default advanced overrides for a high-level preset."""
    if preset is FMMPreset.FAST:
        return FMMAdvancedConfig()
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
        nearfield=replace(cfg.nearfield, mode="baseline"),
        runtime=replace(cfg.runtime, host_refine_mode="on"),
    )


def _normalize_preset(preset: Union[FMMPreset, str]) -> FMMPreset:
    """Normalize user preset input to :class:`FMMPreset`."""
    if isinstance(preset, FMMPreset):
        return preset
    return FMMPreset(str(preset).strip().lower())


class _BasisResolution(NamedTuple):
    """Resolved basis routing for public API and runtime backend."""

    public_name: str
    runtime_basis: Basis
    basis_impl: Optional[BasisInterface]


def _resolve_basis_input(basis: Union[Basis, BasisInterface, str]) -> _BasisResolution:
    """Normalize basis string/object to runtime expansion basis + metadata."""
    if isinstance(basis, str):
        basis_norm = basis.strip().lower()
        if basis_norm in ("solidfmm", "complex"):
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
    """Consume legacy constructor kwargs and map them to modern arguments."""
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
    complex_rotation: str
    tree_type: Optional[str]
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
    """Resolve runtime-facing legacy kwargs while preserving old behavior."""
    complex_rotation = advanced_cfg.farfield.rotation
    if complex_rotation is None:
        complex_rotation = "solidfmm" if basis == "solidfmm" else "cached"
    legacy_rotation = legacy_kwargs.pop("complex_rotation", None)
    if legacy_rotation is not None:
        complex_rotation = str(legacy_rotation)
        legacy_used = True

    tree_type = advanced_cfg.tree.tree_type
    legacy_tree_type = legacy_kwargs.pop("tree_type", None)
    if legacy_tree_type is not None:
        tree_type = str(legacy_tree_type)
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

    expanse_preset = "fast" if preset_norm is FMMPreset.FAST else None
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
    """Simplified, preset-first high-level FMM API."""

    def __init__(
        self,
        *,
        preset: Union[FMMPreset, str] = FMMPreset.FAST,
        basis: Union[Basis, BasisInterface, str] = "complex",
        m2l_impl: Optional[str] = None,
        adaptive_order: bool = False,
        p_gears: Optional[Sequence[int]] = None,
        use_pallas: bool = False,
        reuse_topology: bool = False,
        rebuild_every: int = 1,
        mac_force_scale_mode: str = "prev",
        theta: float = 0.6,
        G: float = 1.0,
        softening: float = 1e-3,
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
        basis_resolution = _resolve_basis_input(basis)
        runtime_basis = basis_resolution.runtime_basis
        resolved_m2l_impl = m2l_impl
        if resolved_m2l_impl is None and basis_resolution.public_name == "real":
            resolved_m2l_impl = "rot_scale"

        preset_norm = _normalize_preset(preset)
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
            adaptive_order=adaptive_order,
            p_gears=p_gears,
            use_pallas=use_pallas,
            reuse_topology=reuse_topology,
            rebuild_every=rebuild_every,
            mac_force_scale_mode=mac_force_scale_mode,
            complex_rotation=runtime_overrides.complex_rotation,
            tree_type=runtime_overrides.tree_type or "radix",
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
            farfield_mode=runtime_overrides.farfield_mode,
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
            host_refine_mode=legacy_kwargs.pop(
                "host_refine_mode",
                advanced_cfg.runtime.host_refine_mode,
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

    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        target_indices: Optional[Array] = None,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        return_potential: bool = False,
        theta: Optional[float] = None,
        reuse_prepared_state: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Compute accelerations (and optional potentials) for particle data.

        When ``target_indices`` is provided, all particles remain source masses
        but outputs are returned only for the indexed target particles.
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
        )

    def prepare_state(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        theta: Optional[float] = None,
    ) -> FMMPreparedState:
        """Prepare and cache tree/interactions for repeated evaluations."""
        return self._impl.prepare_state(
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            jit_tree=self.advanced.runtime.jit_tree,
        )

    def build_dehnen_error_node_features(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        *,
        p_gears: Optional[Sequence[int]] = None,
        force_scale_nodes: Optional[Array] = None,
    ) -> dict[str, Array]:
        """Build yggdrax node features for ``mac_type='dehnen_error'``.

        When ``force_scale_nodes`` is omitted, the helper uses the values stored
        in ``state`` and falls back to unit scales if the prepared state does not
        carry a previous estimate.
        """

        gears = (
            self._impl.p_gears if p_gears is None else tuple(int(v) for v in p_gears)
        )
        return self._impl.build_dehnen_error_node_features(
            upward=state.upward,
            p_gears=gears,
            force_scale_nodes=(
                state.force_scale_nodes
                if force_scale_nodes is None
                else force_scale_nodes
            ),
        )

    def prepare_upward_sweep(
        self: "FastMultipoleMethod",
        tree: Any,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        max_order: int = 4,
    ) -> Any:
        """Build upward multipole data for a prebuilt tree."""
        return self._impl.prepare_upward_sweep(
            tree,
            positions_sorted,
            masses_sorted,
            max_order=max_order,
        )

    def evaluate_prepared_state(
        self: "FastMultipoleMethod",
        state: FMMPreparedState,
        *,
        target_indices: Optional[Array] = None,
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Evaluate a previously prepared state for all particles or a subset."""
        jit_traversal = (
            True
            if self.advanced.runtime.jit_traversal is None
            else bool(self.advanced.runtime.jit_traversal)
        )
        return self._impl.evaluate_prepared_state(
            state,
            target_indices=target_indices,
            return_potential=return_potential,
            jit_traversal=jit_traversal,
        )

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        """Clear cached prepared states in the runtime backend."""
        self._impl.clear_prepared_state_cache()

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
        """Set grouped-interaction mode and mirror it into advanced config."""
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
