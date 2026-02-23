"""Preset-first FMM solver facade for Jaccpot."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple, Union
import warnings

from jaxtyping import Array, DTypeLike

from .config import (
    Basis,
    FMMAdvancedConfig,
    FMMPreset,
)
from .runtime.fmm import FMMPreparedState
from .runtime.fmm import FastMultipoleMethod as _RuntimeFMM


def _default_advanced_for_preset(preset: FMMPreset) -> FMMAdvancedConfig:
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
    if isinstance(preset, FMMPreset):
        return preset
    return FMMPreset(str(preset).strip().lower())


def _pop_legacy_common_overrides(
    *,
    basis: Basis,
    theta: float,
    G: float,
    softening: float,
    working_dtype: Optional[DTypeLike],
    legacy_kwargs: dict[str, Any],
) -> tuple[Basis, float, float, float, Optional[DTypeLike], bool]:
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


class FastMultipoleMethod:
    """Simplified, preset-first high-level FMM API."""

    def __init__(
        self,
        *,
        preset: Union[FMMPreset, str] = FMMPreset.FAST,
        basis: Basis = "solidfmm",
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

        preset_norm = _normalize_preset(preset)
        advanced_cfg = (
            _default_advanced_for_preset(preset_norm) if advanced is None else advanced
        )

        complex_rotation = advanced_cfg.farfield.rotation
        if complex_rotation is None:
            complex_rotation = "solidfmm" if basis == "solidfmm" else "cached"
        legacy_rotation = legacy_kwargs.pop("complex_rotation", None)
        if legacy_rotation is not None:
            complex_rotation = str(legacy_rotation)
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

        traversal_config = advanced_cfg.runtime.traversal_config
        self._impl = _RuntimeFMM(
            preset=expanse_preset,
            theta=float(theta),
            G=float(G),
            softening=float(softening),
            working_dtype=working_dtype,
            expansion_basis=basis,
            complex_rotation=complex_rotation,
            tree_build_mode=tree_mode,
            target_leaf_particles=target_leaf_particles,
            refine_local=legacy_kwargs.pop("refine_local", advanced_cfg.tree.refine_local),
            max_refine_levels=legacy_kwargs.pop(
                "max_refine_levels",
                advanced_cfg.tree.max_refine_levels,
            ),
            aspect_threshold=legacy_kwargs.pop(
                "aspect_threshold",
                advanced_cfg.tree.aspect_threshold,
            ),
            grouped_interactions=grouped_interactions,
            farfield_mode=farfield_mode,
            m2l_chunk_size=legacy_kwargs.pop(
                "m2l_chunk_size",
                advanced_cfg.farfield.m2l_chunk_size,
            ),
            l2l_chunk_size=legacy_kwargs.pop(
                "l2l_chunk_size",
                advanced_cfg.farfield.l2l_chunk_size,
            ),
            nearfield_mode=nearfield_mode,
            nearfield_edge_chunk_size=nearfield_edge_chunk_size,
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
                legacy_kwargs.pop("dehnen_radius_scale", advanced_cfg.dehnen_radius_scale)
            ),
            use_dense_interactions=legacy_kwargs.pop("use_dense_interactions", None),
            traversal_config=legacy_kwargs.pop("traversal_config", traversal_config),
            fixed_order=fixed_order,
            fixed_max_leaf_size=fixed_max_leaf_size,
            mac_type=mac_type,
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
        self.basis = basis
        self.advanced = advanced_cfg

    def compute_accelerations(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        *,
        bounds: Optional[Tuple[Array, Array]] = None,
        leaf_size: int = 16,
        max_order: int = 4,
        return_potential: bool = False,
        theta: Optional[float] = None,
        reuse_prepared_state: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        return self._impl.compute_accelerations(
            positions,
            masses,
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
        return self._impl.prepare_state(
            positions,
            masses,
            bounds=bounds,
            leaf_size=leaf_size,
            max_order=max_order,
            theta=theta,
            jit_tree=self.advanced.runtime.jit_tree,
        )

    def prepare_upward_sweep(
        self: "FastMultipoleMethod",
        tree: Any,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        max_order: int = 4,
    ) -> Any:
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
        return_potential: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        jit_traversal = (
            True
            if self.advanced.runtime.jit_traversal is None
            else bool(self.advanced.runtime.jit_traversal)
        )
        return self._impl.evaluate_prepared_state(
            state,
            return_potential=return_potential,
            jit_traversal=jit_traversal,
        )

    def clear_prepared_state_cache(self: "FastMultipoleMethod") -> None:
        self._impl.clear_prepared_state_cache()

    @property
    def complex_rotation(self: "FastMultipoleMethod") -> str:
        return str(self._impl.complex_rotation)

    @property
    def mac_type(self: "FastMultipoleMethod") -> str:
        return str(self._impl.mac_type)

    @property
    def farfield_mode(self: "FastMultipoleMethod") -> str:
        return str(self._impl.farfield_mode)

    @property
    def nearfield_mode(self: "FastMultipoleMethod") -> str:
        return str(self._impl.nearfield_mode)

    @property
    def nearfield_edge_chunk_size(self: "FastMultipoleMethod") -> int:
        return int(self._impl.nearfield_edge_chunk_size)

    @property
    def grouped_interactions(self: "FastMultipoleMethod") -> bool:
        return bool(self._impl.grouped_interactions)

    @grouped_interactions.setter
    def grouped_interactions(self: "FastMultipoleMethod", value: bool) -> None:
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
