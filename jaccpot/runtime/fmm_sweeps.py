"""SweepsMixin: fmm_sweeps methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, jaxtyped
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.geometry import compute_tree_geometry
from yggdrax.grouped_interactions import GroupedInteractionBuffers
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
)
from yggdrax.tree import Tree, get_node_levels
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
    TreeDownwardData,
)
from jaccpot.downward.local_expansions import (
    prepare_downward_sweep as prepare_tree_downward_sweep,
)
from jaccpot.downward.local_expansions import (
    run_downward_sweep as run_tree_downward_sweep,
)
from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep
from jaccpot.upward.solidfmm_complex_tree_expansions import (
    prepare_solidfmm_complex_upward_sweep,
)
from jaccpot.upward.tree_expansions import (
    NodeMultipoleData,
    TreeUpwardData,
)
from jaccpot.upward.tree_expansions import (
    prepare_upward_sweep as prepare_tree_upward_sweep,
)

from .kernels.core import _FarPairCOO, _prepare_solidfmm_downward_sweep
from .reference import MultipoleExpansion
from .reference import compute_expansion as reference_compute_expansion
from .reference import direct_sum as reference_direct_sum
from .reference import evaluate_expansion as reference_evaluate_expansion


class SweepsMixin:
    @staticmethod
    @jaxtyped(typechecker=beartype)
    def compute_expansion(
        positions: Array,
        masses: Array,
        order: int = 1,
    ) -> MultipoleExpansion:
        """Return the multipole expansion via the shared reference helper."""

        return reference_compute_expansion(positions, masses, order=order)

    @jaxtyped(typechecker=beartype)
    def evaluate_expansion(
        self: "FastMultipoleMethod",
        expansion: MultipoleExpansion,
        order: int = 1,
        eval_point: Optional[Array] = None,
    ) -> Array:
        """Evaluate multipole expansions via the shared reference helper."""

        return reference_evaluate_expansion(
            expansion,
            order=order,
            eval_point=eval_point,
            G=self.G,
            softening=self.softening,
        )

    @jaxtyped(typechecker=beartype)
    def direct_sum(
        self: "FastMultipoleMethod",
        positions: Array,
        masses: Array,
        eval_point: Array,
        eval_mass: float = 0.0,
    ) -> Array:
        """Compute direct-sum accelerations for diagnostic comparisons."""

        _ = eval_mass  # Unused but preserved for backwards compatibility.
        return reference_direct_sum(
            positions,
            masses,
            eval_point,
            G=self.G,
            softening=self.softening,
        )

    def _resolve_upward_num_levels(self, tree: Tree) -> Optional[int]:
        """Return the concrete (unpadded) tree depth for the M2M level loop.

        When ``tree`` is concrete (full prepare / template build) this computes
        the actual depth and stashes it on ``self``. When ``tree`` is a JAX
        tracer (the fused device-resident refresh) its array values are
        unavailable, so we return the previously stashed value. The stash is
        populated by the eager full-prepare that always precedes the traced
        refresh, so the hot path gets the concrete depth. Returns ``None`` only
        if nothing concrete has been seen yet (callers then fall back to the
        padded shape-derived depth, which is correct, just slower)."""
        probe = getattr(tree, "parent", None)
        if probe is None or isinstance(probe, jax.core.Tracer):
            return self._static_upward_num_levels
        try:
            levels = get_node_levels(tree)
            if isinstance(levels, jax.core.Tracer):
                return self._static_upward_num_levels
            self._static_upward_num_levels = int(jnp.max(levels)) + 1
        except Exception:
            return self._static_upward_num_levels
        return self._static_upward_num_levels

    def prepare_upward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        positions_sorted: Array,
        masses_sorted: Array,
        *,
        max_order: int = 2,
        center_mode: str = "com",
        explicit_centers: Optional[Array] = None,
        max_leaf_size: Optional[int] = None,
        precomputed_geometry: Optional[Any] = None,
        defer_geometry: bool = False,
    ) -> TreeUpwardData:
        """Bundle geometry, raw moments, and packed expansions for a tree."""
        self._ensure_execution_backend_supported(tree=tree)

        if self.expansion_basis == "solidfmm":
            if self._solidfmm_basis_mode() == "real":
                # Native real (Dehnen no-sqrt2) upward sweep: P2M + M2M produce real
                # multipoles directly -- NO complex intermediate, NO complex<->real
                # conversion (the "real everywhere / never convert bases" contract).
                # prepare_real_upward_sweep uses COM centers (what the large-N fast
                # lane uses) and is bit-equivalent to the complex sweep +
                # complex_to_dehnen_real_coeffs it replaces.
                center_mode_norm = str(center_mode).strip().lower()
                if center_mode_norm != "com":
                    raise ValueError(
                        "native real-basis upward sweep supports center_mode='com' "
                        f"only (got '{center_mode}'); the large-N fast lane uses COM"
                    )
                resolved_leaf_cap = max_leaf_size
                if resolved_leaf_cap is None:
                    num_internal_real = int(jnp.asarray(tree.left_child).shape[0])
                    leaf_ranges_real = jax.device_get(tree.node_ranges)[
                        num_internal_real:
                    ]
                    if leaf_ranges_real.shape[0] == 0:
                        resolved_leaf_cap = 0
                    else:
                        counts_real = (
                            leaf_ranges_real[:, 1] - leaf_ranges_real[:, 0] + 1
                        )
                        resolved_leaf_cap = int(counts_real.max())
                geometry_real = (
                    precomputed_geometry
                    if precomputed_geometry is not None
                    else (
                        None
                        if bool(defer_geometry)
                        else compute_tree_geometry(
                            tree,
                            positions_sorted,
                            max_leaf_size=int(resolved_leaf_cap),
                        )
                    )
                )
                mass_moments_real = compute_tree_mass_moments(
                    tree, positions_sorted, masses_sorted
                )
                real_upward = prepare_real_upward_sweep(
                    tree,
                    positions_sorted,
                    masses_sorted,
                    max_order=max_order,
                    max_leaf_size=int(resolved_leaf_cap),
                    leaf_batch_size=self.upward_leaf_batch_size,
                    static_num_levels=self._resolve_upward_num_levels(tree),
                )
                real_multipoles = NodeMultipoleData(
                    order=int(real_upward.multipoles.order),
                    centers=real_upward.multipoles.centers,
                    moments=None,  # type: ignore[arg-type]
                    packed=real_upward.multipoles.packed,
                    component_matrix=None,
                    source_motion_packed=None,
                )
                return TreeUpwardData(
                    geometry=geometry_real,
                    mass_moments=mass_moments_real,
                    multipoles=real_multipoles,
                )

            def _record_upward_stage(name: str, elapsed: float) -> None:
                if not bool(getattr(self, "_refresh_timing_active", False)):
                    return
                attr_by_name = {
                    "geometry": "_refresh_timing_upward_geometry_seconds",
                    "mass_moments": "_refresh_timing_upward_mass_moments_seconds",
                    "p2m": "_refresh_timing_upward_p2m_seconds",
                    "m2m": "_refresh_timing_upward_m2m_seconds",
                    "source_motion": "_refresh_timing_upward_source_motion_seconds",
                }
                attr = attr_by_name.get(str(name))
                if attr is None:
                    return
                setattr(self, attr, float(getattr(self, attr, 0.0)) + float(elapsed))

            complex_upward = prepare_solidfmm_complex_upward_sweep(
                tree,
                positions_sorted,
                masses_sorted,
                max_order=max_order,
                center_mode=center_mode,
                explicit_centers=explicit_centers,
                max_leaf_size=max_leaf_size,
                leaf_batch_size=self.upward_leaf_batch_size,
                rotation=self.complex_rotation,
                precomputed_geometry=precomputed_geometry,
                upward_timing_callback=_record_upward_stage,
                defer_geometry=bool(defer_geometry),
                static_num_levels=self._resolve_upward_num_levels(tree),
            )

            multipoles = NodeMultipoleData(
                order=int(complex_upward.multipoles.order),
                centers=complex_upward.multipoles.centers,
                moments=None,  # type: ignore[arg-type]
                packed=complex_upward.multipoles.packed,
                component_matrix=None,
                source_motion_packed=complex_upward.multipoles.source_motion_packed,
            )

            return TreeUpwardData(
                geometry=complex_upward.geometry,
                mass_moments=complex_upward.mass_moments,
                multipoles=multipoles,
            )

        return prepare_tree_upward_sweep(
            tree,
            positions_sorted,
            masses_sorted,
            max_order=max_order,
            center_mode=center_mode,
            explicit_centers=explicit_centers,
            precomputed_geometry=precomputed_geometry,
        )

    def run_downward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        multipoles: NodeMultipoleData,
        interactions: NodeInteractionList,
        *,
        initial_locals: Optional[LocalExpansionData] = None,
        m2l_chunk_size: Optional[int] = None,
        dense_buffers: Optional[DenseInteractionBuffers] = None,
    ) -> LocalExpansionData:
        """Execute an M2L+L2L pass for the provided multipoles."""

        return run_tree_downward_sweep(
            tree,
            multipoles,
            interactions,
            initial_locals=initial_locals,
            m2l_chunk_size=m2l_chunk_size,
            dense_buffers=dense_buffers,
        )

    def prepare_downward_sweep(
        self: "FastMultipoleMethod",
        tree: Tree,
        upward_data: TreeUpwardData,
        *,
        theta: Optional[float] = None,
        mac_type: Optional[MACType] = None,
        initial_locals: Optional[LocalExpansionData] = None,
        interactions: Optional[NodeInteractionList] = None,
        m2l_chunk_size: Optional[int] = None,
        l2l_chunk_size: Optional[int] = None,
        traversal_config: Optional[DualTreeTraversalConfig] = None,
        dense_buffers: Optional[DenseInteractionBuffers] = None,
        retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
        grouped_interactions: bool = False,
        grouped_buffers: Optional[GroupedInteractionBuffers] = None,
        grouped_segment_starts: Optional[Array] = None,
        grouped_segment_lengths: Optional[Array] = None,
        grouped_segment_class_ids: Optional[Array] = None,
        grouped_segment_sort_permutation: Optional[Array] = None,
        grouped_segment_group_ids: Optional[Array] = None,
        grouped_segment_unique_targets: Optional[Array] = None,
        farfield_mode: str = "pair_grouped",
        dehnen_radius_scale: Optional[float] = None,
        far_pairs_coo: Optional[_FarPairCOO] = None,
        far_pairs_by_gear: Optional[tuple[tuple[Array, Array], ...]] = None,
        adaptive_order: Optional[bool] = None,
        p_gears: Optional[tuple[int, ...]] = None,
    ) -> TreeDownwardData:
        """Build interactions and locals needed for the downward sweep."""
        self._ensure_execution_backend_supported(tree=tree)

        theta_val = float(self.theta if theta is None else theta)
        mac_type_val = self.mac_type if mac_type is None else mac_type
        dehnen_scale_val = float(
            self.dehnen_radius_scale
            if dehnen_radius_scale is None
            else dehnen_radius_scale
        )
        config = traversal_config if traversal_config is not None else None
        if config is None:
            config = self.traversal_config
        retry_callback = (
            retry_logger if retry_logger is not None else self.interaction_retry_logger
        )
        if self.expansion_basis == "solidfmm":
            adaptive_order_val = (
                self.adaptive_order if adaptive_order is None else bool(adaptive_order)
            )
            p_gears_val = (
                self.p_gears if p_gears is None else tuple(int(v) for v in p_gears)
            )
            timing_recorder = None
            sync_substage_timing = str(
                os.environ.get("JACCPOT_REFRESH_TIMING_SYNC_SUBSTAGES", "0")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if bool(getattr(self, "_refresh_timing_active", False)) and bool(
                sync_substage_timing
            ):

                def timing_recorder(attr: str, elapsed: float) -> None:
                    setattr(self, attr, float(getattr(self, attr, 0.0)) + elapsed)

            return _prepare_solidfmm_downward_sweep(
                tree,
                upward_data,
                theta=theta_val,
                mac_type=mac_type_val,
                initial_locals=initial_locals,
                interactions=interactions,
                m2l_chunk_size=m2l_chunk_size,
                l2l_chunk_size=l2l_chunk_size,
                complex_rotation=self.complex_rotation,
                basis_mode=self._solidfmm_basis_mode(),
                m2l_impl=self.m2l_impl,
                retry_logger=retry_callback,
                traversal_config=config,
                dense_buffers=dense_buffers,
                grouped_interactions=grouped_interactions,
                grouped_buffers=grouped_buffers,
                grouped_segment_starts=grouped_segment_starts,
                grouped_segment_lengths=grouped_segment_lengths,
                grouped_segment_class_ids=grouped_segment_class_ids,
                grouped_segment_sort_permutation=grouped_segment_sort_permutation,
                grouped_segment_group_ids=grouped_segment_group_ids,
                grouped_segment_unique_targets=grouped_segment_unique_targets,
                farfield_mode=farfield_mode,
                far_pairs_coo=far_pairs_coo,
                far_pairs_by_gear=far_pairs_by_gear,
                adaptive_order=adaptive_order_val,
                p_gears=p_gears_val,
                dehnen_radius_scale=dehnen_scale_val,
                use_pallas=self.use_pallas,
                timing_recorder=timing_recorder,
            )

        return prepare_tree_downward_sweep(
            tree,
            upward_data,
            theta=theta_val,
            mac_type=mac_type_val,
            initial_locals=initial_locals,
            interactions=interactions,
            m2l_chunk_size=m2l_chunk_size,
            retry_logger=retry_callback,
            traversal_config=config,
            max_pair_queue=self.max_pair_queue,
            process_block=self.pair_process_block,
            dense_buffers=dense_buffers,
        )
