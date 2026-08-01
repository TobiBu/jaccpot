"""DiagnosticsMixin: fmm_diagnostics methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

from typing import Any


class DiagnosticsMixin:
    def _record_large_n_eval_shape_diagnostics(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
    ) -> None:
        """Record shape-only local-eval diagnostics outside compiled hot loops."""
        neighbor_list = getattr(state, "neighbor_list", None)
        leaf_nodes = (
            getattr(neighbor_list, "leaf_indices", None)
            if neighbor_list is not None
            else None
        )
        local_data = getattr(state, "local_data", None)
        coefficients = getattr(local_data, "coefficients", None)
        centers = getattr(local_data, "centers", None)
        leaf_particles = getattr(state, "nearfield_leaf_particle_indices", None)
        radix_payload = getattr(state, "radix_fast_payload", None)
        radix_source_particles = (
            getattr(radix_payload, "source_particle_ids", None)
            if radix_payload is not None
            else None
        )
        radix_source_leaves = (
            getattr(radix_payload, "source_leaf_ids", None)
            if radix_payload is not None
            else None
        )
        padded_source_leaves = getattr(
            state, "nearfield_target_block_source_leaf_ids_padded", None
        )

        def _shape_tuple(value: Any) -> tuple[int, ...]:
            shape = getattr(value, "shape", None)
            if shape is None:
                return ()
            return tuple(int(v) for v in shape)

        leaf_shape = _shape_tuple(leaf_nodes)
        leaf_particle_shape = _shape_tuple(leaf_particles)
        radix_source_particle_shape = _shape_tuple(radix_source_particles)
        radix_source_leaf_shape = _shape_tuple(radix_source_leaves)
        padded_source_leaf_shape = _shape_tuple(padded_source_leaves)
        self._large_n_eval_leaf_nodes_shape = leaf_shape
        self._large_n_eval_local_coefficients_shape = _shape_tuple(coefficients)
        self._large_n_eval_local_centers_shape = _shape_tuple(centers)
        self._large_n_eval_active_leaf_count = int(leaf_shape[0]) if leaf_shape else 0
        self._large_n_eval_max_leaf_size = int(getattr(state, "max_leaf_size", 0))
        self._large_n_eval_leaf_particle_slots = (
            int(leaf_particle_shape[0] * leaf_particle_shape[1])
            if len(leaf_particle_shape) >= 2
            else 0
        )
        self._large_n_radix_payload_present = radix_payload is not None
        self._large_n_radix_payload_source_particle_shape = radix_source_particle_shape
        self._large_n_radix_payload_source_particle_slots = (
            int(
                radix_source_particle_shape[0]
                * radix_source_particle_shape[1]
                * radix_source_particle_shape[2]
            )
            if len(radix_source_particle_shape) >= 3
            else 0
        )
        self._large_n_radix_payload_source_leaf_shape = radix_source_leaf_shape
        self._large_n_radix_payload_source_leaf_slots = (
            int(
                radix_source_leaf_shape[0]
                * radix_source_leaf_shape[1]
                * radix_source_leaf_shape[2]
            )
            if len(radix_source_leaf_shape) >= 3
            else 0
        )
        self._large_n_target_block_source_leaf_padded_shape = padded_source_leaf_shape

    def get_runtime_diagnostics(self: "FastMultipoleMethod") -> dict[str, Any]:
        """Return read-only runtime diagnostics for compile/profile reuse audits."""
        return {
            "compiled_profile_fingerprint_last": self._compiled_profile_fingerprint_last,
            "compiled_profile_transitions": int(self._compiled_profile_transitions),
            "refresh_prepare_calls": int(self._compiled_profile_refresh_calls),
            "refresh_prepare_reuse_tier_full": int(
                self._compiled_profile_refresh_reuse_tier_full
            ),
            "refresh_prepare_reuse_tier_topology": int(
                self._compiled_profile_refresh_reuse_tier_topology
            ),
            "refresh_prepare_reuse_tier_overflow": int(
                self._compiled_profile_refresh_reuse_tier_overflow
            ),
            "large_n_same_topology_refresh_attempts": int(
                self._large_n_same_topology_refresh_attempts
            ),
            "large_n_same_topology_refresh_hits": int(
                self._large_n_same_topology_refresh_hits
            ),
            "large_n_same_topology_refresh_misses": int(
                self._large_n_same_topology_refresh_misses
            ),
            "large_n_same_topology_refresh_miss_no_key": int(
                self._large_n_same_topology_refresh_miss_no_key
            ),
            "large_n_same_topology_refresh_miss_topology": int(
                self._large_n_same_topology_refresh_miss_topology
            ),
            "large_n_same_topology_refresh_miss_neighbor": int(
                self._large_n_same_topology_refresh_miss_neighbor
            ),
            "large_n_same_topology_refresh_miss_traced": int(
                self._large_n_same_topology_refresh_miss_traced
            ),
            "large_n_same_topology_refresh_last_error": str(
                self._large_n_same_topology_refresh_last_error
            ),
            "static_radix_refresh_hits": int(self._static_radix_refresh_hits),
            "static_radix_refresh_misses": int(self._static_radix_refresh_misses),
            "static_radix_profile_overflows": int(self._static_radix_profile_overflows),
            "static_radix_compact_pair_reuse_hits": int(
                self._static_radix_compact_pair_reuse_hits
            ),
            "static_radix_compact_pair_reuse_misses": int(
                self._static_radix_compact_pair_reuse_misses
            ),
            "update_multipoles_only_calls": int(
                self._compiled_profile_multipoles_only_calls
            ),
            "rebuild_topology_in_place_calls": int(
                self._compiled_profile_topology_rebuild_calls
            ),
            "large_n_overflow_profile_cap": int(self._large_n_overflow_profile_cap),
            "large_n_overflow_profile_reprofiles": int(
                self._large_n_overflow_profile_reprofiles
            ),
            "large_n_neighbor_edges_profile_cap": int(
                self._large_n_neighbor_edges_profile_cap
            ),
            "large_n_neighbor_edges_profile_reprofiles": int(
                self._large_n_neighbor_edges_profile_reprofiles
            ),
            "interaction_cache_hits": int(self._interaction_cache_hits),
            "interaction_cache_misses": int(self._interaction_cache_misses),
            "refresh_dual_planner_cache_hits": int(
                self._refresh_dual_planner_cache_hits
            ),
            "refresh_dual_planner_cache_misses": int(
                self._refresh_dual_planner_cache_misses
            ),
            "refresh_dual_planner_compile_count": int(
                self._refresh_dual_planner_compile_count
            ),
            "refresh_dual_planner_execute_count": int(
                self._refresh_dual_planner_execute_count
            ),
            "refresh_dual_planner_steady_timing_bypass_count": int(
                self._refresh_dual_planner_steady_timing_bypass_count
            ),
            "refresh_dual_planner_compiled_route_count": int(
                self._refresh_dual_planner_compiled_route_count
            ),
            "refresh_strict_mode_active_count": int(
                self._refresh_strict_mode_active_count
            ),
            "strict_runner_compile_count": int(self._strict_runner_compile_count),
            "strict_runner_execute_count": int(self._strict_runner_execute_count),
            "strict_runner_profile_key_hits": int(self._strict_runner_profile_key_hits),
            "strict_runner_profile_key_misses": int(
                self._strict_runner_profile_key_misses
            ),
            "strict_runner_fail_fast_reject_count": int(
                self._strict_runner_fail_fast_reject_count
            ),
            "strict_v2_compile_count": int(self._strict_v2_compile_count),
            "strict_v2_execute_count": int(self._strict_v2_execute_count),
            "strict_v2_profile_key_hits": int(self._strict_v2_profile_key_hits),
            "strict_v2_profile_key_misses": int(self._strict_v2_profile_key_misses),
            "strict_v2_fail_fast_reject_count": int(
                self._strict_v2_fail_fast_reject_count
            ),
            "large_n_eval_diag_mode": str(
                getattr(self, "_large_n_eval_diag_mode", "full")
            ),
            "large_n_nearfield_diag_mode": str(
                getattr(self, "_large_n_nearfield_diag_mode", "full")
            ),
            "large_n_eval_leaf_nodes_shape": tuple(
                getattr(self, "_large_n_eval_leaf_nodes_shape", ())
            ),
            "large_n_eval_local_coefficients_shape": tuple(
                getattr(self, "_large_n_eval_local_coefficients_shape", ())
            ),
            "large_n_eval_local_centers_shape": tuple(
                getattr(self, "_large_n_eval_local_centers_shape", ())
            ),
            "large_n_eval_active_leaf_count": int(
                getattr(self, "_large_n_eval_active_leaf_count", 0)
            ),
            "large_n_eval_max_leaf_size": int(
                getattr(self, "_large_n_eval_max_leaf_size", 0)
            ),
            "large_n_eval_leaf_particle_slots": int(
                getattr(self, "_large_n_eval_leaf_particle_slots", 0)
            ),
            "large_n_radix_payload_present": bool(
                getattr(self, "_large_n_radix_payload_present", False)
            ),
            "large_n_radix_payload_source_particle_shape": tuple(
                getattr(self, "_large_n_radix_payload_source_particle_shape", ())
            ),
            "large_n_radix_payload_source_particle_slots": int(
                getattr(self, "_large_n_radix_payload_source_particle_slots", 0)
            ),
            "large_n_radix_payload_source_leaf_shape": tuple(
                getattr(self, "_large_n_radix_payload_source_leaf_shape", ())
            ),
            "large_n_radix_payload_source_leaf_slots": int(
                getattr(self, "_large_n_radix_payload_source_leaf_slots", 0)
            ),
            "large_n_target_block_source_leaf_padded_shape": tuple(
                getattr(self, "_large_n_target_block_source_leaf_padded_shape", ())
            ),
            "strict_refresh_diag_mode": str(
                getattr(self, "_strict_refresh_diag_mode", "full")
            ),
            "strict_refresh_diag_tree_active": bool(
                getattr(self, "_strict_refresh_diag_tree_active", True)
            ),
            "strict_refresh_diag_upward_active": bool(
                getattr(self, "_strict_refresh_diag_upward_active", True)
            ),
            "strict_refresh_diag_downward_active": bool(
                getattr(self, "_strict_refresh_diag_downward_active", True)
            ),
            "strict_refresh_diag_eval_active": bool(
                getattr(self, "_strict_refresh_diag_eval_active", True)
            ),
            "strict_refresh_detail_diag_mode": str(
                getattr(self, "_strict_refresh_detail_diag_mode", "full")
            ),
            "static_radix_tree_leaf_count": int(
                getattr(self, "_static_radix_tree_leaf_count", 0)
            ),
            "static_radix_tree_node_count": int(
                getattr(self, "_static_radix_tree_node_count", 0)
            ),
            "static_radix_far_pair_count": int(
                getattr(self, "_static_radix_far_pair_count", 0)
            ),
            "static_radix_compact_pair_reuse_hits": int(
                getattr(self, "_static_radix_compact_pair_reuse_hits", 0)
            ),
            "static_radix_compact_pair_reuse_misses": int(
                getattr(self, "_static_radix_compact_pair_reuse_misses", 0)
            ),
            "static_radix_m2l_chunk_count": int(
                getattr(self, "_static_radix_m2l_chunk_count", 0)
            ),
            "static_radix_l2l_edge_count": int(
                getattr(self, "_static_radix_l2l_edge_count", 0)
            ),
            "strict_fused_mode_active": bool(self._strict_fused_mode_active),
            "strict_fused_compile_count": int(self._strict_fused_compile_count),
            "strict_fused_execute_count": int(self._strict_fused_execute_count),
            "strict_fused_profile_key_hits": int(self._strict_fused_profile_key_hits),
            "strict_fused_profile_key_misses": int(
                self._strict_fused_profile_key_misses
            ),
            "strict_fused_fallback_count": int(self._strict_fused_fallback_count),
            "strict_fused_last_fallback_reason": str(
                self._strict_fused_last_fallback_reason
            ),
            "strict_fused_device_refresh_route_count": int(
                self._strict_fused_device_refresh_route_count
            ),
            "strict_fused_planner_bypassed_count": int(
                self._strict_fused_planner_bypassed_count
            ),
            "strict_velocity_verlet_acceleration_carry_active": bool(
                self._strict_velocity_verlet_acceleration_carry_active
            ),
            "strict_self_force_bootstrap_evaluations": int(
                self._strict_self_force_bootstrap_evaluations
            ),
            "strict_self_force_initial_full_fmm_evaluations": int(
                self._strict_self_force_bootstrap_evaluations
            ),
            "strict_self_force_endpoint_evaluations": int(
                self._strict_self_force_endpoint_evaluations
            ),
            "strict_external_bootstrap_evaluations": int(
                self._strict_external_bootstrap_evaluations
            ),
            "strict_external_endpoint_evaluations": int(
                self._strict_external_endpoint_evaluations
            ),
            "strict_static_target_block_capacity_ok": bool(
                self._strict_static_target_block_capacity_ok
            ),
            "large_n_radix_fast_occupancy_sort": bool(
                self._large_n_radix_fast_occupancy_sort
            ),
            "large_n_radix_fast_skip_empty_tiles": bool(
                self._large_n_radix_fast_skip_empty_tiles
            ),
            "strict_fused_fastlane_diag_enabled": bool(
                self._strict_fused_fastlane_diag_enabled
            ),
            "strict_fused_fastlane_attempts": int(self._strict_fused_fastlane_attempts),
            "strict_fused_fastlane_hits": int(self._strict_fused_fastlane_hits),
            "strict_fused_fastlane_misses": int(self._strict_fused_fastlane_misses),
            "strict_fused_fastlane_last_blockers": tuple(
                str(v) for v in self._strict_fused_fastlane_last_blockers
            ),
            "strict_fused_fastlane_block_counts": {
                str(k): int(v)
                for k, v in dict(self._strict_fused_fastlane_block_counts).items()
            },
            "strict_profiled_max_pair_queue": int(self._strict_profiled_max_pair_queue),
            "strict_profiled_pair_process_block": int(
                self._strict_profiled_pair_process_block
            ),
            "strict_profiled_context_key": str(self._strict_profiled_context_key),
            "recent_dual_node_count": int(self._recent_dual_node_count),
            "recent_dual_leaf_count": int(self._recent_dual_leaf_count),
            "recent_dual_neighbor_count": int(self._recent_dual_neighbor_count),
            "recent_dual_far_pair_count": int(self._recent_dual_far_pair_count),
            "recent_dual_far_pairs_by_gear_counts": tuple(
                int(v) for v in self._recent_far_pairs_by_gear_counts
            ),
            "recent_dual_m2l_chunk_size": int(self._recent_dual_m2l_chunk_size),
            "refresh_total_seconds": float(self._refresh_timing_total_seconds),
            "refresh_input_seconds": float(self._refresh_timing_input_seconds),
            "refresh_tree_upward_seconds": float(
                self._refresh_timing_tree_upward_seconds
            ),
            "refresh_tree_build_seconds": float(
                self._refresh_timing_tree_build_seconds
            ),
            "refresh_upward_compute_seconds": float(
                self._refresh_timing_upward_compute_seconds
            ),
            "refresh_upward_geometry_seconds": float(
                self._refresh_timing_upward_geometry_seconds
            ),
            "refresh_upward_mass_moments_seconds": float(
                self._refresh_timing_upward_mass_moments_seconds
            ),
            "refresh_upward_p2m_seconds": float(
                self._refresh_timing_upward_p2m_seconds
            ),
            "refresh_upward_m2m_seconds": float(
                self._refresh_timing_upward_m2m_seconds
            ),
            "refresh_upward_source_motion_seconds": float(
                self._refresh_timing_upward_source_motion_seconds
            ),
            "refresh_dual_downward_seconds": float(
                self._refresh_timing_dual_downward_seconds
            ),
            "refresh_nearfield_seconds": float(self._refresh_timing_nearfield_seconds),
            "refresh_profile_accounting_seconds": float(
                self._refresh_timing_profile_accounting_seconds
            ),
            "refresh_compile_or_sync_suspect_seconds": float(
                self._refresh_timing_compile_or_sync_suspect_seconds
            ),
            "refresh_dual_setup_seconds": float(
                self._refresh_timing_dual_setup_seconds
            ),
            "refresh_dual_artifact_build_seconds": float(
                self._refresh_timing_dual_artifact_build_seconds
            ),
            "refresh_dual_split_shared_far_near_seconds": float(
                self._refresh_timing_dual_split_shared_far_near_seconds
            ),
            "refresh_dual_split_shared_count_seconds": float(
                self._refresh_timing_dual_split_shared_count_seconds
            ),
            "refresh_dual_split_shared_combined_fill_seconds": float(
                self._refresh_timing_dual_split_shared_combined_fill_seconds
            ),
            "refresh_dual_split_shared_far_fill_seconds": float(
                self._refresh_timing_dual_split_shared_far_fill_seconds
            ),
            "refresh_dual_split_shared_near_fill_seconds": float(
                self._refresh_timing_dual_split_shared_near_fill_seconds
            ),
            "refresh_dual_split_far_pairs_seconds": float(
                self._refresh_timing_dual_split_far_pairs_seconds
            ),
            "refresh_dual_split_leaf_neighbors_seconds": float(
                self._refresh_timing_dual_split_leaf_neighbors_seconds
            ),
            "refresh_dual_split_combined_seconds": float(
                self._refresh_timing_dual_split_combined_seconds
            ),
            "refresh_dual_raw_combined_seconds": float(
                self._refresh_timing_dual_raw_combined_seconds
            ),
            "refresh_dual_split_dense_buffers_seconds": float(
                self._refresh_timing_dual_split_dense_buffers_seconds
            ),
            "refresh_dual_far_pair_plan_seconds": float(
                self._refresh_timing_dual_far_pair_plan_seconds
            ),
            "refresh_dual_m2l_autotune_seconds": float(
                self._refresh_timing_dual_m2l_autotune_seconds
            ),
            "refresh_dual_select_interactions_seconds": float(
                self._refresh_timing_dual_select_interactions_seconds
            ),
            "refresh_dual_downward_compute_seconds": float(
                self._refresh_timing_dual_downward_compute_seconds
            ),
            "refresh_dual_m2l_compute_seconds": float(
                self._refresh_timing_dual_m2l_compute_seconds
            ),
            "refresh_dual_l2l_compute_seconds": float(
                self._refresh_timing_dual_l2l_compute_seconds
            ),
            "refresh_dual_final_symmetry_seconds": float(
                self._refresh_timing_dual_final_symmetry_seconds
            ),
            "refresh_dual_source_motion_seconds": float(
                self._refresh_timing_dual_source_motion_seconds
            ),
            "refresh_dual_finalize_seconds": float(
                self._refresh_timing_dual_finalize_seconds
            ),
            "refresh_dual_residual_seconds": float(
                self._refresh_timing_dual_residual_seconds
            ),
            "refresh_nearfield_leaf_groups_seconds": float(
                self._refresh_timing_nearfield_leaf_groups_seconds
            ),
            "refresh_nearfield_precompute_seconds": float(
                self._refresh_timing_nearfield_precompute_seconds
            ),
            "refresh_nearfield_target_blocks_seconds": float(
                self._refresh_timing_nearfield_target_blocks_seconds
            ),
            "refresh_nearfield_block_sort_seconds": float(
                self._refresh_timing_nearfield_block_sort_seconds
            ),
            "refresh_nearfield_speed_layout_seconds": float(
                self._refresh_timing_nearfield_speed_layout_seconds
            ),
            "refresh_nearfield_overflow_profile_seconds": float(
                self._refresh_timing_nearfield_overflow_profile_seconds
            ),
            "refresh_nearfield_radix_payload_seconds": float(
                self._refresh_timing_nearfield_radix_payload_seconds
            ),
            "refresh_nearfield_neighbor_padding_seconds": float(
                self._refresh_timing_nearfield_neighbor_padding_seconds
            ),
            "refresh_nearfield_state_pack_seconds": float(
                self._refresh_timing_nearfield_state_pack_seconds
            ),
            "refresh_nearfield_residual_seconds": float(
                self._refresh_timing_nearfield_residual_seconds
            ),
            "refresh_timing_calls": int(self._refresh_timing_calls),
        }
