"""StrictCapProfileMixin: fmm_strict_cap_profile methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from yggdrax.interactions import DualTreeRetryEvent


class StrictCapProfileMixin:
    def _strict_cap_profile_path(self) -> str:
        return str(
            os.environ.get(
                "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH",
                "/tmp/jaccpot_static_strict_caps.json",
            )
        )

    def _strict_cap_profile_context_key(
        self,
        *,
        tree_mode: str,
        leaf_parameter: int,
        particle_count: int,
    ) -> str:
        return (
            f"tree_mode={str(tree_mode).strip().lower()}|"
            f"leaf={int(leaf_parameter)}|n={int(particle_count)}"
        )

    def _maybe_load_strict_cap_profile(
        self, *, context_key: Optional[str] = None
    ) -> None:
        if self._strict_profile_loaded_once:
            if context_key is not None:
                self._apply_strict_cap_profile_for_key(context_key=context_key)
            return
        self._strict_profile_loaded_once = True
        try:
            path = self._strict_cap_profile_path()
            if not os.path.exists(path):
                return
            payload = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
                self._strict_profile_catalog = {
                    str(k): {
                        "max_pair_queue": int(v.get("max_pair_queue", 0) or 0),
                        "pair_process_block": int(v.get("pair_process_block", 0) or 0),
                    }
                    for k, v in payload["profiles"].items()
                    if isinstance(v, dict)
                }
            else:
                # Backward compatibility with the original single-profile payload.
                q = int(payload.get("max_pair_queue", 0) or 0)
                b = int(payload.get("pair_process_block", 0) or 0)
                self._strict_profile_catalog = {
                    "legacy_default": {
                        "max_pair_queue": q,
                        "pair_process_block": b,
                    }
                }
            if context_key is not None:
                self._apply_strict_cap_profile_for_key(context_key=context_key)
            elif len(self._strict_profile_catalog) > 0:
                # Preserve previous behavior when no context is supplied.
                self._apply_strict_cap_profile_for_key(context_key="legacy_default")
        except Exception:
            return

    def _apply_strict_cap_profile_for_key(self, *, context_key: str) -> None:
        selected_key = ""
        selected = self._strict_profile_catalog.get(context_key)
        if selected is not None:
            selected_key = context_key
        else:
            # Conservative fallback: keep same tree_mode+leaf and pick the largest queue.
            prefix = "|".join(str(context_key).split("|")[:2])
            best_q = 0
            best_entry: Optional[dict[str, int]] = None
            best_key = ""
            for key, entry in self._strict_profile_catalog.items():
                if not str(key).startswith(prefix):
                    continue
                q = int(entry.get("max_pair_queue", 0) or 0)
                if q >= best_q:
                    best_q = q
                    best_entry = entry
                    best_key = str(key)
            if best_entry is not None:
                selected = best_entry
                selected_key = best_key
            else:
                selected = self._strict_profile_catalog.get("legacy_default")
                selected_key = "legacy_default" if selected is not None else ""
        if selected is None:
            return
        q = int(selected.get("max_pair_queue", 0) or 0)
        b = int(selected.get("pair_process_block", 0) or 0)
        if q > 0:
            self._strict_profiled_max_pair_queue = q
        if b > 0:
            self._strict_profiled_pair_process_block = b
        if selected_key:
            self._strict_profiled_context_key = selected_key

    def _record_strict_cap_profile_from_retries(
        self,
        retry_events: Tuple[DualTreeRetryEvent, ...],
        *,
        context_key: Optional[str] = None,
    ) -> None:
        if len(retry_events) == 0:
            return
        max_queue = int(self._strict_profiled_max_pair_queue)
        max_block = int(self._strict_profiled_pair_process_block)
        for ev in retry_events:
            try:
                q = int(getattr(ev, "queue_capacity", 0) or 0)
            except Exception:
                q = 0
            if q > max_queue:
                max_queue = q
        block_hint = int(self.pair_process_block or 0)
        if block_hint > max_block:
            max_block = block_hint
        if max_queue <= 0 and max_block <= 0:
            return
        self._strict_profiled_max_pair_queue = max_queue
        self._strict_profiled_pair_process_block = max_block
        if context_key is not None:
            self._strict_profiled_context_key = str(context_key)
            existing = self._strict_profile_catalog.get(str(context_key), {})
            self._strict_profile_catalog[str(context_key)] = {
                "max_pair_queue": max(
                    int(existing.get("max_pair_queue", 0) or 0),
                    int(max_queue),
                ),
                "pair_process_block": max(
                    int(existing.get("pair_process_block", 0) or 0),
                    int(max_block),
                ),
            }
        if not bool(getattr(self, "_strict_cap_record_enabled", True)):
            return
        try:
            path = self._strict_cap_profile_path()
            payload = {
                "version": 2,
                "active_context_key": str(self._strict_profiled_context_key),
                "profiles": self._strict_profile_catalog,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception:
            return

    def _compiled_profile_from_prepared_state(
        self: "FastMultipoleMethod",
        state: PreparedStateLike,
    ) -> dict[str, Any]:
        """Build a stable-shape profile summary for compile-reuse diagnostics."""

        def _shape0(value: Any) -> int:
            if value is None:
                return 0
            return int(jnp.asarray(value).shape[0])

        def _shape_last(value: Any) -> int:
            if value is None:
                return 0
            arr = jnp.asarray(value)
            return int(arr.shape[-1]) if arr.ndim >= 1 else 0

        leaves, _ = jax.tree_util.tree_flatten(state)
        leaf_shapes: list[tuple[int, ...]] = []
        for leaf in leaves:
            shape = getattr(leaf, "shape", None)
            if shape is None:
                continue
            leaf_shapes.append(tuple(int(v) for v in shape))

        tree_parent = getattr(state.tree, "parent", None)
        neighbor_leaf_indices = getattr(state.neighbor_list, "leaf_indices", None)
        node_count = (
            int(jnp.asarray(tree_parent).shape[0]) if tree_parent is not None else 0
        )
        leaf_count = (
            int(jnp.asarray(neighbor_leaf_indices).shape[0])
            if neighbor_leaf_indices is not None
            else 0
        )
        nearfield_blocks = _shape0(getattr(state, "nearfield_target_leaf_ids", None))
        nearfield_target_block_slots = _shape0(
            getattr(state, "nearfield_target_block_source_leaf_ids", None)
        )
        leaf_particle_slots = _shape_last(
            getattr(state, "nearfield_leaf_particle_indices", None)
        )
        order = 0
        local_data = getattr(state, "local_data", None)
        if local_data is not None:
            order = int(getattr(local_data, "order", 0))
        else:
            downward = getattr(state, "downward", None)
            locals_view = (
                getattr(downward, "locals", None) if downward is not None else None
            )
            order = (
                int(getattr(locals_view, "order", 0)) if locals_view is not None else 0
            )

        return {
            "preset": str(self.preset),
            "runtime_path": str(self.runtime_path),
            "tree_type": str(self.tree_type),
            "execution_backend": str(getattr(state, "execution_backend", "unknown")),
            "expansion_basis": str(
                getattr(state, "expansion_basis", self.expansion_basis)
            ),
            "working_dtype": str(
                jnp.dtype(getattr(state, "working_dtype", self.working_dtype))
            ),
            "max_leaf_size": int(getattr(state, "max_leaf_size", 0)),
            "max_order": int(order),
            "max_nodes": int(node_count),
            "max_leaves": int(leaf_count),
            "max_nearfield_blocks": int(nearfield_blocks),
            "max_nearfield_target_block_slots": int(nearfield_target_block_slots),
            "max_leaf_particle_slots": int(leaf_particle_slots),
            "leaf_shapes": tuple(leaf_shapes),
        }

    def _compiled_profile_fingerprint(
        self: "FastMultipoleMethod", profile: dict[str, Any]
    ) -> str:
        payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _compiled_profile_capacity_compatible(
        self: "FastMultipoleMethod",
        base_profile: dict[str, Any],
        candidate_profile: dict[str, Any],
    ) -> bool:
        """Return True when candidate usage fits within base profile capacities."""
        capacity_fields = (
            "max_nodes",
            "max_leaves",
            "max_nearfield_blocks",
            "max_nearfield_target_block_slots",
            "max_leaf_particle_slots",
        )
        return all(
            int(candidate_profile.get(name, 0)) <= int(base_profile.get(name, 0))
            for name in capacity_fields
        )

    def _compiled_profile_record_transition(
        self: "FastMultipoleMethod",
        profile_fingerprint: str,
    ) -> None:
        prev = self._compiled_profile_fingerprint_last
        if prev is not None and profile_fingerprint != prev:
            self._compiled_profile_transitions += 1
        self._compiled_profile_fingerprint_last = profile_fingerprint

    def _strict_fused_profile_allows_n(self: "FastMultipoleMethod", n: int) -> bool:
        raw = str(getattr(self, "_strict_fused_profile_set_raw", "")).strip()
        if raw == "":
            return True
        allowed: set[int] = set()
        for token in raw.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                allowed.add(int(t))
            except Exception:
                continue
        if not allowed:
            return True
        return int(n) in allowed
