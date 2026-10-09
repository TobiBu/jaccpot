"""StrictCapProfileMixin: fmm_strict_cap_profile methods extracted from the FMMEngine
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.

What is left are the compiled-profile fingerprints (shape summaries for the
compile-reuse diagnostics) and the fused lane's ``PROFILE_SET`` gate. The on-disk
strict cap profile (``JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH``, by default
``/tmp/jaccpot_static_strict_caps.json``), which the strict lanes read to widen
their traversal caps and wrote after every retry, went in the 2026-10 cleanup
(X4): a run's caps no longer depend on what an earlier run left in that file.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # The engine lives in `_fmm_impl`, which imports *these mixins* -- so this import
    # must stay under TYPE_CHECKING or it would form the cycle ARCHITECTURE §8
    # forbids. Inheriting `_EngineBase` makes each mixin *be* the engine under a type
    # checker, so every `self.<engine attribute>` resolves; at runtime the alias is
    # `object`, leaving the MRO exactly as it was. The audit's E.2 records why this
    # beats annotating `self`, and what it does not buy at runtime.
    from ._fmm_impl import FMMEngine, PreparedStateLike

    _EngineBase = FMMEngine
else:  # pragma: no cover - the mixin is only ever mixed into the engine
    _EngineBase = object

__all__ = [
    "StrictCapProfileMixin",
]


class StrictCapProfileMixin(_EngineBase):
    def _compiled_profile_from_prepared_state(
        self,
        state: PreparedStateLike,
    ) -> dict[str, Any]:
        """Build a stable-shape profile summary for compile-reuse diagnostics.

        Shapes only -- it flattens the state's pytree and records dimensions,
        never values. Two states holding different numbers under an identical
        topology therefore produce the same profile, which is the point: the
        profile answers "will this recompile?", not "is this the same problem?".

        Parameters
        ----------
        state : PreparedStateLike
            The prepared state to summarise. Duck-typed, and every field is read
            defensively -- a missing array records 0 rather than raising, so a
            state from a different lane still profiles.

        Returns
        -------
        dict[str, Any]
            JSON-serialisable, which is load-bearing:
            :meth:`_compiled_profile_fingerprint` hashes
            ``json.dumps(..., sort_keys=True)`` of it, so anything unserialisable
            added here breaks fingerprinting rather than degrading it. The
            ``max_*`` entries are capacities and are what
            :meth:`_compiled_profile_capacity_compatible` compares.
        """

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

    def _compiled_profile_fingerprint(self, profile: dict[str, Any]) -> str:
        payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _compiled_profile_capacity_compatible(
        self,
        base_profile: dict[str, Any],
        candidate_profile: dict[str, Any],
    ) -> bool:
        """Return True when candidate usage fits within base profile capacities.

        Asymmetric on purpose: a candidate that needs *less* than the base is
        compatible, since the compiled executable's padded shapes still hold it.
        So this is "fits inside", not "equals" -- reversing the arguments is a
        different question and generally gives a different answer.

        Only five capacity fields are compared. Any other difference between the
        two profiles is ignored here, so a ``True`` does not mean the profiles
        match; it means the capacities do not force a recompile.

        Parameters
        ----------
        base_profile : dict[str, Any]
            Profile of the already-compiled state -- the capacities available.
        candidate_profile : dict[str, Any]
            Profile of the state being considered for reuse.

        Returns
        -------
        bool
            Whether every compared capacity in the candidate is within the base.
            Missing keys read as 0 on both sides, so a profile lacking a field
            compares as needing none of it -- which makes an unrecognised
            profile look compatible rather than incompatible.
        """
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
        self,
        profile_fingerprint: str,
    ) -> None:
        prev = self._compiled_profile_fingerprint_last
        if prev is not None and profile_fingerprint != prev:
            self._compiled_profile_transitions += 1
        self._compiled_profile_fingerprint_last = profile_fingerprint

    def _strict_fused_profile_allows_n(self, n: int) -> bool:
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
