"""Preset configurations for :class:`FastMultipoleMethod`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

from yggdrasil.interactions import DualTreeTraversalConfig


class FMMPreset(str, Enum):
    """Named presets offering curated FMM parameter selections."""

    FAST = "fast"


@dataclass(frozen=True)
class FMMPresetConfig:
    """Resolved preset parameters used by :class:`FastMultipoleMethod`."""

    name: FMMPreset
    tree_build_mode: str
    target_leaf_particles: int
    refine_local: bool
    max_refine_levels: int
    aspect_threshold: float
    m2l_chunk_size: Optional[int]
    l2l_chunk_size: Optional[int]
    max_pair_queue: Optional[int]
    pair_process_block: Optional[int]
    traversal_config: Optional[DualTreeTraversalConfig]
    use_dense_interactions: bool
    jit_tree: Union[bool, Literal["auto"]]
    jit_traversal: bool
    description: str


_FAST_TRAVERSAL_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=65536,
    process_block=512,
    max_interactions_per_node=8192,
    max_neighbors_per_leaf=4096,
)

_FAST_PRESET = FMMPresetConfig(
    name=FMMPreset.FAST,
    tree_build_mode="fixed_depth",
    target_leaf_particles=64,
    refine_local=False,
    max_refine_levels=0,
    aspect_threshold=16.0,
    m2l_chunk_size=512,
    l2l_chunk_size=None,
    max_pair_queue=None,
    pair_process_block=None,
    traversal_config=_FAST_TRAVERSAL_CONFIG,
    use_dense_interactions=False,
    jit_tree="auto",
    jit_traversal=True,
    description=(
        "Single-tree gravitational preset optimised for throughput. Uses a "
        "fixed-depth builder, disables host-side refinement, relies on a "
        "single dual-tree traversal, and favours compiled evaluation while "
        "keeping memory usage bounded."
    ),
)


def resolve_preset(name: Union[str, FMMPreset]) -> FMMPreset:
    """Normalise preset identifiers to :class:`FMMPreset`."""

    if isinstance(name, FMMPreset):
        return name
    if hasattr(name, "value"):
        normalized = str(getattr(name, "value")).strip().lower()
    else:
        normalized = str(name).strip().lower()
    if normalized.startswith("fmmpreset."):
        normalized = normalized.split(".", 1)[1]
    try:
        return FMMPreset(normalized)
    except ValueError:  # pragma: no cover - defensive guard
        known = ", ".join(p.value for p in FMMPreset)
        message = f"Unknown FMM preset '{name}'. Known presets: {known}"
        raise ValueError(message)


def get_preset_config(name: Union[str, FMMPreset]) -> FMMPresetConfig:
    """Return the :class:`FMMPresetConfig` for ``name``."""

    preset = resolve_preset(name)
    if preset is FMMPreset.FAST:
        return _FAST_PRESET
    raise AssertionError(f"Missing config for preset {preset!r}")


__all__ = [
    "FMMPreset",
    "FMMPresetConfig",
    "get_preset_config",
    "resolve_preset",
]
