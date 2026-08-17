"""Preset configurations for :class:`FMMEngine`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot.config import FMMPreset


@dataclass(frozen=True)
class FMMPresetConfig:
    """Resolved preset parameters used by :class:`FMMEngine`.
    A frozen bundle applied BEFORE the individual constructor keywords, so an
    explicit keyword always wins over its preset value.

    Attributes
    ----------
    name : FMMPreset
        Which preset this is.
    tree_build_mode : str
        Builder the preset selects.
    target_leaf_particles : int
        Leaf occupancy target.
    refine_local : bool
        Whether the host-side refinement pass runs.
    max_refine_levels : int
        Depth cap for that pass.
    aspect_threshold : float
        Leaf aspect ratio above which refinement splits a leaf.
    m2l_chunk_size : Optional[int]
        Pairs per M2L chunk; ``None`` leaves it to the runtime.
    l2l_chunk_size : Optional[int]
        Nodes per L2L chunk; ``None`` as above.
    max_pair_queue : Optional[int]
        Dual-tree pair-queue capacity.
    pair_process_block : Optional[int]
        Traversal process-block width.
    traversal_config : Optional[DualTreeTraversalConfig]
        Explicit traversal capacities.
    use_dense_interactions : bool
        Materialise the interaction list densely.
    jit_tree : Union[bool, Literal['auto']]
        JIT the tree build; ``"auto"`` defers the decision to the runtime.
    jit_traversal : bool
        JIT the traversal/evaluation path.
    description : str
        Human-readable summary, for diagnostics.
    """

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
    tree_build_mode="lbvh",
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

_LARGE_N_GPU_TRAVERSAL_CONFIG = DualTreeTraversalConfig(
    max_pair_queue=262144,
    process_block=256,
    max_interactions_per_node=4096,
    max_neighbors_per_leaf=1024,
)

_LARGE_N_GPU_PRESET = FMMPresetConfig(
    name=FMMPreset.LARGE_N_GPU,
    tree_build_mode="lbvh",
    target_leaf_particles=64,
    refine_local=False,
    max_refine_levels=0,
    aspect_threshold=16.0,
    m2l_chunk_size=None,
    l2l_chunk_size=None,
    max_pair_queue=None,
    pair_process_block=None,
    traversal_config=_LARGE_N_GPU_TRAVERSAL_CONFIG,
    use_dense_interactions=False,
    jit_tree=True,
    jit_traversal=True,
    description=(
        "Large-N GPU preset prioritizing stable memory behavior and high "
        "throughput on streamed/grouped far-field execution."
    ),
)


def resolve_preset(name: Union[str, FMMPreset]) -> FMMPreset:
    """Normalise preset identifiers to :class:`FMMPreset`.

    Parameters
    ----------
    name : Union[str, FMMPreset]
        An enum member, or its value as a string. Strings are stripped and
        lowercased, so ``" Fast "`` is accepted.

    Returns
    -------
    FMMPreset
        The normalised member.

    Raises
    ------
    ValueError
        If the string matches no preset. Unlike much of the runtime's option
        handling, this does NOT fall back to a default -- a misspelled preset
        would otherwise silently select different accuracy.
    """

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
    """Return the :class:`FMMPresetConfig` for ``name``.

    Parameters
    ----------
    name : Union[str, FMMPreset]
        Preset identifier, normalised by :func:`resolve_preset` first.

    Returns
    -------
    FMMPresetConfig
        The frozen configuration for that preset.

    Raises
    ------
    AssertionError
        For ``BALANCED`` and ``ACCURATE``. Only ``FAST`` and ``LARGE_N_GPU`` have
        engine-level bundles here; the other two resolve through
        ``solver._default_advanced_for_preset`` and are documented never to reach
        this function (see :class:`~jaccpot.config.FMMPreset` and ARCHITECTURE
        section 6). Reaching it with either means the solver's routing was
        bypassed -- ``FMMEngine(preset="balanced")`` does exactly that -- which is
        an internal wiring fault, hence an assertion rather than a ValueError.
    """

    preset = resolve_preset(name)
    if preset is FMMPreset.FAST:
        return _FAST_PRESET
    if preset is FMMPreset.LARGE_N_GPU:
        return _LARGE_N_GPU_PRESET
    raise AssertionError(f"Missing config for preset {preset!r}")


__all__ = [
    "FMMPreset",
    "FMMPresetConfig",
    "get_preset_config",
    "resolve_preset",
]
