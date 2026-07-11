"""Optional Pallas kernels for Jaccpot hot paths."""

from .m2l_core_z_real import m2l_core_z_real_pallas, pallas_m2l_real_supported
from .nearfield_fused_leaf import (
    nearfield_fused_leaf,
    nearfield_fused_leaf_backend,
    nearfield_fused_leaf_jax,
    nearfield_fused_leaf_pallas,
    pallas_nearfield_fused_supported,
)
from .treecode_walk_pallas import (
    pallas_treecode_walk_supported,
    treecode_leaf_walk_backend,
    treecode_leaf_walk_pallas,
)

__all__ = [
    "m2l_core_z_real_pallas",
    "nearfield_fused_leaf",
    "nearfield_fused_leaf_backend",
    "nearfield_fused_leaf_jax",
    "nearfield_fused_leaf_pallas",
    "pallas_nearfield_fused_supported",
    "pallas_m2l_real_supported",
    "pallas_treecode_walk_supported",
    "treecode_leaf_walk_backend",
    "treecode_leaf_walk_pallas",
]
