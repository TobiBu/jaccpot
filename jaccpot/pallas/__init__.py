"""Optional Pallas kernels for Jaccpot hot paths."""

from .m2l_core_z_real import m2l_core_z_real_pallas, pallas_m2l_real_supported
from .nearfield_fused_leaf import (
    nearfield_fused_leaf,
    nearfield_fused_leaf_backend,
    nearfield_fused_leaf_jax,
    nearfield_fused_leaf_pallas,
    pallas_nearfield_fused_supported,
)
from .nearfield_tile_pair import (
    nearfield_tile_pair_accel,
    nearfield_tile_pair_accel_jax,
    nearfield_tile_pair_accel_pallas,
    nearfield_tile_pair_backend,
    pallas_nearfield_tile_pair_supported,
)
from .nearfield_unique_updates import (
    apply_packed_particle_vector_updates,
    apply_packed_particle_vector_updates_jax,
    nearfield_unique_updates_backend,
    pack_unique_particle_vector_updates,
    pallas_nearfield_unique_updates_supported,
)

__all__ = [
    "apply_packed_particle_vector_updates",
    "apply_packed_particle_vector_updates_jax",
    "m2l_core_z_real_pallas",
    "nearfield_fused_leaf",
    "nearfield_fused_leaf_backend",
    "nearfield_fused_leaf_jax",
    "nearfield_fused_leaf_pallas",
    "nearfield_tile_pair_accel",
    "nearfield_tile_pair_accel_jax",
    "nearfield_tile_pair_accel_pallas",
    "nearfield_tile_pair_backend",
    "nearfield_unique_updates_backend",
    "pack_unique_particle_vector_updates",
    "pallas_nearfield_fused_supported",
    "pallas_nearfield_tile_pair_supported",
    "pallas_m2l_real_supported",
    "pallas_nearfield_unique_updates_supported",
]
