"""Halo-source P2P mechanism de-risk (distributed FMM, Phase 4b).

Distributed near-field P2P must let a local target leaf draw its source
particles from an imported HALO buffer, not just the local particle array.
jaccpot's ``compute_leaf_p2p_accelerations`` is single-array, but its
``leaf_particle_indices_override`` gathers each leaf's particles from arbitrary
indices. This test concatenates ``[local ; halo]`` into one buffer and wires a
target(local)-leaf ← source(halo)-leaf neighbor edge via the overrides, then
checks the local particles' accelerations equal a direct Plummer sum over the
halo particles (single device, fast, no shard_map).
"""

import jax.numpy as jnp
import numpy as np
from yggdrax import build_prepared_tree_artifacts
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations

_G = 1.0
_SOFT = 0.05


def _plummer(tgt, src, src_mass, G, soft):
    """Plummer accel on each `tgt` from all `src` (self pairs give 0: diff=0)."""
    diff = tgt[:, None, :] - src[None, :, :]  # r_i - r_j
    d2 = (diff**2).sum(-1) + soft**2
    inv = d2 ** (-1.5)
    return -G * (src_mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def test_halo_source_p2p_matches_direct_sum():
    rng = np.random.default_rng(3)
    n_loc, n_halo = 10, 12
    # Well-separated so softening/self effects are clean; disjoint sets.
    local = rng.uniform(-1.0, -0.3, size=(n_loc, 3)).astype(np.float32)
    halo = rng.uniform(0.3, 1.0, size=(n_halo, 3)).astype(np.float32)
    loc_mass = rng.uniform(0.5, 2.0, size=(n_loc,)).astype(np.float32)
    halo_mass = rng.uniform(0.5, 2.0, size=(n_halo,)).astype(np.float32)

    pos = jnp.asarray(np.concatenate([local, halo], axis=0))  # [nL+nH, 3]
    mass = jnp.asarray(np.concatenate([loc_mass, halo_mass], axis=0))

    # Throwaway real Tree + NodeNeighborList only to satisfy the typed signature;
    # every field is bypassed by the overrides below.
    art = build_prepared_tree_artifacts(pos, mass, leaf_size=16)

    K = max(n_loc, n_halo)
    # leaf 0 = local particles, leaf 1 = halo particles (indices into `pos`).
    idx = np.zeros((2, K), dtype=np.int64)
    m = np.zeros((2, K), dtype=bool)
    idx[0, :n_loc] = np.arange(n_loc)
    m[0, :n_loc] = True
    idx[1, :n_halo] = np.arange(n_loc, n_loc + n_halo)
    m[1, :n_halo] = True

    accel = compute_leaf_p2p_accelerations(
        art.tree,
        art.neighbors,
        pos,
        mass,
        G=_G,
        softening=_SOFT,
        nearfield_mode="baseline",
        node_ranges_override=jnp.zeros((2, 2), INDEX_DTYPE),
        leaf_nodes_override=jnp.asarray([0, 1], INDEX_DTYPE),
        neighbor_offsets_override=jnp.asarray(
            [0, 1, 1], INDEX_DTYPE
        ),  # leaf0 -> [edge0]
        neighbor_indices_override=jnp.asarray(
            [1], INDEX_DTYPE
        ),  # edge0 source = leaf node 1
        neighbor_counts_override=jnp.asarray([1, 0], INDEX_DTYPE),
        leaf_particle_indices_override=jnp.asarray(idx, INDEX_DTYPE),
        leaf_particle_mask_override=jnp.asarray(m),
    )
    accel = np.asarray(accel)

    # The kernel always adds each leaf's self-block, so local particles feel the
    # local self-block (local<-local) PLUS the halo neighbour (local<-halo).
    ref = _plummer(local, local, loc_mass, _G, _SOFT) + _plummer(
        local, halo, halo_mass, _G, _SOFT
    )
    np.testing.assert_allclose(accel[:n_loc], ref, rtol=1e-4, atol=1e-4)
    # the halo contribution is a real, non-trivial part of the total
    halo_only = _plummer(local, halo, halo_mass, _G, _SOFT)
    assert np.abs(halo_only).max() > 0
