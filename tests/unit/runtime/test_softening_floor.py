"""``softening_floor``: the walk keeps far pairs out of the softening kernel's reach.

The fused lane's far field is the UNSOFTENED expansion and the geometric MAC knows
nothing about epsilon, so cells closer than a few softening lengths were accepted
with the Newtonian force law (the 2e6 disc: rel-L2 0.45 at its production
softening). ``FMMAdvancedConfig.softening_floor = c`` accepts a pair only if also
``|c_B - c_A| - r_A - r_B >= c * softening``. Pinned here:

* CPU: the option is validated, keys the interaction cache, and a dual-tree build
  that cannot apply it raises instead of ignoring it (the walk-level contract --
  both walks, same sets, every far pair at least the floor apart -- is in
  ``test_mutual_walk_pallas.py``);
* GPU (the strict fused lane exists only there): a cluster smaller than epsilon
  inside a Plummer sphere, against a softened fp64 direct sum. Without the floor
  the cluster's sub-cells attract each other with Newton's law; with it the
  cluster is near field. With epsilon far below every accepted separation the
  floor changes nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.runtime._interaction_cache import _interaction_cache_key

_GPU = jax.default_backend() == "gpu"


def test_negative_floor_is_refused():
    with pytest.raises(ValueError, match="softening_floor"):
        FastMultipoleMethod(
            theta=0.6,
            softening=1e-2,
            advanced=FMMAdvancedConfig(softening_floor=-1.0),
        )


def _key(tree, **over):
    kw = dict(
        topology_key="fixed-topology",
        tree_mode="radix",
        leaf_parameter=8,
        theta=0.6,
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
        expansion_basis="solidfmm",
        center_mode="com",
        max_pair_queue=None,
        pair_process_block=None,
        traversal_config=None,
        refine_local=False,
        max_refine_levels=0,
        aspect_threshold=16.0,
        pair_policy_identity="none",
    )
    kw.update(over)
    return _interaction_cache_key(tree, **kw)


def test_floor_is_part_of_the_cache_key():
    from yggdrax.tree import Tree

    points = jnp.asarray(
        np.random.default_rng(0).uniform(size=(256, 3)), dtype=jnp.float32
    )
    tree = Tree.from_particles(
        points, jnp.ones((256,), jnp.float32), leaf_size=8, tree_type="radix"
    )
    base = _key(tree)
    assert base is not None
    assert _key(tree, separation_floor=0.0) == base
    a, b = _key(tree, separation_floor=0.1), _key(tree, separation_floor=0.2)
    assert len({base, a, b}) == 3


def _plummer_with_cluster(
    n, n_cluster, cluster_radius, cluster_mass, seed=0, r_clip=20.0
):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    r = np.minimum(r, r_clip)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    mass = np.full(n, (1.0 - cluster_mass) / (n - n_cluster))
    if n_cluster:
        # the cluster: a uniform ball inside one softening length, at the centre
        u = rng.normal(size=(n_cluster, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u *= cluster_radius * rng.uniform(size=(n_cluster, 1)) ** (1.0 / 3.0)
        pos[:n_cluster] = u
        mass[:n_cluster] = cluster_mass / n_cluster
    return pos.astype(np.float32), mass.astype(np.float32)


def _direct(pos, mass, idx, eps):
    p = pos.astype(np.float64)
    m = mass.astype(np.float64)
    out = np.empty((idx.size, 3))
    for k, i in enumerate(idx):
        d = p - p[i]
        r2 = np.einsum("ij,ij->i", d, d) + eps * eps
        w = m / (r2 * np.sqrt(r2))
        w[i] = 0.0
        out[k] = w @ d
    return out


_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "auto",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "4194304",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "1048576",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}
_N = 70_000
_LEAF = 64
_THETA = 0.6
_ORDER = 4


def _force(monkeypatch, pos, mass, eps, floor):
    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", str(pos.shape[0]))
    monkeypatch.delenv("JACCPOT_WALK_SEPARATION_FLOOR", raising=False)
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=eps,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=_LEAF),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
            softening_floor=floor,
        ),
        fixed_order=_ORDER,
    )
    prepared, ev = solver.strict_fused_prepared_eval_fn(
        positions=jnp.asarray(pos),
        masses=jnp.asarray(mass),
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
    )
    acc = np.asarray(jax.block_until_ready(ev(prepared)), np.float64)
    caps = dict(getattr(solver._impl, "_strict_fused_validated_caps", None) or {})
    assert caps.get("flat_walk") is True, "the floor lives in the flat walk"
    return acc


def _rel(got, ref):
    return np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)


@pytest.mark.skipif(not _GPU, reason="the strict fused large-N lane is GPU-only")
def test_a_cluster_smaller_than_epsilon_is_near_field(monkeypatch):
    # The box must resolve the cluster into leaves that pass the MAC against each
    # other: inside the r <= 20 clip a 0.005-wide cluster stays below the Morton
    # grid, its leaves overlap, and no pair inside it is ever far.
    eps = 0.3
    n_cluster = 4096
    pos, mass = _plummer_with_cluster(
        _N, n_cluster, cluster_radius=eps / 3.0, cluster_mass=0.05, r_clip=3.0
    )
    rng = np.random.default_rng(1)
    idx = np.concatenate(
        [np.arange(0, n_cluster, 16), rng.choice(np.arange(n_cluster, _N), 512)]
    )
    ref = _direct(pos, mass, idx, eps)
    bare = _rel(_force(monkeypatch, pos, mass, eps, 0.0)[idx], ref)
    floored = _rel(_force(monkeypatch, pos, mass, eps, 5.0)[idx], ref)
    inner = idx < n_cluster
    # Measured on an A100 (2026-10-07): cluster median / max 0.15 / 4.0 without
    # the floor -- its sub-cells pull on each other with Newton's law -- and
    # 1.3e-4 / 4.9e-4 with it; the rest of the sphere 2.5e-2 -> 7.7e-4 (median).
    assert np.median(bare[inner]) > 0.05 and bare[inner].max() > 1.0
    assert np.median(floored[inner]) < 1e-3, np.median(floored[inner])
    assert floored[inner].max() < 2e-3, floored[inner].max()
    assert np.median(floored[~inner]) < 3e-3, np.median(floored[~inner])
    assert np.median(bare[~inner]) > 10.0 * np.median(floored[~inner])


@pytest.mark.skipif(not _GPU, reason="the strict fused large-N lane is GPU-only")
def test_a_floor_below_every_far_gap_changes_nothing(monkeypatch):
    pos, mass = _plummer_with_cluster(
        _N, 0, cluster_radius=0.0, cluster_mass=0.0, seed=2
    )
    eps = 1e-7
    bare = _force(monkeypatch, pos, mass, eps, 0.0)
    floored = _force(monkeypatch, pos, mass, eps, 5.0)
    rel = np.linalg.norm(floored - bare) / np.linalg.norm(bare)
    # same lists; a fresh compile may still sum in another order (fp32)
    assert rel < 1e-6, rel
