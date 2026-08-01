"""On-device graft validation: treecode far/near through the REAL solidfmm lane.

Injects the treecode-derived far COO + near CSR (from
``build_treecode_far_pairs_and_neighbors``) into the large-N fused fast-lane at its
single walk site (``_interaction_cache.build_compact_far_pairs_and_leaf_neighbor_lists``)
via a test-scoped monkeypatch, then checks ``FastMultipoleMethod.compute_accelerations``
still matches direct N-body. This is the pre-graft physics gate for step 3: it proves
the treecode's leaf-only far targets feed the real solidfmm M2L -> L2L -> L2P with L2L
acting as a no-op (no double-count), and that the self-excluded near CSR drives the real
fused near-field correctly.

Requires an Ampere+ GPU (the large-N fused lane is GPU-only); skipped otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

import jaccpot.runtime._interaction_cache as _ic
from jaccpot import FastMultipoleMethod
from jaccpot.experimental.treecode_far_near import (
    build_treecode_far_pairs_and_neighbors,
)
from jaccpot.pallas.treecode_walk_pallas import pallas_treecode_walk_supported

try:
    from yggdrax._interactions_impl import CompactTaggedFarPairs
except Exception:  # pragma: no cover
    from yggdrax.interactions import CompactTaggedFarPairs

_FAR_CAP = 131072

# Strict-fused device-only knobs (mirror odisseo _large_n_environment_overrides),
# applied per-test via monkeypatch so FastMultipoleMethod.__init__ captures them and
# they are CLEANLY RESTORED afterwards. Previously these were written to os.environ at
# module import time, which leaked into unrelated tests under `pytest -n`: every xdist
# worker imports this module (even though its GPU-only tests are skipped), permanently
# setting e.g. STRICT_FUSED_MODE/DISALLOW_HOST_SEGMENT_FALLBACK process-wide.
_STRICT_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": str(_FAR_CAP),
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
}
_THETA, _G, _SOFT, _LEAF, _ORDER = 0.5, 1.0, 1e-2, 16, 4


@pytest.fixture(autouse=True)
def _apply_strict_fused_env(monkeypatch):
    for _k, _v in _STRICT_FUSED_ENV.items():
        monkeypatch.setenv(_k, _v)


def _direct(pos, mass):
    p = np.asarray(pos, np.float64)
    m = np.asarray(mass, np.float64)
    d = p[:, None, :] - p[None, :, :]
    d2 = (d**2).sum(-1) + _SOFT**2
    np.fill_diagonal(d2, np.inf)
    return -_G * (m[None, :, None] * d * (d2**-1.5)[..., None]).sum(axis=1)


def _make_treecode_builder(orig, *, use_pallas):
    """Wrap the yggdrax builder: reuse its near template shapes, swap in treecode."""

    def _patched(tree, geometry, **kw):
        real_cfp, real_nbr = orig(tree, geometry, **kw)
        topo = tree.topology
        num_internal = int(topo.num_internal_nodes)
        total_nodes = int(topo.parent.shape[0])
        num_leaves = total_nodes - num_internal
        idx = topo.parent.dtype
        left_full = jnp.concatenate(
            [topo.left_child.astype(idx), jnp.full((num_leaves,), -1, idx)]
        )
        right_full = jnp.concatenate(
            [topo.right_child.astype(idx), jnp.full((num_leaves,), -1, idx)]
        )
        leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
        root_idx = jnp.argmin(topo.parent).astype(idx)
        cap = total_nodes
        prod = build_treecode_far_pairs_and_neighbors(
            leaf_nodes,
            geometry.center,
            geometry.max_extent,
            left_full,
            right_full,
            jnp.asarray(_THETA * _THETA, geometry.center.dtype),
            root_idx,
            num_internal=num_internal,
            max_far=cap,
            max_near=cap,
            max_stack=2 * cap + 4,
            max_iters=total_nodes + 1,
            far_pair_capacity=_FAR_CAP,
            near_capacity=num_leaves * cap,
            idx_dtype=idx,
            use_pallas=use_pallas,
        )
        assert not bool(prod.overflow)
        cfp = CompactTaggedFarPairs(
            sources=prod.far_sources,
            targets=prod.far_targets,
            tags=prod.far_tags,
            far_pair_count=prod.far_pair_count,
        )
        nbr = real_nbr._replace(
            offsets=prod.near_offsets,
            neighbors=prod.near_neighbors,
            leaf_indices=prod.near_leaf_indices,
            counts=prod.near_counts,
        )
        return cfp, nbr

    return _patched


def _run(pos, mass):
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        theta=_THETA,
        G=_G,
        softening=_SOFT,
        working_dtype=jnp.float32,
        use_pallas=False,
    )
    acc = fmm.compute_accelerations(pos, mass, leaf_size=_LEAF, max_order=_ORDER)
    return np.asarray(acc, np.float64)


@pytest.mark.skipif(
    not pallas_treecode_walk_supported(),
    reason="large-N fused lane + treecode-walk kernel require an Ampere+ GPU",
)
@pytest.mark.parametrize("use_pallas_walk", [False, True])
@pytest.mark.parametrize("n", [2000, 3000])
def test_treecode_graft_matches_direct(n, use_pallas_walk, monkeypatch):
    kp, km = jax.random.split(jax.random.PRNGKey(1))
    pos = jax.random.uniform(kp, (n, 3), dtype=jnp.float32, minval=-1.0, maxval=1.0)
    mass = jnp.abs(jax.random.normal(km, (n,), dtype=jnp.float32)) + 0.5
    ref = _direct(pos, mass)

    # Baseline (dual-tree) must match direct -> confirms the lane + harness.
    base = _run(pos, mass)
    base_rel = np.linalg.norm(base - ref, axis=1) / (
        np.linalg.norm(ref, axis=1) + 1e-12
    )
    assert np.median(base_rel) < 0.09

    # Graft the treecode far/near and re-evaluate.
    orig = _ic.build_compact_far_pairs_and_leaf_neighbor_lists
    monkeypatch.setattr(
        _ic,
        "build_compact_far_pairs_and_leaf_neighbor_lists",
        _make_treecode_builder(orig, use_pallas=use_pallas_walk),
    )
    got = _run(pos, mass)
    rel = np.linalg.norm(got - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
    # float32 large-N tolerance (cf. tests/test_gravity_vs_direct.py float32 gate).
    assert np.median(rel) < 0.09, f"median rel {np.median(rel):.3e}"
    assert np.percentile(rel, 90) < 0.15, f"p90 rel {np.percentile(rel, 90):.3e}"
