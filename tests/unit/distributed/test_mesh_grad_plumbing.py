"""The multi-GPU gradient's plumbing refuses what its seams cannot replay (WP-G, G1).

Fast, host-only checks of the refusals (no force is traced):

* ``make_cross_hook(grad_sink=...)`` refuses every control exchange, a hook without
  its near sink and the table near layout -- the reverse seams replay the default
  symmetric exchange and the direct near layout only;
* ``cross_reverse`` (seam A's slot in the downward sweep) raises off the CSR M2L
  lanes kernel, without cross-domain sources, on a list that is not target-sorted
  and with the far force scale: anywhere else the cross source terms would silently
  leave the gradient;
* the placeholder seams' forwards are the production launches, their reverses
  raise; the cross pair hashes are order-free and agree at both ends of a pair.

The end-to-end gates (the forward bitwise ``FusedMeshForce.shard``'s at 2 and 4
devices, the ndev=1 gradient) are ``tests/distributed/test_mesh_grad_forward.py``.

    JAX_PLATFORMS=cpu pytest tests/unit/distributed/test_mesh_grad_plumbing.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.export")

from jaccpot.distributed.cross import (  # noqa: E402
    CrossCapacities,
    _peer_pair_hashes,
    make_cross_hook,
)


def _hook(**kwargs):
    kw = dict(
        ndev=2,
        theta=0.6,
        caps=CrossCapacities(),
        near_sink={},
        grad_sink={},
    )
    kw.update(kwargs)
    return make_cross_hook(**kw)


def test_the_default_exchange_takes_a_grad_sink():
    hook = _hook()
    assert callable(hook)


@pytest.mark.parametrize(
    "env, kwargs",
    [
        ({"JACCPOT_CROSS_SYMMETRIC": "0"}, {}),
        ({"JACCPOT_CROSS_TWO_SIDED": "0"}, {}),
        ({}, {"two_sided": False}),
        ({"JACCPOT_CROSS_NEAR_TILES": "1"}, {}),
        ({"JACCPOT_CROSS_FAR_RECEIVER_WALK": "1"}, {}),
        ({"JACCPOT_CROSS_NEAR_RECEIVER_WALK": "1"}, {}),
        ({}, {"near_theta": 0.0}),
        ({}, {"export_theta": 0.3}),
        ({}, {"caps": CrossCapacities(max_leaves_per_cell=4)}),
        ({}, {"near_sink": None}),
        ({"JACCPOT_NEARFIELD_LAYOUT": "table"}, {}),
        ({"JACCPOT_NEARFIELD_ACCUM": "wide"}, {}),
    ],
)
def test_a_grad_sink_refuses_what_the_seams_cannot_replay(monkeypatch, env, kwargs):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(NotImplementedError, match="grad_sink"):
        _hook(**kwargs)
    # the same configuration without a sink is the production's, untouched
    kwargs = dict(kwargs, grad_sink=None)
    assert callable(_hook(**kwargs))


def _accumulate(**kwargs):
    from jaccpot.runtime.kernels._downward_prep import (
        _solidfmm_downward_accumulate_from_multipoles,
    )

    n, c = 4, 9
    base = dict(
        n_targets=n,
        centers=jnp.zeros((n + 2, 3)),
        src=jnp.zeros((3,), jnp.int32),
        tgt=jnp.zeros((n + 1,), jnp.int32),
        pair_count=3,
        active_pair_count=jnp.int32(0),
        order=2,
        total_nodes=n,
        chunk_size=8,
        targets_sorted=True,
        cross_reverse=lambda *a: pytest.fail("the seam must not be called"),
    )
    base.update(kwargs)
    return _solidfmm_downward_accumulate_from_multipoles(
        jnp.zeros((n, c)), jnp.zeros((n + 2, c)), **base
    )


@pytest.mark.parametrize(
    "env, kwargs",
    [
        ({"JACCPOT_STATIC_STRICT_FUSED_M2L_CSR": "0"}, {}),
        ({"JACCPOT_M2L_CSR_INTERPRET": "1"}, {"n_targets": None}),
        ({"JACCPOT_M2L_CSR_INTERPRET": "1"}, {"targets_sorted": False}),
        ({"JACCPOT_M2L_CSR_INTERPRET": "1"}, {"force_scale": "far"}),
    ],
)
def test_cross_reverse_raises_where_the_cross_terms_would_vanish(
    monkeypatch, env, kwargs
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if kwargs.get("force_scale") == "far":
        # a real FarForceScaleInputs (not a sentinel): CI's unit jobs run the
        # runtime type checker, which would refuse anything else first
        from jaccpot.runtime._force_scale_levels import FarForceScaleInputs

        kwargs = dict(
            kwargs,
            force_scale=FarForceScaleInputs(
                node_mass=jnp.ones((6,)),
                node_radii=jnp.zeros((6,)),
                softening_sq=jnp.asarray(0.0),
                gravitational_constant=1.0,
            ),
        )
    with pytest.raises(ValueError, match="cross_reverse"):
        _accumulate(**kwargs)


@pytest.fixture(scope="module")
def tree_and_upward():
    """A real ``Tree`` and ``TreeUpwardData`` (CI's unit jobs type-check the
    sweep's arguments, so placeholders would be refused before the check)."""
    from jaccpot.runtime import _fmm_impl as fmm_impl_private

    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (128, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, (128,)))
    solver = fmm_impl_private.FMMEngine(expansion_basis="solidfmm", mac_type="bh")
    state = solver.prepare_state(positions, masses, leaf_size=8, max_order=2)
    return state.tree, state.upward


def test_cross_reverse_raises_without_cross_sources_in_the_sweep(tree_and_upward):
    from jaccpot.runtime.kernels._l2l import _prepare_solidfmm_downward_sweep

    tree, upward = tree_and_upward
    with pytest.raises(ValueError, match="cross_reverse"):
        _prepare_solidfmm_downward_sweep(
            tree,
            upward,
            theta=0.5,
            mac_type="dehnen",
            cross_reverse=lambda *a: None,
        )


def test_the_placeholder_seams_raise_in_reverse_only(monkeypatch):
    """Seam A's placeholder: the production launch forward (bitwise), and its
    reverse raises the not-wired error at trace time."""
    from jaccpot.distributed.grad import placeholder_far_seam
    from jaccpot.pallas.m2l_real_csr import m2l_real_csr_tables
    from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas

    order = 2
    c = int(np.asarray(m2l_real_csr_tables(order)["idx"]).max()) + 1
    rng = np.random.default_rng(0)
    n_local, k = 3, 2
    mp = jnp.asarray(rng.normal(size=(n_local + k, c)))
    cen = jnp.asarray(rng.normal(size=(n_local + k, 3)) * 0.1)
    cen = cen.at[n_local:].add(3.0)
    # CSR over the local rows: each local target reads the two imported rows
    src = jnp.asarray([3, 4, 3, 4, 3, 4], jnp.int32)
    offsets = jnp.asarray([0, 2, 4, 6], jnp.int32)
    args = (mp, cen, src, offsets, jnp.int32(6), order, 32, True, "triton", 1)
    seam = placeholder_far_seam(None)
    got = seam(*args, n_local, True, False)
    want = m2l_real_csr_lanes_pallas(
        mp,
        cen,
        src,
        offsets,
        order=order,
        active_pair_count=jnp.int32(6),
        n_targets=n_local,
        k_lanes=32,
        interpret=True,
        backend="triton",
        num_warps=1,
        presorted=True,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert float(jnp.max(jnp.abs(got))) > 0.0
    with pytest.raises(NotImplementedError, match="cross seam A not wired yet"):
        jax.grad(lambda m: jnp.sum(seam(m, *args[1:], n_local, True, False)))(mp)


def test_the_pair_hashes_agree_at_both_ends_and_ignore_order():
    """Device 0 and device 1 key their shared pairs alike; a permuted list hashes
    the same; a pair the other end lacks breaks the agreement."""
    S, ndev = 8, 2
    pairs = [(0, 3), (2, 5), (7, 1)]  # (device 0's index, device 1's index)
    live = jnp.ones((3,), bool)
    h0 = _peer_pair_hashes(
        live,
        jnp.asarray([a for a, _ in pairs]),
        jnp.full((3,), 1),
        jnp.asarray([b for _, b in pairs]),
        jnp.int32(0),
        S,
        ndev,
    )
    perm = [2, 0, 1]
    h1 = _peer_pair_hashes(
        live,
        jnp.asarray([pairs[i][1] for i in perm]),
        jnp.zeros((3,), jnp.int32),
        jnp.asarray([pairs[i][0] for i in perm]),
        jnp.int32(1),
        S,
        ndev,
    )
    H = np.stack([np.asarray(h0), np.asarray(h1)])
    assert np.all(H[0, 0] == 0) and np.all(H[1, 1] == 0)
    assert np.array_equal(H[0, 1], H[1, 0]) and np.any(H[0, 1] != 0)
    h1_short = _peer_pair_hashes(
        jnp.asarray([True, True, False]),
        jnp.asarray([pairs[i][1] for i in perm]),
        jnp.zeros((3,), jnp.int32),
        jnp.asarray([pairs[i][0] for i in perm]),
        jnp.int32(1),
        S,
        ndev,
    )
    assert not np.array_equal(H[0, 1], np.asarray(h1_short)[0])
