"""The fused mesh lane conserves momentum across devices to round-off (M1, #173).

Every device walks the gathered summary trees and evaluates ITS half of every cross
pair from the other device's shipped data (``cross._symmetric_exchange``): the far
half as an M2L of the peer's multipole into its own local expansion, the near half
as a direct sum over the peer's particles. One card does the same with its own pairs
(the directed lists of ``_directed_csr_from_canonical``: each canonical pair once per
direction), and both ends of a cross pair read bitwise-identical multipoles, centres
and particles. So a cross pair cancels in ``sum_i m_i a_i`` as well as a local pair
does, and no reaction has to travel back (track M1's Step 0: fp64, N = 8192 Plummer +
disc on 2 forced CPU devices, ``|sum m a| / sum m |a|`` 1.9e-17 with the cross part
alone at 2.6e-17 while it carries 57 % of ``sum m |a|``).

This file pins that, in fp64 where round-off is unambiguous: two devices, the start
force with and without the cross hook (the local-only force), and

* the total ``|sum m a| / sum m |a|`` and the cross part's own residual at round-off
  (``TOL``), with the cross part a real share of the force (not a vacuous pass);
* the control: dropping the cross far pairs on ONE device (the other keeps its
  half: what a one-sided exchange would do) breaks it by orders of magnitude.

The CPU end-to-end pattern and toy size of ``test_mesh_potential.py`` (its ``mac``
fixture: the large-N profile reached on CPU, the all-gather exchange, the Pallas
kernels interpreted; N = 2048 uniform cube, leaf 64, p4, eps 1e-2), with a float64
working dtype. Marker ``distributed_segment``.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_mesh_cross_momentum.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import decompose  # noqa: E402
from jaccpot.distributed.template_force import FusedMeshForce  # noqa: E402
from tests.distributed import test_mesh_potential as tmp  # noqa: E402
from tests.distributed.test_mesh_potential import mac  # noqa: E402,F401

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

#: ``|sum m a| / sum m |a|`` in fp64: Step 0 measured 1.9e-17 (total) and 2.6e-17
#: (the cross part alone) at N = 8192; a summation over ~1e3 rows per device leaves
#: room for a few hundred ulps
TOL = 1e-13
#: the cross part's share of ``sum m |a|`` on two devices of the uniform cube: about
#: one half (a cross pair is half the force of a particle near the cut)
MIN_CROSS_SHARE = 0.1


def _solver(mac):
    criterion = mac == "dehnen_error"
    extra = (
        dict(
            use_pallas=True,
            adaptive_eps=tmp.EPS,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if criterion
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=tmp.THETA,
        G=1.0,
        softening=tmp.SOFT,
        working_dtype=jnp.float64,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=tmp.LEAF,
                leaf_partition="cells",
                leaf_capacity=tmp.LEAF_CAP,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=tmp.M2L_CHUNK),
            mac_type=mac,
        ),
        fixed_order=tmp.ORDER,
        **extra,
    )


def _one_sided_far(hook):
    """``hook`` with the cross far pairs dropped on device 0 only: device 1 still
    applies device 0's multipoles, device 0 no longer applies device 1's."""

    def dropped(*args, **kwargs):
        out = hook(*args, **kwargs)
        first = jax.lax.axis_index(tmp.AXIS) == 0
        dead = tuple(jnp.where(first, -1, part) for part in out[2:4])
        return (out[0], out[1], *dead, *out[4:])

    for attr in ("flag_sink", "needs_sink", "error_criterion"):
        setattr(dropped, attr, getattr(hook, attr))
    return dropped


def _momentum(force, mesh, shard, f_b, hook):
    """``(acceleration, per-device [sum m a, sum m |a|])`` of the start force with
    ``hook`` as the cross hook (``None``: the local-only force), fp64 sums."""
    from jax.sharding import PartitionSpec as P

    keep = force.hook, force.near_sink
    sink = keep[1] if hook is not None else None

    def body(x, m, n, ids, *fb):
        live = jnp.minimum(n[0], int(x.shape[0]))
        force.hook, force.near_sink = hook, sink
        try:
            out = force.shard(x, m, live, ids, *fb)
        finally:
            force.hook, force.near_sink = keep
        rows = (jnp.arange(int(x.shape[0])) < live)[:, None]
        acc = jnp.where(rows, out[0].astype(jnp.float64), 0.0)
        mm = m.astype(jnp.float64)[:, None]
        row = jnp.concatenate([jnp.sum(mm * acc, 0), jnp.sum(mm * jnp.abs(acc), 0)])
        return acc, row[None], jnp.asarray(out[1]).reshape(1)

    spec = P(tmp.AXIS)
    args = (shard.positions, shard.masses, shard.count, shard.ids)
    args += () if f_b is None else (f_b,)
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * len(args),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )
    )
    force.before_trace()
    with force.context():
        acc, rows, bits = program(*args)
    assert int(np.bitwise_or.reduce(np.asarray(bits))) == 0, bits
    return np.asarray(acc), np.asarray(rows)


def _residual(rows):
    return float(np.linalg.norm(rows[:, :3].sum(0)))


def test_cross_pairs_cancel_in_the_momentum_to_round_off(mac):
    """Two devices, fp64: ``sum m a`` of the mesh force and of its cross part alone
    at round-off, the cross part a real share of the force; dropping the cross far
    pairs on one device only (under the geometric MAC) breaks it."""
    mesh = tmp._mesh(tmp.NDEV)
    pos, vel, mass = tmp._cube(tmp.N, seed=0)
    pos, vel, mass = (np.asarray(a, np.float64) for a in (pos, vel, mass))
    shard, _ = decompose(mesh, pos, vel, mass, cap=tmp.CAP, num_samples=64)
    force = FusedMeshForce(
        _solver(mac),
        mesh,
        leaf_size=tmp.LEAF,
        max_order=tmp.ORDER,
        theta=tmp.THETA,
        cross_caps=CrossCapacities(max_cells=tmp.LEAF_CAP, leaf_width=tmp.LEAF),
    )
    force.prepare(shard.positions, shard.masses, shard.count)
    assert force.hook is not None
    f_b = force.force_scale_seed if force.scale_carry else None

    acc, rows = _momentum(force, mesh, shard, f_b, force.hook)
    acc_local, rows_local = _momentum(force, mesh, shard, f_b, None)
    m = np.asarray(shard.masses, np.float64)
    scale = float(np.sum(m * np.linalg.norm(acc, axis=1)))
    cross = acc - acc_local
    cross_scale = float(np.sum(m * np.linalg.norm(cross, axis=1)))
    total = _residual(rows) / scale
    cross_alone = _residual(rows - rows_local) / cross_scale
    local_alone = _residual(rows_local) / scale
    print(
        f"{mac}: |sum m a| / sum m|a| total {total:.2e}, local part {local_alone:.2e},"
        f" cross part {cross_alone:.2e} (share {cross_scale / scale:.2f})"
    )
    assert cross_scale / scale > MIN_CROSS_SHARE, cross_scale / scale
    assert total < TOL, total
    assert cross_alone < TOL, cross_alone
    # each device's cross part is real: the two halves cancel, they are not zero
    per_device = np.linalg.norm((rows - rows_local)[:, :3], axis=1)
    assert np.all(per_device > 1e3 * TOL * cross_scale), per_device

    if mac == "dehnen":
        _, rows_one_sided = _momentum(
            force, mesh, shard, f_b, _one_sided_far(force.hook)
        )
        broken = _residual(rows_one_sided) / scale
        print(f"one-sided cross far pairs: {broken:.2e}")
        assert broken > 1e6 * TOL, broken
