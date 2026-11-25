"""The fused mesh force's ``targets=`` hook (HH4c): each device's field at points that are not sources.

nornax's clusters on the mesh evaluate, on every device, the field of ITS galaxy rows
at all the clusters' stars on the topology its self-gravity just built
(``FusedMeshForce.shard(..., targets=...)``), and sum the partial fields over the
mesh. Here, two forced CPU devices, fp64, ``test_mesh_potential.py``'s toy system
(N = 2048 uniform cube, leaf 64, p4, geometric MAC):

* the hook does not change the force: acceleration, bits and needs equal the plain
  ``shard``'s, bitwise;
* the partial fields summed over the devices are the field of all the sources at
  the points (against the fp64 direct sum, within the FMM's error);
* ``potential_shard(..., targets=)`` returns the potential at the points too.

Marker ``distributed_segment`` (minutes of compile on CPU).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_mesh_force_targets.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jax.sharding import PartitionSpec as P  # noqa: E402

from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import decompose  # noqa: E402
from jaccpot.distributed.template_force import FusedMeshForce  # noqa: E402
from jaccpot.runtime.target_points import TargetPlan  # noqa: E402
from tests.distributed import test_mesh_potential as tmp  # noqa: E402
from tests.distributed.test_mesh_cross_momentum import _solver  # noqa: E402
from tests.distributed.test_mesh_potential import mac  # noqa: E402,F401

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

NPTS = 64


@pytest.mark.parametrize("mac", ["dehnen"], indirect=True)
def test_the_targets_hook_sums_to_the_field_and_leaves_the_force_alone(mac):
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
    rng = np.random.default_rng(3)
    pts = jnp.asarray(rng.uniform(-0.9, 0.9, (NPTS, 3)))
    plan = TargetPlan(
        num_points=NPTS,
        far_capacity=1 << 16,
        near_capacity=1 << 14,
        queue_capacity=1 << 15,
    )

    def targets(provider, topology):
        return provider.target_field(topology, pts, plan, potential=True)

    def body(x, m, n, ids):
        live = jnp.minimum(n[0], int(x.shape[0]))
        plain = force.shard(x, m, live, ids)
        hooked = force.shard(x, m, live, ids, targets=targets)
        field = hooked[-1]
        phi, bits, pfield = force.potential_shard(x, m, live, ids, targets=targets)
        same = jnp.all(plain[0] == hooked[0]) & (plain[1] == hooked[1])
        same = same & jnp.all(plain[2] == hooked[2])
        return (
            same.reshape(1),
            field.acc[None],
            field.ok.reshape(1),
            pfield.potential[None],
        )

    spec = P(tmp.AXIS)
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=(spec,) * 4,
            check_vma=False,
        )
    )
    force.before_trace()
    with force.context():
        same, acc, ok, pot = program(
            shard.positions, shard.masses, shard.count, shard.ids
        )
    assert bool(np.all(np.asarray(same)))
    assert bool(np.all(np.asarray(ok)))
    total = np.asarray(acc).sum(0)
    total_pot = np.asarray(pot).sum(0)
    # the fp64 direct sum (the engine's kernel, its softening) at the points
    from jaccpot.softening import (
        masked_pair_factors,
        resolve_softening_kernel,
        softening_params,
    )

    eng = force.engine
    kernel = resolve_softening_kernel(getattr(eng, "softening_kernel", None))
    params = softening_params(kernel, float(eng.softening), jnp.float64)
    dx = np.asarray(pts)[:, None, :] - pos[None, :, :]
    r2 = jnp.asarray(np.sum(dx * dx, -1))
    g, psi, _ = masked_pair_factors(
        r2, jnp.ones(r2.shape, bool), params, kernel, potential=True
    )
    want = -float(eng.G) * np.sum((mass[None, :] * np.asarray(g))[..., None] * dx, 1)
    want_pot = -float(eng.G) * np.sum(mass[None, :] * np.asarray(psi), 1)
    rel = np.linalg.norm(total - want) / np.linalg.norm(want)
    rel_pot = np.linalg.norm(total_pot - want_pot) / np.linalg.norm(want_pot)
    print(f"targets: rel-L2 field {rel:.2e}, potential {rel_pot:.2e}")
    assert rel < 1e-2 and rel_pot < 1e-2
