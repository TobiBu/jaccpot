"""The fused mesh lane's self-gravity potential on forced CPU devices (S11 session 2, E).

``FusedMeshForce.potential_shard`` (``template_force.mesh_potential_shard``): the
force's rebuild, then ``phi_i = -G sum_{j != i} m_j psi(r_ij)`` over every device's
particles -- the local near and far fields (``FusedForceProvider.potential``), the
cross far pairs through the local expansions their M2L wrote, and the cross near
pairs from the cross near kernel's fourth lane. The energy diagnostic's potential
(``W = 1/2 sum m phi``), in a program of its own.

The CPU end-to-end pattern of ``test_mesh_segment_runner.py`` (the large-N profile
reached on CPU with ``jax.default_backend`` reporting ``"gpu"``, the all-gather
``"buf"`` exchange, the Pallas kernels interpreted) at its toy size: N = 2048 uniform
cube, leaf 64 (leaf capacity 128: one card holds all 2048), p4, eps 1e-2 (ferrers3),
on the geometric MAC (theta 0.8) and on ``dehnen_error`` (eq 16a at ``adaptive_eps =
1e-4``, with ``test_mesh_segment_dehnen_error.py``'s CSR near-field switches):

* one device: the mesh potential equals the single card's ``FusedForceProvider
  .potential`` on the same particles in the same box, bitwise (under the criterion
  from the same seed f_b);
* two devices: phi against the fp64 direct sum (all pairs, the solver's pair kernel)
  in class with one card on the same particles, ``W`` within the bound; dropping the
  cross near term or the cross far pairs fails it; padding rows zero; under the
  criterion the potential's lists are the force's at the same f_b.

Each case is under a minute of compilation on a loaded 64-core host (a potential
program compiles in ~20 s at this size), so the file runs in the segment job
(marker ``distributed_segment``).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_potential.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

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
from jaccpot.softening import pair_factors, softening_params_np  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

AXIS = "gpus"
NDEV = 2
N = 2048
LEAF = 64
LEAF_CAP = 128
ORDER = 4
THETA = 0.8
SOFT = 1e-2
EPS = 1e-4
CAP = int(N / NDEV * 1.3)
#: the flat M2L's chunk switch (test_mesh_segment_runner.py's M2L_CHUNK)
M2L_CHUNK = 512
#: W's relative error against the direct sum: one card on these particles measured
#: 1.94e-5 (geometric) and 1.84e-8 (criterion: nearly every pair direct); dropping
#: the cross near term or the cross far pairs moves W by 7.1e-2 / 3.3e-1
#: (geometric) and 3.5e-1 / 4.5e-2 (criterion)
W_BOUND = {"dehnen": 1e-4, "dehnen_error": 1e-6}
#: phi's rel-L2 against the direct sum, absolute: one card 1.88e-4 / 1.28e-6
PHI_BOUND = {"dehnen": 5e-4, "dehnen_error": 1e-5}

_CPU_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR": "0",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
}
#: eq (16b)'s near half is the CSR near-field lane's (test_mesh_segment_dehnen_error.py)
_CRITERION_ENV = {
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(params=["dehnen", "dehnen_error"])
def mac(request, monkeypatch):
    """The MAC of the case, with the segment tests' CPU switches (and, under the
    criterion, the CSR near field) and an empty level-shape registry."""
    from jaccpot.runtime import _level_shapes

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    resolve = ygg_comm.resolve_ragged_method
    monkeypatch.setattr(
        ygg_comm,
        "resolve_ragged_method",
        lambda method="auto": "buf" if method == "auto" else resolve(method),
    )
    env = dict(_CPU_ENV, **(_CRITERION_ENV if request.param == "dehnen_error" else {}))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})
    return request.param


def _solver(mac):
    criterion = mac == "dehnen_error"
    extra = (
        dict(
            use_pallas=True,
            adaptive_eps=EPS,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if criterion
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=LEAF_CAP,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=M2L_CHUNK),
            mac_type=mac,
        ),
        fixed_order=ORDER,
        **extra,
    )


def _mesh(ndev):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _cube(n, seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(n, 3))).astype(np.float32)
    return pos, vel, np.full(n, 1.0 / n, np.float32)


def _live_rows(count, cap):
    count = np.asarray(count)
    return np.concatenate(
        [np.arange(d * cap, d * cap + int(count[d])) for d in range(count.shape[0])]
    )


def _assert_bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{what}: {differ} of {got.size} elements differ, max "
        f"{float(np.nanmax(np.abs(got.astype(np.float64) - want))):.3e}"
    )


def direct_potential(positions, masses, kernel):
    """fp64 ``-G sum_{j != i} m_j psi(r_ij)`` over all pairs, the solver's kernel."""
    x = np.asarray(positions, np.float64)
    m = np.asarray(masses, np.float64)
    d = x[:, None, :] - x[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    _, psi, _ = pair_factors(
        r2, softening_params_np(kernel, SOFT), kernel, potential=True, xp=np
    )
    np.fill_diagonal(psi, 0.0)
    return -(psi * m[None, :]).sum(axis=1)


def _errors(phi, ref, masses):
    """``(rel-L2, max relative, W's relative error)`` of ``phi`` against ``ref``."""
    phi = np.asarray(phi, np.float64)
    m = np.asarray(masses, np.float64)
    w, w_ref = 0.5 * np.sum(m * phi), 0.5 * np.sum(m * ref)
    return (
        float(np.linalg.norm(phi - ref) / np.linalg.norm(ref)),
        float(np.max(np.abs(phi - ref) / np.abs(ref))),
        float(abs(w - w_ref) / abs(w_ref)),
    )


def _potential(
    force, mesh, positions, masses, count, ids, force_scale=None, needs=None
):
    """The mesh potential through a ``shard_map``, as a runner's energy program
    calls it: ``(phi flat, bits per device)`` as host arrays, then with ``needs``
    (:func:`_capture_needs`'s dict) the rebuild's needs per device. A fresh
    program each call: a changed hook or sink is traced."""
    from jax.sharding import PartitionSpec as P

    spec = P(AXIS)
    scale = () if force_scale is None else (force_scale,)

    def body(x, m, n, i, *fb):
        live = jnp.minimum(n[0], int(x.shape[0]))
        phi, bits = force.potential_shard(x, m, live, i, *fb)
        if needs is None:
            return phi, bits.reshape(1)
        return phi, bits.reshape(1), needs["needs"].reshape(1, -1)

    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * (4 + len(scale)),
            out_specs=(spec,) * (2 if needs is None else 3),
            check_vma=False,
        )
    )
    with force.context():
        force.before_trace()
        out = program(positions, masses, count, ids, *scale)
    return tuple(np.asarray(o) for o in out)


def _capture_needs(monkeypatch):
    """Record the needs ``mesh_potential_shard`` returns (``potential_shard`` drops
    them) in the trace that ran it."""
    import jaccpot.distributed.template_force as template_force

    captured = {}
    real = template_force.mesh_potential_shard

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured["needs"] = out[2]
        return out

    monkeypatch.setattr(template_force, "mesh_potential_shard", spy)
    return captured


def _single_card(mac, pos, mass, box):
    """The single card on ``pos`` in ``box``: its caps-only prepare, its seed f_b
    (input order; ``None`` without the criterion) and its potential program (a
    rebuild in the same box, ``FusedForceProvider.potential``)."""
    from jaccpot import FusedForceProvider
    from jaccpot.distributed.fused import cube_bounds

    single = _solver(mac)
    prov, topology = FusedForceProvider.prepare(
        single,
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        bounds=box,
        caps_only=True,
    )
    seed = None
    if prov.scale_carry:
        perm = np.asarray(topology.prepared.stub.tree.particle_indices)
        seed = np.zeros(pos.shape[0], np.float32)
        seed[perm] = np.asarray(single._impl._last_force_scale_particles)
    del topology

    @jax.jit
    def potential(x, m, f_b):
        # the box global_mesh_bounds builds over these particles
        box = cube_bounds(jnp.min(x, axis=0), jnp.max(x, axis=0), pad=1e-6)
        scale = {} if f_b is None else {"force_scale": f_b}
        topo = prov.rebuild(x, m, bounds=box, **scale)
        return prov.potential(topo), prov.capacity_ok(topo)

    phi, ok = potential(
        jnp.asarray(pos), jnp.asarray(mass), None if seed is None else jnp.asarray(seed)
    )
    assert bool(ok)
    return np.asarray(phi), seed


def test_one_device_mesh_potential_is_the_single_cards(mac):
    """One device, no padding rows, no cross hook: the mesh potential (its
    shard_map, the box's all-reduce, the merged caps-only template, ``num_valid``,
    the installed plan) equals the single card's ``FusedForceProvider.potential``
    in the same box, bitwise; under the criterion from the same seed f_b (the
    mesh's seed is the single card's, bitwise)."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.fused import host_mesh_bounds
    from jaccpot.distributed.template_force import FusedMeshForce

    n_one = N // 2
    mesh = _mesh(1)
    pos, _, mass = _cube(n_one, seed=1)
    sharding = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(pos, sharding)
    m = jax.device_put(mass, sharding)
    count = jax.device_put(np.asarray([n_one], np.int32), sharding)
    ids = jax.device_put(np.arange(n_one, dtype=np.int32), sharding)
    force = FusedMeshForce(
        _solver(mac), mesh, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    force.prepare(x, m, count)
    assert force.hook is None
    seed_mesh = force.force_scale_seed if force.scale_carry else None
    got, bits = _potential(force, mesh, x, m, count, ids, seed_mesh)
    assert int(np.bitwise_or.reduce(bits)) == 0, bits

    want, seed = _single_card(mac, pos, mass, host_mesh_bounds([pos], [n_one]))
    if seed is not None:
        _assert_bitwise(seed_mesh, seed, "the prepare's f_b seed")
    assert np.all(np.isfinite(want)) and np.all(want < 0)
    _assert_bitwise(got, want, "one-device mesh potential vs the single card's")


def _no_cross_far(hook):
    """``hook`` with every cross far pair dropped (dead on both sides), its sinks kept."""

    def dropped(*args, **kwargs):
        out = hook(*args, **kwargs)
        dead = jnp.full_like(out[2], -1), jnp.full_like(out[3], -1)
        return (out[0], out[1], *dead, *out[4:])

    for attr in ("flag_sink", "needs_sink", "error_criterion"):
        setattr(dropped, attr, getattr(hook, attr))
    return dropped


def test_two_device_mesh_potential_against_the_direct_sum(mac, monkeypatch):
    """Two devices: phi against the fp64 direct sum in class with one card on the
    same particles; ``W = 1/2 sum m phi`` within ``W_BOUND``, and outside it with the
    cross near term or the cross far pairs dropped -- both reach phi; padding rows
    zero. Under the criterion the rebuild runs on the given f_b: its needs (every
    list and cross capacity it measured) are the force's at the same f_b and move
    with it."""
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.fused import host_mesh_bounds
    from jaccpot.distributed.template_force import FusedMeshForce

    mesh = _mesh(NDEV)
    pos, vel, mass = _cube(N, seed=0)
    shard, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    solver = _solver(mac)
    kernel = solver._impl.softening_kernel
    force = FusedMeshForce(
        solver,
        mesh,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF),
    )
    force.prepare(shard.positions, shard.masses, shard.count)
    assert force.hook is not None
    seed = force.force_scale_seed if force.scale_carry else None
    rows = _live_rows(shard.count, CAP)
    ids = np.asarray(shard.ids)[rows]
    pad = np.setdiff1d(np.arange(NDEV * CAP), rows)
    assert pad.size > 0
    captured = _capture_needs(monkeypatch)

    def run(f_b=seed):
        phi, bits, needs = _potential(
            force,
            mesh,
            shard.positions,
            shard.masses,
            shard.count,
            shard.ids,
            f_b,
            needs=captured,
        )
        assert int(np.bitwise_or.reduce(bits)) == 0, bits
        assert np.all(phi[pad] == 0.0), "padding rows carry a potential"
        by_id = np.empty(N, phi.dtype)
        by_id[ids] = phi[rows]
        return by_id, needs

    ref = direct_potential(pos, mass, kernel)
    phi, needs = run()
    got = _errors(phi, ref, mass)
    box = host_mesh_bounds([pos], [N], cubic=bool(solver._impl._tree_cubic_bounds))
    phi_one, _ = _single_card(mac, pos, mass, box)
    one = _errors(phi_one, ref, mass)
    print(f"{mac}: mesh rel-L2 / max / W {got}, one card {one}")
    assert got[0] < PHI_BOUND[mac] and one[0] < PHI_BOUND[mac], (got, one)
    assert got[0] <= 1.5 * one[0] and got[1] <= 1.5 * one[1], (got, one)
    assert got[2] < W_BOUND[mac] and one[2] < W_BOUND[mac], (got, one)

    if seed is not None:
        # the potential's lists are the force's at the same f_b, and follow it
        spec = P(AXIS)

        def force_needs(x, m, n, i, f_b):
            out = force.shard(x, m, jnp.minimum(n[0], int(x.shape[0])), i, f_b)
            return out[2].reshape(1, -1)

        program = jax.jit(
            jax.shard_map(
                force_needs,
                mesh=mesh,
                in_specs=(spec,) * 5,
                out_specs=spec,
                check_vma=False,
            )
        )
        with force.context():
            force.before_trace()
            want = np.asarray(
                program(shard.positions, shard.masses, shard.count, shard.ids, seed)
            )
        np.testing.assert_array_equal(needs, want)
        assert not np.array_equal(run(4.0 * seed)[1], want)

    # mutations: each cross term dropped in turn fails the energy's bound
    sink = force.near_sink
    force.near_sink = None
    no_near = _errors(run()[0], ref, mass)
    force.near_sink = sink
    hook = force.hook
    force.hook = _no_cross_far(hook)
    no_far = _errors(run()[0], ref, mass)
    force.hook = hook
    print(f"{mac}: W without the cross near {no_near[2]:.3e}, far {no_far[2]:.3e}")
    assert no_near[2] > 100 * W_BOUND[mac], no_near
    assert no_far[2] > 100 * W_BOUND[mac], no_far
