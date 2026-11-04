"""Scenarios for ``test_multiprocess_runtime.py``: each runs in a child process.

Every function takes the device count ``n`` and returns host arrays and
JSON-able values; the harness (``_multiprocess_harness``) runs it as ONE process
with ``n`` forced CPU devices and as ``n`` processes with one device each, and
the test requires the two results bitwise equal. Inputs: the single process
passes the whole initial condition, each of the ``n`` processes only its id
range (:class:`~jaccpot.distributed.multiprocess.ProcessLocalParticles`).

Not a test module (no ``test_`` prefix): the children import it after
``jax.distributed`` is initialised.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.distributed import multiprocess as mp

AXIS = "gpus"


def _input(state: np.ndarray, masses: np.ndarray, mesh) -> tuple:
    """The run's input: the whole set (one process) or this process's ids."""
    if not mp.is_multiprocess():
        return state, masses
    lo, hi = mp.local_id_range(int(state.shape[0]), mesh)
    return (
        mp.ProcessLocalParticles(state[lo:hi], masses[lo:hi], int(state.shape[0])),
        None,
    )


def _edges(res) -> list:
    """What every edge reported: diagnostics, counts, plan record."""
    rows = []
    for edge in res.edges:
        plan = edge.plan or {}
        diag = edge.diagnostics or {}
        rows.append(
            {
                "step": edge.step,
                "retries": edge.retries,
                "diagnostics": {
                    k: (None if v is None else np.asarray(v).tolist())
                    for k, v in diag.items()
                },
                "plan": plan,
            }
        )
    return rows


def _reports(handle) -> list:
    return [
        {
            "retries": r.retries,
            "bits": r.bits,
            "needs": r.needs,
            "failures": r.failures,
            "counts": r.counts,
            "migrated": r.migrated,
        }
        for r in handle.reports
    ]


# ----------------------------------------------------------------------
# The helpers.
# ----------------------------------------------------------------------


def helpers(n: int) -> dict:
    """``mesh_sum``, ``mesh_moments``, ``host_value``, ``decompose(_local)``."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.rollout import decompose, decompose_local

    mesh = mp.global_mesh(AXIS)
    rows = 512
    rng = np.random.default_rng(3)
    full = rng.standard_normal((n * rows, 2, 3)) * np.exp(
        rng.standard_normal((n * rows, 1, 1)) * 4
    )
    masses = np.abs(rng.standard_normal(n * rows))
    sharding = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(full, sharding)
    m = jax.device_put(masses, sharding)
    summed = jax.jit(
        jax.shard_map(
            lambda a: mp.mesh_sum(jnp.sum(a[:, 0], axis=0), AXIS),
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(),
            check_vma=False,
        )
    )(x)
    moments = mp.mesh_moments(x, m, mesh, AXIS)
    out = {
        "mesh_sum": mp.host_value(summed),
        "host_value": mp.host_value(x),
        **{f"moment_{k}": np.asarray(v) for k, v in moments.items()},
    }
    # the decomposition: the whole set from one host, or each process's ids
    num = 1000
    pos = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
    vel = rng.normal(size=(num, 3)).astype(np.float32)
    mass = np.full(num, 1.0 / num, np.float32)
    cap = int(np.ceil(1.4 * num / n))
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(num, mesh)
        shard, diag = decompose_local(
            mesh,
            pos[lo:hi],
            vel[lo:hi],
            mass[lo:hi],
            num_particles=num,
            cap=cap,
            num_samples=64,
            axis_name=AXIS,
        )
    else:
        shard, diag = decompose(
            mesh, pos, vel, mass, cap=cap, num_samples=64, axis_name=AXIS
        )
    for name in ("positions", "velocities", "masses", "ids", "count"):
        out[f"decompose_{name}"] = mp.host_value(getattr(shard, name))
    for key, value in diag.items():
        out[f"decompose_diag_{key}"] = np.asarray(value)
    # one entry per process, each this process's own object, types kept
    gathered = mp.allgather_object((jax.process_index(), np.int32(7)))
    out["allgather_object_ok"] = gathered == [
        (p, np.int32(7)) for p in range(jax.process_count())
    ] and all(type(v[1]) is np.int32 for v in gathered)
    return out


# ----------------------------------------------------------------------
# The runner with the direct force.
# ----------------------------------------------------------------------


def harmonic(state_rows, params):
    """A harmonic trap toward ``params["centre"]`` (module level: one object)."""
    return -params["k"] * (state_rows[:, 0] - jnp.asarray(params["centre"]))


def harmonic_potential(state_rows, params):
    """Its potential per unit mass."""
    d = state_rows[:, 0] - jnp.asarray(params["centre"])
    return 0.5 * params["k"] * jnp.sum(d * d, axis=1)


def direct_runner(n: int) -> dict:
    """``ProductionRunner(mesh=...)`` with ``DirectMeshForce`` (N = 256, float64).

    Migrating and non-migrating, a repartition before every segment, the energy
    and moments at every edge, a run continued in memory from its
    ``RunResult``; under several processes a restart file is refused.
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig

    num, g, soft = 256, 1e-3, 0.05
    mesh = mp.global_mesh(AXIS)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1.0, size=(num, 3))
    vel = 0.5 * rng.normal(size=(num, 3)) + np.array([0.6, 0.0, 0.0])
    state, masses = np.stack([pos, vel], axis=1), np.full(num, 1.0 / num)
    params = {"k": 0.5, "centre": (0.5, 0.5, 0.5)}
    out: dict = {}

    def runner(migrate, **segment):
        return ProductionRunner(
            FastMultipoleMethod(preset="large_n_gpu", G=g, softening=soft),
            segment=SegmentConfig(steps=2, diagnostics_every_energy=1, **segment),
            external_acceleration_fn=harmonic,
            external_potential_fn=harmonic_potential,
            mesh=MeshRunConfig(
                mesh,
                cap=int(np.ceil(1.6 * num / n)),
                repartition_every=1,
                num_samples=64,
                migrate=migrate,
                force=DirectMeshForce(G=g, softening=soft, axis_name=AXIS),
            ),
        )

    for migrate in (False, True):
        run = runner(migrate)
        s, m = _input(state, masses, mesh)
        first = run.run(s, m, dt=0.02, num_steps=8, external_params=params)
        edges = _edges(first)
        res = run.run(dt=0.02, num_steps=12, external_params=params, restart_from=first)
        got = res.handle.gather()
        tag = f"migrate{int(migrate)}"
        out[f"{tag}_state"] = res.state
        out[f"{tag}_acceleration"] = got["acceleration"]
        out[f"{tag}_owner"] = got["owner"]
        out[f"{tag}_edges"] = edges + _edges(res)
        out[f"{tag}_reports"] = _reports(res.handle)
    # restart files: refused under several processes, before anything runs
    refused = "one process"
    if mp.is_multiprocess():
        refused = "not refused"
        try:
            s, m = _input(state, masses, mesh)
            runner(True, restart_every=1, restart_path="r_{segment}.npz").run(
                s, m, dt=0.02, num_steps=2, external_params=params
            )
        except NotImplementedError as exc:
            refused = "refused" if "orbax" in str(exc) else str(exc)
    out["restart_refused"] = refused
    return out


# ----------------------------------------------------------------------
# The runner with the fused force.
# ----------------------------------------------------------------------


def _cpu_fused_lane(criterion: bool) -> None:
    """The CPU end-to-end switches of ``test_mesh_segment_runner.py`` (for good)."""
    import yggdrax.distributed.comm as ygg_comm

    jax.default_backend = lambda: "gpu"
    resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = lambda method="auto": (
        "buf" if method == "auto" else resolve(method)
    )
    os.environ["JACCPOT_NEARFIELD_PALLAS_INTERPRET"] = "1"
    os.environ["JACCPOT_STATIC_STRICT_FUSED_M2L_CSR"] = "0"
    os.environ["JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"] = "65536"
    os.environ["JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF"] = "32"
    if criterion:
        os.environ["JACCPOT_NEARFIELD_LEAFPAIR_CSR"] = "1"
        os.environ["JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB"] = "0"


def fused_runner(n: int) -> dict:
    """``ProductionRunner(mesh=...)`` on ``FusedMeshForce`` (the CPU fused lane).

    ``M2_MAC`` (``dehnen`` / ``dehnen_error``), ``M2_MIGRATE`` (0/1),
    ``M2_LEAF_CAPACITY`` (64) and ``M2_IC`` (``cube``; ``collapse``: the A3
    start, whose trees outgrow a leaf capacity of 44 on two devices -- a TREE
    re-plan -- with the library's cross sizing) select the case. (On four
    devices the A3 start does not fit 44 leaves at all, in one process or four.)
    N = 2048 uniform cube (30x the segment tests' speeds when migrating, so rows
    change device), leaf 64, p4, theta 0.8, eps 1e-2, 3 segments of 2 steps, a
    repartition before every segment after the first, the energy at every edge,
    and two cross buffers too small for the start (the first segment's cross
    flag fires and is re-planned).
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import (
        FarFieldConfig,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )
    from jaccpot.distributed.cross import CrossCapacities
    from jaccpot.production import (
        GalaxySettings,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
    )

    mac = os.environ.get("M2_MAC", "dehnen")
    migrate = os.environ.get("M2_MIGRATE", "1") == "1"
    leaf_capacity = int(os.environ.get("M2_LEAF_CAPACITY", "64"))
    collapse = os.environ.get("M2_IC", "cube") == "collapse"
    criterion = mac == "dehnen_error"
    _cpu_fused_lane(criterion)
    if collapse:
        # the library's own walk caps: a NAMED near cap is never widened
        os.environ.pop("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", None)
    num, leaf, order, theta, soft = 2048, 64, 4, 0.8, 1e-2
    extra = (
        dict(
            use_pallas=True,
            adaptive_eps=1e-4,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if criterion
        else {}
    )
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=theta,
        G=1.0,
        softening=soft,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=leaf,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=512),
            mac_type=mac,
        ),
        fixed_order=order,
        **extra,
    )
    settings = GalaxySettings(
        leaf_size=leaf,
        max_order=order,
        theta=theta,
        mac_type=mac,
        adaptive_eps=1e-4 if criterion else None,
    )
    cross: object = CrossCapacities(max_cells=64, leaf_width=leaf)._with(
        recv_node_cap=64, send_particle_cap=512
    )
    mesh = mp.global_mesh(AXIS)
    dt, num_steps = 1e-3, 6
    if collapse:
        # test_mesh_segment_runner's A3 start: a uniform cube drifting onto a
        # clipped Plummer sphere over 4 steps (masses 1e-9): the trees outgrow the
        # leaf capacity and the walks their lists (a TREE re-plan); the library
        # sizes the cross capacities (at the start and at the drift)
        cross, num_steps = "auto", 4
        rng = np.random.default_rng(0)
        start = rng.uniform(-1.0, 1.0, (num, 3))
        radius = np.linalg.norm(start, axis=1)
        rank = (np.argsort(np.argsort(radius)) + 0.5) / num
        radius_end = np.minimum(0.7 / np.sqrt(rank ** (-2.0 / 3.0) - 1.0), 1.5)
        end = start / radius[:, None] * radius_end[:, None]
        pos = start.astype(np.float32)
        vel = ((end - start) / (num_steps * dt)).astype(np.float32)
        mass = np.full(num, 1e-9, np.float32)
    else:
        rng = np.random.default_rng(0)
        pos = rng.uniform(-1.0, 1.0, (num, 3)).astype(np.float32)
        vel = (0.1 * rng.normal(size=(num, 3))).astype(np.float32)
        if migrate:
            vel = vel * np.float32(30.0)
        mass = np.full(num, 1.0 / num, np.float32)
    state = np.stack([pos, vel], axis=1)
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=2, diagnostics_every_energy=1),
        mesh=MeshRunConfig(
            mesh,
            cap=int(num / n * 1.3),
            num_samples=64,
            cross_caps=cross,
            migrate=migrate,
        ),
    )
    s, m = _input(state, mass, mesh)
    res = runner.run(s, m, dt=dt, num_steps=num_steps)
    got = res.handle.gather()
    out = {
        "state": res.state,
        "acceleration": got["acceleration"],
        "owner": got["owner"],
        "edges": _edges(res),
        "reports": _reports(res.handle),
    }
    if "force_scale" in got:
        out["force_scale"] = got["force_scale"]
    return out
