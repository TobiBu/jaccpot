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

import dataclasses
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


class KDK:
    """``nornax.schemes.LeapfrogKDK``'s expressions (I2: ``M2_SCHEME=kdk``)."""

    rowwise = True

    def key(self):
        return ("test.KDK", 1)

    def open_step(self, state, acceleration, dt):
        return state.at[:, 0].set(
            state[:, 0] + state[:, 1] * dt + 0.5 * acceleration * dt**2
        )

    def close_step(self, state_open, acceleration, acceleration_new, dt):
        return state_open.at[:, 1].set(
            state_open[:, 1] + 0.5 * (acceleration + acceleration_new) * dt
        )


def _scheme():
    """``M2_SCHEME``: ``kdk`` runs the runner with :class:`KDK` (I2), else none."""
    return KDK() if os.environ.get("M2_SCHEME") == "kdk" else None


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
            scheme=_scheme(),
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
# The escaper path on the mesh (MESC).
# ----------------------------------------------------------------------


def escaper_runner(n: int) -> dict:
    """``ProductionRunner(mesh=..., escapers=...)`` with escapers crossing.

    ESC's system (``test_production_runner_mesh_escapers._system``: a Plummer
    sphere, eight fliers leaving ``r_escape = 5`` at several edges and a returner
    coming back inside ``r_return = 4.5``), 3 segments of 10 steps continued in
    memory to 6, the energy at every edge, migrating. ``MESC_FORCE``: ``direct``
    (the default, :class:`~jaccpot.distributed.segment.DirectMeshForce`) or
    ``fused`` (the CPU fused lane, ``MESC_MAC`` ``dehnen`` / ``dehnen_error``).
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.escapers import EscaperPolicy
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig
    from tests.distributed import test_production_runner_mesh_escapers as tme

    fused = os.environ.get("MESC_FORCE", "direct") == "fused"
    mac = os.environ.get("MESC_MAC", "dehnen")
    mesh = mp.global_mesh(AXIS)
    state, masses = tme._system()
    if fused:
        from tests.integration.strict import test_escapers as te

        _cpu_fused_lane(True)
        solver, settings, force = te._solver(mac), te._settings(mac), None
    else:
        solver = FastMultipoleMethod(preset="large_n_gpu", G=tme.G, softening=tme.SOFT)
        settings = None
        force = DirectMeshForce(G=tme.G, softening=tme.SOFT, axis_name=AXIS)
    runner = ProductionRunner(
        solver,
        **({} if settings is None else {"settings": settings}),
        segment=SegmentConfig(steps=tme.STEPS, diagnostics_every_energy=1),
        escapers=EscaperPolicy(tme.R_ESCAPE, tme.R_RETURN),
        mesh=MeshRunConfig(
            mesh,
            cap=int(np.ceil(1.5 * state.shape[0] / n)),
            num_samples=64,
            migrate=True,
            force=force,
        ),
    )
    s, m = _input(state, masses, mesh)
    half = tme.STEPS * tme.SEGMENTS // 2
    first = runner.run(s, m, dt=tme.DT, num_steps=half)
    edges = _edges(first)
    res = runner.run(dt=tme.DT, num_steps=2 * half, restart_from=first)
    got = res.handle.gather()
    out = {
        "state": res.state,
        "owner": got["owner"],
        "edges": edges + _edges(res),
        "reports": _reports(res.handle),
        "split": res.escapers,
        "tail_ids": mp.host_value(res.handle.tail.ids),
    }
    # (a split that changed at the last edge dropped the carried acceleration)
    for name in ("acceleration", "force_scale"):
        if name in got:
            out[name] = got[name]
    return out


# ----------------------------------------------------------------------
# The runner with the fused force.
# ----------------------------------------------------------------------


def _cpu_fused_lane(criterion: bool) -> None:
    """The CPU end-to-end switches of ``test_mesh_segment_runner.py`` (for good)."""
    import yggdrax.distributed.comm as ygg_comm

    if jax.default_backend() == "gpu":
        return  # a real GPU: the real lane, the library's walk caps
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
    """``ProductionRunner(mesh=...)`` on ``FusedMeshForce`` (the CPU fused lane;
    on a GPU backend the real lane with the library's walk caps).

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
        scheme=_scheme(),
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


# ----------------------------------------------------------------------
# The collectives the mesh runner uses (the GPU spike: NCCL across processes).
# ----------------------------------------------------------------------


def _ragged_sizes(n: int) -> np.ndarray:
    """``[n, n]`` rows device ``s`` sends device ``i``: above the ~512-row
    threshold of the NCCL-fallback defect (#398), all distinct."""
    s, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (600 + 37 * s + 101 * i).astype(np.int32)


def collectives(n: int) -> dict:
    """Every collective of the mesh runner, inside ``jit(shard_map)``, 3 calls each.

    ``psum`` (float32 / float64, elementwise over 4096 x 7 and of scalars; int32),
    ``pmax``, ``pmin``, ``all_gather``, ``all_to_all``, ``ppermute`` and the ragged
    exchange (``M2_RAGGED``: ``native`` -- ``jax.lax.ragged_all_to_all`` --,
    ``exchange`` -- yggdrax's ``ragged_all_to_all_exchange`` (``auto``), with its
    flag guard -- or ``none``), each called three times with new data (the NCCL
    fallback's defect delivers no cross-device rows on the first call and hangs on
    the second). Returns every result and whether the exact ones match a host
    reference (``*_ok``).
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    ragged = os.environ.get("M2_RAGGED", "native")
    mesh = mp.global_mesh(AXIS)
    rows, feat = 4096, 7
    sharding = NamedSharding(mesh, P(AXIS))
    sizes = _ragged_sizes(n)
    c_in = int(sizes.sum(axis=1).max()) + 64
    c_out = int(sizes.sum(axis=0).max()) + 64
    width = 56

    def body(x32, x64, ints, payload):
        me = jax.lax.axis_index(AXIS)
        out = {
            "psum_f32": jax.lax.psum(x32, AXIS),
            "psum_f64": jax.lax.psum(x64, AXIS),
            "psum_f32_scalar": jax.lax.psum(jnp.sum(x32), AXIS)[None],
            "psum_f64_scalar": jax.lax.psum(jnp.sum(x64), AXIS)[None],
            "psum_i32": jax.lax.psum(ints, AXIS),
            "pmax": jax.lax.pmax(x32, AXIS),
            "pmin": jax.lax.pmin(ints, AXIS),
            "all_gather": jax.lax.all_gather(x32[:8], AXIS, tiled=True),
            "all_to_all": jax.lax.all_to_all(
                x32.reshape(n, rows // n, feat), AXIS, 0, 0, tiled=True
            ).reshape(rows, feat),
            "ppermute": jax.lax.ppermute(
                x64, AXIS, perm=[(d, (d + 1) % n) for d in range(n)]
            ),
        }
        out = {k: v[None] for k, v in out.items()}
        if ragged != "none":
            my = jnp.asarray(sizes)[me]
            if ragged == "exchange":
                from yggdrax.distributed.comm import ragged_all_to_all_exchange

                got, _, _ = ragged_all_to_all_exchange(
                    payload, my, output_capacity=c_out, axis_name=AXIS, fill_value=-1.0
                )
            else:
                full = jnp.asarray(sizes)
                in_off = jnp.cumsum(full, axis=1) - full
                out_off = jnp.cumsum(full, axis=0) - full
                got = jax.lax.ragged_all_to_all(
                    payload,
                    jnp.full((c_out, width), -1.0, payload.dtype),
                    in_off[me],
                    full[me],
                    out_off[me],
                    full[:, me],
                    axis_name=AXIS,
                )
            out["ragged"] = got[None]
        return out

    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * 4,
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    result: dict = {"ragged_mode": ragged, "backend": jax.default_backend()}
    for call in range(3):
        rng = np.random.default_rng(100 + call)
        x32 = rng.standard_normal((n * rows, feat)).astype(np.float32) * np.float32(
            10.0
        ) ** rng.integers(-6, 6, (n * rows, 1)).astype(np.float32)
        x64 = rng.standard_normal((n * rows, feat)) * 10.0 ** rng.integers(
            -12, 12, (n * rows, 1)
        )
        ints = rng.integers(-1000, 1000, (n * rows, feat)).astype(np.int32)
        # sender s's rows for i: value s * 1e6 + i * 1e4 + row (exact in f32)
        payload = np.full((n, c_in, width), -2.0, np.float32)
        for s in range(n):
            lo = 0
            for i in range(n):
                k = int(sizes[s, i])
                payload[s, lo : lo + k] = (
                    s * 1e6 + i * 1e4 + np.arange(k, dtype=np.float32)
                )[:, None]
                lo += k
        args = [
            jax.device_put(a, sharding)
            for a in (x32, x64, ints, payload.reshape(n * c_in, width))
        ]
        got = {k: mp.host_value(v) for k, v in program(*args).items()}
        for k, v in got.items():
            result[f"call{call}_{k}"] = v
        # the host references of the exact ones
        blocks = lambda a: a.reshape(n, rows, feat)  # noqa: E731
        ok = {
            "psum_i32": np.array_equal(
                got["psum_i32"][0], blocks(ints).sum(0, dtype=np.int32)
            ),
            "pmax": np.array_equal(got["pmax"][0], blocks(x32).max(0)),
            "pmin": np.array_equal(got["pmin"][0], blocks(ints).min(0)),
            "all_gather": all(
                np.array_equal(
                    got["all_gather"][d], blocks(x32)[:, :8].reshape(-1, feat)
                )
                for d in range(n)
            ),
            "all_to_all": all(
                np.array_equal(
                    got["all_to_all"][d].reshape(n, rows // n, feat),
                    blocks(x32)[:, d * (rows // n) : (d + 1) * (rows // n)],
                )
                for d in range(n)
            ),
            "ppermute": all(
                np.array_equal(got["ppermute"][(d + 1) % n], blocks(x64)[d])
                for d in range(n)
            ),
        }
        # an elementwise float psum: in mesh order (a0 + a1) + ..., or not?
        fold32, fold64 = blocks(x32)[0].copy(), blocks(x64)[0].copy()
        for d in range(1, n):
            fold32, fold64 = fold32 + blocks(x32)[d], fold64 + blocks(x64)[d]
        result[f"call{call}_psum_f32_mesh_order_differs"] = int(
            np.sum(got["psum_f32"][0] != fold32)
        )
        result[f"call{call}_psum_f64_mesh_order_differs"] = int(
            np.sum(got["psum_f64"][0] != fold64)
        )
        if ragged != "none":
            want = np.full((n, c_out, width), -1.0, np.float32)
            for i in range(n):
                lo = 0
                for s in range(n):
                    k = int(sizes[s, i])
                    want[i, lo : lo + k] = (
                        s * 1e6 + i * 1e4 + np.arange(k, dtype=np.float32)
                    )[:, None]
                    lo += k
            ok["ragged"] = np.array_equal(got["ragged"], want)
            result[f"call{call}_ragged_rows_wrong"] = int(
                np.any(got["ragged"] != want, axis=-1).sum()
            )
        result[f"call{call}_ok"] = {k: bool(v) for k, v in ok.items()}
    result["process_count"] = jax.process_count()
    result["devices"] = [str(d) for d in np.asarray(mesh.devices).reshape(-1)]
    return result


# ----------------------------------------------------------------------
# The fused force at a size that exercises the cross field (GPU).
# ----------------------------------------------------------------------


def _plummer(num: int, seed: int) -> tuple:
    """A G = M = a = 1 Plummer sphere (radii clipped at 20), float32, mass 1/N."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1.0 - 1e-6, num)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu, phi = rng.uniform(-1.0, 1.0, num), rng.uniform(0.0, 2.0 * np.pi, num)
    st = np.sqrt(1.0 - mu * mu)
    direction = np.stack([st * np.cos(phi), st * np.sin(phi), mu], 1)
    pos = r[:, None] * direction
    q = np.empty(num)
    todo = np.arange(num)
    while todo.size:
        x = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < x * x * (1.0 - x * x) ** 3.5
        q[todo[ok]] = x[ok]
        todo = todo[~ok]
    speed = q * np.sqrt(2.0) * (1.0 + r * r) ** -0.25
    mu, phi = rng.uniform(-1.0, 1.0, num), rng.uniform(0.0, 2.0 * np.pi, num)
    st = np.sqrt(1.0 - mu * mu)
    vel = speed[:, None] * np.stack([st * np.cos(phi), st * np.sin(phi), mu], 1)
    state = np.stack([pos, vel], axis=1).astype(np.float32)
    return state, np.full(num, 1.0 / num, np.float32)


def fused_runner_large(n: int) -> dict:
    """The production preset's fused mesh force on a Plummer sphere (GPU-sized).

    ``preset="galaxy"`` (leaf 64, p6, ``dehnen_error`` at 1e-4) or, with
    ``M2_MAC=dehnen``, the same with the ``dehnen`` MAC; ``M2_N_PER_DEVICE``
    particles per device (default 2e5), ``M2_SEGMENTS`` (3) segments of
    ``M2_STEPS`` (2) steps at ``M2_DT`` (5e-3; ``M2_SPEEDUP`` (4) times the
    equilibrium speeds so rows change device), a repartition before every
    segment after the first, the energy at every edge, migration unless
    ``M2_MIGRATE=0``. ``M2_SMALL_CROSS=1`` (default) starts from cross
    capacities far too small (the first segment's cross flag fires: a cross
    re-plan and a retry); ``0`` the library's sizing.
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.cross import CrossCapacities
    from jaccpot.production import (
        GALAXY_SETTINGS,
        GalaxySettings,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
    )

    mac = os.environ.get("M2_MAC", "dehnen_error")
    migrate = os.environ.get("M2_MIGRATE", "1") == "1"
    per_device = int(float(os.environ.get("M2_N_PER_DEVICE", "2e5")))
    segments = int(os.environ.get("M2_SEGMENTS", "3"))
    steps = int(os.environ.get("M2_STEPS", "2"))
    dt = float(os.environ.get("M2_DT", "5e-3"))
    speedup = float(os.environ.get("M2_SPEEDUP", "4"))
    small_cross = os.environ.get("M2_SMALL_CROSS", "1") == "1"
    num = per_device * n
    _cpu_fused_lane(mac == "dehnen_error")  # a no-op on a GPU
    if mac == "dehnen_error":
        solver = FastMultipoleMethod(preset="galaxy", G=1.0, softening=1e-3)
        settings = GALAXY_SETTINGS
    else:
        base = FastMultipoleMethod(preset="galaxy").advanced
        solver = FastMultipoleMethod(
            preset="galaxy",
            G=1.0,
            softening=1e-3,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None)
    cross: object = "auto"
    if small_cross:
        cross = CrossCapacities(max_cells=64, leaf_width=64)._with(
            recv_node_cap=64, send_particle_cap=512
        )
    state, mass = _plummer(num, seed=11)
    state[:, 1] *= np.float32(speedup)
    mesh = mp.global_mesh(AXIS)
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(steps=steps, diagnostics_every_energy=1),
        mesh=MeshRunConfig(mesh, cross_caps=cross, migrate=migrate),
    )
    s, m = _input(state, mass, mesh)
    res = runner.run(s, m, dt=dt, num_steps=steps * segments)
    got = res.handle.gather()
    out = {
        "state": res.state,
        "acceleration": got["acceleration"],
        "owner": got["owner"],
        "edges": _edges(res),
        "reports": _reports(res.handle),
        "num_particles": num,
        "peak_bytes": [
            int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
            for d in jax.local_devices()
        ],
    }
    if "force_scale" in got:
        out["force_scale"] = got["force_scale"]
    return out


# ----------------------------------------------------------------------
# The built-in writers, and how the run was launched.
# ----------------------------------------------------------------------


def writers(n: int) -> dict:
    """``SnapshotWriter`` / ``DiagnosticsLog`` on the direct-force runner.

    ``M2_WRITER_DIR`` is the side's directory: the default writers write there
    (process 0 only under several processes), the ``all_processes=True`` ones
    into ``.../p<process>/``. Returns the files' contents and, per process, what
    the writers kept and wrote.
    """
    import glob
    import json

    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.production import (
        DiagnosticsLog,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
        SnapshotWriter,
    )

    base = os.environ["M2_WRITER_DIR"]
    me = jax.process_index()
    mesh = mp.global_mesh(AXIS)
    num, g, soft = 128, 1e-3, 0.05
    rng = np.random.default_rng(1)
    state = np.stack(
        [rng.uniform(0.0, 1.0, (num, 3)), 0.3 * rng.normal(size=(num, 3))], axis=1
    )
    masses = np.full(num, 1.0 / num)
    shared_dir = os.path.join(base, "snapshots")
    log_path = os.path.join(base, "diagnostics.jsonl")
    own_dir = os.path.join(base, f"p{me}")
    snap = SnapshotWriter(shared_dir)
    log = DiagnosticsLog(log_path)
    snap_all = SnapshotWriter(own_dir, all_processes=True)
    log_all = DiagnosticsLog(os.path.join(own_dir, "d.jsonl"), all_processes=True)
    runner = ProductionRunner(
        FastMultipoleMethod(preset="large_n_gpu", G=g, softening=soft),
        segment=SegmentConfig(steps=2, diagnostics_every_energy=1),
        callbacks=(snap, log, snap_all, log_all),
        mesh=MeshRunConfig(
            mesh,
            cap=int(np.ceil(1.6 * num / n)),
            num_samples=64,
            force=DirectMeshForce(G=g, softening=soft, axis_name=AXIS),
        ),
    )
    s, m = _input(state, masses, mesh)
    runner.run(s, m, dt=0.01, num_steps=4)
    kept = mp.allgather_object(
        {
            "paths": [os.path.relpath(p, base) for p in snap.paths],
            "rows": len(log.rows),
            "paths_all": [os.path.relpath(p, base) for p in snap_all.paths],
            "rows_all": len(log_all.rows),
        }
    )
    out: dict = {"kept": kept}
    files = sorted(glob.glob(os.path.join(shared_dir, "*.npz")))
    out["snapshot_files"] = [os.path.basename(p) for p in files]
    for p in files:
        with np.load(p) as data:
            out[f"snapshot_{os.path.basename(p)}"] = np.asarray(data["state"])
    with open(log_path) as fh:
        rows = [json.loads(line) for line in fh]
    for row in rows:
        row.pop("wall_s", None)
    out["log_rows"] = rows
    out["own_dirs"] = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(base, "p*"))
    )
    out["own_files"] = sorted(
        os.path.relpath(p, base) for p in glob.glob(os.path.join(base, "p*", "*"))
    )
    return out


def launch(n: int) -> dict:
    """How this process joined: its devices and what ``initialize`` passed.

    Under the harness's SLURM launcher ``initialize()`` had no arguments (jax's
    SLURM detection; on a one-card task ``slurm_bound_device_ids``). A psum
    over the global mesh checks the processes are joined.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = mp.global_mesh(AXIS)
    info = mp.allgather_object(
        {
            "process": jax.process_index(),
            "count": jax.process_count(),
            "local": [d.id for d in jax.local_devices()],
            "bound": mp.slurm_bound_device_ids(),
        }
    )
    x = jax.device_put(np.arange(n, dtype=np.float32), NamedSharding(mesh, P(AXIS)))
    total = jax.jit(
        jax.shard_map(
            lambda a: jax.lax.psum(a, AXIS),
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(),
            check_vma=False,
        )
    )(x)
    return {"info": info, "psum": mp.host_value(total)}
