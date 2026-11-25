"""A carried scheme's MESH FORM (HH4c), forced CPU devices.

``ProductionRunner(mesh=..., scheme=<carried scheme with bind_mesh>)`` binds the
scheme to the mesh and runs ``jaccpot.carried``'s loop through it (no
``MeshBackend``); a carried scheme without ``bind_mesh`` is refused, as before. A
test-local carried scheme (a drift of mesh-sharded rows; nornax's
``MeshClustersRunnerScheme`` is tested against this seam in nornax) checks the
dispatch, the result's mode, restart files written from sharded arrays and
restored onto ``restore_specs``' shardings, and the refusals. Also here:
``jaccpot.distributed.clusters_mesh.MortonPartition`` (the clusters' ownership:
rows routed with their payload, every live row owned by the device holding it).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_carried_mesh_seam.py -q
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.clusters_mesh import MortonPartition  # noqa: E402
from jaccpot.production import ProductionRunner, SegmentConfig  # noqa: E402
from jaccpot.scheme import check_mesh_scheme  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the mesh seam needs >= 2 devices"
)

AXIS = "gpus"
N, DT, STEPS = 64, 0.01, 2


def _mesh(n: int = 2) -> Mesh:
    return Mesh(np.array(jax.devices()[:n]), (AXIS,))


@dataclasses.dataclass
class _Carry:
    x: Any
    v: Any
    m: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _DriftOnMesh:
    """A carried scheme whose carry is mesh-sharded rows (positions drift)."""

    mesh: Any = None
    carried: ClassVar[bool] = True

    def bind_mesh(self, mesh: Any) -> "_DriftOnMesh":
        return dataclasses.replace(self, mesh=mesh)

    def _sh(self) -> Any:
        return NamedSharding(self.mesh, P(AXIS))

    def key(self) -> tuple:
        return ("test.DriftOnMesh",)

    def record(self) -> dict:
        return {"ndev": int(self.mesh.devices.size)}

    def start(self, solver, settings, state, masses, *, dt, t0, escapers):
        s = np.asarray(state)
        put = lambda a: jax.device_put(np.asarray(a), self._sh())  # noqa: E731
        return _Carry(put(s[:, 0]), put(s[:, 1]), put(masses))

    def segment(self, carry, dt, steps):
        x = jax.jit(lambda x, v: x + (dt * steps) * v)(carry.x, carry.v)
        assert x.sharding == self._sh()
        return dataclasses.replace(carry, x=x)

    def verdict(self, carry):
        return []

    def replanned(self, start, failed):  # pragma: no cover - never fails
        return start

    def describe(self, carry):
        return {
            "n_live": N,
            "n_stripped": 0,
            "bound_mass": [0.0],
            "strip_ledger": 0.0,
            "reserve": 0,
            "n_escapers": 0,
            "n_galaxy_rows": N,
        }

    def edge(self, carry, *, index, escapers_due, energy):
        return carry, self.describe(carry)

    def moments(self, carry):
        x, v, m = (np.asarray(a, np.float64) for a in (carry.x, carry.v, carry.m))
        return {
            "mass": np.sum(m),
            "momentum": np.sum(m[:, None] * v, 0),
            "angular_momentum": np.sum(m[:, None] * np.cross(x, v), 0),
            "mass_moment": np.sum(m[:, None] * x, 0),
            "kinetic": 0.5 * np.sum(m * np.sum(v * v, 1)),
            "momentum_scale": np.sum(m * np.linalg.norm(v, axis=1)),
        }

    def energy(self, carry):
        return {"W_gg": 0.0, "W_gc": 0.0, "W_cc": 0.0, "W_cc_inter": 0.0}

    def galaxy(self, carry):
        s = jnp.stack(
            [jnp.asarray(np.asarray(carry.x)), jnp.asarray(np.asarray(carry.v))], 1
        )
        return s, jnp.asarray(np.asarray(carry.m)), jnp.zeros_like(s[:, 0])

    def fingerprint_arrays(self, carry):
        return carry.x, carry.m

    def escaper_split(self, carry):
        return {"n_bound": N, "n_escapers": 0, "transfers": 0, "order": None}

    def checkpoint(self, carry):
        return {"x": carry.x, "v": carry.v, "m": carry.m}, {"names": ["x", "v", "m"]}

    def restore_specs(self, meta):
        return {name: (self._sh(), None) for name in meta["names"]}

    def restore(self, arrays, meta, *, solver, settings, dt, escapers):
        put = lambda a: (  # noqa: E731
            a
            if isinstance(a, jax.Array) and a.sharding == self._sh()
            else jax.device_put(np.asarray(a), self._sh())
        )
        return _Carry(put(arrays["x"]), put(arrays["v"]), put(arrays["m"])), []


def _system():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, 3))
    v = 0.1 * rng.normal(size=(N, 3))
    return np.stack([x, v], 1), np.full(N, 1.0 / N)


def _solver():
    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(preset="galaxy", G=1.0, softening=1e-2)


def _segment(**kw):
    return SegmentConfig(
        steps=STEPS,
        diagnostics_every=1,
        diagnostics_every_energy=1,
        domain_guard="off",
        **kw,
    )


class _NoMeshForm(_DriftOnMesh):
    bind_mesh = None  # type: ignore[assignment]


def test_a_carried_scheme_needs_a_mesh_form():
    assert check_mesh_scheme(_DriftOnMesh()) is not None
    with pytest.raises(ValueError, match="no mesh form"):
        check_mesh_scheme(_NoMeshForm())
    with pytest.raises(ValueError, match="no mesh form"):
        ProductionRunner(_solver(), mesh=_mesh(), scheme=_NoMeshForm())


def test_the_runner_binds_the_scheme_and_runs_its_loop(tmp_path):
    state, masses = _system()
    runner = ProductionRunner(
        _solver(),
        segment=_segment(
            restart_every=1, restart_path=str(tmp_path / "r{segment}.npz")
        ),
        mesh=_mesh(),
        scheme=_DriftOnMesh(),
    )
    assert runner.scheme.mesh is not None and runner._mesh_backend is None
    res = runner.run(state, masses, dt=DT, num_steps=3 * STEPS)
    assert res.mode == "mesh" and res.step == 3 * STEPS
    want = state[:, 0] + 3 * STEPS * DT * state[:, 1]
    np.testing.assert_allclose(np.asarray(res.state)[:, 0], want, rtol=1e-12)
    assert len(res.edges) == 4 and res.edges[-1].diagnostics["n_live"] == N
    # a restart file written from sharded arrays, restored onto their shardings
    again = ProductionRunner(
        _solver(), segment=_segment(), mesh=_mesh(), scheme=_DriftOnMesh()
    )
    cont = again.run(dt=DT, num_steps=3 * STEPS, restart_from=str(tmp_path / "r1.npz"))
    assert cont.restored["bitwise"], cont.restored
    np.testing.assert_array_equal(np.asarray(cont.state), np.asarray(res.state))
    with pytest.raises(ValueError, match="escaper"):
        from jaccpot.escapers import EscaperPolicy

        ProductionRunner(
            _solver(), mesh=_mesh(), scheme=_DriftOnMesh(), escapers=EscaperPolicy(5.0)
        )


@pytest.mark.distributed_checkpoint
def test_an_orbax_restart_restores_onto_the_mesh(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    state, masses = _system()
    runner = ProductionRunner(
        _solver(),
        segment=_segment(restart_every=1, restart_path=str(tmp_path / "c{segment}")),
        mesh=_mesh(),
        scheme=_DriftOnMesh(),
    )
    res = runner.run(state, masses, dt=DT, num_steps=2 * STEPS)
    again = ProductionRunner(
        _solver(), segment=_segment(), mesh=_mesh(), scheme=_DriftOnMesh()
    )
    cont = again.run(dt=DT, num_steps=2 * STEPS, restart_from=str(tmp_path / "c1"))
    assert cont.restored["bitwise"], cont.restored
    assert cont.carried.x.sharding == NamedSharding(_mesh(), P(AXIS))
    np.testing.assert_array_equal(np.asarray(cont.state), np.asarray(res.state))


def _repartition(part, x, m, n_live, payload, mesh):
    spec = P(AXIS)

    def body(x, m, n, ids):
        out = part.repartition(x, m, n[0], {"id": ids}, axis_name=AXIS)
        x2, m2, n2, moved, cuts, ok = out
        owners = part.owners(x2, cuts, axis_name=AXIS)
        me = jax.lax.axis_index(AXIS).astype(jnp.int32)
        live = jnp.arange(x.shape[0]) < n2
        mine = jnp.all(jnp.where(live, owners == me, True))
        return x2, m2, n2.reshape(1), moved["id"], mine.reshape(1), ok.reshape(1)

    prog = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=(spec,) * 6,
            check_vma=False,
        )
    )
    return prog(x, m, n_live, payload)


@pytest.mark.parametrize("cubic", [True, False])
def test_the_morton_partition_routes_rows_to_their_owners(cubic):
    mesh = _mesh()
    ndev, cap = 2, 48
    rng = np.random.default_rng(1)
    n = 70
    x = rng.normal(size=(n, 3))
    m = rng.uniform(0.5, 1.5, n)
    flat_x = np.zeros((ndev * cap, 3))
    flat_m = np.zeros(ndev * cap)
    ids = np.full(ndev * cap, -1, np.int32)
    counts = np.asarray([40, 30], np.int32)
    flat_x[:40], flat_m[:40], ids[:40] = x[:40], m[:40], np.arange(40)
    flat_x[cap : cap + 30], flat_m[cap : cap + 30] = x[40:], m[40:]
    ids[cap : cap + 30] = np.arange(40, 70)
    sh = NamedSharding(mesh, P(AXIS))
    put = lambda a: jax.device_put(a, sh)  # noqa: E731
    x2, m2, n2, id2, mine, ok = _repartition(
        MortonPartition(cubic=cubic),
        put(flat_x),
        put(flat_m),
        put(counts),
        put(ids),
        mesh,
    )
    assert bool(np.all(np.asarray(ok))) and bool(np.all(np.asarray(mine)))
    n2 = np.asarray(n2)
    assert int(n2.sum()) == n
    x2, m2, id2 = np.asarray(x2), np.asarray(m2), np.asarray(id2)
    got = np.concatenate([id2[d * cap : d * cap + n2[d]] for d in range(ndev)])
    np.testing.assert_array_equal(np.sort(got), np.arange(n))
    for d in range(ndev):
        rows = slice(d * cap, d * cap + n2[d])
        np.testing.assert_array_equal(x2[rows], x[id2[rows]])  # rows travel bitwise
        np.testing.assert_array_equal(m2[rows], m[id2[rows]])
        assert np.all(m2[d * cap + n2[d] : (d + 1) * cap] == 0)
        assert np.all(id2[d * cap + n2[d] : (d + 1) * cap] == -1)
