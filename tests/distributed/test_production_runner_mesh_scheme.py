"""The multi-GPU mode takes an integration scheme (multi-node plan I2), forced CPU devices.

``ProductionRunner(..., mesh=..., scheme=...)``: the scheme's ``open_step`` /
``close_step`` replace the segment step's drift and kick on each device's rows,
inside the segment's ``shard_map``. The gate: a rowwise KDK scheme -- a test-local
copy of ``nornax.schemes.LeapfrogKDK``'s expressions (nornax is not a jaccpot
dependency; the real one is tested against this runner in nornax's and Odisseo's
suites) -- is the built-in mesh runner BITWISE: the state and every edge's state,
gathers and diagnostics by id, with repartitions, migration, a retried segment,
chained runs and a restart file continued in memory and from disk, on 2 and (with
4 forced devices) 4 devices. A scheme that reads other rows must declare a mesh
form (``jaccpot.scheme.ShardedSplitStepScheme``) or is refused.

The direct force (partition independent) keeps this tier cheap; the fused force,
both MACs: ``test_production_runner_mesh_scheme_fused.py``.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_production_runner_mesh_scheme.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.segment import DirectMeshForce  # noqa: E402
from jaccpot.production import (  # noqa: E402
    MeshRunConfig,
    ProductionRunner,
    SegmentConfig,
)
from jaccpot.scheme import ShardedSplitStepScheme, check_mesh_scheme  # noqa: E402
from tests.distributed import test_production_runner_mesh as base  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the runner's mesh mode needs >= 2 devices"
)

AXIS, G, SOFT, DT = base.AXIS, base.G, base.SOFT, base.DT
PARAMS = base.PARAMS


class KDK:
    """``nornax.schemes.LeapfrogKDK``'s expressions (rowwise)."""

    rowwise = True

    def key(self):
        return ("test.KDK", 1)

    def open_step(self, state, acceleration, dt):
        return state.at[:, 0].set(
            state[:, 0] + dt * (state[:, 1] + (0.5 * dt) * acceleration)
        )

    def close_step(self, state_open, acceleration, acceleration_new, dt):
        return state_open.at[:, 1].set(
            state_open[:, 1] + 0.5 * (acceleration + acceleration_new) * dt
        )


class MeanFrameKDK(KDK):
    """NOT rowwise: the kick removes the mesh-wide mean velocity change of the live
    rows (equal masses: the momentum stays). Its mesh form reduces over the mesh."""

    rowwise = False

    def key(self):
        return ("test.MeanFrameKDK", 1)

    def close_step(self, state_open, acceleration, acceleration_new, dt):
        dv = 0.5 * (acceleration + acceleration_new) * dt
        return state_open.at[:, 1].set(state_open[:, 1] + dv - jnp.mean(dv, axis=0))

    def open_step_shard(self, state, acceleration, dt, *, live_rows, axis_name):
        return self.open_step(state, acceleration, dt)

    def close_step_shard(
        self, state_open, acceleration, acceleration_new, dt, *, live_rows, axis_name
    ):
        from jaccpot.distributed.multiprocess import _fold

        dv = 0.5 * (acceleration + acceleration_new) * dt
        w = jnp.where(live_rows, 1.0, 0.0)
        local = jnp.concatenate([jnp.sum(w[:, None] * dv, axis=0), jnp.sum(w)[None]])
        total = _fold(jax.lax.all_gather(local, axis_name, tiled=False))
        mean = total[:3] / total[3]
        return state_open.at[:, 1].set(
            state_open[:, 1] + jnp.where(live_rows[:, None], dv - mean, 0.0)
        )


class GlobalOnly(KDK):
    """Not rowwise, no mesh form."""

    rowwise = False


def _devices(ndev):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    return ndev


def _runner(
    scheme, *, ndev=2, migrate=False, force=None, callbacks=(), escapers=None, **segment
):
    config = dict(steps=2, diagnostics_every_energy=1)
    config.update(segment)
    return ProductionRunner(
        base._solver(),
        segment=SegmentConfig(**config),
        external_acceleration_fn=base.harmonic,
        external_potential_fn=base.harmonic_potential,
        callbacks=callbacks,
        scheme=scheme,
        escapers=escapers,
        mesh=MeshRunConfig(
            base._mesh(ndev),
            cap=int(np.ceil(1.6 * base.N / ndev)),
            repartition_every=1,
            num_samples=base.SAMPLES,
            migrate=migrate,
            force=(
                DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS)
                if force is None
                else force
            ),
        ),
    )


def _run(scheme, num_steps=8, **kw):
    state, masses = base._ic()
    snaps = base._Snapshots()
    kw["callbacks"] = tuple(kw.get("callbacks", ())) + (snaps,)
    res = _runner(scheme, **kw).run(
        state, masses, dt=DT, num_steps=num_steps, external_params=PARAMS
    )
    return res, snaps


def _same_run(got, want, what):
    """Two runs' results and every edge: bitwise by id."""
    (res_a, snaps_a), (res_b, snaps_b) = got, want
    base._bitwise(res_a.state, res_b.state, f"{what}: final state")
    assert [e.step for e in res_a.edges] == [e.step for e in res_b.edges]
    assert [e.retries for e in res_a.edges] == [e.retries for e in res_b.edges]
    for ea, eb in zip(res_a.edges, res_b.edges):
        assert ea.diagnostics == eb.diagnostics, (what, ea.step)
    assert snaps_a.states.keys() == snaps_b.states.keys()
    for step in snaps_a.states:
        base._bitwise(snaps_a.states[step], snaps_b.states[step], f"{what}: {step}")
        assert snaps_a.gathers[step].keys() == snaps_b.gathers[step].keys()
        for name in snaps_a.gathers[step]:
            base._bitwise(
                snaps_a.gathers[step][name],
                snaps_b.gathers[step][name],
                f"{what}: {name} at {step}",
            )


@pytest.mark.parametrize("migrate", [False, True], ids=["repartition", "migrate"])
@pytest.mark.parametrize("ndev", [2, 4])
def test_a_rowwise_scheme_is_the_built_in_mesh_runner_bitwise(ndev, migrate):
    """4 segments of 2 steps, a repartition (or a rebalance under migration) before
    each after the first, an external field with traced parameters, the energy at
    every edge: the KDK scheme == ``scheme=None``, bitwise by id."""
    _devices(ndev)
    built_in = _run(None, ndev=ndev, migrate=migrate)
    with_scheme = _run(KDK(), ndev=ndev, migrate=migrate)
    _same_run(with_scheme, built_in, f"{ndev} devices, migrate={migrate}")
    repart = [e.plan["repartition"] for e in with_scheme[0].edges[2:]]
    assert all(r["repartitioned"] for r in repart)


def test_chained_runs_and_restart_files_with_a_scheme(tmp_path):
    """Steps 0-4 then a continuation from the RunResult, and from the restart file
    (another runner, the same mesh): each == the straight scheme run == the
    built-in, bitwise; the file's arrays are the built-in's, and its meta records
    the scheme (only then). A restore under another scheme says so and is not
    called bitwise."""
    straight = _run(KDK())
    built_in = _run(None)
    _same_run(straight, built_in, "straight")

    first, _ = _run(
        KDK(), num_steps=4, restart_every=2, restart_path=str(tmp_path / "s.npz")
    )
    second = _runner(KDK()).run(
        dt=DT, num_steps=8, restart_from=first, external_params=PARAMS
    )
    assert second.restored["bitwise"]
    base._bitwise(second.state, built_in[0].state, "chained")
    assert second.edges[-1].diagnostics == built_in[0].edges[-1].diagnostics

    _run(None, num_steps=4, restart_every=2, restart_path=str(tmp_path / "n.npz"))
    import json

    with np.load(tmp_path / "s.npz") as fs, np.load(tmp_path / "n.npz") as fn:
        assert sorted(fs.files) == sorted(fn.files)
        for name in fs.files:
            if name != "meta":
                base._bitwise(fs[name], fn[name], f"restart file {name}")
        meta_s = json.loads(bytes(fs["meta"]).decode())
        meta_n = json.loads(bytes(fn["meta"]).decode())
    assert meta_s["scheme"]["key"] == repr(KDK().key()) and "scheme" not in meta_n

    resumed = _runner(KDK()).run(
        dt=DT, num_steps=8, restart_from=str(tmp_path / "s.npz"), external_params=PARAMS
    )
    assert resumed.restored["bitwise"], resumed.restored
    base._bitwise(resumed.state, built_in[0].state, "restored from the file")
    with pytest.warns(UserWarning, match="scheme"):
        other = _runner(None).run(
            dt=DT,
            num_steps=8,
            restart_from=str(tmp_path / "s.npz"),
            external_params=PARAMS,
        )
    assert "scheme" in other.restored["mismatches"]


def test_a_retried_segment_with_a_scheme_is_the_built_in_retry():
    """Armed after step 4, the third segment fails, is re-planned and re-run: with
    the scheme the run is the built-in's retried run, bitwise."""
    runs = []
    for scheme in (None, KDK()):
        force = base._Armed(retryable=True, G=G, softening=SOFT, axis_name=AXIS)

        def arm(edge, force=force):
            if edge.step == 4:
                force.armed = True

        runs.append(_run(scheme, force=force, callbacks=(arm,)))
    assert [e.retries for e in runs[1][0].edges] == [0, 0, 0, 1, 0]
    _same_run(runs[1], runs[0], "retried")


def test_a_scheme_that_reads_other_rows_needs_a_mesh_form():
    """A non-rowwise scheme without a mesh form is refused at construction (one
    card takes it); with one, it runs on every device's rows and its mesh-wide
    reduction holds: the momentum stays at its initial value (equal masses, a
    field that moves it otherwise)."""
    with pytest.raises(ValueError, match="not rowwise"):
        _runner(GlobalOnly())
    with pytest.raises(ValueError, match="not rowwise"):
        check_mesh_scheme(GlobalOnly())
    assert check_mesh_scheme(KDK()) is not None and check_mesh_scheme(None) is None
    assert isinstance(MeanFrameKDK(), ShardedSplitStepScheme)
    assert not isinstance(KDK(), ShardedSplitStepScheme)
    ProductionRunner(base._solver(), scheme=GlobalOnly())  # one card: fine

    state, masses = base._ic()
    p0 = np.sum(masses[:, None] * state[:, 1], axis=0)
    res, _ = _run(MeanFrameKDK())
    p_end = np.sum(res.masses[:, None] * res.state[:, 1], axis=0)
    np.testing.assert_allclose(p_end, p0, rtol=0, atol=1e-14)
    built_in, _ = _run(None)
    p_built_in = np.sum(built_in.masses[:, None] * built_in.state[:, 1], axis=0)
    assert np.abs(p_built_in - p0).max() > 1e-6, "the field must move the momentum"


def test_the_adapter_is_the_runners_force():
    """``ShardedMeshForce`` (nornax's protocol over a jaccpot mesh force): its
    acceleration on the runner's layout is the start force's self-gravity,
    bitwise, zero on padding rows; its carry folds the verdict."""
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed.sharded_force import ShardedMeshForce

    res, snaps = _run(None, num_steps=2)
    msr = res.handle
    adapter = ShardedMeshForce(msr.force)
    assert adapter.axis_name == AXIS
    layout = msr.state
    cap = msr.cap
    live = (jnp.arange(layout.ids.shape[0]) % cap) < jnp.repeat(layout.count, cap)

    def body(pos, m, ids, live_rows):
        acc, carry = adapter.shard_accelerations(pos, m, ids, live_rows)
        return acc, carry["bits"][None]

    with adapter.context():
        acc, bits = jax.jit(
            jax.shard_map(
                body,
                mesh=msr.mesh,
                in_specs=(P(AXIS),) * 4,
                out_specs=(P(AXIS), P(AXIS)),
                check_vma=False,
            )
        )(layout.state[:, 0], layout.masses, layout.ids, live)
    acc_h, live_h = np.asarray(acc), np.asarray(live)
    assert np.all(acc_h[~live_h] == 0) and np.all(np.asarray(bits) == 0)
    # the carried total minus the field = the self-gravity, row for row
    field = np.asarray(
        jax.vmap(lambda s: base.harmonic(s[None], PARAMS)[0])(layout.state)
    )
    total = np.asarray(layout.acceleration)
    np.testing.assert_allclose(
        acc_h[live_h], (total - field)[live_h], rtol=1e-12, atol=1e-14
    )
    carry0 = adapter.initial_carry(2)
    assert carry0["bits"].shape == (2,) and "force_scale" not in carry0


def test_a_scheme_with_escapers_is_refused_on_the_mesh():
    """Stacked on MESC: the escaper tail is stepped by the built-in KDK, so the mesh
    mode refuses a scheme together with escapers (at construction), and a segment
    runner with a scheme refuses an escaper tail. Either alone is fine; one card
    takes both."""
    from jaccpot.escapers import EscaperPolicy

    policy = EscaperPolicy(2.0)
    with pytest.raises(ValueError, match="escaper"):
        _runner(KDK(), escapers=policy)
    ProductionRunner(base._solver(), scheme=KDK(), escapers=policy)  # one card
    res, _ = _run(KDK(), num_steps=2)
    with pytest.raises(ValueError, match="escaper tail"):
        res.handle.attach_escapers(object(), G)
