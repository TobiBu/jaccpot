"""``ProductionRunner(scheme=<a CarriedScheme>)``: the carried-scheme path (HH3b).

nornax's ``ClustersRunnerScheme`` (direct-sum clusters inside the FMM galaxy) owns its
carry and its programs; the runner keeps the loop (``jaccpot.carried``). jaccpot does
not import nornax, so these gates use a test-local carried scheme: a leapfrog over
the fused provider (``FusedForceProvider.step``) at the runner tests' operating point
(N = 512, CPU, the geometric MAC), with its force's plan recorded and pinned through
``jaccpot.carried``'s helpers the way nornax's scheme does. nornax's own suite runs
the clusters through this runner (``tests/integration/test_jaccpot_runner.py``).

* chained runs (``restart_from=RunResult``) equal one run, bitwise;
* a restart file (``.npz``) and an orbax checkpoint directory, continued in a fresh
  runner (a re-prepare at the restart's positions, the recorded plan installed),
  equal the uninterrupted run bitwise; under another scheme configuration the
  restore says it is not bitwise;
* a segment whose verdict fails is re-planned and re-run (the runner's retry loop);
* the diagnostics rows carry the moments, the energy and the scheme's columns;
* refusals: the mesh, an external field, ``potential_energy``, a scheme that lacks
  a member, a restart of another kind of run.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.carried import (
    install_recorded_tree,
    pinned_trace_state,
    provider_record,
    provider_with_record,
    trace_state,
)
from jaccpot.production import ProductionRunner, SegmentConfig, load_restart
from jaccpot.scheme import check_mesh_scheme, check_scheme, is_carried_scheme
from tests.integration.strict.test_production_runner import (  # noqa: F401
    DT,
    LEAF,
    ORDER,
    SETTINGS,
    _lane,
    _solver,
    _system,
)

STEPS, SEGMENTS = 2, 3


@dataclasses.dataclass
class _Carry:
    x: Any
    v: Any
    m: Any
    acc: Any
    aux: Any
    time: Any
    provider: Any
    pin: Any
    edges: int = 0


@dataclasses.dataclass(frozen=True, eq=False)
class _Leapfrog:
    """A carried scheme: KDK over ``FusedForceProvider.step``, the plan recorded and pinned."""

    tag: str = "a"
    fail_once_at: int = -1
    carried: ClassVar[bool] = True

    def __post_init__(self):
        object.__setattr__(self, "_progs", {})
        object.__setattr__(self, "_failed", [False])

    def key(self):
        return ("test.Leapfrog", self.tag)

    def record(self):
        return {"tag": self.tag}

    def _pinned(self, c):
        return pinned_trace_state([c.provider.engine], c.pin)

    def _prepare(self, solver, x, m):
        from jaccpot.runtime.fused_provider import FusedForceProvider

        prov, topo = FusedForceProvider.prepare(
            solver, x, m, leaf_size=LEAF, max_order=ORDER
        )
        return prov, trace_state([prov.engine])

    def start(self, solver, settings, state, masses, *, dt, t0, escapers):
        state = jnp.asarray(state)
        prov, pin = self._prepare(solver, state[:, 0], masses)
        c = _Carry(
            state[:, 0], state[:, 1], masses, None, None, jnp.asarray(t0), prov, pin
        )
        with self._pinned(c):
            acc, aux = jax.jit(lambda x, m: prov.step(x, m, None))(c.x, c.m)
        c.acc, c.aux = acc, aux
        return c

    def _segment_program(self, prov, dt, steps):
        key = (id(prov), dt, steps)
        if key not in self._progs:

            def seg(x, v, acc, aux, m):
                def body(carry, _):
                    x, v, acc, aux = carry
                    v = v + 0.5 * dt * acc
                    x = x + dt * v
                    acc, aux = prov.step(x, m, None, aux=aux)
                    v = v + 0.5 * dt * acc
                    return (x, v, acc, aux), None

                return jax.lax.scan(body, (x, v, acc, aux), None, length=steps)[0]

            self._progs[key] = (prov, jax.jit(seg))
        return self._progs[key][1]

    def segment(self, c, dt, steps):
        prog = self._segment_program(c.provider, float(dt), int(steps))
        with self._pinned(c):
            x, v, acc, aux = prog(c.x, c.v, c.acc, c.aux, c.m)
        return dataclasses.replace(
            c, x=x, v=v, acc=acc, aux=aux, time=c.time + steps * dt
        )

    def verdict(self, c):
        if (
            self.fail_once_at >= 0
            and c.edges == self.fail_once_at
            and not self._failed[0]
        ):
            self._failed[0] = True
            return ["test: walk"]
        from jaccpot.runtime.fused_topology import overflow_ok

        if (
            c.aux is not None
            and c.aux[1] is not None
            and not bool(overflow_ok(c.aux[1]))
        ):
            return ["walk/tree"]
        return []

    def replanned(self, start, failed):
        del failed
        return dataclasses.replace(start)

    def describe(self, c):
        n = int(c.x.shape[0])
        return {
            "n_live": n,
            "n_escapers": 0,
            "n_galaxy_rows": n,
            "n_stripped": 0,
            "bound_mass": [0.0],
            "strip_ledger": 0.0,
            "reserve": 0,
            "plan": {"tag": self.tag},
        }

    def edge(self, c, *, index, escapers_due, energy):
        c = dataclasses.replace(c, edges=c.edges + 1)
        return c, dict(self.describe(c), n_take=0, replans=[])

    def moments(self, c):
        from jaccpot.production import _moments

        return _moments(jnp.stack([c.x, c.v], 1), c.m)

    def energy(self, c):
        prov = c.provider
        with self._pinned(c):
            phi = jax.jit(lambda x, m: prov.potential(prov.rebuild(x, m)))(c.x, c.m)
        w = 0.5 * float(
            np.sum(np.asarray(c.m, np.float64) * np.asarray(phi, np.float64))
        )
        return {"W_gg": w, "W_gc": 0.0, "W_cc": 0.0, "W_cc_inter": 0.0}

    def galaxy(self, c):
        return jnp.stack([c.x, c.v], 1), c.m, c.acc

    def fingerprint_arrays(self, c):
        return c.x, c.m

    def escaper_split(self, c):
        return {"n_bound": int(c.x.shape[0]), "transfers": 0, "order": None}

    def checkpoint(self, c):
        arrays = {"x": c.x, "v": c.v, "m": c.m, "acc": c.acc, "time": c.time}
        aux = [None if a is None else f"aux{i}" for i, a in enumerate(c.aux)]
        arrays.update({k: a for k, a in zip(aux, c.aux) if k is not None})
        with self._pinned(c):
            prec = provider_record(c.provider)
        return arrays, {
            "pin": c.pin,
            "provider": prec,
            "edges": c.edges,
            "scheme": self.record(),
            "aux": aux,
        }

    def restore(self, arrays, meta, *, solver, settings, dt, escapers):
        x = jnp.asarray(arrays["x"])
        m = jnp.asarray(arrays["m"])
        # the writer's tree bounds before the re-prepare (HH5 item 3)
        install_recorded_tree([solver], meta["pin"])
        prov, _ = self._prepare(solver, x, m)
        prov, ok = provider_with_record(prov, meta["provider"])
        mismatches = [] if ok else ["template"]
        with pinned_trace_state([prov.engine], meta["pin"]):
            now = provider_record(prov)
        if now["provider_key"] != meta["provider"]["provider_key"]:
            mismatches.append("provider_key")
        if meta["scheme"] != self.record():
            mismatches.append("scheme")
        aux = tuple(None if k is None else jnp.asarray(arrays[k]) for k in meta["aux"])
        c = _Carry(
            x,
            jnp.asarray(arrays["v"]),
            m,
            jnp.asarray(arrays["acc"]),
            aux,
            jnp.asarray(arrays["time"]),
            prov,
            meta["pin"],
            int(meta["edges"]),
        )
        return c, mismatches


def _runner(scheme=None, **segment):
    segment.setdefault("steps", STEPS)
    segment.setdefault("domain_guard", "off")
    segment.setdefault("diagnostics_every_energy", 1)
    return ProductionRunner(
        _solver(),
        settings=SETTINGS,
        segment=SegmentConfig(**segment),
        scheme=scheme or _Leapfrog(),
    )


def _final(res):
    c = res.carried
    return np.asarray(c.x), np.asarray(c.v), np.asarray(c.acc)


def _same(a, b):
    for p, q in zip(a, b):
        np.testing.assert_array_equal(p, q)


@pytest.fixture(scope="module")
def one_go():
    state, masses = _system()
    res = _runner().run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    return _final(res), res


def test_the_protocol_checks():
    s = _Leapfrog()
    assert is_carried_scheme(s) and check_scheme(s) is s
    with pytest.raises(ValueError, match="multi-GPU"):
        check_mesh_scheme(s)

    class Half:
        carried = True

        def key(self):
            return ()

    with pytest.raises(TypeError, match="CarriedScheme"):
        check_scheme(Half())


def test_chained_runs_are_one_run(one_go):
    want, res = one_go
    state, masses = _system()
    runner = _runner()
    first = runner.run(state, masses, dt=DT, num_steps=STEPS)
    assert first.carried is not None and first.handle is None
    again = runner.run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=first)
    _same(_final(again), want)
    assert again.restored["bitwise"] and again.step == STEPS * SEGMENTS
    # the diagnostics rows
    rows = [e.diagnostics for e in res.edges]
    assert len(rows) == SEGMENTS + 1
    for row in rows:
        for name in (
            "momentum_drift",
            "energy",
            "energy_drift",
            "n_live",
            "n_stripped",
            "bound_mass",
            "strip_ledger",
            "W_gg",
            "energy_ledgered",
        ):
            assert name in row, name
    # (the blobs collapse: the drift is physical, as the built-in runner's own)
    assert rows[0]["energy_drift"] == 0.0 and np.isfinite(rows[-1]["energy"])
    assert rows[-1]["energy_ledgered"] == rows[-1]["energy"]


@pytest.mark.parametrize("fmt", ["npz", "orbax"])
def test_a_restart_continues_bitwise(fmt, one_go, tmp_path):
    if fmt == "orbax":
        pytest.importorskip("orbax.checkpoint")
    want, _ = one_go
    state, masses = _system()
    path = str(tmp_path / ("r.npz" if fmt == "npz" else "ckpt"))
    runner = _runner(restart_every=1, restart_path=path)
    runner.run(state, masses, dt=DT, num_steps=STEPS * 2)
    if fmt == "npz":
        _, meta = load_restart(path)
        assert meta["carried"]["scheme"] == {"tag": "a"}
        assert meta["fingerprint"]["carried"] == {"tag": "a"}
    else:
        from jaccpot.checkpoint import is_checkpoint

        assert is_checkpoint(path)
    fresh = _runner()
    res = fresh.run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=path)
    assert res.restored["bitwise"], res.restored
    _same(_final(res), want)
    # another configuration: correct, not bitwise, said so
    other = _runner(_Leapfrog(tag="b"))
    with pytest.warns(UserWarning, match="not bitwise"):
        res = other.run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=path)
    assert not res.restored["bitwise"] and "scheme" in res.restored["mismatches"]


def test_a_failed_verdict_reruns_the_segment(one_go):
    want, _ = one_go
    state, masses = _system()
    res = _runner(_Leapfrog(fail_once_at=1)).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    assert [e.retries for e in res.edges] == [0, 0, 1, 0]
    assert res.edges[2].plan["replans"] == [["test: walk"]]
    _same(_final(res), want)


def test_the_carried_path_refuses_what_it_lacks(tmp_path):
    with pytest.raises(ValueError, match="external field"):
        ProductionRunner(
            _solver(),
            settings=SETTINGS,
            scheme=_Leapfrog(),
            external_acceleration_fn=lambda s: s[:, 0],
        )
    state, masses = _system()
    runner = _runner()
    with pytest.raises(ValueError, match="force_params"):
        runner.run(state, masses, dt=DT, num_steps=2, external_params={"a": 1.0})
    with pytest.raises(ValueError, match="potential_energy"):
        runner.potential_energy(state, masses, None)
    with pytest.raises(RuntimeError, match="edge"):
        _runner().save_restart(str(tmp_path / "x.npz"))
    # a built-in run's restart is not a carried one's
    plain = ProductionRunner(
        _solver(),
        settings=SETTINGS,
        segment=SegmentConfig(steps=STEPS, domain_guard="off"),
    )
    plain.run(state, masses, dt=DT, num_steps=STEPS)
    path = plain.save_restart(str(tmp_path / "plain.npz"))
    with pytest.raises(ValueError, match="carried"):
        runner.run(dt=DT, num_steps=4, restart_from=path)
    with pytest.raises(ValueError, match="carried"):
        runner.run(
            dt=DT,
            num_steps=4,
            restart_from=plain.run(state, masses, dt=DT, num_steps=STEPS),
        )
    with pytest.raises(TypeError):
        runner.run(dt=DT, num_steps=4, restart_from=3)


def test_the_edge_sees_the_carry():
    seen = []
    state, masses = _system()
    runner = ProductionRunner(
        _solver(),
        settings=SETTINGS,
        segment=SegmentConfig(steps=STEPS, domain_guard="off", diagnostics_every=0),
        scheme=_Leapfrog(),
        callbacks=[lambda e: seen.append((e.carried, e.state.shape)) or True],
    )
    res = runner.run(state, masses, dt=DT, num_steps=4)
    assert res.stopped and len(seen) == 1 and seen[0][0] is not None
    assert seen[0][1] == (512, 2, 3) and res.edges[0].diagnostics is None
    json.dumps(res.edges[0].plan)


@dataclasses.dataclass(frozen=True, eq=False)
class _Donating(_Leapfrog):
    """``_Leapfrog`` with ``keep_start``/``restore_start``: the runner donates."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_dprogs", {})
        object.__setattr__(self, "log", [])

    def segment(self, c, dt, steps, donate=False):
        if not donate:
            return super().segment(c, dt, steps)
        base = self._segment_program(c.provider, float(dt), int(steps))
        key = (id(c.provider), float(dt), int(steps))
        if key not in self._dprogs:
            self._dprogs[key] = jax.jit(base.__wrapped__, donate_argnums=(0, 1, 2))
        start = (c.x, c.v, c.acc)
        with self._pinned(c):
            x, v, acc, aux = self._dprogs[key](c.x, c.v, c.acc, c.aux, c.m)
        self.log.append(("donated", all(a.is_deleted() for a in start)))
        return dataclasses.replace(
            c, x=x, v=v, acc=acc, aux=aux, time=c.time + steps * dt
        )

    def keep_start(self, c):
        self.log.append(("keep",))
        # a copy of a device-side copy: device_get of ``a`` itself leaves a
        # zero-copy NumPy view on it, and JAX then refuses the donation (CPU)
        return c, tuple(
            (np.asarray(jax.device_get(jnp.copy(a))), a.sharding)
            for a in (c.x, c.v, c.acc)
        )

    def restore_start(self, kept):
        c, arrays = kept
        self.log.append(("restore",))
        x, v, acc = (jax.device_put(a, s) for a, s in arrays)
        return dataclasses.replace(c, x=x, v=v, acc=acc)


def test_a_donated_segment_is_rerun_from_its_host_copy(one_go):
    """keep_start/restore_start: later segments donate, a retry restores the start bitwise."""
    want, _ = one_go
    state, masses = _system()
    scheme = _Donating(fail_once_at=1)
    res = _runner(scheme).run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    assert [e.retries for e in res.edges] == [0, 0, 1, 0]
    _same(_final(res), want)
    # segment 1 keeps the caller's state (no donate_input); 2 and its rerun and 3
    # donate, each from a fresh host copy or its restore
    assert scheme.log == [
        ("keep",),
        ("donated", True),
        ("restore",),
        ("donated", True),
        ("keep",),
        ("donated", True),
    ]
    assert not state.is_deleted() and not masses.is_deleted()  # the caller's


def test_donate_input_donates_the_first_segment_too(one_go):
    want, _ = one_go
    state, masses = _system()
    scheme = _Donating()
    res = _runner(scheme, donate_input=True).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    _same(_final(res), want)
    assert [e for e in scheme.log if e[0] == "donated"] == [("donated", True)] * 3
    # the caller lent its state (the start copied it): released, as the plain
    # runner's donate_input; the masses (the carry's own leaf here) are kept
    assert state.is_deleted()
    assert not masses.is_deleted()


def _grown_solver(capacity: int = 2048):
    """A solver whose leaf capacity grew past what a prepare at the run's
    positions sizes (a tree re-plan, ``_replan_tree_from_needs``, sets it so; and
    ``leaf_capacity="auto"`` is fixed at a solver's FIRST prepare, at another
    state than any restart's)."""
    solver = _solver()
    solver._impl._tree_leaf_capacity = int(capacity)
    return solver


@pytest.mark.parametrize("fmt", ["npz", "orbax"])
def test_a_restart_keeps_the_writers_tree_and_target_block_preflight(
    fmt, tmp_path, monkeypatch
):
    """HH5 item 3: the writer's leaf capacity differs from a fresh prepare's.

    The writer's programs (and its recorded template) have the leaf count its
    engine settled on (here grown, as a tree re-plan grows it); a restore's
    re-prepare at the restart's positions in a fresh process settles on another.
    Then (1) the fused payload's static target-block preflight was keyed with the
    restore's leaf count, so the restored traced refresh refused to start ("fused
    payload static target-block cap was not preflighted", HH4b at 1e7); (2) the
    static-radix topology key differed (a restore reported not bitwise,
    ``provider_key``). Now the pinned record carries the writer's preflight key
    and cap, and ``install_recorded_tree`` gives the re-prepare the writer's tree
    bounds: the restore continues, bitwise, with no mismatch. A scheme that does
    not call it still continues (the record's key), and says ``provider_key``.
    """
    if fmt == "orbax":
        pytest.importorskip("orbax.checkpoint")
    state, masses = _system()
    seg = dict(steps=STEPS, domain_guard="off", diagnostics_every_energy=1)

    def runner(solver, **more):
        return ProductionRunner(
            solver,
            settings=SETTINGS,
            segment=SegmentConfig(**seg, **more),
            scheme=_Leapfrog(),
        )

    whole = runner(_grown_solver()).run(
        state, masses, dt=DT, num_steps=STEPS * SEGMENTS
    )
    written = whole.carried.pin["engines"][0]["static_target_blocks"]["preflight"]
    assert written is not None and written[0] == 2048  # the grown leaf count
    path = str(tmp_path / ("r.npz" if fmt == "npz" else "ckpt"))
    runner(_grown_solver(), restart_every=1, restart_path=path).run(
        state, masses, dt=DT, num_steps=STEPS
    )
    # a fresh engine: a prepare at the restart's positions sizes 1024 leaves
    res = runner(_solver()).run(dt=DT, num_steps=STEPS * SEGMENTS, restart_from=path)
    assert res.restored["bitwise"] and not res.restored["mismatches"], res.restored
    _same(_final(res), _final(whole))
    # a scheme whose restore re-prepares without the writer's tree bounds
    monkeypatch.setattr(
        sys.modules[__name__], "install_recorded_tree", lambda engines, record: None
    )
    with pytest.warns(UserWarning, match="not bitwise"):
        res = runner(_solver()).run(
            dt=DT, num_steps=STEPS * SEGMENTS, restart_from=path
        )
    assert "provider_key" in res.restored["mismatches"]
    _same(_final(res), _final(whole))
