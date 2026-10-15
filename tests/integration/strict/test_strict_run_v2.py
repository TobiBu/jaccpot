"""The fused ``strict_run_v2`` scan, on the GPU: one class per former module.

Merged in the 2026-10 cleanup (Z-a, rule R5) from the seven
``tests/integration/test_strict_run_v2_*.py`` modules; each class keeps its
module's docstring, environment fixture and tests unchanged. The shared setup is
``_solver`` / ``_state`` and the ``_FUSED_ENV`` profile (``donation``'s, which
four of the modules imported or duplicated); helpers that differed between modules
carry the module's name. Every test here is slow and GPU-only.
"""

from __future__ import annotations

import collections
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    ),
]


_N = 20_000


_LEAF, _ORDER, _THETA = 64, 4, 0.8


_FUSED_ENV = {
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
}


def _solver():
    from jaccpot import (
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=1e-4,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity=2048,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=_ORDER,
    )


def _state():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, _N)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, _N)
    phi = rng.uniform(0.0, 2.0 * np.pi, _N)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (_N, 3))
    masses = jnp.full((_N,), 1.0 / _N, jnp.float32)
    state = jnp.stack(
        [jnp.asarray(pos, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    return state, masses


def _donation_run(solver, state, masses, prepared=None, donate=False):
    out, prepared_out, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=2,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        donate_prepared_state=donate,
    )
    return out, prepared_out


def _chain(donate):
    state, masses = _state()
    solver = _solver()
    s1, p1 = _donation_run(solver, state, masses)
    far = p1.compact_far_pairs
    s2, p2 = _donation_run(solver, s1, masses, p1, donate=donate)
    return solver, p1, far, s2, p2


def _arrays(tree):
    return [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]


def _deleted_reachable(obj, path="", seen=None, out=None):
    """Paths of deleted arrays reachable from ``obj`` (engine attributes, states)."""
    seen = set() if seen is None else seen
    out = [] if out is None else out
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, jax.core.Tracer):
        return out
    if isinstance(obj, jax.Array):
        if obj.is_deleted():
            out.append(path)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _deleted_reachable(v, f"{path}[{k!r}]", seen, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _deleted_reachable(v, f"{path}[{i}]", seen, out)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _deleted_reachable(
                getattr(obj, f.name, None), f"{path}.{f.name}", seen, out
            )
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith(
        ("jaccpot", "yggdrax")
    ):
        for k, v in vars(obj).items():
            _deleted_reachable(v, f"{path}.{k}", seen, out)
    return out


_STEPS = 6


_DONATE_STATE_ENV = {
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
    "JACCPOT_STRICT_CARRY": "particles",
}


def _drifting_system():
    """A uniform cube contracting inside its own box, with nearly no gravity.

    Each coordinate drifts from ``u`` towards ``sign(u) u**2`` (the eight corners
    stay put, so the box does not change); after the 6 steps it is 0.2 of the
    way. The directed far count rises 21510 -> 32632 by s = 0.12, past the cap
    the eager prepare sizes on the cube (32768 = 1.5 x 21510, rounded up to a
    power of two) near s = 0.125: steps 1-3 (s <= 0.1) fit, step 4 (s = 0.133)
    overflows.
    """
    rng = np.random.default_rng(0)
    cube = rng.uniform(-1.0, 1.0, (_N, 3))
    cube[:8] = [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    target = np.sign(cube) * cube**2
    dt = 1e-2
    vel = 0.2 * (target - cube) / (_STEPS * dt)
    state = jnp.stack(
        [jnp.asarray(cube, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    masses = jnp.full((_N,), 1e-12 / _N, jnp.float32)
    return state, masses, dt


def _donate_state_run(solver, state, masses, dt, *, donate_state=False, seen=None):
    def callback(step, _state):
        jax.debug.callback(lambda s: seen.append(int(s)), step)

    out, _, history = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=dt,
        num_steps=_STEPS,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        return_prepared_state=True,
        return_history=True,
        step_callback=callback if seen is not None else None,
        donate_state=donate_state,
    )
    jax.effects_barrier()
    return (
        np.asarray(jax.block_until_ready(out), np.float64),
        np.asarray(history, np.float64),
    )


def _fallbacks(solver):
    diag = dict(solver.get_runtime_diagnostics() or {})
    return int(diag.get("strict_fused_fallback_count") or 0), diag.get(
        "strict_fused_last_fallback_reason"
    )


_MASSES_ENV = {
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "1048576",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
}


def _masses_systems():
    rng = np.random.default_rng(0)
    cube = rng.uniform(-1.0, 1.0, (_N, 3)).astype(np.float32)
    x = rng.uniform(0.0, 1.0, _N)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, _N)
    phi = rng.uniform(0.0, 2.0 * np.pi, _N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (_N, 3))
    masses = jnp.full((_N,), 1.0 / _N, jnp.float32)
    state = jnp.stack(
        [jnp.asarray(plummer, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    return jnp.asarray(cube), state, masses


def _masses_run(solver, state, masses, prepared=None, steps=2):
    out, prepared_out, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
    )
    return np.asarray(jax.block_until_ready(out), np.float64), prepared_out


_FAR_CAP = "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP"


_NEAR_CAP = "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP"


def _retry_systems():
    rng = np.random.default_rng(0)
    cube = rng.uniform(-1.0, 1.0, (_N, 3)).astype(np.float32)
    x = rng.uniform(0.0, 1.0, _N)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, _N)
    phi = rng.uniform(0.0, 2.0 * np.pi, _N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (_N, 3))
    masses = jnp.full((_N,), 1.0 / _N, jnp.float32)
    state = jnp.stack(
        [jnp.asarray(plummer, jnp.float32), jnp.asarray(vel, jnp.float32)], axis=1
    )
    return jnp.asarray(cube), state, masses


def _retry_run(solver, state, masses, prepared=None, steps=2):
    out, prepared_out, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
    )
    return np.asarray(jax.block_until_ready(out), np.float64), prepared_out


def _diag(solver):
    return dict(solver.get_runtime_diagnostics() or {})


_SOFT = 2e-3


def _dehnen_solver(mac_type="dehnen_error", eps=1e-5):
    from jaccpot import (
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    extra = (
        dict(
            adaptive_eps=eps,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac_type == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=_SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity=2048,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac_type,
        ),
        fixed_order=_ORDER,
        **extra,
    )


def _dehnen_run(solver, state, masses, *, steps, carry, prepared=None):
    return solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        carry=carry,
    )


def _direct(pos, masses, idx):
    pos = np.asarray(pos, np.float64)
    m = np.asarray(masses, np.float64)
    d = pos[idx][:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    r2[np.arange(idx.size), idx] = np.inf
    from jaccpot.softening import pair_factors, softening_params_np

    params = softening_params_np("ferrers3", _SOFT)
    g = pair_factors(r2, params, "ferrers3", xp=np)[0]
    g = np.where(np.isfinite(r2), g, 0.0)
    return -np.einsum("ij,ijk->ik", g * m[None, :], d)


def _force_error(solver, state_out, masses):
    prepared = solver.prepare_state(
        state_out[:, 0, :], masses, leaf_size=_LEAF, max_order=_ORDER
    )
    acc = np.asarray(solver.evaluate_prepared_state(prepared), np.float64)
    idx = np.random.default_rng(1).choice(_N, 512, replace=False)
    ref = _direct(state_out[:, 0, :], masses, idx)
    return float(np.linalg.norm(acc[idx] - ref) / np.linalg.norm(ref))


def _carry_run(solver, state, masses, *, steps, carry, prepared=None):
    return solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        carry=carry,
    )


def _maxdiff(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


_REFRESH_ENV = {
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "131072",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
}


def _plummer(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def _rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class TestDonation:
    """``strict_run_v2`` donating its carry, and the far list riding outside the scan.

    Step 3 of the memory plan. The prepared state is the scan's carry, so without
    donation the compiled runner held it twice -- argument and output, 5.7 GiB each at
    2.5e7 particles. With ``donate_prepared_state=True`` (and always for a state the
    call prepares itself) the runner writes the returned state into the passed one's
    buffers. On the fused lane's default fresh far-pair rebuild the carried far list is
    a placeholder the refresh never reads; it is detached before the scan (so a
    caller's list is never donated) and not put back: it is the START positions'
    list, and the returned state's tree and positions are the end's (the gradient
    path read it: 7.5e-3 against 7.9e-5, review 2026-10-13).

    Pinned here: donating changes no number (A-vs-A controlled), consumes exactly what
    it says, leaves no deleted array reachable from the engine, and a later prepare and
    evaluation still work.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _FUSED_ENV.items():
            monkeypatch.setenv(key, val)
        for key in (
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_donating_changes_no_number(self):
        _, _, _, kept, _ = _chain(donate=False)
        _, _, _, kept_again, _ = _chain(donate=False)
        _, _, _, donated, _ = _chain(donate=True)
        control = float(jnp.max(jnp.abs(kept_again - kept)))
        assert float(jnp.max(jnp.abs(donated - kept))) <= max(10.0 * control, 1e-7)

    def test_donation_consumes_the_state_and_only_when_asked(self):
        _, p1, far, _, p2 = _chain(donate=False)
        assert not any(x.is_deleted() for x in _arrays(p1))
        _, p1, far, _, p2 = _chain(donate=True)
        # a returned state carries no far list: it would be the start positions'
        assert far is None and p2.compact_far_pairs is None
        assert _arrays(p1) and all(x.is_deleted() for x in _arrays(p1))
        assert not any(x.is_deleted() for x in _arrays(p2))
        # a caller's fused-lane state with a far list: the list rides outside the
        # scan, so donating consumes everything else and never the list
        state, masses = _state()
        solver = _solver()
        prepared, _ = solver.strict_fused_prepared_eval_fn(
            positions=state[:, 0],
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        far = prepared.compact_far_pairs
        assert far is not None
        _, p_out = _donation_run(solver, state, masses, prepared, donate=True)
        far_leaves = {id(x) for x in _arrays(far)}
        rest = [x for x in _arrays(prepared) if id(x) not in far_leaves]
        assert rest and all(x.is_deleted() for x in rest)
        assert not any(x.is_deleted() for x in _arrays(far))
        assert p_out.compact_far_pairs is None

    def test_nothing_deleted_stays_reachable_and_the_engine_keeps_working(self):
        solver, _, _, s2, p2 = _chain(donate=True)
        assert _deleted_reachable(solver._impl, "engine") == []
        # chain once more, then a fresh eager prepare and evaluation on the same engine
        _, masses = _state()
        s3, p3 = _donation_run(solver, s2, masses, p2, donate=True)
        assert _deleted_reachable(solver._impl, "engine") == []
        prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
            positions=s3[:, 0],
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        acc = np.asarray(eval_fn(prepared))
        assert np.all(np.isfinite(acc))

    def test_the_eval_closure_donates_only_when_asked(self):
        state, masses = _state()
        solver = _solver()
        kwargs = dict(
            positions=state[:, 0],
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        prepared, eval_fn = solver.strict_fused_prepared_eval_fn(**kwargs)
        ref = np.asarray(eval_fn(prepared))
        again = np.asarray(eval_fn(prepared))  # reusable
        prepared_d, eval_d = solver.strict_fused_prepared_eval_fn(
            **kwargs, donate_prepared=True
        )
        got = np.asarray(eval_d(prepared_d))
        assert any(x.is_deleted() for x in _arrays(prepared_d))
        control = float(np.max(np.abs(again - ref)))
        assert float(np.max(np.abs(got - ref))) <= max(10.0 * control, 1e-7)


class TestDonateState:
    """``strict_run_v2(carry="particles", donate_state=...)`` and the capacity retry.

    ``donate_state`` hands the input state's buffer to the scan; a segment whose
    refresh overflows a list is re-run once from its start with re-planned caps, so
    the call keeps a host copy of the start state when donating. The scenario forces
    the overflow MID-RUN: the particles drift ballistically (negligible masses) from
    a uniform cube, where the eager prepare sizes the caps, into a contracted version
    of it inside the same box. The first three steps fit; the fourth refresh does
    not. The re-run must equal a run whose caps never overflow (a large cap
    headroom), its history included; the stream stops at the failed step, so the
    steps before it are streamed once per attempt and never a failed step's state.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _DONATE_STATE_ENV.items():
            monkeypatch.setenv(key, val)
        for key in (
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
            "JACCPOT_FLAT_WALK_CAP_HEADROOM",
            "JACCPOT_STRICT_SEGMENT_RETRY",
        ):
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture(scope="class")
    def reference(self):
        """The same rollout with caps that never overflow."""
        mp = pytest.MonkeyPatch()
        for key, val in _DONATE_STATE_ENV.items():
            mp.setenv(key, val)
        mp.setenv("JACCPOT_FLAT_WALK_CAP_HEADROOM", "16")
        try:
            state, masses, dt = _drifting_system()
            solver = _solver()
            out, history = _donate_state_run(solver, state, masses, dt)
            assert _fallbacks(solver)[0] == 0, "the reference must fit throughout"
            return out, history
        finally:
            mp.undo()

    @pytest.mark.parametrize("donate", [False, True])
    def test_a_mid_run_overflow_is_rerun_from_the_start(self, reference, donate):
        ref_out, ref_history = reference
        state, masses, dt = _drifting_system()
        solver = _solver()
        seen: list = []
        before = _fallbacks(solver)[0]
        out, history = _donate_state_run(
            solver, state, masses, dt, donate_state=donate, seen=seen
        )
        count, reason = _fallbacks(solver)
        assert count == before + 1 and reason == "capacity_segment_retry"
        failed = solver._impl._strict_particle_failed_step
        assert (
            0 < failed < _STEPS
        ), f"the overflow should come mid-run, came at {failed}"
        # the lists do not depend on their caps: the re-run is the same run
        assert np.array_equal(out, ref_out)
        assert history.shape == ref_history.shape
        assert np.array_equal(history, ref_history)
        # the first attempt streams the steps before the failure, the re-run all
        want = {s: (2 if s < failed else 1) for s in range(_STEPS)}
        assert dict(collections.Counter(seen)) == want
        assert state.is_deleted() == donate

    def test_without_donation_the_input_state_is_kept(self):
        state, masses, dt = _drifting_system()
        before = np.asarray(state)
        _donate_state_run(_solver(), state, masses, dt)
        assert not state.is_deleted()
        assert np.array_equal(np.asarray(state), before)

    def test_donate_state_needs_the_particle_carry(self, monkeypatch):
        monkeypatch.setenv("JACCPOT_STRICT_CARRY", "state")
        state, masses, dt = _drifting_system()
        with pytest.raises(ValueError, match="donate_state"):
            _donate_state_run(_solver(), state, masses, dt, donate_state=True)


class TestMasses:
    """``strict_run_v2`` honours the masses of every call.

    The compiled runner closed over ``masses`` while its cache key held only their shape
    and dtype, so a second call with different masses of the same shape reused the first
    call's: every refresh inside the scan rebuilt the multipoles and the near field with
    stale masses (the eager prepare and the initial force used the new ones, so the error
    started at the first refreshed step). The masses are now a runner argument.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _MASSES_ENV.items():
            monkeypatch.setenv(key, val)

    def test_a_second_call_with_new_masses_uses_them(self):
        _, state, masses = _masses_systems()
        heavy = masses.at[: _N // 2].multiply(3.0)
        solver = _solver()
        first, _ = _masses_run(solver, state, masses)
        second, _ = _masses_run(solver, state, heavy)
        fresh, _ = _masses_run(_solver(), state, heavy)
        fresh_again, _ = _masses_run(_solver(), state, heavy)
        control = np.max(np.abs(fresh_again - fresh))
        # the masses matter at this resolution ...
        assert np.max(np.abs(first - fresh)) > 1e3 * max(control, 1e-9)
        # ... and the cached runner now reads the ones it is given
        assert np.max(np.abs(second - fresh)) <= max(10.0 * control, 1e-7)


class TestSegmentRetry:
    """``strict_run_v2``: the segment retry for outgrown walk caps.

    Unnamed flat-walk list caps are sized from the counts the eager prepare measured
    (``JACCPOT_FLAT_WALK_CAP_HEADROOM`` x count), so a rollout whose counts grow past
    the headroom trips the scan's capacity flag. The scan carries the running maximum
    of what its walks needed out of the trace; on a failed segment ``strict_run_v2``
    re-plans the caps from it, re-prepares from the segment's start and runs it once
    more. The scenario here forces that: the state is prepared on a uniform cube
    (21k far / 11k near pairs at N = 2e4) and stepped on a Plummer sphere (84k / 55k),
    so all three walk capacities overflow on the first refresh. The retried segment
    must equal a run that was prepared on the right positions; a named cap, and the
    retry switched off, must still raise.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _FUSED_ENV.items():
            monkeypatch.setenv(key, val)
        for key in (
            _FAR_CAP,
            _NEAR_CAP,
            "JACCPOT_FLAT_WALK_CAP_HEADROOM",
            "JACCPOT_STRICT_SEGMENT_RETRY",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_an_outgrown_segment_is_replanned_once_and_matches_a_fitting_run(self):
        cube, state, masses = _retry_systems()
        ref_solver = _solver()
        ref, _ = _retry_run(ref_solver, state, masses)
        ref_again, _ = _retry_run(_solver(), state, masses)
        # A-vs-A control: two fitting runs agree to this level
        control = np.max(np.abs(ref_again - ref))

        solver = _solver()
        small, _ = solver.strict_fused_prepared_eval_fn(
            positions=cube,
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        caps_before = dict(solver._impl._strict_fused_validated_caps)
        before = int(_diag(solver).get("strict_fused_fallback_count") or 0)
        got, _ = _retry_run(solver, state, masses, prepared=small)
        diag = _diag(solver)
        assert int(diag.get("strict_fused_fallback_count") or 0) == before + 1
        assert diag.get("strict_fused_last_fallback_reason") == "capacity_segment_retry"
        caps_after = solver._impl._strict_fused_validated_caps
        assert (
            caps_after["compact_far_pair_capacity"]
            > caps_before["compact_far_pair_capacity"]
        )
        assert caps_after["near_edge_capacity"] > caps_before["near_edge_capacity"]
        assert np.max(np.abs(got - ref)) <= max(10.0 * control, 1e-6)

    def test_the_retry_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("JACCPOT_STRICT_SEGMENT_RETRY", "0")
        cube, state, masses = _retry_systems()
        solver = _solver()
        small, _ = solver.strict_fused_prepared_eval_fn(
            positions=cube,
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _retry_run(solver, state, masses, prepared=small)
        assert "saturated" in str(excinfo.value.__cause__ or excinfo.value)

    def test_a_named_cap_is_never_widened(self, monkeypatch):
        cube, state, masses = _retry_systems()
        probe = _solver()
        probe.strict_fused_prepared_eval_fn(
            positions=cube,
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        caps = probe._impl._strict_fused_validated_caps
        # caps that fit the cube exactly as named
        monkeypatch.setenv(_FAR_CAP, str(caps["compact_far_pair_capacity"]))
        monkeypatch.setenv(_NEAR_CAP, str(caps["near_edge_capacity"]))
        solver = _solver()
        small, _ = solver.strict_fused_prepared_eval_fn(
            positions=cube,
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _retry_run(solver, state, masses, prepared=small)
        assert "saturated" in str(excinfo.value.__cause__ or excinfo.value)
        assert int(_diag(solver).get("strict_fused_fallback_count") or 0) == 0


class TestDehnenError:
    """``mac_type='dehnen_error'`` inside the fused scan: eq (16a) per step, f_b carried.

    The strict fused lane evaluates Dehnen's criterion inside its flat walk on every
    traced refresh, with each node's threshold ``eps * min_b f_b`` from the force
    scale the PREVIOUS step's evaluation returned (near half from the near-field
    kernel's lane, far half from the prepare's far pairs). Pinned on a GPU:

    * both carries (state and particles) run a traced rollout with the criterion and
      agree with each other;
    * the scale the particle carry hands back equals a fresh evaluation's at the
      returned positions (to the stale-by-one-step lists' tolerance);
    * the criterion is live in the traced steps: at a tight ``eps`` the rollout's final
      force is closer to the direct sum than the geometric MAC's at the same theta;
    * a handle without a force scale cannot continue a criterion run.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _FUSED_ENV.items():
            monkeypatch.setenv(key, val)
        for key in (
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
            "JACCPOT_STRICT_CARRY",
            "JACCPOT_STRICT_CARRY_ORDER",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_both_carries_run_the_criterion_and_agree(self):
        state, masses = _state()
        solver = _dehnen_solver()
        out_state, _, _ = _dehnen_run(solver, state, masses, steps=3, carry="state")
        out_part, handle, _ = _dehnen_run(
            _dehnen_solver(), state, masses, steps=3, carry="particles"
        )
        assert bool(jnp.all(jnp.isfinite(out_state))) and bool(
            jnp.all(jnp.isfinite(out_part))
        )
        scale = np.asarray(handle.force_scale)
        assert scale.shape == (_N,) and np.all(np.isfinite(scale)) and np.all(scale > 0)
        np.testing.assert_allclose(
            np.asarray(out_part), np.asarray(out_state), rtol=0, atol=2e-5
        )

    def test_the_carried_scale_is_the_evaluations_own(self):
        from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state

        state, masses = _state()
        solver = _dehnen_solver()
        out, handle, _ = _dehnen_run(solver, state, masses, steps=2, carry="particles")
        prepared = solver._impl.prepare_state(
            out[:, 0, :],
            masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            fused_device_mode=True,
        )
        _, fresh = evaluate_large_n_state(
            solver._impl,
            prepared,
            target_indices=None,
            return_potential=False,
            max_acc_derivative_order=0,
            return_force_scale=True,
        )
        carried, fresh = np.asarray(handle.force_scale), np.asarray(fresh)
        # the same positions; only the far lists differ (their thresholds were one
        # step older), which moves the monopole lower bound a little
        rel = np.abs(carried - fresh) / fresh
        assert np.median(rel) < 0.05, float(np.median(rel))

    def test_the_criterion_is_live_in_the_traced_steps(self):
        state, masses = _state()
        geo = _dehnen_solver(mac_type="dehnen")
        de = _dehnen_solver(eps=1e-5)
        out_geo, _, _ = _dehnen_run(geo, state, masses, steps=2, carry="particles")
        out_de, _, _ = _dehnen_run(de, state, masses, steps=2, carry="particles")
        err_geo = _force_error(geo, out_geo, masses)
        err_de = _force_error(de, out_de, masses)
        assert err_de < 0.5 * err_geo, (err_de, err_geo)

    def test_a_handle_without_a_scale_cannot_continue_a_criterion_run(self):
        state, masses = _state()
        out, handle, _ = _dehnen_run(
            _dehnen_solver(mac_type="dehnen"), state, masses, steps=1, carry="particles"
        )
        assert handle.force_scale is None
        with pytest.raises(RuntimeError, match="force scale"):
            _dehnen_run(
                _dehnen_solver(),
                out,
                masses,
                steps=1,
                carry="particles",
                prepared=handle,
            )


class TestParticleCarry:
    """``strict_run_v2(carry="particles")``: the fused scan carries the particles only.

    Pinned here (GPU, the fused strict lane):

    * the premise: with the default fresh far-pair rebuild a scan step READS no
      leaf of the carried prepared state -- jax's dead-code elimination of the step,
      with only the particle outputs marked used, keeps none of its inputs;
    * the particle carry follows the state carry's trajectory (A-vs-A controlled);
    * chaining through the returned handle is one long run, and the handle's
      self-gravity is the force at the returned positions;
    * the call leaves less resident (no prepared state, no far list kept);
    * the handle is refused where it does not belong;
    * the tree-order carry (``JACCPOT_STRICT_CARRY_ORDER=tree``) is the input-order
      carry: the same states and handle, back in input order.
    """

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for key, val in _FUSED_ENV.items():
            monkeypatch.setenv(key, val)
        for key in (
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
            "JACCPOT_STRICT_CARRY",
            "JACCPOT_STRICT_CARRY_ORDER",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_a_step_reads_no_carried_state_leaf(self, monkeypatch):
        """The premise of the particle carry, measured on the state carry's scan body."""
        from dataclasses import replace

        from jax._src.interpreters import partial_eval as pe

        from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS

        captured = {}
        orig = jax.lax.scan

        def spy(f, init, xs=None, length=None, **kw):
            captured["f"] = f
            return orig(f, init, xs=xs, length=length, **kw)

        monkeypatch.setattr(jax.lax, "scan", spy)
        state, masses = _state()
        solver = _solver()
        s1, p1, _ = _carry_run(solver, state, masses, steps=1, carry="state")
        monkeypatch.setattr(jax.lax, "scan", orig)
        impl = solver._impl
        assert impl._strict_far_pairs_ride_outside_the_scan(p1)
        p_in = replace(p1, compact_far_pairs=None)
        carry = (
            p_in,
            s1,
            jnp.zeros_like(s1[:, 0]),
            jnp.asarray(True),
            jnp.zeros((len(WALK_NEEDS_FIELDS),), jnp.int32),
            # the force-scale slot: None (no leaf) unless mac_type='dehnen_error'
            None,
        )
        f = captured["f"]
        closed = jax.make_jaxpr(lambda c: f(c, None)[0])(carry)
        n_prep = len(jax.tree_util.tree_leaves(p_in))
        n_prep_out = len(
            jax.tree_util.tree_leaves(jax.eval_shape(lambda c: f(c, None)[0], carry)[0])
        )
        n_out = len(closed.jaxpr.outvars)
        _, used_in = pe.dce_jaxpr(
            closed.jaxpr, [False] * n_prep_out + [True] * (n_out - n_prep_out)
        )
        assert n_prep > 20  # non-vacuous: the state really is a big pytree
        assert not any(used_in[:n_prep])
        assert all(used_in[n_prep:])  # the particles are read

    def test_particle_carry_follows_the_state_carry(self):
        state, masses = _state()
        a, _, _ = _carry_run(_solver(), state, masses, steps=3, carry="state")
        a2, _, _ = _carry_run(_solver(), state, masses, steps=3, carry="state")
        b, handle, _ = _carry_run(_solver(), state, masses, steps=3, carry="particles")
        from jaccpot.runtime.strict_carry import StrictParticleCarry

        assert isinstance(handle, StrictParticleCarry)
        control = _maxdiff(a2, a)
        assert _maxdiff(b, a) <= max(10.0 * control, 1e-6)
        assert bool(jnp.all(jnp.isfinite(b)))

    def test_chaining_through_the_handle_is_one_long_run(self):
        state, masses = _state()
        solver = _solver()
        one, _, _ = _carry_run(solver, state, masses, steps=4, carry="particles")
        one_again, _, _ = _carry_run(
            _solver(), state, masses, steps=4, carry="particles"
        )
        mid, handle, _ = _carry_run(solver, state, masses, steps=2, carry="particles")
        end, handle2, _ = _carry_run(
            solver, mid, masses, steps=2, carry="particles", prepared=handle
        )
        control = _maxdiff(one_again, one)
        assert _maxdiff(end, one) <= max(10.0 * control, 1e-6)
        # the handle's force is the self-gravity at the returned positions
        prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
            positions=end[:, 0],
            masses=masses,
            leaf_size=_LEAF,
            max_order=_ORDER,
            theta=_THETA,
        )
        eager = np.asarray(eval_fn(prepared))
        scale = float(np.max(np.abs(eager)))
        assert _maxdiff(handle2.self_acceleration, eager) <= 1e-4 * scale

    def test_the_particle_carry_leaves_less_resident(self):
        state, masses = _state()

        def resident_after(carry):
            solver = _solver()
            out = _carry_run(solver, state, masses, steps=2, carry=carry)
            jax.block_until_ready(out[0])
            live = sum(x.nbytes for x in jax.live_arrays() if not x.is_deleted())
            del out, solver
            return live

        by_state = resident_after("state")
        by_particles = resident_after("particles")
        # the state carry keeps the prepared state + far list (several MB at 2e4);
        # the handle holds one [N, 3] force
        assert by_particles < by_state - 1_000_000

    def test_the_handle_is_refused_where_it_does_not_belong(self):
        state, masses = _state()
        solver = _solver()
        out, handle, _ = _carry_run(solver, state, masses, steps=1, carry="particles")
        with pytest.raises(ValueError):
            _carry_run(solver, out, masses, steps=1, carry="state", prepared=handle)
        with pytest.raises(ValueError):
            _carry_run(
                solver,
                out[:-1],
                masses[:-1],
                steps=1,
                carry="particles",
                prepared=handle,
            )

    def test_the_tree_order_carry_is_the_input_order_carry(self, monkeypatch):
        """The scan carries the rows in Morton order and puts them back once: the same
        trajectory as the input-order carry -- bitwise unless two particles share a
        Morton code (ties keep the carry's order) -- and the same handle, chained."""
        state, masses = _state()
        a, ha, _ = _carry_run(_solver(), state, masses, steps=3, carry="particles")
        a2, ha2, _ = _carry_run(_solver(), state, masses, steps=3, carry="particles")
        monkeypatch.setenv("JACCPOT_STRICT_CARRY_ORDER", "tree")
        solver = _solver()
        b, hb, _ = _carry_run(solver, state, masses, steps=3, carry="particles")
        # non-vacuous: the runner that ran is the tree-order one
        keys = list(solver._impl._strict_fused_jit_function_cache)
        runner_keys = [k for k in keys if k[0] == "strict_velocity_verlet_particles"]
        assert runner_keys and all(k[-1] is True for k in runner_keys)
        for got, ref, ref2 in (
            (b, a, a2),
            (hb.self_acceleration, ha.self_acceleration, ha2.self_acceleration),
        ):
            control = _maxdiff(ref2, ref)
            if control == 0.0:
                assert np.array_equal(np.asarray(got), np.asarray(ref))
            else:
                assert _maxdiff(got, ref) <= 10.0 * control
        # chained through the handle, still one run
        mid, handle, _ = _carry_run(solver, state, masses, steps=2, carry="particles")
        end, _, _ = _carry_run(
            solver, mid, masses, steps=1, carry="particles", prepared=handle
        )
        control = _maxdiff(a2, a)
        if control == 0.0:
            assert np.array_equal(np.asarray(end), np.asarray(a))
        else:
            assert _maxdiff(end, a) <= 10.0 * control


class TestRefreshCapacity:
    """The fused ``strict_run_v2`` refresh must not truncate its interaction lists.

    Regression test for the 2026-09-06 finding: on ``large_n_gpu``/static_radix at
    N=200k, leaf 256, theta 0.6 the eager prepare built exact lists (rows up to
    781, via yggdrax's retry ladder) while the traced refresh inside the compiled
    velocity-Verlet scan ran the same walk with the preset capacities (256
    neighbours per leaf, queue 65536), could not read the overflow flags under
    ``jit``, and silently kept 15 % of the near field.  Every step after the first
    then carried a wrong force -- ~60 % at theta 0.6, 6 % at theta 1.0 -- while
    ``fallback_count`` stayed 0 and no overflow diagnostic fired.  An energy check
    between two lanes with the same bug, or any single-step probe, cannot see it.

    The check recovers the force the scan actually applied from the trajectory --
    from rest ``a0 = 2 (x1 - x0) / dt^2`` and ``a_k = (x_{k+1} - 2 x_k + x_{k-1})
    / dt^2`` -- and compares it with an eager prepare+evaluate AT THE SAME
    POSITIONS ``x_k``.  Comparing against the force at ``x0`` instead is wrong:
    this sample has unresolved close pairs whose accelerations dominate the L2 norm
    and move far in one step, so the true field changes by order unity between
    steps (that false alarm cost half a day).
    """

    @pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    )
    @pytest.mark.parametrize("flat_walk", ["0", "1"], ids=["dual_walk", "flat_walk"])
    def test_strict_run_v2_refresh_keeps_full_neighbor_lists(
        self, monkeypatch, flat_walk
    ):
        n = 200_000
        leaf, order, theta = 256, 3, 0.6
        for key, val in _REFRESH_ENV.items():
            monkeypatch.setenv(key, val)
        monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", str(n))
        # The flat-emission walk (plan "tree walk", 2026-09-10) has no per-leaf row
        # cap: its neighbour list is a CSR of width JACCPOT_LARGE_N_NEIGHBOR_EDGE_
        # PROFILE_FIXED_CAP, and every overflow saturates the far-pair count into
        # the same guard. Same force, same lists as sets, different fp32 order.
        monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK", flat_walk)

        from jaccpot import (
            FastMultipoleMethod,
            FMMAdvancedConfig,
            NearFieldConfig,
            TreeConfig,
        )

        pos, mass = _plummer(n)
        solver = FastMultipoleMethod(
            preset="large_n_gpu",
            basis="real",
            theta=theta,
            G=1.0,
            softening=1e-7,
            working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=leaf),
                nearfield=NearFieldConfig(mode="auto"),
                mac_type="dehnen",
            ),
            fixed_order=order,
        )
        P = jnp.asarray(pos)
        M = jnp.asarray(mass)

        # eager reference: the eval-only seam builds the exact lists
        prepared_eager, eval_fn = solver.strict_fused_prepared_eval_fn(
            positions=P, masses=M, leaf_size=leaf, max_order=order, theta=theta
        )
        a_eager = np.asarray(jax.block_until_ready(eval_fn(prepared_eager)))
        eager_rows = np.asarray(prepared_eager.neighbor_list.counts)
        assert (
            int(eager_rows.max()) > 256
        ), "this test needs rows longer than the preset cap to be meaningful"

        def eager_force_at(x):
            p, ev = solver.strict_fused_prepared_eval_fn(
                positions=jnp.asarray(x, jnp.float32),
                masses=M,
                leaf_size=leaf,
                max_order=order,
                theta=theta,
            )
            return np.asarray(jax.block_until_ready(ev(p)), np.float64)

        dt = 1e-2
        state0 = jnp.stack([P, jnp.zeros((n, 3), jnp.float32)], axis=1)
        final, prepared_out, history = solver.strict_run_v2(
            state=state0,
            masses=M,
            dt=dt,
            num_steps=2,
            refresh_every=1,
            leaf_size=leaf,
            max_order=order,
            theta=theta,
            prepared_state=None,
            return_prepared_state=True,
            return_history=True,
        )
        hist = np.asarray(jax.block_until_ready(history), np.float64)
        xs = [pos.astype(np.float64)] + [hist[k, :, 0, :] for k in range(hist.shape[0])]
        a0_scan = 2.0 * (xs[1] - xs[0]) / dt**2  # force at x0 (fresh prepare)
        a1_scan = (xs[2] - 2.0 * xs[1] + xs[0]) / dt**2  # force at x1 (traced refresh)

        err0 = _rel_l2(a0_scan, a_eager)
        err1 = _rel_l2(a1_scan, eager_force_at(xs[1]))
        # float32 positions limit the recovery to a few 1e-3; the truncated refresh
        # gave ~0.6 here (rows capped at 128 against 781).
        assert err0 < 1e-2, err0
        assert err1 < 1e-2, (err0, err1)

        # and the refreshed state's lists are full-length, not a capped subset
        refreshed_rows = np.asarray(prepared_out.neighbor_list.counts)
        assert int(refreshed_rows.max()) > 256, int(refreshed_rows.max())
        assert abs(int(refreshed_rows.max()) - int(eager_rows.max())) <= 8, (
            int(refreshed_rows.max()),
            int(eager_rows.max()),
        )
        caps = solver._impl._strict_fused_traced_caps
        if flat_walk == "1":
            assert caps["flat_walk"] is True
            assert caps["near_edge_capacity"] > int(refreshed_rows.sum())
            validated = solver._impl._strict_fused_validated_caps
            assert validated["peak_wavefront"] > 0
            assert caps["queue_capacity"] >= validated["peak_wavefront"]
        else:
            assert caps["max_neighbors_per_leaf_used"] > int(refreshed_rows.max())
