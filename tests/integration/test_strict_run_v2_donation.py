"""``strict_run_v2`` donating its carry, and the far list riding outside the scan.

Step 3 of the memory plan. The prepared state is the scan's carry, so without
donation the compiled runner held it twice -- argument and output, 5.7 GiB each at
2.5e7 particles. With ``donate_prepared_state=True`` (and always for a state the
call prepares itself) the runner writes the returned state into the passed one's
buffers. On the fused lane's default fresh far-pair rebuild the carried far list is
a placeholder the refresh never reads; it is detached before the scan and
re-attached to the returned state, which the gradient path reads.

Pinned here: donating changes no number (A-vs-A controlled), consumes exactly what
it says, leaves no deleted array reachable from the engine, and a later prepare and
evaluation still work.
"""

from __future__ import annotations

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
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(_N),
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
    ):
        monkeypatch.delenv(key, raising=False)


def _solver():
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
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
            farfield=FarFieldConfig(mode="auto"),
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


def _run(solver, state, masses, prepared=None, donate=False):
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
    s1, p1 = _run(solver, state, masses)
    far = p1.compact_far_pairs
    s2, p2 = _run(solver, s1, masses, p1, donate=donate)
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


def test_donating_changes_no_number():
    _, _, _, kept, _ = _chain(donate=False)
    _, _, _, kept_again, _ = _chain(donate=False)
    _, _, _, donated, _ = _chain(donate=True)
    control = float(jnp.max(jnp.abs(kept_again - kept)))
    assert float(jnp.max(jnp.abs(donated - kept))) <= max(10.0 * control, 1e-7)


def test_donation_consumes_the_state_and_only_when_asked():
    _, p1, far, _, p2 = _chain(donate=False)
    assert not any(x.is_deleted() for x in _arrays(p1))
    _, p1, far, _, p2 = _chain(donate=True)
    far_leaves = {id(x) for x in _arrays(far)}
    rest = [x for x in _arrays(p1) if id(x) not in far_leaves]
    assert rest and all(x.is_deleted() for x in rest)
    # the far list rode outside the scan: the returned state carries the same one
    assert p2.compact_far_pairs is far
    assert not any(x.is_deleted() for x in _arrays(far))
    assert not any(x.is_deleted() for x in _arrays(p2))


def test_nothing_deleted_stays_reachable_and_the_engine_keeps_working():
    solver, _, _, s2, p2 = _chain(donate=True)
    assert _deleted_reachable(solver._impl, "engine") == []
    # chain once more, then a fresh eager prepare and evaluation on the same engine
    _, masses = _state()
    s3, p3 = _run(solver, s2, masses, p2, donate=True)
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


def test_the_eval_closure_donates_only_when_asked():
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
