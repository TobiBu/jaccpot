"""Track C's user-facing half: nornax's block-step integrator over a device mesh.

The single-device counterpart is ``tests/integration/test_mutual_fmm_nornax.py``, and
this file follows it deliberately -- same protocol checks, same rollout, same momentum
criterion -- so that "the distributed lane satisfies the same contract" is a claim
backed by the same assertions rather than by weaker ones.

Two devices AND nornax, so it skips twice::

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
        JAX_ENABLE_X64=1 pytest -q -o addopts="" \
        tests/integration/test_mutual_distributed_nornax.py

**What made this possible, and what it cost.** ``block_kdk_rollout`` walks its base
steps with a ``lax.scan``, so it needs a force it can TRACE. The distributed driver
originally assembled its result with a NumPy scatter and raised on a host-read overflow
flag, neither of which survives a trace. Both moved:
:class:`~jaccpot.mutual.distributed.DistributedMutualEvaluator` scatters with
``jnp``, and the "every real particle appears exactly once" check moved to the
partition -- where it belongs, since it depends only on the frozen gid layout and not
on any evaluation. The overflow read is attempted rather than gated on
``isinstance(..., Tracer)``, so it raises on every eager call and is skipped under
trace. A traced driver consequently gets no overflow check, which is why the rollout
tests below evaluate once eagerly first.

Momentum is the criterion, per level and across the rollout, because it is what the
scheme is *defined* by: each level's contribution must be applied antisymmetrically or
an inactive coarse partner never receives its equal-and-opposite kick. It is checked on
a GLOBAL sum -- both failure modes of a cross-domain exchange, dropping the ``-f`` and
double-counting it, leave every per-device sum exact.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from yggdrax.distributed import device_count

from jaccpot.nornax_adapter import DistributedBlockStepFMM

# `except Exception` rather than ImportError, as the single-device file does: an
# import-time incompatibility should report as skipped, not errored.
try:  # pragma: no cover - environment dependent
    from nornax.forces.base import FusedMutualForceModel, MutualForceModel
    from nornax.solvers.leapfrog_kdk import (
        advance_base_step,
        block_kdk_rollout,
        fused_boundary_model,
        initialize_block_state,
        supports_traced_level_weights,
        total_acceleration,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"nornax unavailable: {exc!r}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="the distributed mutual force needs >= 2 devices"
)

SOFT = 1.0e-2
K_MAX = 1
N = 128
LEAF = 8


def _ndev():
    return min(4, device_count())


def _system(seed=9):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(size=(N, 3))),
        jnp.asarray(rng.normal(scale=0.05, size=(N, 3))),
        jnp.asarray(rng.uniform(0.5, 1.5, size=N)),
        jnp.asarray(rng.integers(0, K_MAX + 1, size=N).astype(np.int32)),
    )


def _model(**kw):
    return DistributedBlockStepFMM(
        softening=SOFT,
        k_max=K_MAX,
        theta=kw.pop("theta", 0.5),
        cross_theta=kw.pop("cross_theta", 0.0),
        max_order=4,
        leaf_size=LEAF,
        ndev=_ndev(),
        **kw,
    )


def _relative_momentum(masses, vectors):
    terms = masses[:, None] * vectors
    scale = float(jnp.sum(jnp.abs(terms)))
    return float(jnp.linalg.norm(jnp.sum(terms, axis=0))) / (scale + 1e-300)


def _energy(positions, velocities, masses, softening):
    """Softened total energy, by direct summation -- an independent reference.

    Computed here rather than taken from a diagnostic, because the point is to watch a
    quantity the force lane has no hand in: a leapfrog's energy should stay bounded,
    and a rollout that quietly went unstable would still conserve momentum exactly.
    """
    kinetic = 0.5 * float(jnp.sum(masses * jnp.sum(velocities**2, axis=1)))
    d = positions[None, :, :] - positions[:, None, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + softening**2)
    pair = masses[:, None] * masses[None, :] / r
    off = pair - jnp.diag(jnp.diag(pair))
    return kinetic - 0.5 * float(jnp.sum(off))


@pytest.fixture(scope="module")
def prepared():
    """One prepared model, shared: every evaluation after the first reuses its program.

    Module-scoped for the reason the class exists -- the first evaluation compiles the
    mapped program (~20 s on 2 forced CPU devices at this size) and later ones do not.
    A per-test model would pay that compile in every test.
    """
    pos, vel, mass, rung = _system()
    fmm = _model()
    fmm.prepare(pos, mass)
    return fmm, pos, vel, mass, rung


def test_the_distributed_model_satisfies_both_nornax_protocols(prepared):
    """Structural conformance, with no nornax import in the library.

    The same claim the single-device class makes, and worth re-asserting rather than
    assumed: the two classes share no base class, so nothing propagates the conformance
    from one to the other.
    """
    fmm = prepared[0]
    assert isinstance(fmm, MutualForceModel)
    assert isinstance(fmm, FusedMutualForceModel)
    assert fused_boundary_model(fmm, K_MAX) is fmm
    # And the boundaries get scanned rather than unrolled, which on this lane is not a
    # micro-optimisation: each unrolled kick would inline a whole distributed program.
    assert supports_traced_level_weights(fmm)


def test_prepare_is_required_rather_than_implicit():
    """Refused, not done silently -- an implicit partition recompiles on every call."""
    pos, _vel, mass, rung = _system()
    fmm = _model()
    with pytest.raises(RuntimeError, match="call prepare"):
        fmm.level_accelerations(pos, mass, rung=rung, level=0)


def test_the_levels_partition_the_total_through_the_model(prepared):
    """``sum_k level_accelerations(k) == total_accelerations()``, via nornax's own sum.

    Driven through ``nornax.solvers.leapfrog_kdk.total_acceleration`` rather than a
    local loop, so what is checked is the contract nornax actually relies on.
    """
    fmm, pos, _vel, mass, rung = prepared
    total = fmm.total_accelerations(pos, mass)
    summed = total_acceleration(fmm, pos, mass, rung, k_max=K_MAX)
    rel = float(jnp.linalg.norm(summed - total)) / float(jnp.linalg.norm(total))
    assert rel < 1e-13, f"the levels sum to {rel:.3e} away from the total"


@pytest.mark.parametrize("level", range(K_MAX + 1))
def test_every_level_conserves_momentum_across_the_mesh(prepared, level):
    """The defining property, on a GLOBAL sum over all devices."""
    fmm, pos, _vel, mass, rung = prepared
    acc = fmm.level_accelerations(pos, mass, rung=rung, level=level)
    res = _relative_momentum(mass, acc)
    assert res < 1e-14, f"level {level} momentum residual {res:.3e}"


def test_a_level_outside_the_ladder_is_refused(prepared):
    """A level with no weight would have to be invented or dropped; say so instead."""
    fmm, pos, _vel, mass, rung = prepared
    with pytest.raises(ValueError, match=r"\[0, k_max=1\]"):
        fmm.level_accelerations(pos, mass, rung=rung, level=K_MAX + 1)


def test_a_rung_above_k_max_is_refused(prepared):
    """Caught at the boundary rather than clamped into a wrong integration."""
    fmm, pos, _vel, mass, _rung = prepared
    bad = np.zeros(N, dtype=np.int32)
    bad[0] = K_MAX + 5
    with pytest.raises(ValueError, match=r"\[0, k_max=1\]"):
        fmm.total_accelerations(pos, mass)  # warm, then the real call
        fmm.level_accelerations(pos, mass, rung=jnp.asarray(bad), level=0)


@pytest.mark.slow
def test_nornax_advance_base_step_conserves_momentum(prepared):
    """One base step of nornax's own integrator, on the distributed force.

    ``advance_base_step`` is the layer ``block_kdk_rollout`` scans over, so this
    isolates a single step's boundary schedule from the rollout's scan.
    """
    fmm, pos, vel, mass, rung = prepared
    state = initialize_block_state(pos, vel, mass, fmm, k_max=K_MAX, rung=rung)
    p0 = jnp.sum(mass[:, None] * state.velocities, axis=0)
    out = advance_base_step(state, 2.0e-3, fmm, k_max=K_MAX)
    p1 = jnp.sum(mass[:, None] * out.velocities, axis=0)
    scale = float(jnp.sum(jnp.abs(mass[:, None] * out.velocities)))
    drift = float(jnp.linalg.norm(p1 - p0)) / scale
    assert drift < 1e-13, f"base-step momentum drift {drift:.3e}"
    assert bool(jnp.all(jnp.isfinite(out.positions)))


@pytest.mark.slow
def test_nornax_block_rollout_conserves_momentum_and_bounds_energy(prepared):
    """The C3 acceptance criterion: nornax's real scanned rollout over >= 2 devices.

    Multi-rung, momentum and energy both tracked. Momentum is the exact one -- it
    cancels structurally, so it must land at round-off and nothing about the rollout
    length or ``dt`` should move it. Energy is the LOOSE one, and it is here because it
    fails independently: a rollout that went unstable, or one whose cross-domain force
    was simply wrong, would conserve momentum perfectly and drift in energy. The bound
    is a sanity bound on a second-order symplectic map over a couple of steps, not an
    accuracy claim.
    """
    fmm, pos, vel, mass, rung = prepared
    state = initialize_block_state(pos, vel, mass, fmm, k_max=K_MAX, rung=rung)
    e0 = _energy(state.positions, state.velocities, mass, SOFT)
    p0 = jnp.sum(mass[:, None] * state.velocities, axis=0)

    final = block_kdk_rollout(
        state,
        2.0e-3,
        fmm,
        k_max=K_MAX,
        n_base=2,
        checkpoint=False,
        reassign_rungs=False,
    )

    p1 = jnp.sum(mass[:, None] * final.velocities, axis=0)
    scale = float(jnp.sum(jnp.abs(mass[:, None] * final.velocities)))
    drift = float(jnp.linalg.norm(p1 - p0)) / scale
    assert drift < 1e-13, f"rollout momentum drift {drift:.3e}"

    assert bool(jnp.all(jnp.isfinite(final.positions)))
    assert bool(jnp.all(jnp.isfinite(final.velocities)))
    e1 = _energy(final.positions, final.velocities, mass, SOFT)
    assert abs(e1 - e0) / abs(e0) < 1e-3, (
        f"energy moved {abs(e1 - e0) / abs(e0):.3e} over 2 base steps -- a symplectic "
        "map should not, so either the rollout is unstable or the force is wrong"
    )
    # Vacuity guard: a rollout that did not move would pass both checks above.
    moved = float(jnp.max(jnp.abs(final.positions - state.positions)))
    assert moved > 0.0, "the rollout did not move any particle"


@pytest.mark.slow
def test_the_models_own_base_step_matches_nornax_advance_base_step(prepared):
    """jaccpot's ``advance_base_step`` and nornax's must integrate the same equations.

    They are separate implementations of the same palindrome -- jaccpot's so a caller
    can drive the lane without nornax at all -- so one of them drifting is exactly the
    kind of thing no single-implementation test can see.
    """
    fmm, pos, vel, mass, rung = prepared
    state = initialize_block_state(pos, vel, mass, fmm, k_max=K_MAX, rung=rung)
    theirs = advance_base_step(state, 2.0e-3, fmm, k_max=K_MAX)
    mine_p, mine_v, _acc = fmm.advance_base_step(
        pos, vel, mass, rung=rung, dt_max=2.0e-3
    )
    dp = float(jnp.linalg.norm(mine_p - theirs.positions)) / float(
        jnp.linalg.norm(theirs.positions)
    )
    dv = float(jnp.linalg.norm(mine_v - theirs.velocities)) / float(
        jnp.linalg.norm(theirs.velocities)
    )
    assert dp < 1e-12, f"positions differ by {dp:.3e}"
    assert dv < 1e-12, f"velocities differ by {dv:.3e}"


@pytest.mark.slow
def test_a_starved_capacity_raises_rather_than_returning_a_wrong_force(prepared):
    """The one thing this model does that the driver does not: raise, eagerly.

    A starved capacity drops a canonical pair, which drops BOTH its halves, so the
    global momentum sum stays exact and no norm on the result reveals it. Reported is
    not good enough for a force an integrator will step with, so this lane raises.
    """
    pos, _vel, mass, rung = _system()
    fmm = _model(cross_caps={"near_cap": 1, "max_pair_queue": 8})
    fmm.prepare(pos, mass)
    with pytest.raises(RuntimeError, match="capacity overflowed"):
        fmm.level_accelerations(pos, mass, rung=rung, level=0)


def test_an_unknown_cross_cap_name_is_refused():
    """A misspelt capacity would silently keep the heuristic default."""
    with pytest.raises(ValueError, match="unknown cross_caps keys"):
        _model(cross_caps={"near_capacity": 10})
