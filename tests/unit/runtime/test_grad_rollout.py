"""``differentiable_rollout`` on the softened direct sum (S9, track B).

The rollout is the production velocity Verlet over any force with the provider's
step contract. On the direct sum (``direct_sum_step_fn``), with no capacity
anywhere:

* the trajectory IS a hand-written scan of the particle carry's expressions, bit
  for bit, under every checkpoint mode, with an external field and a traced
  ``dt``; and 2 + 3 chained steps are one rollout of 5;
* the three checkpoint modes compute the same forward bit for bit, and gradients
  that agree to round-off -- NOT bitwise (measured; see the test);
* the gradient is the finite differences' (float64, 4 steps) in the start
  positions, velocities, masses, ``dt``, ``G``, the softening and the external
  field's parameter;
* ``"step"`` saves exactly the carry per step: the residuals are three ``[T, N, 3]``
  stacks and the masses are stored once, and the compiled temporaries grow by the
  carry's bytes per step (36 B/particle float32); ``"nested"`` stores one carry per
  segment plus one per step of a segment;
* a failed verdict in the aux is refused on the host, gradient included.

CPU, N = 64.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N, STEPS, DT, SOFTENING = 64, 5, 0.05, 0.05
#: the external field's spring constant
K_EXT = 0.3


def _system(dtype=jnp.float64, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = jnp.asarray(rng.normal(size=(N, 3)), dtype)
    v0 = jnp.asarray(0.3 * rng.normal(size=(N, 3)), dtype)
    m = jnp.asarray((np.abs(rng.normal(size=N)) + 0.5) / N, dtype)
    wx = jnp.asarray(rng.normal(size=(N, 3)), dtype)
    wv = jnp.asarray(rng.normal(size=(N, 3)), dtype)
    return x0, v0, m, wx, wv


def _harmonic(state_rows, args):
    """``strict_run_v2``'s two-argument field form: a row-by-row harmonic pull."""
    return -args["k"] * state_rows[:, 0]


def _step():
    from jaccpot.runtime.grad_rollout import direct_sum_step_fn

    return direct_sum_step_fn(G=1.0, softening=SOFTENING)


def _rollout(x0, v0, m, dt, k, *, checkpoint, steps=STEPS, params=None, **kw):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    return differentiable_rollout(
        _step(),
        x0,
        v0,
        m,
        dt,
        steps,
        params=params,
        external_fn=_harmonic,
        external_args={"k": k},
        checkpoint=checkpoint,
        **kw,
    )


MODES = [None, "step", "nested"]


@pytest.mark.parametrize("checkpoint", MODES, ids=["plain", "step", "nested"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["f32", "f64"])
def test_the_trajectory_is_a_hand_written_scan(dtype, checkpoint):
    """The caller's velocity Verlet of ``test_strict_runners_on_provider.py``, with
    the start force and the field written out, against the rollout: bitwise. Five
    steps, so ``"nested"`` (segments of round(sqrt(5)) = 2) runs a remainder
    segment of 1 too."""
    x0, v0, m, _, _ = _system(dtype)
    step = _step()

    @jax.jit
    def by_hand(x0, v0, m, dt, k):
        def field(x, v):
            return _harmonic(jnp.stack([x, v], axis=1), {"k": k})

        def verlet(carry, _):
            x, v, a = carry
            x = x + v * dt + 0.5 * a * dt**2
            a_new = step(x, m)[0] + field(x, v)
            return (x, v + 0.5 * (a + a_new) * dt, a_new), None

        a0 = step(x0, m)[0] + field(x0, v0)
        return jax.lax.scan(verlet, (x0, v0, a0), None, length=STEPS)[0]

    dt, k = jnp.asarray(DT, dtype), jnp.asarray(K_EXT, dtype)
    want = by_hand(x0, v0, m, dt, k)
    got = jax.jit(lambda *a: _rollout(*a, checkpoint=checkpoint))(x0, v0, m, dt, k)
    assert got.aux is None
    for g, w in zip(got[:3], want):
        assert g.dtype == dtype
        assert np.array_equal(np.asarray(g), np.asarray(w))


def test_chained_rollouts_are_one_rollout():
    """2 + 3 steps, the second started from the first's acceleration and aux, are
    one rollout of 5, bitwise."""
    x0, v0, m, _, _ = _system()
    run = jax.jit(
        lambda x, v, m, a, steps: _rollout(
            x, v, m, DT, K_EXT, checkpoint="step", steps=steps, acceleration=a
        ),
        static_argnums=4,
    )
    whole = run(x0, v0, m, None, 5)
    mid = run(x0, v0, m, None, 2)
    end = run(mid.positions, mid.velocities, m, mid.acceleration, 3)
    for g, w in zip(end[:3], whole[:3]):
        assert np.array_equal(np.asarray(g), np.asarray(w))


@pytest.mark.parametrize(
    "dtype, rtol", [(jnp.float32, 3e-5), (jnp.float64, 1e-13)], ids=["f32", "f64"]
)
def test_the_checkpoint_modes_agree_to_round_off(dtype, rtol):
    """The gradients, and the forward computed with them, agree to round-off.

    Measured (CPU, 2026-10-31, N = 64, 4-6 steps): the largest gradient difference
    between the plain scan and ``"step"`` or ``"nested"`` is 1.5e-15 of the
    largest component in float64 and 2.7e-7 in float32 (the tolerances here are
    about 70x and 100x that). The forward agrees bitwise in every case. Three
    things move the gradient's last bits, none of them a different map:

    * a checkpoint changes how the cotangents of a value used inside and outside
      it are grouped: the start positions' cotangent is the start force's own sum
      plus the scan's under ``jax.checkpoint``, one running sum of every use
      without it;
    * ``"nested"`` sums a scan constant's cotangent (masses, ``dt``, the field's
      parameter) per segment and then over segments, where ``"step"`` keeps one
      running sum over all steps -- carrying the constants instead would store
      them per step, which is what the checkpoint is for avoiding;
    * the reverse pass recomputes the forward inside its own fusions, and XLA's CPU
      backend contracts a multiply and an add into one FMA within a fusion (a
      ``1 + 2^-30`` product and sum: 8.7e-19 jitted, 0 in numpy); with FMAs off
      (``--xla_cpu_max_isa=AVX``) the reductions still differ.

    An elementwise force with ``acceleration`` given gives bitwise position and
    velocity gradients between ``"step"`` and ``"nested"``; the direct sum's
    reductions do not.

    The forward-only rollouts are bitwise in every mode
    (:func:`test_the_trajectory_is_a_hand_written_scan`), but the forward that a
    ``value_and_grad`` program computes is not always: the plain scan's forward
    also writes every residual, and ``"step"``'s compiles in another context, so
    XLA fuses (and contracts) differently (CPU, 2026-10-31, 5 steps: plain
    float32 3 / 23 / 62 of 192 elements of x / v / a off the forward-only
    rollout, float64 1 / 30 / 48; ``"step"`` float64 the same; ``"nested"``
    bitwise). An exact comparison against a forward-only run therefore belongs to
    the forward (G1), not to the primal of a gradient.
    """
    x0, v0, m, wx, wv = _system(dtype)
    dt, k = jnp.asarray(DT, dtype), jnp.asarray(K_EXT, dtype)

    def loss(x0, v0, m, dt, k, checkpoint):
        out = _rollout(x0, v0, m, dt, k, checkpoint=checkpoint)
        return jnp.sum(wx * out.positions) + jnp.sum(wv * out.velocities), out

    fn = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True),
        static_argnums=5,
    )
    (_, ref_out), ref = fn(x0, v0, m, dt, k, None)
    for mode in ("step", "nested"):
        (_, out), grads = fn(x0, v0, m, dt, k, mode)
        named = zip(("x", "v", "a"), out[:3], ref_out[:3])
        for name, g, w in [*named, *zip(("x0", "v0", "m", "dt", "k"), grads, ref)]:
            g, w = np.asarray(g), np.asarray(w)
            scale = float(np.max(np.abs(w)))
            assert scale > 0.0
            err = float(np.max(np.abs(g - w))) / scale
            assert err <= rtol, f"{mode} {name}: {err:.2e} of the largest component"


@pytest.mark.parametrize("checkpoint", MODES, ids=["plain", "step", "nested"])
def test_the_gradient_is_the_finite_differences(checkpoint):
    """``jax.grad`` of a linear functional of the state after 4 steps (float64)
    against central differences, in every input the rollout differentiates: the
    start positions, velocities and masses (along a random unit direction), ``dt``,
    ``G`` and the softening squared (traced ``ForceParams``), the field's spring
    constant. The start force is computed inside (``acceleration=None``), so its
    derivative is in it.

    The step is measured, not inherited (CPU, 2026-10-31): relative to each
    input's largest entry, the error falls as h^2 from h = 1e-2 (3e-4 in x0, 6e-4
    in dt) to a round-off floor at h = 1e-5, where every input agrees to 4e-10 -
    3e-9 (h = 1e-4: up to 6e-8 in dt; h = 1e-6: up to 2e-8 in v0)."""
    from jaccpot import ForceParams

    x0, v0, m, wx, wv = _system()
    args = (
        x0,
        v0,
        m,
        jnp.asarray(DT),
        jnp.asarray(1.0),
        jnp.asarray(SOFTENING**2),
        jnp.asarray(K_EXT),
    )

    @jax.jit
    def loss(x0, v0, m, dt, g, eps2, k):
        out = _rollout(
            x0,
            v0,
            m,
            dt,
            k,
            checkpoint=checkpoint,
            steps=4,
            params=ForceParams(G=g, eps2=eps2),
        )
        return jnp.sum(wx * out.positions) + jnp.sum(wv * out.velocities)

    grads = jax.jit(jax.grad(loss, argnums=tuple(range(len(args)))))(*args)
    rng = np.random.default_rng(7)
    for i, name in enumerate(("x0", "v0", "m", "dt", "G", "eps2", "k")):
        a = args[i]
        d = jnp.asarray(rng.normal(size=a.shape)) if a.ndim else jnp.asarray(1.0)
        if a.ndim:
            d = d / jnp.linalg.norm(d)
        h = 1e-5 * float(jnp.max(jnp.abs(a)))
        plus, minus = list(args), list(args)
        plus[i], minus[i] = a + h * d, a - h * d
        fd = (float(loss(*plus)) - float(loss(*minus))) / (2.0 * h)
        exact = float(jnp.sum(grads[i] * d))
        assert exact != 0.0, name
        assert abs(fd - exact) <= 1e-7 * abs(
            exact
        ), f"{name}: grad {exact:.10e}, finite difference {fd:.10e}"


def _residuals(capsys, fn, *args) -> list[str]:
    from jax.ad_checkpoint import print_saved_residuals

    capsys.readouterr()
    print_saved_residuals(fn, *args)
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "checkpoint, steps, traced_dt, stacked, stacks",
    [
        ("step", 4, False, 4, 1),
        ("nested", 9, False, 3, 3),
        ("step", 4, True, 4, 3),
    ],
    ids=["step", "nested", "step-traced-dt"],
)
def test_a_checkpointed_rollout_saves_what_the_force_needs(
    capsys, checkpoint, steps, traced_dt, stacked, stacks
):
    """``print_saved_residuals`` of the gradient, with an external field whose
    constant is differentiated.

    ``"step"`` checkpoints each step's force and kick with the drifted positions
    as input (track C's placement): the scan saves ONE ``[T, N, 3]`` stack, those
    positions -- the drift and the kick are linear in ``(x, v, a)``, so their
    reverse reads no primal, and the field reads only the position row
    (``_field_reads_velocities``), so the velocities are not saved for it.
    ``"nested"`` (9 steps in 3 segments of 3): the outer scan saves three
    ``[3, N, 3]``, the carry ``(x, v, a)`` at each segment's start. A
    DIFFERENTIATED ``dt`` makes the drift bilinear: its cotangent needs ``v`` and
    ``a`` of every step: three stacks in all, ``(x, v, a)`` (four under session 1's
    ``jax.checkpoint`` placement, which kept the undrifted positions too; the
    hand-written reverse transposes the drift itself). The masses are an argument,
    stored once (CPU, 2026-11-01)."""
    x0, v0, m, wx, _ = _system(jnp.float32)
    k = jnp.asarray(K_EXT, jnp.float32)

    def loss(x0, v0, m, k, *dt):
        out = _rollout(
            x0, v0, m, dt[0] if dt else DT, k, checkpoint=checkpoint, steps=steps
        )
        return jnp.sum(wx * out.positions)

    args = (x0, v0, m, k) + ((jnp.asarray(DT, jnp.float32),) if traced_dt else ())
    lines = _residuals(capsys, loss, *args)
    per_step = [ln for ln in lines if "output of scan" in ln]
    assert [ln.split()[0] for ln in per_step] == [f"f32[{stacked},{N},3]"] * stacks
    # nothing else is a stack, and the masses are an argument, stored once
    shapes = [ln.split()[0] for ln in lines]
    assert sum(s.startswith(f"f32[{stacked},") for s in shapes) == stacks
    assert sum(f"f32[{N}]" == s for s in shapes) == 1
    assert any(ln.startswith(f"f32[{N}] from the argument") for ln in lines)


def test_the_velocity_rows_reach_a_field_that_reads_them():
    """A field that reads the velocity row gets the velocities (and they become a
    residual again); a halo-like field that reads only ``state[:, 0]`` gets zeros
    there -- with the same output."""
    from jaccpot.runtime.grad_rollout import _field_reads_velocities

    x = jnp.zeros((N, 3))

    def drag(state, args):
        return -args["k"] * state[:, 1]

    def halo(state):
        pos = state[:, 0, :]
        r2 = jnp.sum(pos * pos, axis=1, keepdims=True)
        return -pos / (r2 + 1.0) ** 1.5

    assert _field_reads_velocities(drag, {"k": 1.0}, x)
    assert not _field_reads_velocities(_harmonic, {"k": 1.0}, x)
    assert not _field_reads_velocities(halo, None, x)
    # a field it cannot see into is assumed to read them
    assert _field_reads_velocities(jax.jit(halo), None, x)
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def no_force(positions, masses, params=None, aux=None):
        return jnp.zeros_like(positions), None

    x0, v0, m, _, _ = _system()
    out = jax.jit(
        lambda x, v, m: differentiable_rollout(
            no_force, x, v, m, DT, 3, external_fn=drag, external_args={"k": 2.0}
        )
    )(x0, v0, m)
    # a pure drag, about (1 - k dt) = 0.9 per kick: the field saw the velocities
    # (zeros would have left them unchanged)
    assert float(jnp.sum(out.velocities**2)) < 0.8 * float(jnp.sum(v0**2))


def _temp_bytes(checkpoint, steps, inner=None, n=256):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.normal(size=(n, 3)), jnp.float32)
    m = jnp.full((n,), 1.0 / n, jnp.float32)

    def loss(x0, v0, m):
        out = differentiable_rollout(
            _step(),
            x0,
            v0,
            m,
            DT,
            steps,
            checkpoint=checkpoint,
            inner_steps=inner,
        )
        return jnp.sum(out.positions * out.velocities)

    compiled = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(x0, x0, m).compile()
    return compiled.memory_analysis().temp_size_in_bytes


def test_the_compiled_temporaries_grow_by_the_saved_positions():
    """``compiled.memory_analysis()``, per step: ``"step"``'s temporaries grow by
    the drifted positions, 12 B/particle in float32 (within 5 %: measured exact, CPU
    N = 256 - 4096, ``bench/grad_rollout_memory.py``; 36 with the checkpoint around
    the whole step, before track C's placement); ``"nested"``'s by one carry (36 B)
    per segment and 12 B per step of a segment (9 -> 25 steps at segments of 3 ->
    5: +2 carries, +2 steps, 96 B); the plain scan's by the force's own residuals,
    the ``[N, N]`` pair arrays (thousands of times more)."""
    n = 256
    step = (_temp_bytes("step", 8) - _temp_bytes("step", 4)) / 4
    assert abs(step / (12 * n) - 1.0) <= 0.05, step / n
    nested = _temp_bytes("nested", 25) - _temp_bytes("nested", 9)
    assert abs(nested / ((2 * 36 + 2 * 12) * n) - 1.0) <= 0.05, nested / n
    plain = (_temp_bytes(None, 8) - _temp_bytes(None, 4)) / 4
    assert plain > 1000 * 12 * n


# --------------------------------------------------------------------------
# The verdict.
# --------------------------------------------------------------------------


#: a particle beyond this radius "overflows" the toy step's capacity
_RADIUS = 2.5


def _capped_step(positions, masses, params=None, aux=None):
    """The direct sum with the provider's aux: ``OVERFLOW_TREE`` once a particle is
    beyond ``_RADIUS``, folded like the provider's (OR over the steps)."""
    from jaccpot.runtime.fused_topology import OVERFLOW_TREE

    acc, _ = _step()(positions, masses, params)
    bits = jnp.where(
        jnp.max(jnp.abs(positions)) > _RADIUS, jnp.uint32(OVERFLOW_TREE), jnp.uint32(0)
    )
    if aux is not None and aux[1] is not None:
        bits = bits | aux[1]
    return acc, (None, bits, jnp.zeros((2,), jnp.int32))


def _verdict_loss(x0, v0, m, steps):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    out = differentiable_rollout(_capped_step, x0, v0, m, DT, steps)
    return jnp.sum(out.positions**2), out


def test_a_failed_verdict_refuses_the_state_and_the_gradient():
    from jaccpot.runtime.grad_rollout import (
        RolloutVerdictError,
        check_rollout,
        rollout_ok,
        value_and_grad_checked,
    )

    x0, v0, m, _, _ = _system()
    x0 = x0 / jnp.max(jnp.abs(x0))  # inside the radius
    # every step inside: the verdict holds and the gradient is jax's own
    run = value_and_grad_checked(_verdict_loss, argnums=(0, 2), static_argnums=3)
    (value, out), grads = run(x0, v0, m, 3)
    assert rollout_ok(out) and rollout_ok(out.aux)
    check_rollout(out)
    (want, _), want_grads = jax.value_and_grad(_verdict_loss, (0, 2), has_aux=True)(
        x0, v0, m, 3
    )
    np.testing.assert_allclose(float(value), float(want), rtol=1e-14)
    for g, w in zip(grads, want_grads):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-12)
    # one particle flies out at 40 per unit time: beyond the radius at the 2nd
    # step's positions (2 * 0.05 * 40 = 4), and the fold keeps the bit to the end
    v_out = v0.at[0].set(jnp.asarray([40.0, 0.0, 0.0]))
    _, out = jax.jit(_verdict_loss, static_argnums=3)(x0, v_out, m, 4)
    assert not rollout_ok(out)
    with pytest.raises(RolloutVerdictError, match="OVERFLOW_TREE"):
        check_rollout(out.aux)
    with pytest.raises(RolloutVerdictError, match="refused"):
        run(x0, v_out, m, 4)


def test_a_step_without_a_verdict_always_passes_and_an_unknown_aux_is_refused():
    from jaccpot.runtime.grad_rollout import check_rollout, rollout_ok

    x0, v0, m, _, _ = _system()
    out = jax.jit(lambda *a: _rollout(*a, checkpoint="step"))(x0, v0, m, DT, K_EXT)
    assert out.aux is None and rollout_ok(out)
    check_rollout(out)
    with pytest.raises(TypeError, match="no capacity verdict"):
        rollout_ok({"bits": 0})


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(num_steps=0), "num_steps must be >= 1"),
        (dict(checkpoint="remat"), "checkpoint must be one of"),
        (dict(inner_steps=2), "inner_steps needs checkpoint='nested'"),
        (dict(checkpoint="nested", inner_steps=0), "inner_steps must be >= 1"),
        (dict(external_args={"k": 1.0}), "external_args needs an external_fn"),
        (dict(masses=jnp.ones((N + 1,))), r"masses must be \[64\]"),
        (dict(velocities=jnp.zeros((N, 3), jnp.float32)), "the state has one dtype"),
        (dict(acceleration=jnp.zeros((N, 2))), r"acceleration must be \[64, 3\]"),
    ],
)
def test_bad_arguments_are_refused(kw, match):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    x0, v0, m, _, _ = _system()
    call = dict(
        step_fn=_step(),
        positions=x0,
        velocities=v0,
        masses=m,
        dt=DT,
        num_steps=2,
    )
    call.update(kw)
    with pytest.raises(ValueError, match=match):
        differentiable_rollout(**call)


def test_the_gradient_verdict_reads_the_gradient_bits():
    """A gradient is refused on ``GRAD_FAIL_MASK`` (the force's bits and the
    differentiable step's ``OVERFLOW_GRAD_LISTS``, 1 << 11, which the forward does
    not read); the state on ``OVERFLOW_FAIL_MASK`` only. Bit 11 set alone: the
    state passes, the gradient does not."""
    from jaccpot.runtime import fused_topology
    from jaccpot.runtime.grad_rollout import (
        RolloutVerdictError,
        check_rollout,
        rollout_ok,
    )

    grad_bit = fused_topology.OVERFLOW_GRAD_LISTS
    assert grad_bit == 1 << 11
    aux = (None, jnp.uint32(grad_bit), jnp.zeros((2,), jnp.int32))
    assert rollout_ok(aux)
    check_rollout(aux)
    assert not rollout_ok(aux, gradient=True)
    with pytest.raises(RolloutVerdictError, match="gradient verdict failed"):
        check_rollout(aux, gradient=True)
