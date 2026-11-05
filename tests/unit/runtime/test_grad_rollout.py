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
* ``"offload"`` stores exactly ``"nested"``'s residuals, in host memory (explicit
  ``device_put`` s in the hand-written reverse): the same forward and, on the
  direct sum, the same gradient bit for bit; the GPU's compiled device
  temporaries are flat in the horizon and the host's grow by ``"nested"``'s
  carries and step residuals;
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


MODES = [None, "step", "nested", "offload"]
MODE_IDS = ["plain", "step", "nested", "offload"]


@pytest.mark.parametrize("checkpoint", MODES, ids=MODE_IDS)
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
    for mode in ("step", "nested", "offload"):
        (_, out), grads = fn(x0, v0, m, dt, k, mode)
        named = zip(("x", "v", "a"), out[:3], ref_out[:3])
        for name, g, w in [*named, *zip(("x0", "v0", "m", "dt", "k"), grads, ref)]:
            g, w = np.asarray(g), np.asarray(w)
            scale = float(np.max(np.abs(w)))
            assert scale > 0.0
            err = float(np.max(np.abs(g - w))) / scale
            assert err <= rtol, f"{mode} {name}: {err:.2e} of the largest component"


@pytest.mark.parametrize("checkpoint", MODES, ids=MODE_IDS)
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


def _temp_bytes(checkpoint, steps, inner=None, n=256, host=False):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.normal(size=(n, 3)), jnp.float32)
    m = jnp.full((n,), 1.0 / n, jnp.float32)

    # the start acceleration is an input: the start force's reverse is a unit of
    # the reverse loop of its own, and on the direct sum at this N its pair arrays
    # are a peak beside the stored carries but not beside a segment's steps
    def loss(x0, v0, m, a0):
        out = differentiable_rollout(
            _step(),
            x0,
            v0,
            m,
            DT,
            steps,
            acceleration=a0,
            checkpoint=checkpoint,
            inner_steps=inner,
        )
        return jnp.sum(out.positions * out.velocities)

    compiled = (
        jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3))).lower(x0, x0, m, x0).compile()
    )
    stats = compiled.memory_analysis()
    if host:
        return stats.temp_size_in_bytes, stats.host_temp_size_in_bytes
    return stats.temp_size_in_bytes


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


def _host_device_puts(jaxpr) -> list:
    """Every ``device_put`` to host memory in a jaxpr, sub-jaxprs included: the
    dtypes and shapes it moves."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "device_put" and any(
            "Host" in str(d) for d in eqn.params.get("devices", ())
        ):
            found += [(v.aval.dtype.name, v.aval.shape) for v in eqn.invars]
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                sub = getattr(sub, "jaxpr", sub)  # a ClosedJaxpr's
                if hasattr(sub, "eqns"):
                    found += _host_device_puts(sub)
    return found


def test_offload_stores_nesteds_residuals_in_host_memory(capsys):
    """``"offload"`` (9 steps in 3 segments of 3, float32): the forward saves the
    same three ``[3, N, 3]`` carry stacks as ``"nested"``
    (:func:`test_a_checkpointed_rollout_saves_what_the_force_needs`), typed
    ``<host>``, and the program moves exactly the stored values to the host: the
    carry's three arrays in the forward's outer scan and the drifted positions in
    a segment's replay -- not the kick's linear ``v`` and ``a``. The forward is
    ``"nested"``'s and so is the gradient, bit for bit, and a second call repeats
    it (CPU, direct sum: the transfers are copies between views of host
    memory)."""
    x0, v0, m, wx, wv = _system(jnp.float32)
    k = jnp.asarray(K_EXT, jnp.float32)

    def loss(x0, v0, m, k, checkpoint):
        out = _rollout(x0, v0, m, DT, k, checkpoint=checkpoint, steps=9, inner_steps=3)
        return jnp.sum(wx * out.positions) + jnp.sum(wv * out.velocities), out

    lines = _residuals(capsys, lambda *a: loss(*a, "offload")[0], x0, v0, m, k)
    stacks = [ln.split()[0] for ln in lines if "output of scan" in ln]
    assert stacks == [f"f32<host>[3,{N},3]"] * 3
    jaxpr = jax.make_jaxpr(
        jax.grad(lambda *a: loss(*a, "offload")[0], argnums=(0, 1, 2, 3))
    )(x0, v0, m, k).jaxpr
    assert sorted(_host_device_puts(jaxpr)) == [("float32", (N, 3))] * 4
    fn = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True),
        static_argnums=4,
    )
    (_, out_n), g_n = fn(x0, v0, m, k, "nested")
    (_, out_o), g_o = fn(x0, v0, m, k, "offload")
    (_, _), g_o2 = fn(x0, v0, m, k, "offload")
    for a, b, c in zip((*out_o[:3], *g_o), (*out_n[:3], *g_n), (*out_o[:3], *g_o2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
        assert np.array_equal(np.asarray(a), np.asarray(c))


@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="host memory is a separate space on an accelerator only",
)
def test_offload_keeps_the_device_temporaries_flat():
    """On a GPU: ``"offload"``'s compiled device temporaries are the same at every
    horizon and segment length, and its host temporaries grow by one carry (36
    B/particle, float32 direct sum) per segment and the drifted positions (12) per
    step of a segment: ``"nested"``'s device slopes, moved to the host."""
    n = 256
    tb = {
        key: _temp_bytes("offload", *key, n=n, host=True)
        for key in ((9, 3), (25, 5), (16, 2))
    }
    (d9, h9), (d25, h25), (d16, h16) = tb[(9, 3)], tb[(25, 5)], tb[(16, 2)]
    assert d9 == d25 == d16
    # 9 -> 25 steps at segments of 3 -> 5: +2 carries, +2 step residuals
    assert h25 - h9 == (2 * 36 + 2 * 12) * n
    # 9:3 -> 16:2: +5 carries, -1 step residual
    assert h16 - h9 == (5 * 36 - 12) * n


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
        (dict(inner_steps=2), "inner_steps needs checkpoint='nested' or 'offload'"),
        (dict(checkpoint="nested", inner_steps=0), "inner_steps must be >= 1"),
        (dict(external_args={"k": 1.0}), "external_args needs an external_fn"),
        (dict(masses=jnp.ones((N + 1,))), r"masses must be \[64\]"),
        (dict(velocities=jnp.zeros((N, 3), jnp.float32)), "the state has one dtype"),
        (dict(acceleration=jnp.zeros((N, 2))), r"acceleration must be \[64, 3\]"),
        (
            dict(acceleration=jnp.zeros((N, 3)), start_aux=lambda a, b: b),
            "start_aux needs acceleration=None",
        ),
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


# ---------------------------------------------------------------------------
# The aux guard: a gradient through the aux is refused where it would be dropped.
# ---------------------------------------------------------------------------


def _carrying_acc():
    """The direct sum, its acceleration also carried in the aux (as the production
    runner's gradient chain carried a segment's last self-gravity)."""
    base = _step()

    def step(x, m, params, aux):
        acc, _ = base(x, m, params, None)
        return acc, (acc,)

    return step


def _with_scale():
    """The direct sum with a provider-like aux: a gradient-stopped float (f_b's
    role) and an integer count (the verdict bits')."""
    base = _step()

    def step(x, m, params, aux):
        acc, _ = base(x, m, params, None)
        scale = jax.lax.stop_gradient(jnp.sum(acc * acc, axis=1))
        return acc, (scale, aux[1] + jnp.uint32(1))

    return step


@pytest.mark.parametrize("checkpoint", MODES, ids=MODE_IDS)
def test_a_gradient_through_the_aux_is_refused_when_it_would_be_dropped(checkpoint):
    """A loss that reads a value the step carries in the aux: under the
    hand-written reverses (``"step"``, ``"nested"``, ``"offload"``) the reverse
    pass refuses it while it is TRACED (``make_jaxpr``: nothing is compiled or
    run), naming the mode and the leaf; a plain scan differentiates it. Measured
    before the guard (2026-11-01): the gradient through the aux was exactly 0.0 in
    the three modes, 9.59 (norm) under ``None``."""
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    x0, v0, m, wx, _ = _system()
    a0 = jnp.zeros_like(x0)

    def loss(x):
        out = differentiable_rollout(
            _carrying_acc(),
            x,
            v0,
            m,
            DT,
            3,
            aux=(a0,),
            acceleration=a0,
            checkpoint=checkpoint,
            inner_steps=2 if checkpoint in ("nested", "offload") else None,
        )
        return jnp.sum(wx * out.aux[0])

    if checkpoint is None:
        grad = jax.jit(jax.grad(loss))(x0)
        assert float(jnp.linalg.norm(grad)) > 0.0
        return
    with pytest.raises(ValueError, match=rf"checkpoint={checkpoint!r}.*aux\[0\]"):
        jax.make_jaxpr(jax.grad(loss))(x0)


@pytest.mark.parametrize("checkpoint", MODES, ids=MODE_IDS)
def test_a_chain_that_passes_its_aux_on_is_not_refused(checkpoint):
    """Two rollouts, the second continuing the first's acceleration and aux (a
    gradient-stopped float and an integer, the provider's ``(f_b, bits)`` shape),
    a loss on the state: the aux the second reads gets a symbolic-zero cotangent,
    so the first's guard stays quiet; the gradient is ``None``'s to round-off."""
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    x0, v0, m, wx, wv = _system()
    aux0 = (jnp.zeros((N,)), jnp.uint32(0))

    def loss(x, mode):
        kw = dict(
            checkpoint=mode,
            inner_steps=2 if mode in ("nested", "offload") else None,
        )
        one = differentiable_rollout(_with_scale(), x, v0, m, DT, 2, aux=aux0, **kw)
        two = differentiable_rollout(
            _with_scale(),
            one.positions,
            one.velocities,
            m,
            DT,
            3,
            aux=one.aux,
            acceleration=one.acceleration,
            **kw,
        )
        return jnp.sum(wx * two.positions) + jnp.sum(wv * two.velocities)

    got = jax.jit(jax.grad(lambda x: loss(x, checkpoint)))(x0)
    want = jax.jit(jax.grad(lambda x: loss(x, None)))(x0)
    scale = float(jnp.max(jnp.abs(want)))
    assert float(jnp.max(jnp.abs(got - want))) <= 1e-12 * scale


def _counting():
    """The direct sum, its aux a float that each force call increments by one."""
    base = _step()

    def step(x, m, params, aux):
        acc, _ = base(x, m, params, None)
        return acc, aux + 1.0

    return step


@pytest.mark.parametrize("checkpoint", MODES, ids=MODE_IDS)
def test_start_aux_maps_the_start_forces_aux_for_the_first_step(checkpoint):
    """``start_aux(aux_in, aux_out)`` replaces the aux the start force returned
    (what the production chain's ``"seed_kept"`` segment needs: the seed again
    for the first step): the steps continue from it, and nothing else moves -- the
    state and the gradient are the rollout's without it, bitwise."""
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    x0, v0, m, wx, wv = _system()
    steps = 3
    kw = dict(
        checkpoint=checkpoint,
        inner_steps=2 if checkpoint in ("nested", "offload") else None,
    )

    def run(x, start_aux):
        extra = {} if start_aux is None else {"start_aux": start_aux}
        out = differentiable_rollout(
            _counting(), x, v0, m, DT, steps, aux=jnp.asarray(7.0), **kw, **extra
        )
        return jnp.sum(wx * out.positions) + jnp.sum(wv * out.velocities), out

    def seed_again(aux_in, aux_out):
        del aux_out
        return aux_in + 100.0

    (plain_value, plain), plain_grad = jax.jit(
        jax.value_and_grad(lambda x: run(x, None), has_aux=True)
    )(x0)
    (value, mapped), grad = jax.jit(
        jax.value_and_grad(lambda x: run(x, seed_again), has_aux=True)
    )(x0)
    assert float(plain.aux) == 7.0 + 1.0 + steps
    assert float(mapped.aux) == 7.0 + 100.0 + steps
    assert np.array_equal(np.asarray(value), np.asarray(plain_value))
    for got, want in zip(mapped[:3], plain[:3]):
        assert np.array_equal(np.asarray(got), np.asarray(want))
    assert np.array_equal(np.asarray(grad), np.asarray(plain_grad))


# ---------------------------------------------------------------------------
# Forward mode (ad_mode="forward"): the plain scan, jvp and jacfwd.
# ---------------------------------------------------------------------------


def test_forward_mode_rollout_is_the_plain_scan_and_its_jvp_the_adjoints_transpose():
    """``ad_mode="forward"`` runs the ``checkpoint=None`` scan (the default
    ``"step"`` means the same; the forward is bitwise ``checkpoint=None``'s), its
    jvp matches the reverse gradient (the adjoint identity) in every input, and
    ``jax.jacfwd`` over ``(dt, k)`` matches ``jax.jacrev``."""
    x0, v0, m, wx, wv = _system()
    dt, k = jnp.asarray(DT), jnp.asarray(K_EXT)

    def loss(ad_mode, checkpoint):
        def f(x, v, mm, dd, kk):
            out = _rollout(x, v, mm, dd, kk, checkpoint=checkpoint, ad_mode=ad_mode)
            return jnp.sum(wx * out.positions) + jnp.sum(wv * out.velocities)

        return f

    args = (x0, v0, m, dt, k)
    plain = jax.jit(loss("reverse", None))(*args)
    for checkpoint in (None, "step"):
        assert np.asarray(jax.jit(loss("forward", checkpoint))(*args)).tobytes() == (
            np.asarray(plain).tobytes()
        )
    grads = jax.jit(jax.grad(loss("reverse", "step"), argnums=tuple(range(5))))(*args)
    rng = np.random.default_rng(3)
    dirs = tuple(jnp.asarray(rng.normal(size=np.shape(a)), a.dtype) for a in args)
    value, tangent = jax.jit(lambda p, t: jax.jvp(loss("forward", None), p, t))(
        args, dirs
    )
    assert np.asarray(value).tobytes() == np.asarray(plain).tobytes()
    want = sum(float(jnp.vdot(g, d)) for g, d in zip(grads, dirs))
    assert abs(float(tangent) - want) <= 1e-12 * abs(want)

    def theta_fn(ad_mode, checkpoint):
        def f(theta):
            return jnp.stack(
                [
                    loss(ad_mode, checkpoint)(x0, v0, m, theta[0], theta[1]),
                    loss(ad_mode, checkpoint)(x0, v0, 2.0 * m, theta[0], theta[1]),
                ]
            )

        return f

    theta = jnp.asarray([DT, K_EXT])
    jf = np.asarray(jax.jit(jax.jacfwd(theta_fn("forward", None)))(theta))
    jr = np.asarray(jax.jit(jax.jacrev(theta_fn("reverse", "step")))(theta))
    assert np.max(np.abs(jf - jr)) <= 1e-12 * np.max(np.abs(jr))


@pytest.mark.parametrize("checkpoint", ["nested", "offload"])
def test_forward_mode_refuses_reverse_storage_schedules(checkpoint):
    x0, v0, m, _, _ = _system()
    with pytest.raises(ValueError, match="ad_mode='forward' stores no residuals"):
        _rollout(x0, v0, m, DT, K_EXT, checkpoint=checkpoint, ad_mode="forward")
