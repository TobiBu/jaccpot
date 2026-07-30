"""Distributed (multi-GPU) gradient correctness for the differentiable FMM.

Phase 5 of the differentiability work: ``make_force_evaluator(...,
differentiable=True)`` puts the ``shard_map`` force pipeline under ``jax.grad``
at fixed topology. What is asserted here:

* the forward is **bit-identical** to the shipped forward path (the seam is
  ``stop_gradient`` plus re-gathers, so it may not move a single bit);
* FD vs AD along a random direction, for positions and for masses;
* ``grad(FMM)`` against ``grad(direct sum)`` -- the physics gate, which must
  track the forward's own FMM force accuracy;
* a forward evaluated **after** a gradient is still correct. That is a
  regression guard, not a formality: the native ``jax.lax.ragged_all_to_all``
  halo exchange silently dropped the whole cross-domain near field from every
  later evaluation once its reverse pass had run (see
  ``docs/differentiable_fmm_distributed_audit.md``).

Run on >= 2 GPUs:

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        pytest tests/test_distributed_grad_correctness.py -o addopts="" -q
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import DistributedFMMConfig
from jaccpot.distributed.fmm import make_force_evaluator, partition_for_devices

pytestmark = [
    pytest.mark.skipif(
        device_count() < 2, reason="distributed gradients need >= 2 devices"
    ),
    pytest.mark.slow,
]

NDEV = 2
PER = 32


def _clusters(ndev: int, per: int, seed: int = 4):
    """One spatially separated cluster per Morton domain (engages cross-domain)."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float64,
    )[:ndev]
    positions = np.concatenate(
        [centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    )
    return positions, rng.uniform(0.5, 2.0, size=(per * ndev,))


def _rel_l2(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


@pytest.fixture(scope="module")
def setup():
    """One partitioned system + both evaluators, shared across the tests."""
    config = dataclasses.replace(
        DistributedFMMConfig(), nearfield_backend="baseline", local_walk="dual_tree"
    )
    positions, masses = _clusters(NDEV, PER)
    part = partition_for_devices(positions, masses, NDEV, leaf_size=config.leaf_size)
    mesh = make_mesh(NDEV)
    return {
        "config": config,
        "positions": positions,
        "masses": masses,
        "pos": jnp.asarray(part["pos_flat"]),
        "mass": jnp.asarray(part["mass_flat"]),
        "gid": jnp.asarray(part["gid_flat"]),
        "counts": jnp.asarray(part["counts"]),
        "forward": make_force_evaluator(
            config, NDEV, part["cap"], mesh, jit=True, differentiable=False
        ),
        "grad_path": make_force_evaluator(
            config, NDEV, part["cap"], mesh, jit=True, differentiable=True
        ),
    }


def _loss(setup, positions, masses):
    accel = setup["grad_path"](positions, masses, setup["gid"], setup["counts"])[0]
    return jnp.sum(accel**2)


def test_differentiable_forward_is_bit_identical(setup):
    """differentiable=True must not perturb the forward force at all."""
    args = (setup["pos"], setup["mass"], setup["gid"], setup["counts"])
    shipped = np.asarray(setup["forward"](*args)[0])
    grad_path = np.asarray(setup["grad_path"](*args)[0])
    assert np.array_equal(shipped, grad_path), (
        "differentiable=True changed the forward force "
        f"(max abs diff {np.max(np.abs(shipped - grad_path)):.3e})"
    )


@pytest.mark.parametrize("wrt", ["positions", "masses"])
def test_fd_vs_ad(setup, wrt):
    """Directional finite differences against jax.grad, at fixed topology."""
    pos, mass = setup["pos"], setup["mass"]
    argnum = 0 if wrt == "positions" else 1
    base = pos if argnum == 0 else mass

    grad = jax.grad(lambda p, m: _loss(setup, p, m), argnums=argnum)(pos, mass)
    grad = np.asarray(grad)
    assert np.all(np.isfinite(grad)), f"{wrt} gradient has non-finite entries"
    assert np.linalg.norm(grad) > 0, f"{wrt} gradient is identically zero"

    direction = np.asarray(
        jax.random.normal(jax.random.PRNGKey(0), base.shape, dtype=base.dtype)
    )
    analytic = float(np.sum(grad * direction))
    # 1e-5 is comfortably inside the fixed-topology regime for this IC (no
    # particle crosses a cell/MAC boundary) and far above the float64 noise floor.
    step = 1e-5
    d = jnp.asarray(direction)
    if argnum == 0:
        plus = float(_loss(setup, pos + step * d, mass))
        minus = float(_loss(setup, pos - step * d, mass))
    else:
        plus = float(_loss(setup, pos, mass + step * d))
        minus = float(_loss(setup, pos, mass - step * d))
    numeric = (plus - minus) / (2 * step)
    rel = abs(numeric - analytic) / (abs(numeric) + 1e-300)
    assert (
        rel < 1e-5
    ), f"{wrt} FD-vs-AD rel err {rel:.3e} (fd={numeric:.6e}, ad={analytic:.6e})"


def test_grad_matches_direct_sum_oracle(setup):
    """grad(FMM) vs grad(exact direct sum), the strongest cross-check.

    The direct sum is exactly differentiable, so agreement is bounded by the
    FMM's own force accuracy -- asserted against the *measured* forward error so
    the gradient can never be looser than the force it differentiates.
    """
    config = setup["config"]
    pos, mass = setup["pos"], setup["mass"]
    gid = np.asarray(setup["gid"]).reshape(-1)
    valid = gid >= 0
    order_map = gid[valid].astype(int)

    g_pos, g_mass = jax.grad(lambda p, m: _loss(setup, p, m), argnums=(0, 1))(pos, mass)

    positions0 = jnp.asarray(setup["positions"])
    masses0 = jnp.asarray(setup["masses"])

    def direct_accel(p, m):
        d = p[:, None, :] - p[None, :, :]
        dist_sq = jnp.sum(d * d, -1) + config.softening**2
        eye = jnp.eye(p.shape[0], dtype=p.dtype)
        return -config.G * jnp.einsum(
            "ij,ijk->ik", m[None, :] * dist_sq**-1.5 * (1.0 - eye), d
        )

    dg_pos, dg_mass = jax.grad(
        lambda p, m: jnp.sum(direct_accel(p, m) ** 2), argnums=(0, 1)
    )(positions0, masses0)

    # forward accuracy of this configuration, as the tolerance reference
    accel = np.asarray(setup["grad_path"](pos, mass, setup["gid"], setup["counts"])[0])
    scattered = np.zeros((positions0.shape[0], 3))
    scattered[order_map] = accel[valid]
    force_err = _rel_l2(scattered, np.asarray(direct_accel(positions0, masses0)))

    fmm_gp = np.zeros_like(np.asarray(dg_pos))
    fmm_gm = np.zeros_like(np.asarray(dg_mass))
    fmm_gp[order_map] = np.asarray(g_pos)[valid]
    fmm_gm[order_map] = np.asarray(g_mass)[valid]
    err_pos = _rel_l2(fmm_gp, np.asarray(dg_pos))
    err_mass = _rel_l2(fmm_gm, np.asarray(dg_mass))

    budget = max(10 * force_err, 1e-4)
    assert err_pos < budget, (
        f"position gradient rel-L2 {err_pos:.3e} exceeds {budget:.3e} "
        f"(forward force error {force_err:.3e})"
    )
    assert err_mass < budget, (
        f"mass gradient rel-L2 {err_mass:.3e} exceeds {budget:.3e} "
        f"(forward force error {force_err:.3e})"
    )


def test_forward_survives_a_gradient(setup):
    """A forward evaluated after a gradient must still be correct.

    Regression guard for the native ragged-all-to-all corruption: executing the
    reverse pass of ``jax.lax.ragged_all_to_all`` left every later ragged
    exchange returning nothing, so the forward silently degraded to the
    LOCAL-ONLY sum (it reproduced it to 2e-16, rel-L2 0.42 against the true
    force) -- on evaluators built *before* the gradient as well.
    """
    args = (setup["pos"], setup["mass"], setup["gid"], setup["counts"])
    before_grad_path = np.asarray(setup["grad_path"](*args)[0])
    before_forward = np.asarray(setup["forward"](*args)[0])

    g = jax.grad(lambda p, m: _loss(setup, p, m), argnums=0)(
        setup["pos"], setup["mass"]
    )
    jax.block_until_ready(g)

    after_grad_path = np.asarray(setup["grad_path"](*args)[0])
    after_forward = np.asarray(setup["forward"](*args)[0])
    assert np.array_equal(before_grad_path, after_grad_path), (
        "the differentiable evaluator's forward changed after a gradient "
        f"(rel-L2 drift {_rel_l2(after_grad_path, before_grad_path):.3e})"
    )
    assert np.array_equal(before_forward, after_forward), (
        "the forward-only evaluator's force changed after a gradient elsewhere "
        f"(rel-L2 drift {_rel_l2(after_forward, before_forward):.3e})"
    )


def test_nearfield_chunk_rejected_on_grad_path():
    """The chunked near field has no autodiff rule; it must raise, not degrade."""
    config = dataclasses.replace(
        DistributedFMMConfig(), nearfield_backend="pallas", nearfield_chunk=256
    )
    with pytest.raises(NotImplementedError, match="nearfield_chunk"):
        make_force_evaluator(
            config, NDEV, 32, make_mesh(NDEV), jit=False, differentiable=True
        )


def test_rejects_unknown_halo_exchange():
    """Only the vetted halo-exchange implementations are selectable."""
    with pytest.raises(ValueError, match="halo_exchange"):
        make_force_evaluator(
            DistributedFMMConfig(),
            NDEV,
            32,
            make_mesh(NDEV),
            jit=False,
            differentiable=True,
            halo_exchange="ragged",
        )


@pytest.mark.skipif(
    os.environ.get("JACCPOT_CHECK_UPSTREAM_RAGGED_FIX", "0") != "1",
    reason=(
        "opt-in: asserts the upstream jax.lax.ragged_all_to_all bug is FIXED. "
        "Run with JACCPOT_CHECK_UPSTREAM_RAGGED_FIX=1 after a JAX upgrade; if it "
        "passes, make halo_exchange='native' the default and drop the shim."
    ),
)
def test_native_halo_exchange_is_fixed_upstream(setup):
    """Tripwire for the JAX fix: is `native` safe to differentiate through yet?

    Today this FAILS by design -- executing a gradient through
    ``jax.lax.ragged_all_to_all`` makes every later forward drop the cross-domain
    near field (see ``bench/repro_jax_ragged_all_to_all_grad.py``). It is the
    one-argument check that tells us when the workaround can go away.
    """
    config = setup["config"]
    positions, masses = _clusters(NDEV, PER)
    part = partition_for_devices(positions, masses, NDEV, leaf_size=config.leaf_size)
    args = (
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        jnp.asarray(part["counts"]),
    )
    native = make_force_evaluator(
        config,
        NDEV,
        part["cap"],
        make_mesh(NDEV),
        jit=True,
        differentiable=True,
        halo_exchange="native",
    )
    before = np.asarray(native(*args)[0])
    grad = jax.grad(lambda p, m: jnp.sum(native(p, m, args[2], args[3])[0] ** 2))(
        args[0], args[1]
    )
    jax.block_until_ready(grad)
    after = np.asarray(native(*args)[0])
    assert np.array_equal(before, after), (
        "jax.lax.ragged_all_to_all still corrupts the forward after a gradient "
        f"(rel-L2 drift {_rel_l2(after, before):.3e}) -- keep halo_exchange='buf'"
    )


def test_l2l_num_levels_requires_differentiable():
    """The static L2L bound is grad-path-only; the forward resolves it exactly."""
    with pytest.raises(ValueError, match="l2l_num_levels"):
        make_force_evaluator(
            DistributedFMMConfig(),
            NDEV,
            32,
            make_mesh(NDEV),
            jit=False,
            differentiable=False,
            l2l_num_levels=4,
        )


def test_truncated_l2l_bound_is_reported(setup):
    """An l2l_num_levels below the tree depth must surface l2l_level_overflow."""
    from jaccpot.distributed.fmm import DIAG_FIELDS

    config = setup["config"]
    positions, masses = _clusters(NDEV, PER)
    part = partition_for_devices(positions, masses, NDEV, leaf_size=config.leaf_size)
    evaluator = make_force_evaluator(
        config,
        NDEV,
        part["cap"],
        make_mesh(NDEV),
        jit=True,
        differentiable=True,
        l2l_num_levels=0,
    )
    _, _, diag = evaluator(
        jnp.asarray(part["pos_flat"]),
        jnp.asarray(part["mass_flat"]),
        jnp.asarray(part["gid_flat"]),
        jnp.asarray(part["counts"]),
    )
    flag = np.asarray(diag)[:, DIAG_FIELDS.index("l2l_level_overflow")]
    assert np.any(
        flag > 0
    ), "an L2L bound of 0 truncated the cascade without setting l2l_level_overflow"
