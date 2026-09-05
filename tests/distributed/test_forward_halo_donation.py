"""The FORWARD force under a donating loop with moving positions -- the leapfrog case.

Every distributed rollout before 2026-09-04 was wrong from its second force
evaluation on jax 0.9.0: the native ``ragged_all_to_all`` halo exchange kept its
fill value once buffer donation moved its buffers, the cross-domain near field
vanished, and no invariant or overflow flag noticed. The forward path was not gated
because the defect had only ever been triggered through a gradient; identical-input
reproducibility tests are blind to it because they never move a buffer.

This test does what a leapfrog does -- donate the positions, move them a little,
evaluate again -- and compares EVERY step against an fp64 direct sum at that step's
positions. Step 0 sets the accuracy class (it is always clean); a later step that
leaves it by more than 3x is the defect.

Run on >= 2 GPUs (or 2 host devices for the ``buf`` path only):

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        pytest tests/distributed/test_forward_halo_donation.py -o addopts="" -q
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")

from yggdrax.distributed import make_mesh  # noqa: E402

from jaccpot.distributed.fmm import (  # noqa: E402
    DistributedFMMConfig,
    make_force_evaluator,
    partition_for_devices,
)

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="needs >= 2 devices for a cross-domain halo"
    ),
    pytest.mark.slow,
]

NDEV = 2
PER = 48
LEAF = 8
STEPS = 6
SOFT = 0.05


def _clusters(ndev: int, per: int, seed: int = 4):
    """``ndev`` spatially separated clusters, so the cross-domain halo carries mass."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]
    )[:ndev]
    pos = np.concatenate([c + rng.normal(scale=1.0, size=(per, 3)) for c in centers])
    mass = rng.uniform(0.5, 1.5, size=pos.shape[0])
    return pos, mass


def _direct(pos, mass, soft, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(d * d, axis=-1) + soft * soft
    inv = g * mass[None, :] / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv, 0.0)
    return np.einsum("ij,ijk->ik", inv, d)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _evaluator_and_partition(halo_exchange):
    config = dataclasses.replace(
        DistributedFMMConfig(leaf_size=LEAF, softening=SOFT),
        nearfield_backend="baseline",
        local_walk="dual_tree",
    )
    pos0, mass = _clusters(NDEV, PER)
    part = partition_for_devices(pos0, mass, NDEV, leaf_size=config.leaf_size)
    ev = make_force_evaluator(
        config,
        NDEV,
        part["cap"],
        make_mesh(NDEV),
        jit=True,
        halo_exchange=halo_exchange,
    )
    return ev, part, pos0, mass


def _forces_in_input_order(out, n):
    """Map the evaluator's (accel, gid, ...) back to input rows; padded rows have gid < 0."""
    acc = np.asarray(out[0], dtype=np.float64)
    gid = np.asarray(out[1]).reshape(-1)
    acc = acc.reshape(gid.shape[0], -1)
    valid = gid >= 0
    res = np.full((n, 3), np.nan)
    res[gid[valid]] = acc[valid]
    assert np.isfinite(res).all(), "an input row received no force"
    return res


def _run(halo_exchange):
    ev, part, pos0, mass = _evaluator_and_partition(halo_exchange)
    n = pos0.shape[0]
    # A leapfrog donates its state. Positions drift a little every step so the LET
    # contents change; the partition (gid/counts) stays frozen, as in a real rollout.
    run = jax.jit(lambda p, m, g, c: ev(p, m, g, c), donate_argnums=(0,))
    mass_d = jnp.asarray(part["mass_flat"])
    gid_d = jnp.asarray(part["gid_flat"])
    counts_d = jnp.asarray(part["counts"])
    perm = np.asarray(part["gid_flat"]).reshape(-1)
    rng = np.random.default_rng(1)
    drift = rng.normal(scale=0.02, size=pos0.shape)
    errs = []
    for step in range(STEPS):
        pos = pos0 + step * drift
        # rebuild the per-device flat positions from the frozen partition order
        flat = np.zeros_like(np.asarray(part["pos_flat"]))
        valid = perm >= 0
        flat[valid] = pos[perm[valid]]
        junk = jnp.ones(
            (int(rng.integers(1 << 12, 1 << 20)),)
        )  # allocator churn, as a real loop has
        out = run(jnp.asarray(flat), mass_d, gid_d, counts_d)
        acc = _forces_in_input_order(out, n)
        errs.append(_rel_l2(acc, _direct(pos, mass, SOFT)))
        del junk
    return errs


def test_forward_force_survives_donation_and_drift():
    """``halo_exchange="auto"`` must give the step-0 accuracy class on EVERY step."""
    errs = _run("auto")
    e0 = errs[0]
    assert e0 < 0.1, f"step 0 is not in an FMM accuracy class at all: {e0:.3e}"
    bad = [(i, e) for i, e in enumerate(errs) if e > 3 * e0 + 1e-6]
    assert not bad, (
        "the forward lost accuracy on later steps under donation -- the halo exchange "
        f"returned its fill value (step 0 {e0:.3e}; later {bad})"
    )


@pytest.mark.skipif(
    os.environ.get("JACCPOT_CHECK_UPSTREAM_RAGGED_FIX", "0") != "1",
    reason=(
        "opt-in: forces halo_exchange='native' in a donating loop. Expected to FAIL "
        "on jax < 0.9.1 (that is the measurement JAX_RAGGED_FIXED_VERSION encodes) "
        "and pass on >= 0.9.1. Needs GPUs -- there is no native path on CPU."
    ),
)
def test_native_forward_is_fixed_upstream():
    if jax.default_backend() == "cpu":
        pytest.skip("ragged_all_to_all has no XLA:CPU lowering")
    errs = _run("native")
    e0 = errs[0]
    bad = [(i, e) for i, e in enumerate(errs) if e > 3 * e0 + 1e-6]
    assert (
        not bad
    ), f"native ragged_all_to_all still corrupts the forward under donation: {bad}"
