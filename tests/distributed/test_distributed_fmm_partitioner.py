"""``DistributedFMMConfig.partitioner`` has to reach the decomposition.

WHY THIS FILE EXISTS. #212 added an RCB partitioner alongside the Morton one;
#215 flipped the default to RCB, on the strength of a win in every geometry
measured, and argued -- correctly -- that flipping a decomposition default
deserves a deliberate step *because* it moves the domain assignment of every
distributed run and therefore every accuracy and timing baseline taken with one.
But ``distributed_fmm_accelerations`` called the partitioner with no argument and
``DistributedFMMConfig`` carried no field, so the one lane the flip actually moved
was the one lane with no way to move it back.

The failure mode is silence: an ignored ``partitioner="morton"`` returns a
perfectly good force computed from the RCB decomposition, and the caller sees a
number that simply is not the baseline they asked for.

WHY THIS IC. ``_separated_clusters`` at ``ndev=2`` is the geometry the two
partitioners disagree about most sharply, and
``test_distributed_fmm_driver.py``'s docstring already records why: the global
box is 7x1x1, so the Morton code's leading bits are not the axis that separates
the clusters, and a contiguous split of the Morton order cuts straight across
them. RCB splits on the long axis and gets one cluster per device. Measured here
on two forced CPU devices, 64 particles per device:

    partitioner   cross_far  cross_near  self_far  self_near   rel L2 vs direct
    morton               18         106        14         94              < 1e-5
    rcb                   2           0         0        112              < 1e-5

which is the whole point in one table: the same physics off two decompositions
that could hardly be less alike.

The assertions follow from it, and each is load-bearing:

* the far field must be ENGAGED under both, or nothing here says anything about
  the FMM. Several tests in this repo once passed while accepting no far pairs at
  all, and an earlier draft of this file used a disk IC that did exactly that --
  ``cross_far_pairs == self_far_pairs == 0``, a green test over a pure direct
  near-field sum;
* both must agree with a direct sum, so "they agree with each other" cannot be
  satisfied by two equally wrong answers; and
* the two must DISAGREE on the decomposition, which is the vacuity guard: with
  the field ignored, every other assertion here passes unchanged.

Run with two devices, e.g. forced CPU ones::

    XLA_FLAGS="--xla_force_host_platform_device_count=2" JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_distributed_fmm_partitioner.py -o addopts="" -q
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from yggdrax.distributed import device_count

from jaccpot.distributed import (
    DistributedFMMConfig,
    distributed_fmm_accelerations,
)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed FMM needs >= 2 devices"
)

_NDEV = 2
_PER_DEVICE = 64
_SOFT = 0.02
_G = 1.0


def _separated_clusters(ndev, per, seed=4):
    """``ndev`` spatially separated clusters in a long box.

    The same IC as ``test_distributed_fmm_driver.py``, and for the same reason:
    it is the configuration in which the Morton split lands across the clusters
    rather than between them.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    pts = np.concatenate(
        [centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return pts, mass


def _direct(pos, mass):
    diff = pos[:, None, :] - pos[None, :, :]
    inv = ((diff**2).sum(-1) + _SOFT**2) ** (-1.5)
    return -_G * (mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def _run(pos, mass, partitioner):
    config = dataclasses.replace(
        DistributedFMMConfig(softening=_SOFT, G=_G, leaf_size=8, order=4, theta=0.4),
        partitioner=partitioner,
    )
    return distributed_fmm_accelerations(
        pos, mass, config=config, ndev=_NDEV, auto_scale_caps=True
    )


def _diag(result, field):
    return int(np.asarray(result.diagnostics[field]).sum())


@pytest.fixture(scope="module")
def runs():
    """One IC decomposed both ways. Module-scoped: each run is a full shard_map."""
    pos, mass = _separated_clusters(_NDEV, _PER_DEVICE)
    return pos, mass, _run(pos, mass, "morton"), _run(pos, mass, "rcb")


@pytest.mark.parametrize("which", [2, 3], ids=["morton", "rcb"])
def test_the_far_field_is_engaged_under_both_partitioners(runs, which):
    """Without this the accuracy assertion below would say nothing about the FMM."""
    result = runs[which]
    assert not bool(result.diagnostics["overflow"])
    assert _diag(result, "cross_far_pairs") > 0, (
        "no cross-domain far pairs were accepted, so the coarse M2L is not "
        "exercised and this configuration proves nothing about the far field"
    )


@pytest.mark.parametrize("which", [2, 3], ids=["morton", "rcb"])
def test_both_partitioners_agree_with_a_direct_sum(runs, which):
    """A decomposition is a decomposition: neither choice changes the physics."""
    pos, mass = runs[0], runs[1]
    accel = np.asarray(runs[which].accelerations)
    reference = _direct(pos, mass)
    rel = float(np.linalg.norm(accel - reference) / np.linalg.norm(reference))
    assert rel < 1e-3, f"relative L2 vs direct sum {rel:.2e}"


def test_the_partitioner_field_moves_the_decomposition(runs):
    """The vacuity guard, stated as the property RCB exists to have.

    Off-device near pairs are what a partitioner is judged on, and on this
    geometry RCB drives them to zero while Morton leaves 106 of them. Asserting
    the ordering rather than mere inequality means a field that arrives but is
    misrouted -- ``"rcb"`` reaching the partitioner as ``"morton"`` -- fails here
    too.
    """
    _, _, morton, rcb = runs
    assert _diag(rcb, "cross_near_pairs") < _diag(morton, "cross_near_pairs")
