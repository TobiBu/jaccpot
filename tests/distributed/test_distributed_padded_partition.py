"""The distributed force on a *padded* partition, pinned against the direct sum.

WHY THIS FILE EXISTS. ``partition_for_devices`` pads every device up to a common
capacity whenever its particle count is not a multiple of ``leaf_size`` -- which is
exactly what dividing a fixed N across an arbitrary device count produces. Until
2026-08-22 no test constructed such a configuration: every distributed IC in this
suite is ``ndev * per`` with ``per`` a leaf multiple, so ``cap == count`` and the
padding branch never ran. The gap was found from the outside, by
``docs/distributed_padding_force_defect.md`` reporting a 40% force error on every
padded point of a strong-scaling sweep.

WHAT THE FORCES ACTUALLY DO. They are correct, and the two tests below say so and
say why the sweep disagreed:

1. ``test_padded_partition_matches_direct`` -- the driver on a deliberately padded
   partition matches a direct sum to the same 1% the unpadded ICs are held to, with
   the cross-domain far field engaged. The padding is numerically inert.
2. ``test_padding_permutes_the_device_row_order`` -- the padded per-device row order
   is *not* the input row order, so reading the accelerations with the input
   ``gid_flat`` instead of the ``gid`` the evaluator returns produces a plausible,
   entirely wrong answer. That readout, not the force, is what the report measured.

THE MECHANISM behind (2). ``partition_for_devices`` places its padding rows at
``positions[chunk[0]]``: the device's *first* particle in Morton order, hence the
smallest Morton code on that device. The local tree build re-sorts by code, so the
padding lands at the front and displaces every real particle after the first by the
padding count. With ``cap == count`` there is no padding, the permutation is the
identity, and a caller that indexes the output by input row is right by accident --
right until the counts stop dividing, and silently wrong from then on.

    XLA_FLAGS=--xla_force_host_platform_device_count=3 \\
        pytest tests/distributed/test_distributed_padded_partition.py -o addopts="" -q
"""

import numpy as np
import pytest
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import (
    DistributedFMMConfig,
    distributed_fmm_accelerations,
    make_force_evaluator,
    partition_for_devices,
    scatter_to_input_order,
)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed FMM needs >= 2 devices"
)

# Particles per cluster. 57 is not a multiple of the default leaf_size of 8, which is
# the whole point: it pads each device by 64 - 57 = 7 rows. Do not round it up.
_PER = 57
_LEAF = DistributedFMMConfig().leaf_size


def _direct(all_pos, all_mass, G, soft):
    """Exact softened acceleration on every particle from every other.

    float64 throughout, so the reference is never the noisy side of the comparison:
    the IC is float32 and the FMM runs in it, but at order 3 its truncation error is
    ~1e-5, two orders above anything the reference contributes.
    """
    pos = np.asarray(all_pos, np.float64)
    mass = np.asarray(all_mass, np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = (diff**2).sum(-1) + np.float64(soft) ** 2
    inv = d2 ** (-1.5)
    np.fill_diagonal(inv, 0.0)
    return -np.float64(G) * (mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def _clusters(ndev, per, seed=4):
    """``ndev`` clusters of ``per`` particles, the driver tests' IC.

    The Morton split cuts across the clusters at ndev=2 and 3 and lands on them at
    ndev=4; see ``test_distributed_fmm_driver.py``'s module docstring. Either way the
    cross-domain path is engaged, which is all this file needs.
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


def _padding_rows(part):
    """Padding rows per device for a partition, i.e. ``cap - count``."""
    return int(part["cap"]) - np.asarray(part["counts"]).min()


@pytest.mark.parametrize(
    "drop",
    [
        pytest.param(0, id="equal-counts"),
        # N not divisible by ndev, so the counts differ per device on top of the
        # padding -- the shape of the report's own ndev=5 and ndev=6 points
        # ([6554, 6554, 6554, 6553, 6553] and so on), where cap is shared but
        # ``count`` is not. Padding and ragged counts are separate things and this
        # file has to cover both.
        pytest.param(1, id="ragged-counts"),
    ],
)
def test_padded_partition_matches_direct(drop):
    """A padded partition is as accurate as an unpadded one, not 40% wrong.

    Both arms run the same driver on the same kind of IC at the same ``ndev``; the
    only difference is whether ``per`` divides ``leaf_size``. The padded arm is
    asserted against the same 1% bar the rest of the distributed suite uses, and
    against the unpadded arm's own error, so a future padding defect cannot hide
    inside a bar that is loose for both.
    """
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    config = DistributedFMMConfig()

    padded_pts, padded_mass = _clusters(ndev, _PER)
    if drop:
        padded_pts, padded_mass = padded_pts[:-drop], padded_mass[:-drop]
    part = partition_for_devices(padded_pts, padded_mass, ndev, leaf_size=_LEAF)
    pad = _padding_rows(part)
    counts = np.asarray(part["counts"])
    assert pad > 0, (
        f"_PER={_PER} no longer pads at leaf_size={_LEAF} (cap={part['cap']}, "
        f"counts={counts.tolist()}). This file measures nothing "
        "until it does -- pick a per-device count that is not a leaf multiple."
    )
    assert (counts.min() != counts.max()) == bool(drop), (
        f"the {'ragged' if drop else 'equal'}-counts arm got counts "
        f"{counts.tolist()}, which is the other case"
    )

    padded = distributed_fmm_accelerations(
        padded_pts, padded_mass, config=config, mesh=mesh, jit=False
    )
    ref = np.asarray(_direct(padded_pts, padded_mass, config.G, config.softening))
    padded_err = float(
        np.linalg.norm(padded.accelerations - ref) / (np.linalg.norm(ref) + 1e-30)
    )

    # The control: the same IC rounded up to a leaf multiple, so cap == count.
    clean_pts, clean_mass = _clusters(ndev, 64)
    clean_part = partition_for_devices(clean_pts, clean_mass, ndev, leaf_size=_LEAF)
    assert _padding_rows(clean_part) == 0, "the control arm is padded too"
    clean = distributed_fmm_accelerations(
        clean_pts, clean_mass, config=config, mesh=mesh, jit=False
    )
    clean_ref = np.asarray(_direct(clean_pts, clean_mass, config.G, config.softening))
    clean_err = float(
        np.linalg.norm(clean.accelerations - clean_ref)
        / (np.linalg.norm(clean_ref) + 1e-30)
    )

    print(
        f"ndev={ndev} counts={counts.tolist()} cap={part['cap']} pad={pad}  "
        f"padded aggL2={padded_err:.6f}  unpadded aggL2={clean_err:.6f}"
    )
    assert not padded.overflow, "traversal buffers overflowed -- grow the caps"
    assert padded_err < 1e-2, f"padded aggL2 err {padded_err:.6f} exceeds 1%"
    # The padding is inert, so the two arms must sit in the same accuracy regime.
    # A padded arm an order of magnitude worse than its control is the signature of
    # a real padding defect even when both clear the 1% bar.
    assert padded_err < 10 * max(clean_err, 1e-6), (
        f"padded aggL2 {padded_err:.6f} is far worse than the unpadded control "
        f"{clean_err:.6f} -- the padding is no longer numerically inert"
    )
    far = float(np.asarray(padded.diagnostics["cross_far_pairs"]).sum())
    assert far > 0, (
        "no cross-domain far pairs were accepted, so the padded arm never exercised "
        "the coarse M2L and this assertion says nothing about it"
    )


def test_padding_permutes_the_device_row_order():
    """Padding displaces the per-device rows, so only the returned ``gid`` maps back.

    This is the defect of ``docs/distributed_padding_force_defect.md``, pinned as the
    property it actually is. ``make_force_evaluator`` returns ``(accel, gid, diag)``
    with ``accel`` in the per-device *tree* order; the input ``gid_flat`` is in the
    per-device *partition* order. Padding is what separates the two.
    """
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    config = DistributedFMMConfig()
    pts, mass = _clusters(ndev, _PER)
    part = partition_for_devices(pts, mass, ndev, leaf_size=_LEAF)
    assert _padding_rows(part) > 0, "this test needs a padded partition"

    evaluate = make_force_evaluator(config, ndev, part["cap"], mesh, jit=False)
    accel, gid_out, _diag = evaluate(
        part["pos_flat"], part["mass_flat"], part["gid_flat"], part["counts"]
    )
    accel = np.asarray(accel)
    gid_out = np.asarray(gid_out).reshape(-1)
    gid_in = np.asarray(part["gid_flat"]).reshape(-1)

    # The contract: the returned gid names every real particle exactly once.
    real = np.sort(gid_out[gid_out >= 0])
    assert np.array_equal(real, np.arange(part["n"])), (
        "the returned gid is not a permutation of the input particles: "
        f"{part['n']} expected, {real.size} returned"
    )

    # The premise. If this ever fails because the two orders now agree, the padded
    # layout changed (padding that sorts to the END of the Morton order would do it)
    # -- re-audit every caller that reassembles ``accel`` by hand before relaxing it,
    # because the identity they were relying on would have become true by accident a
    # second time rather than by contract.
    assert not np.array_equal(gid_out, gid_in), (
        "padding no longer permutes the per-device row order, so this file's premise "
        "is stale -- see the note above the assertion"
    )

    ref = np.asarray(_direct(pts, mass, config.G, config.softening))

    def rel_l2(scattered):
        """Relative L2 of a reassembled force against the direct sum."""
        return float(np.linalg.norm(scattered - ref) / (np.linalg.norm(ref) + 1e-30))

    # The supported readout, through the public helper.
    correct = rel_l2(scatter_to_input_order(accel, gid_out, part["n"]))
    # The hand-rolled one the harness had. Spelled out rather than routed through the
    # helper, because the helper is what makes it impossible to write by mistake.
    by_input = np.zeros_like(ref)
    rows_in = np.flatnonzero(gid_in >= 0)
    by_input[gid_in[rows_in]] = accel[rows_in]
    wrong = rel_l2(by_input)

    print(f"readout via returned gid={correct:.6f}  via input gid={wrong:.6f}")
    assert correct < 1e-2, f"the returned gid is not the right map: {correct:.6f}"
    assert wrong > 1e-1, (
        "reading the padded output with the INPUT gid no longer produces a wrong "
        f"answer ({wrong:.6f}) -- see the premise assertion above"
    )
