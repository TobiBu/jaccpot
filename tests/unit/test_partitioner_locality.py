"""The partitioner's job is to keep near neighbours on the same device.

Measured as the fraction of each particle's ``k`` nearest neighbours that live on
another device -- the quantity the near field has to import and the cross walk has to
pair up. Host-side and device-free, so it runs in ordinary CI rather than only under a
multi-device harness.

Why this file exists. A Morton code's three most significant bits are ``(z, y, x)``, so
its very first cut bisects all three axes at once, including the thin one. For a disk
centred in its box that cut slices through the thickness, separating particles 0.1 apart
while keeping particles 20 apart together -- and no choice of bounds fixes it, because
rescaling the box does not move the disk off its own centre. A galaxy disk is exactly
this geometry, which is what makes it worth a test rather than a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yggdrax")

from jaccpot.distributed.fmm import partition_for_devices

_K = 16


def _owner(pos, mass, ndev, leaf, partitioner):
    part = partition_for_devices(
        pos, mass, ndev, leaf_size=leaf, partitioner=partitioner
    )
    cap = part["cap"]
    gid = np.asarray(part["gid_flat"])
    owner = np.full(len(pos), -1, np.int64)
    for d in range(ndev):
        g = gid[d * cap : (d + 1) * cap]
        owner[g[g >= 0]] = d
    assert (owner >= 0).all(), "a particle was not assigned to any device"
    return owner, part["counts"]


def _foreign_fraction(pos, owner, k=_K):
    """Fraction of each particle's ``k`` nearest neighbours owned by another device."""
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    return float((owner[nn] != owner[:, None]).mean())


def _disk(n=1024, radius=10.0, thickness=0.2, seed=9):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack(
        [r * np.cos(th), r * np.sin(th), rng.normal(scale=thickness, size=n)], axis=1
    ).astype(np.float32)
    return pos, rng.uniform(0.8, 1.2, n).astype(np.float32)


@pytest.mark.parametrize("ndev", [2, 4])
def test_rcb_keeps_a_disks_neighbours_together_and_morton_does_not(ndev):
    """The case that motivated the option, with both halves asserted.

    Both directions matter. That RCB is compact is the claim; that Morton is NOT is
    what stops this passing for free if the geometry or the encoder changes such that
    the comparison stops meaning anything.
    """
    pos, mass = _disk()
    m_owner, _ = _owner(pos, mass, ndev, 16, "morton")
    r_owner, _ = _owner(pos, mass, ndev, 16, "rcb")
    m_frac = _foreign_fraction(pos, m_owner)
    r_frac = _foreign_fraction(pos, r_owner)

    assert m_frac > 0.25, (
        f"Morton put only {m_frac:.3f} of near neighbours off-device on a flattened "
        "system, so the comparison below no longer demonstrates anything -- re-derive "
        "it before trusting this file"
    )
    assert r_frac < m_frac / 3.0, (
        f"RCB {r_frac:.3f} vs Morton {m_frac:.3f}: expected at least 3x fewer "
        "off-device neighbours on a disk"
    )


def test_rcb_balances_the_devices():
    """Imbalance is wasted capacity on EVERY device, not just the biggest one.

    The lane pads each device up to the largest count, so a lopsided split inflates
    every per-device buffer. RCB places its cut by device share rather than at the
    plain median precisely so this holds for non-powers-of-two too.
    """
    pos, mass = _disk(n=1000)
    for ndev in (2, 3, 4, 8):
        _, counts = _owner(pos, mass, ndev, 16, "rcb")
        assert (
            max(counts) - min(counts) <= 1
        ), f"ndev={ndev}: counts {list(counts)} differ by more than one particle"


def test_rcb_is_not_worse_on_an_isotropic_system():
    """The flattened case is where it pays, but it must not cost anything elsewhere.

    A partitioner that fixed disks by ruining balls would be a bad trade, and the
    default for the mutual lane rests on this not being one.
    """
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(1024, 3)).astype(np.float32)
    mass = rng.uniform(0.8, 1.2, 1024).astype(np.float32)
    m_owner, _ = _owner(pos, mass, 4, 16, "morton")
    r_owner, _ = _owner(pos, mass, 4, 16, "rcb")
    m_frac = _foreign_fraction(pos, m_owner)
    r_frac = _foreign_fraction(pos, r_owner)
    assert (
        r_frac <= m_frac * 1.05
    ), f"RCB {r_frac:.3f} is worse than Morton {m_frac:.3f} on an isotropic system"


def test_an_unknown_partitioner_is_refused():
    """Named rather than silently falling back to a default."""
    pos, mass = _disk(n=64)
    with pytest.raises(ValueError, match="must be 'morton' or 'rcb'"):
        partition_for_devices(pos, mass, 2, leaf_size=16, partitioner="hilbert")
