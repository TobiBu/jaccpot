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


def _disk(
    n=1024,
    radius=10.0,
    thickness=0.2,
    seed=9,
    bulge_fraction=0.0,
    bulge_radius=1.0,
):
    """A thin exponential-ish disc, optionally with a compact central bulge.

    ``bulge_fraction`` of the particles are drawn uniformly inside a sphere of
    ``bulge_radius`` at the origin; the rest form the disc. A bulge is the one
    component every galaxy IC has, and it is the component that concentrates the
    near field somewhere the disc's flatness argument does not apply.
    """
    rng = np.random.default_rng(seed)
    n_bulge = int(round(bulge_fraction * n))
    n_disk = n - n_bulge
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n_disk))
    th = rng.uniform(0.0, 2.0 * np.pi, n_disk)
    pos = np.stack(
        [
            r * np.cos(th),
            r * np.sin(th),
            rng.normal(scale=thickness, size=n_disk),
        ],
        axis=1,
    )
    if n_bulge:
        direction = rng.normal(size=(n_bulge, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        # r ~ U^(1/3) fills the sphere uniformly rather than piling up at the centre
        radii = bulge_radius * rng.uniform(0.0, 1.0, n_bulge) ** (1.0 / 3.0)
        pos = np.concatenate([pos, direction * radii[:, None]])
    return pos.astype(np.float32), rng.uniform(0.8, 1.2, n).astype(np.float32)


def _foreign_ratio(ndev, seed, **disk_kwargs):
    """Morton's foreign-neighbour fraction divided by RCB's. Above 1 favours RCB."""
    pos, mass = _disk(seed=seed, **disk_kwargs)
    m_owner, _ = _owner(pos, mass, ndev, 16, "morton")
    r_owner, _ = _owner(pos, mass, ndev, 16, "rcb")
    m_frac = _foreign_fraction(pos, m_owner)
    r_frac = _foreign_fraction(pos, r_owner)
    return m_frac, r_frac, (m_frac / r_frac if r_frac > 0 else np.inf)


#: Seeds every locality claim in this file is averaged over. A single draw is not
#: a measurement: the audit this file came from once read one seed in the low tail
#: of a bulged geometry as "a bulge inverts the choice" and produced a wrong
#: recommendation from it. Asserting on the median over a stated number of seeds
#: is what makes that mistake impossible to repeat here.
_SEEDS = tuple(range(16))


@pytest.mark.parametrize("ndev", [2, 4])
def test_rcb_keeps_a_disks_neighbours_together_and_morton_does_not(ndev):
    """The case that motivated the option, with both halves asserted.

    Both directions matter. That RCB is compact is the claim; that Morton is NOT is
    what stops this passing for free if the geometry or the encoder changes such that
    the comparison stops meaning anything.

    Measured over the 16 seeds in :data:`_SEEDS`, ratio Morton/RCB:

        ndev 2   median 11.86   range 9.06-16.93   RCB worse in 0/16
        ndev 4   median  7.26   range 6.75- 8.16   RCB worse in 0/16
    """
    m_fracs, r_fracs, ratios = zip(*(_foreign_ratio(ndev, seed) for seed in _SEEDS))
    m_med, r_med = float(np.median(m_fracs)), float(np.median(r_fracs))
    ratio_med = float(np.median(ratios))

    assert m_med > 0.25, (
        f"Morton put only {m_med:.3f} of near neighbours off-device on a flattened "
        f"system (median over {len(_SEEDS)} seeds), so the comparison below no "
        "longer demonstrates anything -- re-derive it before trusting this file"
    )
    assert ratio_med > 3.0, (
        f"median over {len(_SEEDS)} seeds: RCB {r_med:.3f} vs Morton {m_med:.3f} "
        f"= {ratio_med:.2f}x; expected at least 3x fewer off-device neighbours"
    )


@pytest.mark.parametrize("ndev", [2, 4])
def test_rcb_still_wins_once_the_disc_has_a_bulge(ndev):
    """A bulge is the one component every galaxy IC has, so cover it explicitly.

    The flatness argument in the module docstring is about the DISC: a Morton cut
    bisects the thin axis along with the wide ones. A bulge is the part of a real
    IC that argument says nothing about, and it concentrates the near field into a
    round region where Morton has no particular handicap. If a bulge inverted the
    choice, the default would be wrong for every galaxy run.

    It does not. Measured over the 16 seeds in :data:`_SEEDS`, 10% of particles in
    a bulge of radius 1 inside a disc of radius 10, ratio Morton/RCB:

        ndev 2   median 8.44   range 6.84-9.87   RCB worse in 0/16
        ndev 4   median 5.13   range 4.74-5.93   RCB worse in 0/16

    against 11.86 and 7.26 for the same disc with no bulge. So a 10% bulge narrows
    RCB's margin by roughly a third and comes nowhere near reversing it.

    A NOTE ON PROVENANCE, because this contradicts the number this test was
    commissioned from. `docs/plan_2026-08_D_signals.md` reports a 10% bulge taking
    the ratio to ~1.5x with a per-seed range of 0.19-2.95 and RCB worse in 2/16.
    That does not reproduce with this construction at any bulge fraction or radius
    swept (0.1-0.5 x 0.3-3.0); the ratio only reaches ~1.7 at a 90% bulge, which is
    a round system rather than a bulged disc -- and a round system is separately
    reported there as 1.2-1.4x, which is consistent with what is measured here. The
    threshold below is therefore set from THIS measurement, not from that table.
    """
    m_fracs, r_fracs, ratios = zip(
        *(
            _foreign_ratio(ndev, seed, bulge_fraction=0.10, bulge_radius=1.0)
            for seed in _SEEDS
        )
    )
    ratio_med = float(np.median(ratios))
    worse = sum(1 for r in ratios if r < 1.0)

    assert ratio_med > 2.0, (
        f"median over {len(_SEEDS)} seeds: Morton/RCB = {ratio_med:.2f}x on a "
        f"bulged disc (RCB {np.median(r_fracs):.3f} vs Morton "
        f"{np.median(m_fracs):.3f}); a bulge should narrow RCB's margin, not "
        "reverse it"
    )
    assert worse <= len(_SEEDS) // 4, (
        f"RCB was worse in {worse}/{len(_SEEDS)} seeds, which is more scatter than "
        "this geometry showed when the threshold was set (0/16) -- re-measure the "
        "distribution before adjusting anything"
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
