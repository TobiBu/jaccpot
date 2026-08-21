"""Is the distributed cross-domain far field correct, or only inaccurate?

Written to settle that question for the five failures that appeared the first
time ``tests/distributed/`` was ever executed on two cards (2026-08-21). It
sweeps ``theta_cross`` -- the only knob that moves work between the cross-domain
*far* path and the cross-domain *near* path -- and compares each result against
a **baseline that decides the question on its own**: the exact magnitude of the
whole cross-domain contribution, obtained in float64 by masking same-domain
pairs out of a direct sum.

That baseline is the point. It is the error you would get by deleting the cross
far field entirely, and **an approximation cannot be worse than omission**. So

* ``aggL2 < baseline`` -- the far field is approximating something, and any test
  failure is a tolerance question.
* ``aggL2 > baseline`` -- the far field is *contributing error larger than the
  term it is approximating*, which no choice of expansion order or MAC can
  produce. That is a bug.

MEASURED 2026-08-21, 2xA100, ndev=2, per=64 (N=128), caps grown so
``overflow=False`` on every row, ``order=3 theta=0.4 leaf=8 basis=real``:

===========  ==========  ====================
theta_cross  aggL2       cross far / near
===========  ==========  ====================
1e6          10.090450   2 / 0
1.0          0.260355    42 / 53
0.1          0.018223    22 / 102
0.01         0.000003    0 / 128
0.001        0.000003    0 / 128
===========  ==========  ====================

with baseline ``||a_cross|| / ||a_full|| = 0.008814``. The near limit is exact
(3e-6 is float32 round-off at N=128), so decomposition, halo import, cross P2P
and reassembly are all correct and the whole error sits in the far pairs. At the
default ``theta_cross=0.1`` the error is 2.07x the baseline -- and 4.50x with
``mac_type="bh"`` (0.039704), which merely accepts more pairs as far and so
exposes more of the same path. Note ``2 x 0.008814 = 0.017628`` is within 3% of
the measured 0.018223, which is what a sign error or a double-count looks like;
that is a lead, not a finding.

Do not read the ``theta_cross=1e6`` row as an accuracy reference:
``docs/north_star_phase3_scale_plan.md:133`` records that an under-separated
far field "goes garbage", and 10.09 is that warning reproduced.

Usage (org rule: pick the cards with ``autocvd`` before jax is imported)::

    CVD=$(.venv/bin/autocvd -n 2 -l -o -q | tail -1)
    CUDA_VISIBLE_DEVICES=$CVD JAX_ENABLE_X64=1 \\
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.30 \\
      python -m bench.diagnose_cross_domain_far

Each row rebuilds and recompiles (~50-250 s, see
``docs/north_star_phase3_scale_plan.md:130``), so a five-row sweep is minutes,
not seconds, and a quiet console is compilation rather than a hang.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

_DEFAULT_THETA_CROSS = (1e6, 1.0, 0.1, 0.01, 0.001)

# Grown well past the defaults (512 / 128 / 32768) so that a buffer overflow can
# never be mistaken for a numerical result. Verified not to change the answer:
# the theta_cross=0.1 row reproduces the failing test's 0.018223 exactly.
_ROOMY_CAPS = {
    "max_interactions_per_node": 4096,
    "max_neighbors_per_leaf": 4096,
    "max_pair_queue": 131072,
    "cross_max_interactions_per_node": 4096,
    "cross_max_neighbors_per_leaf": 4096,
    "cross_max_pair_queue": 131072,
}


def separated_clusters(
    ndev: int, per: int, seed: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Build one compact cluster per Morton domain.

    Copied deliberately from ``tests/distributed/test_distributed_fmm_driver.py``
    rather than imported: this must reproduce the failing tests' initial
    condition exactly, and a shared helper that later drifts would quietly stop
    reproducing them.

    Parameters
    ----------
    ndev : int
        Number of devices, hence clusters. At most 4.
    per : int
        Particles per cluster.
    seed : int, optional
        Seed for the position and mass draws, by default 4 (the tests' value).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Positions ``(ndev * per, 3)`` and masses ``(ndev * per,)``, both float32
        as the tests build them.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    positions = np.concatenate(
        [centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    masses = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return positions, masses


def direct_sum(
    positions: np.ndarray,
    masses: np.ndarray,
    mask: np.ndarray,
    softening: float,
    newton_g: float,
) -> np.ndarray:
    """Direct-sum accelerations over the pairs selected by ``mask``.

    Evaluated in float64 regardless of the input dtype, so the baseline this
    feeds is not itself limited by float32 round-off.

    Parameters
    ----------
    positions : numpy.ndarray
        Positions, shape ``(n, 3)``.
    masses : numpy.ndarray
        Masses, shape ``(n,)``.
    mask : numpy.ndarray
        Pair weights, shape ``(n, n)``. Ones include a pair, zeros drop it.
    softening : float
        Plummer softening length.
    newton_g : float
        Gravitational constant.

    Returns
    -------
    numpy.ndarray
        Accelerations, shape ``(n, 3)``, float64.
    """
    pos = positions.astype(np.float64)
    mass = masses.astype(np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    inv = ((diff**2).sum(-1) + softening**2) ** -1.5
    weighted = mass[None, :, None] * diff * inv[..., None] * mask[:, :, None]
    return -newton_g * weighted.sum(axis=1)


def omission_baseline(
    positions: np.ndarray,
    masses: np.ndarray,
    ndev: int,
    per: int,
    softening: float,
    newton_g: float,
) -> float:
    """Relative error you would get by dropping the cross-domain term entirely.

    This is the number that turns the sweep into a verdict rather than a table.
    Because the accelerations are linear in the pair contributions, masking
    same-domain pairs gives the cross-domain term exactly.

    Parameters
    ----------
    positions : numpy.ndarray
        Positions, shape ``(ndev * per, 3)``.
    masses : numpy.ndarray
        Masses, shape ``(ndev * per,)``.
    ndev : int
        Number of domains.
    per : int
        Particles per domain, in domain-contiguous order.
    softening : float
        Plummer softening length.
    newton_g : float
        Gravitational constant.

    Returns
    -------
    float
        ``||a_cross|| / ||a_full||``.
    """
    ones = np.ones((ndev * per, ndev * per), dtype=np.float64)
    domain = np.repeat(np.arange(ndev), per)
    same_domain = (domain[:, None] == domain[None, :]).astype(np.float64)
    a_full = direct_sum(positions, masses, ones, softening, newton_g)
    a_self = direct_sum(positions, masses, same_domain, softening, newton_g)
    return float(np.linalg.norm(a_full - a_self) / (np.linalg.norm(a_full) + 1e-30))


def main() -> int:
    """Run the sweep and report a verdict against the omission baseline.

    Returns
    -------
    int
        ``0`` if every row's error is below the omission baseline, ``1`` if any
        row exceeds it -- i.e. non-zero means the cross-domain far field is
        contributing more error than the term it approximates.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per", type=int, default=64, help="particles per cluster")
    parser.add_argument("--ndev", type=int, default=0, help="devices; 0 = all, max 4")
    parser.add_argument("--mac", default="dehnen", help="mac_type: dehnen or bh")
    parser.add_argument("--basis", default="real", help="basis: real or solidfmm")
    parser.add_argument(
        "--theta-cross",
        type=float,
        action="append",
        default=None,
        help="theta_cross value (repeatable); defaults to the documented sweep",
    )
    parser.add_argument(
        "--default-caps",
        action="store_true",
        help="use the config's own caps instead of the roomy ones (overflow may confound)",
    )
    args = parser.parse_args()

    # Imported here, after argument parsing, so --help works without a GPU and
    # without paying jax's import cost.
    import jax
    from yggdrax.distributed import device_count, make_mesh

    from jaccpot.distributed import (
        DistributedFMMConfig,
        distributed_fmm_accelerations,
    )

    ndev = min(4, device_count()) if args.ndev == 0 else args.ndev
    if ndev < 2:
        print(f"FAIL: need >= 2 devices, jax reports {device_count()}")
        return 2
    print(f"devices: {jax.devices()}")

    base = dataclasses.replace(
        DistributedFMMConfig(), mac_type=args.mac, basis=args.basis
    )
    caps = {} if args.default_caps else _ROOMY_CAPS
    positions, masses = separated_clusters(ndev, args.per)
    mesh = make_mesh(ndev)
    print(
        f"ndev={ndev} per={args.per} N={ndev * args.per} "
        f"mac={args.mac} basis={args.basis} order={base.order} "
        f"theta={base.theta} leaf={base.leaf_size}"
    )

    ones = np.ones((ndev * args.per,) * 2, dtype=np.float64)
    reference = direct_sum(positions, masses, ones, base.softening, base.G)
    ref_norm = np.linalg.norm(reference)
    baseline = omission_baseline(
        positions, masses, ndev, args.per, base.softening, base.G
    )
    print(
        f"\nomission baseline ||a_cross||/||a_full|| = {baseline:.6f}"
        "  <-- the error from DROPPING the cross term; nothing may exceed it"
    )

    theta_values = args.theta_cross or list(_DEFAULT_THETA_CROSS)
    print(
        f"\n{'theta_cross':>12} {'aggL2':>12} {'vs baseline':>12} {'ovf':>5}  far/near"
    )
    worst = 0.0
    for theta_cross in theta_values:
        config = dataclasses.replace(base, theta_cross=float(theta_cross), **caps)
        result = distributed_fmm_accelerations(
            positions, masses, config=config, mesh=mesh, jit=False
        )
        err = float(
            np.linalg.norm(np.asarray(result.accelerations) - reference)
            / (ref_norm + 1e-30)
        )
        worst = max(worst, err / baseline if baseline else float("inf"))
        far = np.asarray(result.diagnostics["cross_far_pairs"]).sum()
        near = np.asarray(result.diagnostics["cross_near_pairs"]).sum()
        print(
            f"{theta_cross:>12g} {err:>12.6f} {err / baseline:>11.2f}x "
            f"{str(bool(result.overflow)):>5}  {far:.0f} / {near:.0f}",
            flush=True,
        )

    print()
    if worst > 1.0:
        print(
            f"VERDICT: BUG. Worst row is {worst:.2f}x the omission baseline. The "
            "cross-domain far field contributes more error than the term it "
            "approximates, which no expansion order or MAC can do."
        )
        return 1
    print(
        f"VERDICT: consistent with approximation. Worst row is {worst:.2f}x the "
        "omission baseline, i.e. the far field is better than dropping it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
