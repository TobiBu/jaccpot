"""Golden-output characterization oracle for the FMM engine.

This is the load-bearing safety net for the ``_fmm_impl`` refactor. It drives the
public :class:`jaccpot.FastMultipoleMethod` across a representative grid
(basis x order x distribution x N) and snapshots the outputs to committed
``.npz`` files under ``tests/characterization/golden/``.

On every run it re-computes and asserts:

1. **Inertness gate** -- outputs match the committed golden to float64
   round-off (``rtol=1e-12, atol=1e-12``). Mechanical code moves (Phase 2/3 of
   the refactor) MUST NOT change these numbers at all. Kernel consolidation
   (Phase 4) may relax this to a *documented* tolerance in a dedicated change.
2. **Physics anchor** -- each FMM acceleration agrees with an O(N^2) direct sum
   to a loose relative-L2 bound, so a regenerated golden can never silently
   snapshot garbage.

Regenerate goldens intentionally with ``JACCPOT_REGEN_GOLDEN=1 pytest ...`` (do
this only when a numerical change is expected and reviewed, then commit the new
``.npz`` files).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import DualTreeTraversalConfig  # noqa: E402

from jaccpot import (  # noqa: E402
    FastMultipoleMethod,
    FMMAdvancedConfig,
    RuntimePolicyConfig,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

G_CONST = 1.0
SOFTENING = 1.0e-2

# Inertness tolerance: golden vs recompute must agree to float64 round-off.
INERT_RTOL = 1.0e-12
INERT_ATOL = 1.0e-12
# Physics anchor: FMM vs direct sum. Loose -- only guards against garbage
# goldens, not precision (the golden match is the precision gate).
ANCHOR_REL_L2 = 0.35

# (id, distribution, N, basis, order). Kept modest but covers the axes the
# refactor touches: real / complex(solidfmm) / cartesian bases, orders 2/4/6,
# uniform + clustered geometry, and a larger-N case.
CASES = [
    ("uni_solidfmm_n256_p2", "uniform", 256, "solidfmm", 2),
    ("uni_solidfmm_n256_p4", "uniform", 256, "solidfmm", 4),
    ("uni_real_n256_p2", "uniform", 256, "real", 2),
    ("uni_real_n256_p4", "uniform", 256, "real", 4),
    ("uni_cartesian_n256_p2", "uniform", 256, "cartesian", 2),
    ("uni_cartesian_n256_p4", "uniform", 256, "cartesian", 4),
    ("clu_solidfmm_n256_p4", "clustered", 256, "solidfmm", 4),
    ("clu_real_n256_p4", "clustered", 256, "real", 4),
    ("clu_cartesian_n256_p4", "clustered", 256, "cartesian", 4),
    ("uni_solidfmm_n256_p6", "uniform", 256, "solidfmm", 6),
    ("uni_real_n256_p6", "uniform", 256, "real", 6),
    ("uni_solidfmm_n1024_p4", "uniform", 1024, "solidfmm", 4),
    ("uni_real_n1024_p4", "uniform", 1024, "real", 4),
]


def _make_inputs(distribution: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic positions in a bounded box + positive masses."""
    key = jax.random.PRNGKey(0xC0FFEE)
    k_pos, k_mass, k_blob = jax.random.split(key, 3)
    dtype = jnp.float64
    if distribution == "uniform":
        positions = jax.random.uniform(
            k_pos, (n, 3), dtype=dtype, minval=-1.0, maxval=1.0
        )
    elif distribution == "clustered":
        # Mixture of a few tight Gaussian blobs (exercises adaptive/MAC paths).
        n_blobs = 4
        centers = jax.random.uniform(
            k_blob, (n_blobs, 3), dtype=dtype, minval=-0.8, maxval=0.8
        )
        assign = jax.random.randint(k_pos, (n,), 0, n_blobs)
        jitter = 0.08 * jax.random.normal(k_mass, (n, 3), dtype=dtype)
        positions = centers[assign] + jitter
        positions = jnp.clip(positions, -1.0, 1.0)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown distribution {distribution!r}")
    masses = jnp.abs(jax.random.normal(k_mass, (n,), dtype=dtype)) + 0.5
    return np.asarray(positions), np.asarray(masses)


def _direct_sum_accelerations(
    positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Reference O(N^2) accelerations (self-interaction removed)."""
    n = int(positions.shape[0])
    out = np.zeros_like(positions)
    eps = np.finfo(positions.dtype).eps
    soft_sq = SOFTENING * SOFTENING
    for i in range(n):
        delta = positions[i] - positions
        dist_sq = np.sum(delta * delta, axis=1) + soft_sq
        dist = np.sqrt(dist_sq)
        inv_dist3 = 1.0 / (dist_sq * dist + eps)
        inv_dist3[i] = 0.0
        out[i] = -G_CONST * np.sum(
            (masses[:, None] * inv_dist3[:, None]) * delta, axis=0
        )
    return out


def _build_fmm(basis: str) -> FastMultipoleMethod:
    # Generous traversal caps so small clustered systems never truncate lists
    # (truncation would make the golden distribution-fragile).
    return FastMultipoleMethod(
        preset="accurate",
        basis=basis,
        theta=0.5,
        G=G_CONST,
        softening=SOFTENING,
        advanced=FMMAdvancedConfig(
            runtime=RuntimePolicyConfig(
                traversal_config=DualTreeTraversalConfig(
                    max_pair_queue=1 << 18,
                    process_block=512,
                    max_interactions_per_node=1 << 16,
                    max_neighbors_per_leaf=1 << 16,
                )
            )
        ),
    )


def _compute(basis: str, order: int, positions: np.ndarray, masses: np.ndarray):
    fmm = _build_fmm(basis)
    accel = fmm.compute_accelerations(
        jnp.asarray(positions),
        jnp.asarray(masses),
        leaf_size=8,
        max_order=order,
    )
    return np.asarray(accel, dtype=np.float64)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="golden characterization requires float64 (JAX_ENABLE_X64=1)",
)
@pytest.mark.parametrize(
    ("case_id", "distribution", "n", "basis", "order"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fmm_golden(
    case_id: str, distribution: str, n: int, basis: str, order: int
) -> None:
    positions, masses = _make_inputs(distribution, n)
    accel = _compute(basis, order, positions, masses)

    # Physics anchor: never trust a golden that is grossly wrong.
    ref = _direct_sum_accelerations(positions, masses)
    rel_l2 = np.linalg.norm(accel - ref) / (np.linalg.norm(ref) + 1e-12)
    assert rel_l2 < ANCHOR_REL_L2, (
        f"{case_id}: FMM vs direct-sum rel-L2 {rel_l2:.3e} exceeds "
        f"anchor {ANCHOR_REL_L2}"
    )

    path = GOLDEN_DIR / f"{case_id}.npz"
    if REGEN or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, accel=accel)
        if not REGEN:
            pytest.skip(f"generated missing golden {path.name} (commit it)")
        return

    golden = np.load(path)["accel"]
    np.testing.assert_allclose(
        accel, golden, rtol=INERT_RTOL, atol=INERT_ATOL,
        err_msg=f"{case_id}: output drifted from committed golden",
    )
