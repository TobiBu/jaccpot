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
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

G_CONST = 1.0
SOFTENING = 1.0e-2

# Inertness tolerance: golden vs recompute must agree to float64 round-off.
INERT_RTOL = 1.0e-12
INERT_ATOL = 1.0e-12
# Physics anchor: FMM vs direct sum. The golden match is the precision gate; this
# only has to catch a golden that is grossly wrong.
#
# A single loose bound did not do that. At 0.35 it accepted an M2M defect that
# dropped up to 23% of the system mass from the far field -- `clu_solidfmm_n256_p4`
# sat at 2.5e-2 and passed, where the correct value is 3.5e-4. The bound was loose
# because it had to accommodate the *cartesian* basis, which is ~1.8e-1 here at
# both p=2 and p=4. Order-independent error is the signature of a divergent series
# rather than truncation, and solidfmm at the same configuration is 8.1e-5, so
# cartesian has a defect of its own that is out of scope here.
#
# Per-basis bounds instead, so the accurate bases get a bound that can actually
# fail and the loose one is confined to the basis that needs it.
ANCHOR_REL_L2 = 0.35
ANCHOR_REL_L2_BY_BASIS = {
    # Worst observed across the grid is 7.2e-4 (uniform, p=2); 1e-2 keeps an order
    # of magnitude of headroom while still catching a mass-loss-scale regression.
    "solidfmm": 1.0e-2,
    "real": 1.0e-2,
    # Pre-existing, unexplained, and tracked separately -- see above.
    "cartesian": 0.35,
}

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


# Second grid, for the axes the grid above does not cover: far-field execution mode,
# near-field execution mode, and the potential output. Kept separate so the 13 committed
# `.npz` above are untouched -- regenerating a golden is a deliberate act, not a side
# effect of widening coverage.
#
# `(id, distribution, N, basis, order, farfield_mode, nearfield_mode, anchor)`.
# `farfield_mode` needs `basis="solidfmm"`: enabling grouped interactions selects AABB
# expansion centres, and the native real upward sweep accepts COM centres only, so a
# real-basis grouped case raises rather than running.
#
# `pair_grouped` was deliberately ABSENT when this grid was written: its error was
# order-independent at 1.253e-02 / 1.246e-02 / 1.245e-02 (orders 2 / 4 / 6 uniform) and
# 3.96e-02 clustered, so a golden would have needed a ~4e-2 anchor -- which would have
# encoded a defect as acceptable. G.11 in `docs/refactor_audit_2026-08.md` identified
# that gap as exactly that: `pair_grouped` gathered its class rotation blocks with
# `GroupedInteractionBuffers.class_ids`, which yggdrax stores in the original pair order
# while `class_sources` / `class_targets` are sorted by class, so ~70% of pairs were
# rotated by another class's stencil. With that fixed the mode lands on `class_major` --
# 8.927e-04 / 2.049e-04 / 1.887e-04 uniform, 3.478e-03 clustered -- and it is now
# goldened on the same two cases and the same anchors as `class_major`.
MODE_CASES = [
    (
        "cm_uni_solidfmm_n256_p4",
        "uniform",
        256,
        "solidfmm",
        4,
        "class_major",
        None,
        1.0e-3,
    ),
    (
        "cm_clu_solidfmm_n256_p4",
        "clustered",
        256,
        "solidfmm",
        4,
        "class_major",
        None,
        1.0e-2,
    ),
    (
        "pg_uni_solidfmm_n256_p4",
        "uniform",
        256,
        "solidfmm",
        4,
        "pair_grouped",
        None,
        1.0e-3,
    ),
    (
        "pg_clu_solidfmm_n256_p4",
        "clustered",
        256,
        "solidfmm",
        4,
        "pair_grouped",
        None,
        1.0e-2,
    ),
    ("bkt_uni_real_n256_p4", "uniform", 256, "real", 4, None, "bucketed", 1.0e-2),
    (
        "bkt_uni_solidfmm_n256_p4",
        "uniform",
        256,
        "solidfmm",
        4,
        None,
        "bucketed",
        1.0e-2,
    ),
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


def _direct_sum_accelerations(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
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
    anchor = ANCHOR_REL_L2_BY_BASIS.get(basis, ANCHOR_REL_L2)
    assert rel_l2 < anchor, (
        f"{case_id}: FMM vs direct-sum rel-L2 {rel_l2:.3e} exceeds "
        f"anchor {anchor} for basis {basis!r}"
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
        accel,
        golden,
        rtol=INERT_RTOL,
        atol=INERT_ATOL,
        err_msg=f"{case_id}: output drifted from committed golden",
    )


MODE_GOLDEN_DIR = Path(__file__).parent / "golden_modes"


def _build_fmm_with_modes(basis, farfield_mode, nearfield_mode):
    """The same solver as :func:`_build_fmm`, with execution-mode overrides applied."""
    farfield = (
        FarFieldConfig(mode=farfield_mode, grouped_interactions=True)
        if farfield_mode is not None
        else FarFieldConfig()
    )
    nearfield = (
        NearFieldConfig(mode=nearfield_mode)
        if nearfield_mode is not None
        else NearFieldConfig()
    )
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
            ),
            farfield=farfield,
            nearfield=nearfield,
        ),
    )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="golden characterization requires float64 (JAX_ENABLE_X64=1)",
)
@pytest.mark.parametrize(
    (
        "case_id",
        "distribution",
        "n",
        "basis",
        "order",
        "farfield_mode",
        "nearfield_mode",
        "anchor",
    ),
    MODE_CASES,
    ids=[c[0] for c in MODE_CASES],
)
def test_fmm_golden_execution_modes(
    case_id, distribution, n, basis, order, farfield_mode, nearfield_mode, anchor
):
    """Golden coverage for the execution-mode axes, and for the potential output.

    ARCHITECTURE section 9 used to describe the forward oracle as covering "farfield
    modes, outputs"; it covered neither -- the grid above is
    (distribution, N, basis, order), accelerations only, one preset. These are the
    missing axes, and they matter for the refactor: splitting the M2L seam in
    `runtime/kernels/core.py` touches the grouped and class-major accumulators
    directly, and until now no golden constrained them.

    Both outputs are snapshotted, so this is also the first golden on the potential.

    The anchors are per case rather than global because the modes do not have the same
    accuracy: `class_major` is ~2.5x looser than the default at order 4 (2.05e-04 vs
    8.15e-05 uniform), while `bucketed` agrees with the default near field to
    round-off (3.4e-13), being the same edge set in a different order.
    """
    positions, masses = _make_inputs(distribution, n)
    fmm = _build_fmm_with_modes(basis, farfield_mode, nearfield_mode)
    accel, potential = fmm.compute_accelerations(
        jnp.asarray(positions),
        jnp.asarray(masses),
        leaf_size=8,
        max_order=order,
        return_potential=True,
    )
    accel = np.asarray(accel, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)

    assert np.all(np.isfinite(accel))
    assert np.all(np.isfinite(potential))

    reference = _direct_sum_accelerations(positions, masses)
    rel_l2 = np.linalg.norm(accel - reference) / (np.linalg.norm(reference) + 1e-12)
    assert (
        rel_l2 < anchor
    ), f"{case_id}: FMM vs direct-sum rel-L2 {rel_l2:.3e} exceeds anchor {anchor}"

    path = MODE_GOLDEN_DIR / f"{case_id}.npz"
    if REGEN or not path.exists():
        MODE_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, accel=accel, potential=potential)
        if not REGEN:
            pytest.skip(f"generated missing golden {path.name} (commit it)")
        return

    golden = np.load(path)
    for label, got in (("accel", accel), ("potential", potential)):
        np.testing.assert_allclose(
            got,
            golden[label],
            rtol=INERT_RTOL,
            atol=INERT_ATOL,
            err_msg=f"{case_id}: {label} drifted from the committed golden",
        )


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="requires float64 (JAX_ENABLE_X64=1)",
)
@pytest.mark.parametrize("mode", ["pair_grouped", "class_major"])
def test_grouped_farfield_plateaus_in_order(mode):
    """Both grouped far-field modes stop converging in ``p``, at the same level.

    Recorded as a test rather than a comment because it is the kind of fact that
    silently changes. Measured relative L2 versus a direct sum, uniform N=256:

        order       default    pair_grouped   class_major
        2         7.230e-04       8.927e-04     8.927e-04
        4         8.148e-05       2.049e-04     2.049e-04
        6         1.128e-05       1.887e-04     1.887e-04

    The default converges as expected; both grouped modes are flat from order 4.
    Order-independent error is the signature of a fixed geometric approximation rather
    than expansion truncation, which is exactly what a class-cached scheme does: it
    rotates by one representative lattice displacement per class instead of by each
    pair's own direction. That residual is inherent to the grouping, and it is the
    reason ``FarFieldConfig.mode`` documents these as an accuracy trade.

    The two modes used to differ by ~60x here (``pair_grouped`` flat at 1.25e-02).
    That was G.11 in ``docs/refactor_audit_2026-08.md`` -- a defect, not the trade:
    ``pair_grouped`` gathered its rotation blocks with ``class_ids``, which yggdrax
    stores in the original rather than the class-sorted pair order. Fixed; the two
    modes now agree, so they share a ceiling here.

    This test asserts only the plateau, not that it is acceptable: the error at order 6
    must not be materially better than at order 4 (which would mean the plateau went
    away and this test is obsolete), and must not be worse than the measured value
    (which would be a regression).
    """
    positions, masses = _make_inputs("uniform", 256)
    reference = _direct_sum_accelerations(positions, masses)

    errors = {}
    for order in (4, 6):
        fmm = _build_fmm_with_modes("solidfmm", mode, None)
        accel = np.asarray(
            fmm.compute_accelerations(
                jnp.asarray(positions),
                jnp.asarray(masses),
                leaf_size=8,
                max_order=order,
            ),
            dtype=np.float64,
        )
        errors[order] = float(
            np.linalg.norm(accel - reference) / np.linalg.norm(reference)
        )

    ceiling = 3.0e-4
    assert errors[6] < ceiling, f"{mode} order-6 error {errors[6]:.3e} regressed"
    # The plateau itself: order 6 buys less than a factor of 2 over order 4.
    assert errors[6] > 0.5 * errors[4], (
        f"{mode} now converges in order (4: {errors[4]:.3e}, 6: {errors[6]:.3e}) -- "
        "the plateau this test records has gone away, so retire it and update G.11"
    )
