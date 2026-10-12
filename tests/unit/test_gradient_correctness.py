"""Fixed-topology gradient correctness for the differentiable FMM force.

Finite-difference vs autodiff of :meth:`FastMultipoleMethod.differentiable_accelerations`.
The finite-difference reference perturbs the SAME frozen-topology function (the same
prepared ``state``), so FD and AD must agree to round-off (this is the self-consistency
contract in ``docs/differentiable_fmm_design.md``). Also covers NaN/inf hygiene and the
``rho == 0`` rotation degeneracy.

The deep real-basis config (theta 0.6, N=64, order 4, leaf 4) is checked FD-vs-AD for
positions AND masses inside ``tests/integration/test_grad_fmm_vs_directsum.py``'s
direct-sum tests, on the reverse pass they already take; it was a second pair of tests
here until 2026-10-12. So was ``test_reorders_to_original_particle_order``, now the
elementwise half of ``test_differentiable_matches_forward_compute_accelerations`` there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    return positions, masses


def _num_far_pairs(state):
    """Number of M2L (multipole->local) interactions in the frozen topology.

    Zero means the tree is too shallow for the MAC to accept any box pair, so the
    far-field (M2M/M2L/L2L) reverse path is never traced and FD-vs-AD reduces to a
    near-field-only self-consistency check. The deep configs assert this is
    positive so they exercise the M2L reverse pass.
    """
    inter = state.interactions
    if inter is not None:
        return int(jnp.sum(inter.counts))
    dual = state.dual_tree_result
    if dual is not None:
        return int(jnp.sum(dual.far_pair_count))
    return 0


def _directional_fd_vs_ad(loss_fn, x, direction, eps=1e-6):
    grad = jax.grad(loss_fn)(x)
    ad = float(jnp.sum(grad * direction))
    fd = float(
        (loss_fn(x + eps * direction) - loss_fn(x - eps * direction)) / (2 * eps)
    )
    finite = bool(jnp.all(jnp.isfinite(grad)))
    rel = abs(fd - ad) / (abs(fd) + 1e-300)
    return finite, rel, fd, ad


# ``leaf``/``min_far`` make the far field explicit. The shallow, low-order config is
# a cheap near-field-only point. The deep real-basis config (theta 0.6, order 4, leaf
# 4: a non-empty M2L list, the rotation-angle M2L path) is FD-vs-AD-checked for
# positions and masses in test_grad_fmm_vs_directsum.py's direct-sum tests, which
# already take that reverse pass (see the module docstring). FD-vs-AD is a
# self-consistency check, so the 1e-4 tolerance is config-independent (it holds to
# ~1e-9 either way). The ``_bound_diff_fmm_compile_cache`` fixture (tests/conftest.py)
# drops JAX's compile cache after each of these tests.
@pytest.mark.parametrize(
    "basis,theta,n,order,leaf,min_far",
    [
        ("real", 0.4, 64, 2, 16, 0),  # shallow, low order: near-field only
    ],
)
def test_fd_vs_ad_positions(basis, theta, n, order, leaf, min_far):
    positions, masses = _system(n, seed=int(10 * theta) + n + order)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)
    n_far = _num_far_pairs(state)
    assert n_far >= min_far, f"expected >= {min_far} M2L pairs, got {n_far}"

    def loss(pos):
        return jnp.sum(fmm.differentiable_accelerations(state, pos, masses) ** 2)

    direction = jax.random.normal(
        jax.random.PRNGKey(0), positions.shape, dtype=jnp.float64
    )
    finite, rel, fd, ad = _directional_fd_vs_ad(loss, positions, direction)
    assert finite, "position gradient contains non-finite entries"
    assert rel < 1e-4, f"FD-vs-AD rel-err {rel:.3e} (fd={fd:.3e}, ad={ad:.3e})"


@pytest.mark.parametrize("basis", ["real"])
@pytest.mark.parametrize("softening", [1e-2, 0.0])
def test_no_nan_near_coincident(softening, basis):
    """Near-coincident particles must not produce NaN/inf gradients."""
    n = 48
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(n, 3))
    pos[1] = pos[0] + 1e-9  # near-coincident
    positions = jnp.asarray(pos, dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=0.6, G=1.0, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=8)

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite grad with softening={softening}"


@pytest.mark.parametrize("basis", ["real"])
def test_no_nan_axis_aligned_grid(basis):
    """A regular lattice keeps gradients finite.

    NOTE: this does NOT exercise the rho == 0 rotation-angle guard, despite what
    the name suggests. Centres are COM rather than geometric box centres, so the
    lattice's leaf COMs are generically off-axis with respect to each other --
    measured, zero of this config's 22 M2L pairs have rho == 0 and the smallest
    rho^2 is 5.64. It is still worth keeping as a NaN-hygiene check on a highly
    symmetric configuration; the guard itself is covered by
    :func:`test_fd_vs_ad_along_a_transverse_direction_at_rho_zero`.
    """
    side = 5
    coords = (
        np.stack(
            np.meshgrid(
                np.arange(side), np.arange(side), np.arange(side), indexing="ij"
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float64)
    )
    positions = jnp.asarray(coords, dtype=jnp.float64)
    masses = jnp.ones((positions.shape[0],), dtype=jnp.float64)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=0.7, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=8)

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "non-finite grad on axis-aligned lattice"


# ---------------------------------------------------------------------------
# The rho == 0 degeneracy (docs/refactor_audit_2026-08.md G.10)
# ---------------------------------------------------------------------------


def _z_stacked_system(num_per_cluster=8, num_clusters=4, z_spacing=6.0, seed=3):
    """Clusters stacked along z that share one ``(x, y)`` point set and masses.

    Every cluster therefore has the *same* centre-of-mass ``(x, y)``, so the M2L
    displacements between them have ``rho == 0`` exactly rather than
    approximately. A uniform lattice does **not** do this -- centres are COM, not
    geometric box centres, so lattice leaf COMs are generically off-axis relative
    to each other (see the WARNING in
    :func:`jaccpot.operators.complex_ops.m2l_complex_reference`).
    """
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(num_per_cluster, 2)) * 0.35
    intra_masses = rng.uniform(0.6, 1.4, size=num_per_cluster)
    offsets = [(i - (num_clusters - 1) / 2) * z_spacing for i in range(num_clusters)]
    positions = np.concatenate(
        [
            np.column_stack([xy[:, 0], xy[:, 1], np.full(num_per_cluster, dz)])
            for dz in offsets
        ]
    )
    masses = np.tile(intra_masses, num_clusters)
    return (
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.asarray(masses, dtype=jnp.float64),
    )


def _count_degenerate_m2l_pairs(state):
    """How many accepted M2L displacements have ``rho == 0`` exactly."""
    interactions = state.interactions
    centers = np.asarray(state.upward.multipoles.centers)
    valid = int(np.sum(np.asarray(interactions.counts)))
    targets = np.asarray(interactions.targets)[:valid]
    sources = np.asarray(interactions.sources)[:valid]
    delta = centers[targets] - centers[sources]
    rho_sq = delta[:, 0] ** 2 + delta[:, 1] ** 2
    return int(np.sum(rho_sq == 0.0)), valid


@pytest.mark.parametrize("basis", ["real"])
def test_fd_vs_ad_along_a_transverse_direction_at_rho_zero(basis):
    """FD and AD must agree when M2L displacements are exactly z-aligned.

    This is the user-facing statement of G.10: the operator-level defect is easy to
    dismiss as an isolated helper's subgradient choice, so what matters is whether it
    survives into ``grad`` of the force. It did, and this is what closes it.

    Two things make this test bite where the existing coverage does not. The
    perturbation direction is **purely transverse** (x and y only), because the radial
    component is unaffected and would dilute the comparison; and both the cotangent and
    the direction are **asymmetric random**, because a symmetric construction lets the
    per-pair errors cancel in the sum -- which is why an earlier measurement recorded in
    ``m2l_complex_reference``'s comment concluded, on a symmetric system, that nothing
    was lost.

    It also checks the loss is SMOOTH through the degeneracy, with one-sided
    derivatives. ``_angles_from_delta_solidfmm`` returns a zero cotangent for the
    azimuth at ``rho == 0``, on the argument that the azimuth is undefined there and
    the rotation degenerates to a pure polar turn. A central difference alone cannot
    confirm that: at a symmetric kink it returns the mean of the two slopes whether or
    not the guard is right. So the one-sided slopes must agree with each other (no
    kink), and AD must equal their mean -- which is the central difference. That was
    ``test_degenerate_azimuth_subgradient_is_correct``, on its own z-stacked system,
    until 2026-10-12.

    It took two fixes, and the second is why this construction is worth keeping rather
    than simplifying. Supplying the ``rho == 0`` derivative analytically
    (:mod:`jaccpot.operators._transverse_degeneracy_jvp`) moved it from 1.9e-03 to
    9.5e-06 (real) and 1.8e-05 to 9.7e-06 (complex) -- and no further, because this
    system *also* contains an L2L displacement of ``(5.551e-17, 0.0, +3.0)``: two nodes
    whose ``(x, y)`` centres are mathematically equal but summed in different orders, so
    ``rho_sq = 3.1e-33`` is strictly positive and the alignment guards never fire.
    Widening that module's band to the measured ``rho/r <= sqrt(eps)`` crossover covers
    it. Both bases now agree with the finite difference to 2.7e-10 at ``step=1e-6`` and
    4.9e-11 at ``1e-7``, and -- the tell that one shared cause is gone -- they produce
    the *same* AD value to every digit printed.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("FD-vs-AD at this tolerance requires float64")

    positions, masses = _z_stacked_system()
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=0.5, G=1.0, softening=1e-3
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=4)

    degenerate, total = _count_degenerate_m2l_pairs(state)
    assert total > 0, "no M2L pairs: the far field is not exercised at all"
    assert degenerate > 0, (
        f"none of the {total} M2L pairs has rho == 0, so the degeneracy guard never "
        "fires and this test would pass without testing anything"
    )

    rng = np.random.default_rng(11)
    cotangent = jnp.asarray(rng.normal(size=positions.shape), dtype=jnp.float64)

    def loss(pos):
        return jnp.sum(cotangent * fmm.differentiable_accelerations(state, pos, masses))

    direction = np.zeros(positions.shape)
    direction[:, 0] = rng.normal(size=positions.shape[0])
    direction[:, 1] = rng.normal(size=positions.shape[0])
    direction /= np.linalg.norm(direction)
    direction_j = jnp.asarray(direction, dtype=jnp.float64)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "non-finite grad at the rho == 0 degeneracy"
    ad = float(jnp.sum(grad * direction_j))
    step = 1.0e-6
    f0 = float(loss(positions))
    f_plus = float(loss(positions + step * direction_j))
    f_minus = float(loss(positions - step * direction_j))
    d_plus = (f_plus - f0) / step
    d_minus = (f0 - f_minus) / step
    fd = (f_plus - f_minus) / (2 * step)  # the central difference

    # Smooth, not a kink: the one-sided derivatives agree rather than cancelling.
    slope_scale = max(abs(d_plus), abs(d_minus), 1e-300)
    assert abs(d_plus - d_minus) / slope_scale < 1e-3, (
        f"transverse derivative is not smooth through rho == 0 "
        f"(d+={d_plus:.6e}, d-={d_minus:.6e}); the zeroed azimuth cotangent may be "
        "dropping a real term"
    )
    # 1e-6 is comfortably above central-difference truncation at this step (the
    # measured FD value is stable to ~1e-9 between h=1e-6 and h=1e-7) and far below
    # the 1.9e-03 / 1.8e-05 discrepancies the defect produces.
    rel = abs(fd - ad) / max(abs(fd), 1e-300)
    assert rel < 1.0e-6, (
        f"{basis} basis: FD {fd:.10e} vs AD {ad:.10e} disagree by {rel:.3e} with "
        f"{degenerate}/{total} M2L pairs at rho == 0"
    )
