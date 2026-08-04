"""Fixed-topology gradient correctness for the differentiable FMM force.

Finite-difference vs autodiff of :meth:`FastMultipoleMethod.differentiable_accelerations`
swept over theta (MAC tightness), N, and expansion order, for gradients w.r.t. both
positions and masses. The finite-difference reference perturbs the SAME frozen-topology
function (the same prepared ``state``), so FD and AD must agree to round-off (this is the
self-consistency contract in ``docs/differentiable_fmm_design.md``). Also covers NaN/inf
hygiene and the public-surface guards.
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


# ``leaf``/``min_far`` make the far field explicit: the deep configs (small leaf)
# build a non-empty M2L list so FD-vs-AD self-consistency actually covers the
# multipole->local reverse pass; the real-basis deep config covers the
# rotation-angle M2L path this PR guards. The first (shallow, low-order) config
# keeps a cheap near-field-only point. FD-vs-AD is a self-consistency check, so
# the 1e-4 tolerance is config-independent (it holds to ~1e-9 either way) -- adding
# the far field does not loosen it. The ``_bound_diff_fmm_compile_cache`` fixture
# (tests/conftest.py) drops JAX's compile cache after each of these tests, so the
# deep-config memory plateaus (~1.5 GB) rather than accumulating -- CI-safe.
@pytest.mark.parametrize(
    "basis,theta,n,order,leaf,min_far",
    [
        ("complex", 0.4, 64, 2, 16, 0),  # shallow, low order: near-field only
        ("complex", 0.6, 64, 4, 4, 1),  # deep: non-empty M2L reverse path
        ("real", 0.6, 64, 4, 4, 1),  # deep real basis: rotation-angle M2L
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


@pytest.mark.parametrize("basis", ["complex", "real"])
def test_fd_vs_ad_masses(basis):
    # Deep tree (leaf_size=4, theta=0.6) so the mass gradient's FD-vs-AD
    # self-consistency covers the far-field (M2L) reverse path, not just P2P.
    n, theta, order, leaf = 64, 0.6, 4, 4
    positions, masses = _system(n, seed=7)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)
    n_far = _num_far_pairs(state)
    assert n_far > 0, f"config must exercise the far field (got {n_far} M2L pairs)"

    def loss(m):
        return jnp.sum(fmm.differentiable_accelerations(state, positions, m) ** 2)

    direction = jax.random.normal(
        jax.random.PRNGKey(1), masses.shape, dtype=jnp.float64
    )
    finite, rel, fd, ad = _directional_fd_vs_ad(loss, masses, direction)
    assert finite, "mass gradient contains non-finite entries"
    assert rel < 1e-4, f"FD-vs-AD rel-err {rel:.3e} (fd={fd:.3e}, ad={ad:.3e})"


@pytest.mark.parametrize("softening", [1e-2, 0.0])
def test_no_nan_near_coincident(softening):
    """Near-coincident particles must not produce NaN/inf gradients."""
    n = 48
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(n, 3))
    pos[1] = pos[0] + 1e-9  # near-coincident
    positions = jnp.asarray(pos, dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.6, G=1.0, softening=softening
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=8)

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), f"non-finite grad with softening={softening}"


def _z_stacked_clusters(n_clusters=4, per_cluster=8, dz=4.0, spread=0.10, seed=3):
    """Particles whose leaf COM displacements are EXACTLY z-aligned (rho == 0).

    Every cluster reuses one ``(x, y)`` point set and differs only in ``z``, and the
    masses are equal, so each leaf's centre of mass shares the same ``(x, y)``. The
    resulting M2L centre displacements are therefore ``(0, 0, dz)`` exactly, which
    is what makes ``_angles_from_delta_solidfmm``'s ``rho == 0`` branch fire.

    This construction exists because the obvious one does not work: a uniform
    lattice does NOT produce rho == 0 displacements, since centres are COM rather
    than geometric box centres and lattice leaf COMs come out generically off-axis.
    Measured on the 5^3 grid in :func:`test_no_nan_axis_aligned_grid`, zero of its
    22 M2L pairs have rho == 0 and the smallest rho^2 is 5.64.
    """
    rng = np.random.default_rng(seed)
    base_xy = rng.normal(scale=spread, size=(per_cluster, 2))
    pos = np.vstack(
        [np.c_[base_xy, np.full(per_cluster, k * dz)] for k in range(n_clusters)]
    )
    positions = jnp.asarray(pos, dtype=jnp.float64)
    masses = jnp.ones((pos.shape[0],), dtype=jnp.float64)
    return positions, masses


def _num_rho_zero_m2l_pairs(state):
    """How many frozen M2L pairs have an exactly z-aligned centre displacement."""
    centers = np.asarray(state.upward.multipoles.centers)
    src = np.asarray(state.interactions.sources).reshape(-1)
    tgt = np.asarray(state.interactions.targets).reshape(-1)
    if src.size == 0:
        return 0
    delta = centers[tgt] - centers[src]
    return int(np.sum(delta[:, 0] ** 2 + delta[:, 1] ** 2 == 0.0))


def test_degenerate_azimuth_subgradient_is_correct():
    """At rho == 0 the zeroed azimuth cotangent must not drop a real term.

    ``_angles_from_delta_solidfmm`` returns a zero cotangent for the azimuth when
    ``rho == 0``, on the argument that the azimuth is undefined there and the
    rotation degenerates to a pure polar turn. That was an argument from
    construction; this asserts it numerically.

    A central difference cannot do the job: at a symmetric kink it returns 0, which
    is exactly what the guard returns, so it would agree whether or not the guard is
    right. This uses ONE-SIDED derivatives along a purely transverse (azimuth-plane)
    direction and checks that they converge to the same value from both sides -- i.e.
    the loss is smooth through the degeneracy -- and that AD equals it.

    Measured agreement is ~1e-9 relative; the 1e-5 bound below is set by the
    one-sided FD truncation error at this step size, not by the gradient.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("one-sided FD comparison requires float64")

    positions, masses = _z_stacked_clusters()
    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.7, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=8)

    n_rho_zero = _num_rho_zero_m2l_pairs(state)
    assert n_rho_zero > 0, (
        "config must actually hit the rho == 0 guard, otherwise this test is "
        f"vacuous (got {n_rho_zero} z-aligned M2L pairs)"
    )

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "non-finite grad at the rho == 0 degeneracy"

    # Purely transverse: perturb x and y only. That is the plane whose cotangent
    # the guard zeroes, so a dropped term would show up here and nowhere else.
    rng = np.random.default_rng(23)
    d = np.zeros_like(np.asarray(positions))
    d[:, 0] = rng.normal(size=d.shape[0])
    d[:, 1] = rng.normal(size=d.shape[0])
    d /= np.linalg.norm(d)
    direction = jnp.asarray(d, dtype=jnp.float64)

    ad = float(jnp.sum(grad * direction))
    h = 1e-6
    f0 = float(loss(positions))
    d_plus = (float(loss(positions + h * direction)) - f0) / h
    d_minus = (f0 - float(loss(positions - h * direction))) / h

    scale = max(abs(d_plus), abs(d_minus), 1e-300)
    # Smooth, not a kink: the one-sided derivatives agree rather than cancelling.
    assert abs(d_plus - d_minus) / scale < 1e-3, (
        f"transverse derivative is not smooth through rho == 0 "
        f"(d+={d_plus:.6e}, d-={d_minus:.6e}); the zeroed azimuth cotangent may be "
        "dropping a real term"
    )
    two_sided = 0.5 * (d_plus + d_minus)
    rel = abs(ad - two_sided) / max(abs(two_sided), 1e-300)
    assert rel < 1e-5, (
        f"AD disagrees with the transverse finite difference at rho == 0: "
        f"ad={ad:.10e}, fd={two_sided:.10e}, rel={rel:.3e}"
    )


def test_no_nan_axis_aligned_grid():
    """A regular lattice keeps gradients finite.

    NOTE: this does NOT exercise the rho == 0 rotation-angle guard, despite what
    the name suggests. Centres are COM rather than geometric box centres, so the
    lattice's leaf COMs are generically off-axis with respect to each other --
    measured, zero of this config's 22 M2L pairs have rho == 0 and the smallest
    rho^2 is 5.64. It is still worth keeping as a NaN-hygiene check on a highly
    symmetric configuration; the guard itself is covered by
    :func:`test_degenerate_azimuth_subgradient_is_correct`.
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
        basis="complex", use_pallas=False, theta=0.7, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=8)

    def loss(p):
        return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

    grad = jax.grad(loss)(positions)
    assert jnp.all(jnp.isfinite(grad)), "non-finite grad on axis-aligned lattice"


def test_rejects_non_solidfmm_basis():
    positions, masses = _system(32, seed=3)
    fmm = FastMultipoleMethod(
        basis="cartesian", use_pallas=False, theta=0.6, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=2, leaf_size=16)
    with pytest.raises(NotImplementedError, match="solidfmm"):
        fmm.differentiable_accelerations(state, positions, masses)


def test_reorders_to_original_particle_order():
    """Output is aligned with the original (unsorted) input order."""
    positions, masses = _system(80, seed=9)
    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.4, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=4, leaf_size=16)
    a = fmm.differentiable_accelerations(state, positions, masses)
    a_fwd = fmm.compute_accelerations(positions, masses, max_order=4, theta=0.4)
    assert a.shape == positions.shape
    assert jnp.allclose(a, a_fwd, rtol=1e-6, atol=1e-6)
