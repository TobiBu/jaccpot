"""Reference tests pinning the Dehnen (2014) mass-dependent MAC to the paper.

Dehnen (2014) "A fast multipole method for stellar dynamics", arXiv:1405.2255,
section 5. These tests pin the implementation against independently written
numpy float64 references rather than against the paper's prose, so the claim
"jaccpot implements Dehnen's mass-dependent multipole acceptance criterion" is
mechanically checkable:

- eq (12)  ``P_n^2 = sum_m (n-m)!(n+m)! |M_n^m|^2``           source power
- eq (13)  ``E = sum_k C(p,k) P_k rho_s^(p-k) / (r^p M_A)``   error estimate
- eq (15)  ``E~ = 8 max(rho_z,rho_s)/(rho_z+rho_s) * E``      improved estimate
- eq (16a) ``theta < 1 and E~ M_A / r^2 < eps min_b a_b``     the criterion

The eq (12) test is the load-bearing one: for a single point mass at distance
``d`` the exact source power is ``P_n = m d^n`` for every degree, independent of
direction and of the packed representation. That single identity pins the
factorial weights, rotation invariance, and basis independence at once.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.tree import Tree

from jaccpot.operators.complex_harmonics import p2m_complex
from jaccpot.operators.real_harmonics import (
    complex_to_dehnen_real_coeffs,
    p2m_real_direct,
)
from jaccpot.runtime._adaptive_policy import (
    _ERROR_MODEL_DEHNEN_PAPER,
    adaptive_pair_policy,
    compute_smallest_enclosing_sphere_geometry,
    compute_tree_merged_sphere_geometry,
    dehnen_multipole_power_by_degree,
    dehnen_paper_pair_error_by_order,
)

ORDER = 4
MASS = 1.7

# Offsets spanning generic directions plus the degenerate cases that a
# rotation-invariance bug hides behind: pure-axis (all but one m vanish) and
# near-origin (rho -> 0).
OFFSETS = [
    (0.31, -0.72, 0.45),
    (-1.20, 0.05, 0.88),
    (0.60, 0.60, 0.60),
    (2.50, 0.00, 0.00),
    (0.00, 1.30, 0.00),
    (0.00, 0.00, -0.90),
    (1e-4, 0.0, 0.0),
    (-0.02, 0.03, -0.01),
]


def _packed_real(delta: tuple[float, float, float], mass: float) -> jnp.ndarray:
    packed = p2m_real_direct(
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
        order=ORDER,
    )
    return jnp.asarray(packed)[None, :]


def _packed_complex(delta: tuple[float, float, float], mass: float) -> jnp.ndarray:
    packed = p2m_complex(
        jnp.asarray(delta, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
        order=ORDER,
    )
    return jnp.asarray(packed)[None, :]


# ---------------------------------------------------------------------------
# eq (12): source power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", OFFSETS)
@pytest.mark.parametrize("basis", ["real", "complex"])
def test_dehnen_power_of_point_mass_equals_mass_times_distance_pow_n(basis, delta):
    """``P_n == m |d|^n`` for a single point mass, in either packed basis.

    This is the exact closed form of eq (12) for a one-particle cell and it is
    rotation invariant, so any error in the factorial weights -- including one
    that only affects ``m != 0`` terms, and therefore only shows up away from
    the coordinate axes -- breaks it.
    """

    packed = (
        _packed_real(delta, MASS) if basis == "real" else _packed_complex(delta, MASS)
    )
    power = np.asarray(dehnen_multipole_power_by_degree(multipole_packed=packed))[0]

    distance = float(np.linalg.norm(np.asarray(delta)))
    expected = np.asarray([MASS * distance**n for n in range(ORDER + 1)])

    np.testing.assert_allclose(power, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("delta", OFFSETS)
def test_dehnen_power_is_basis_invariant(delta):
    """eq (12) is a rotation invariant, so it cannot depend on the packing."""

    complex_packed = _packed_complex(delta, MASS)
    real_packed = jnp.asarray(
        complex_to_dehnen_real_coeffs(complex_packed[0], order=ORDER)
    )[None, :]

    from_complex = np.asarray(
        dehnen_multipole_power_by_degree(multipole_packed=complex_packed)
    )
    from_real = np.asarray(
        dehnen_multipole_power_by_degree(multipole_packed=real_packed)
    )

    np.testing.assert_allclose(from_real, from_complex, rtol=1e-12, atol=1e-13)


def test_dehnen_power_matches_numpy_weight_reference():
    """Explicit numpy transcription of eq (12) for a multi-particle real node."""

    rng = np.random.default_rng(20260731)
    positions = rng.normal(size=(6, 3))
    masses = rng.uniform(0.5, 2.0, size=6)

    packed = np.zeros(((ORDER + 1) ** 2,))
    for pos, mass in zip(positions, masses):
        packed += np.asarray(
            p2m_real_direct(
                jnp.asarray(pos, dtype=jnp.float64),
                jnp.asarray(mass, dtype=jnp.float64),
                order=ORDER,
            )
        )

    expected = []
    for n in range(ORDER + 1):
        total = 0.0
        for m in range(-n, n + 1):
            # Dehnen sums over the complex M_n^m for m = -n..n. The real packing
            # stores (Re M_n^m, Im M_n^m) at (+|m|, -|m|), so each |m| != 0
            # magnitude is spread over two real slots and contributes twice.
            weight = math.factorial(n - abs(m)) * math.factorial(n + abs(m))
            if m != 0:
                weight *= 2.0
            total += weight * packed[n * n + n + m] ** 2
        expected.append(math.sqrt(total))

    got = np.asarray(
        dehnen_multipole_power_by_degree(
            multipole_packed=jnp.asarray(packed, dtype=jnp.float64)[None, :]
        )
    )[0]

    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# eqs (13) + (15): the pair error estimate
# ---------------------------------------------------------------------------


def _reference_eq13_eq15(
    *,
    power: np.ndarray,
    mass: float,
    rho_z: float,
    rho_s: float,
    distance: float,
    orders: tuple[int, ...],
) -> np.ndarray:
    """Hand transcription of eqs (13) and (15) for one pair."""

    out = []
    for p_val in orders:
        total = 0.0
        for k in range(min(p_val, power.shape[0] - 1) + 1):
            total += math.comb(p_val, k) * power[k] * rho_s ** (p_val - k)
        basic = total / (mass * distance**p_val)
        improvement = 8.0 * max(rho_z, rho_s) / (rho_z + rho_s)
        out.append(improvement * basic)
    return np.asarray(out)


@pytest.mark.parametrize("orders", [(4,), (2, 3, 4)])
def test_dehnen_paper_pair_error_matches_numpy_eq13_eq15(orders):
    """The estimator equals an independent numpy transcription of (13)+(15)."""

    rng = np.random.default_rng(7)
    power = rng.uniform(0.1, 2.0, size=ORDER + 1)
    mass, rho_z, rho_s, distance = 1.3, 0.42, 0.31, 2.75

    degree_idx = np.arange(ORDER + 1)
    order_arr = np.asarray(orders)
    binomial = np.zeros((len(orders), ORDER + 1))
    for row, p_val in enumerate(orders):
        for k in range(min(p_val, ORDER) + 1):
            binomial[row, k] = float(math.comb(p_val, k))
    masked_binomial = binomial * (degree_idx[None, :] <= order_arr[:, None])
    exponent = np.maximum(order_arr[:, None] - degree_idx[None, :], 0)

    got = np.asarray(
        dehnen_paper_pair_error_by_order(
            source_power=jnp.asarray(power, dtype=jnp.float64)[None, :],
            source_mass=jnp.asarray([mass], dtype=jnp.float64),
            source_radius=jnp.asarray([rho_z], dtype=jnp.float64),
            target_radius=jnp.asarray([rho_s], dtype=jnp.float64),
            distance=jnp.asarray([distance], dtype=jnp.float64),
            order_values_float=jnp.asarray(order_arr, dtype=jnp.float64),
            masked_binomial_by_order=jnp.asarray(masked_binomial, dtype=jnp.float64),
            exponent_by_order=jnp.asarray(exponent, dtype=jnp.int32),
        )
    )[0]

    expected = _reference_eq13_eq15(
        power=power,
        mass=mass,
        rho_z=rho_z,
        rho_s=rho_s,
        distance=distance,
        orders=orders,
    )

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# eq (16a): the acceptance criterion
# ---------------------------------------------------------------------------


def _paper_state(
    *,
    power: np.ndarray,
    masses: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    thresholds: np.ndarray,
    order: int = ORDER,
):
    from jaccpot.runtime._adaptive_policy import AdaptivePolicyState

    degree_idx = np.arange(power.shape[1])
    binomial = np.zeros((1, power.shape[1]))
    for k in range(min(order, power.shape[1] - 1) + 1):
        binomial[0, k] = float(math.comb(order, k))
    order_arr = np.asarray([order])
    masked_binomial = binomial * (degree_idx[None, :] <= order_arr[:, None])
    exponent = np.maximum(order_arr[:, None] - degree_idx[None, :], 0)
    n = power.shape[0]

    return AdaptivePolicyState(
        source_error_proxy_by_order=jnp.zeros((n, 1), dtype=jnp.float64),
        source_degree_power=jnp.zeros_like(jnp.asarray(power, dtype=jnp.float64)),
        source_dehnen_power=jnp.asarray(power, dtype=jnp.float64),
        source_mass=jnp.asarray(masses, dtype=jnp.float64),
        source_mac_center=jnp.asarray(centers, dtype=jnp.float64),
        target_mac_center=jnp.asarray(centers, dtype=jnp.float64),
        source_radius_bound=jnp.asarray(radii, dtype=jnp.float64),
        target_radius_bound=jnp.asarray(radii, dtype=jnp.float64),
        target_accept_threshold=jnp.asarray(thresholds, dtype=jnp.float64),
        order_tags=jnp.asarray([0], dtype=jnp.int32),
        order_values=jnp.asarray(order_arr, dtype=jnp.int32),
        order_values_float=jnp.asarray(order_arr, dtype=jnp.float64),
        dehnen_binomial_masked_by_order=jnp.asarray(masked_binomial, dtype=jnp.float64),
        dehnen_exponent_by_order=jnp.asarray(exponent, dtype=jnp.int32),
        relaxed_theta_sq=jnp.asarray(1.0, dtype=jnp.float64),
        error_model_code=jnp.asarray(_ERROR_MODEL_DEHNEN_PAPER, dtype=jnp.int32),
    )


def test_dehnen_paper_accept_matches_eq16a_reference():
    """The accept mask equals a numpy evaluation of eq (16a), boundaries included."""

    rng = np.random.default_rng(11)
    num_nodes = 6
    power = rng.uniform(0.05, 1.0, size=(num_nodes, ORDER + 1))
    power[:, 0] = rng.uniform(0.5, 2.0, size=num_nodes)  # P_0 = mass
    masses = power[:, 0].copy()
    radii = np.asarray([0.4, 0.3, 0.25, 0.5, 0.0, 0.35])
    thresholds = rng.uniform(1e-6, 1e-2, size=num_nodes)

    # Pair geometry chosen to include: comfortably separated, marginal, exactly
    # touching (rho_z + rho_s == r), and coincident centres (r at the floor).
    centers = np.zeros((num_nodes, 3))
    centers[1] = [5.0, 0.0, 0.0]
    centers[2] = [0.55, 0.0, 0.0]
    centers[3] = [0.9, 0.0, 0.0]
    centers[4] = [0.0, 0.0, 0.0]
    centers[5] = [0.75, 0.0, 0.0]

    targets = np.asarray([0, 0, 0, 0, 4, 2], dtype=np.int32)
    sources = np.asarray([1, 2, 3, 5, 0, 5], dtype=np.int32)

    state = _paper_state(
        power=power,
        masses=masses,
        centers=centers,
        radii=radii,
        thresholds=thresholds,
    )

    npairs = targets.shape[0]
    actions, _ = adaptive_pair_policy(
        state,
        valid_pairs=jnp.ones((npairs,), dtype=jnp.bool_),
        mac_ok=jnp.zeros((npairs,), dtype=jnp.bool_),
        different_nodes=jnp.ones((npairs,), dtype=jnp.bool_),
        target_leaf=jnp.zeros((npairs,), dtype=jnp.bool_),
        source_leaf=jnp.zeros((npairs,), dtype=jnp.bool_),
        same_node=jnp.zeros((npairs,), dtype=jnp.bool_),
        target_nodes=jnp.asarray(targets),
        source_nodes=jnp.asarray(sources),
        center_target=jnp.asarray(centers[targets], dtype=jnp.float64),
        center_source=jnp.asarray(centers[sources], dtype=jnp.float64),
        dist_sq=jnp.asarray(
            np.sum((centers[targets] - centers[sources]) ** 2, axis=1),
            dtype=jnp.float64,
        ),
        extent_target=jnp.asarray(radii[targets], dtype=jnp.float64),
        extent_source=jnp.asarray(radii[sources], dtype=jnp.float64),
    )
    accepted = np.asarray(actions) == 0

    expected = []
    for tgt, src in zip(targets, sources):
        rho_z = radii[src]
        rho_s = radii[tgt]
        r = max(float(np.linalg.norm(centers[src] - centers[tgt])), 1e-24)
        est = _reference_eq13_eq15(
            power=power[src],
            mass=masses[src],
            rho_z=rho_z,
            rho_s=rho_s,
            distance=r,
            orders=(ORDER,),
        )[0]
        force_error = est * masses[src] / max(r * r, 1e-24)
        expected.append(bool((rho_z + rho_s < r) and (force_error < thresholds[tgt])))

    assert accepted.tolist() == expected
    # The fixture must actually exercise both outcomes, or it proves nothing.
    assert any(expected) and not all(expected)


# ---------------------------------------------------------------------------
# MAC geometry: node spheres must contain their own particles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaf_mode", ["exact", "approx"])
def test_node_sphere_contains_every_particle(leaf_mode):
    """Every node sphere must enclose every particle in its span.

    Uses a >=3-level radix tree on purpose: the merge is only correct if
    children are reduced before parents, and radix internal-node indices are
    not stored in postorder. A tree with a single internal node satisfies any
    ordering vacuously and cannot see the bug.
    """

    rng = np.random.default_rng(3)
    positions = jnp.asarray(rng.normal(size=(512, 3)), dtype=jnp.float64)
    masses = jnp.ones((512,), dtype=jnp.float64)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=16,
        tree_type="radix",
        target_leaf_particles=16,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]

    centers, radii = compute_tree_merged_sphere_geometry(
        tree=tree, positions_sorted=positions_sorted, leaf_mode=leaf_mode
    )
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    pos = np.asarray(positions_sorted)
    node_ranges = np.asarray(tree.node_ranges)

    assert int(tree.num_internal_nodes) >= 4, "fixture must be a multi-level tree"

    failures = []
    for node in range(node_ranges.shape[0]):
        start, end = int(node_ranges[node, 0]), int(node_ranges[node, 1])
        if end < start:
            continue
        span = pos[start : end + 1]
        reach = float(np.max(np.linalg.norm(span - centers[node], axis=1)))
        if reach > radii[node] + 1e-9:
            failures.append((node, reach, float(radii[node])))

    assert (
        not failures
    ), f"{len(failures)} nodes do not contain their particles: {failures[:5]}"


def _upward_fixture(n: int = 512, seed: int = 3):
    from jaccpot.upward.tree_expansions import prepare_upward_sweep

    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 2.0, size=n), dtype=jnp.float64)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=16,
        tree_type="radix",
        target_leaf_particles=16,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]
    masses_sorted = masses[tree.particle_indices]
    upward = prepare_upward_sweep(
        tree,
        positions_sorted,
        masses_sorted,
        max_order=ORDER,
        center_mode="com",
    )
    return tree, positions_sorted, upward


@pytest.mark.parametrize("mode", ["com", "exact", "tree", "tree_approx", "runtime"])
def test_resolved_mac_radius_is_a_valid_bound_about_the_expansion_centre(mode):
    """Every geometry mode must bound the radius about the M2L expansion centre.

    Dehnen's eqs (13)/(15) are evaluated at the pair distance and radii the
    translation actually uses. A mode whose radius is measured about a different
    point is not a bound, and the `theta < 1` guard in eq (16a) then admits
    pairs whose multipole series diverges.
    """

    from jaccpot.runtime._adaptive_policy import resolve_dehnen_geometry

    tree, positions_sorted, upward = _upward_fixture()
    centers, radii = resolve_dehnen_geometry(
        geometry_mode=mode,
        tree=tree,
        positions_sorted=positions_sorted,
        upward=upward,
        dtype=jnp.float64,
    )

    expansion_centers = np.asarray(upward.multipoles.centers)
    np.testing.assert_allclose(np.asarray(centers), expansion_centers, atol=1e-12)

    pos = np.asarray(positions_sorted)
    node_ranges = np.asarray(tree.node_ranges)
    radii = np.asarray(radii)
    failures = []
    for node in range(node_ranges.shape[0]):
        start, end = int(node_ranges[node, 0]), int(node_ranges[node, 1])
        if end < start:
            continue
        reach = float(
            np.max(
                np.linalg.norm(pos[start : end + 1] - expansion_centers[node], axis=1)
            )
        )
        if reach > radii[node] + 1e-9:
            failures.append((node, reach, float(radii[node])))

    assert not failures, f"{mode}: {len(failures)} nodes unbounded: {failures[:5]}"


def test_com_mode_is_no_looser_than_the_other_geometry_modes():
    """The ``com`` default must not be a looser bound than any opt-in mode.

    ``com`` combines a leaf-exact upward merge with the exact farthest-corner
    distance to the node's bounding box and takes the tighter of the two. It is
    therefore expected to dominate ``runtime`` (which inflates the bounding
    sphere by the whole centre offset) everywhere, and to be competitive with
    the host-side sphere fits.
    """

    from jaccpot.runtime._adaptive_policy import resolve_dehnen_geometry

    tree, positions_sorted, upward = _upward_fixture()
    kwargs = dict(
        tree=tree,
        positions_sorted=positions_sorted,
        upward=upward,
        dtype=jnp.float64,
    )
    _, com_radii = resolve_dehnen_geometry(geometry_mode="com", **kwargs)
    _, runtime_radii = resolve_dehnen_geometry(geometry_mode="runtime", **kwargs)

    populated = np.asarray(tree.node_ranges)[:, 1] >= np.asarray(tree.node_ranges)[:, 0]
    com = np.asarray(com_radii)[populated]
    runtime = np.asarray(runtime_radii)[populated]

    assert np.all(com <= runtime + 1e-9)
    # And it must be a strict improvement somewhere, or it is not worth having.
    assert float(np.median(runtime / np.maximum(com, 1e-30))) > 1.0


def test_tree_merged_sphere_is_a_valid_bound_on_the_exact_sphere():
    """The merged sphere must never be tighter than the exact enclosing sphere."""

    rng = np.random.default_rng(5)
    positions = jnp.asarray(rng.normal(size=(256, 3)), dtype=jnp.float64)
    masses = jnp.ones((256,), dtype=jnp.float64)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=16,
        tree_type="radix",
        target_leaf_particles=16,
        refine_local=False,
    )
    positions_sorted = positions[tree.particle_indices]

    _, merged_radii = compute_tree_merged_sphere_geometry(
        tree=tree, positions_sorted=positions_sorted, leaf_mode="exact"
    )
    _, exact_radii = compute_smallest_enclosing_sphere_geometry(
        node_ranges=tree.node_ranges, positions_sorted=positions_sorted
    )

    assert np.all(np.asarray(merged_radii) >= np.asarray(exact_radii) - 1e-9)


# ---------------------------------------------------------------------------
# Force scales and G
# ---------------------------------------------------------------------------


def test_force_scale_min_ignores_empty_leaves():
    """Padded/empty leaves must not drag the `min` force scale to zero.

    Under the paper MAC the per-node force scale is reduced with `min`, and
    ``target_accept_threshold = eps * scale``. A zero from an empty leaf
    propagates to every ancestor and drives the threshold to its floor, so that
    whole chain refuses every far pair. Capacity-fixed trees have padded leaves
    by construction, so this is the common case rather than a corner case.

    The reduction is exercised directly against a tree whose leaf spans have been
    edited to include an empty one -- a freshly built radix tree over random
    points happens to have none, which would make this test vacuous.
    """

    import dataclasses

    from jaccpot.runtime._adaptive_policy import (
        compute_node_force_scale_from_sorted_acc,
    )

    rng = np.random.default_rng(19)
    n = 64
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.ones((n,), dtype=jnp.float64)
    real_tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=8,
        tree_type="radix",
        target_leaf_particles=8,
        refine_local=False,
    )
    num_internal = int(real_tree.num_internal_nodes)
    node_ranges = np.asarray(real_tree.node_ranges).copy()
    assert np.all(
        node_ranges[num_internal:, 1] >= node_ranges[num_internal:, 0]
    ), "fixture assumes the freshly built tree starts with no empty leaves"

    # Blank one leaf so it spans nothing, exactly as capacity padding does.
    blanked = num_internal
    blank_lo = int(node_ranges[blanked, 0])
    blank_hi = int(node_ranges[blanked, 1])
    node_ranges[blanked] = [blank_lo, blank_lo - 1]

    # A real tree carrying the edit, not a namespace imitating one. `node_ranges` is
    # not a field of the tree itself -- it lives on `tree.topology`, a NamedTuple --
    # so the edit goes in through `_replace` and every other field stays consistent
    # instead of simply being absent. `SimpleNamespace` used to stand in here, which
    # violated `compute_node_force_scale_from_sorted_acc`'s declared `tree: Tree` and
    # was the last of F40's runtime-typecheck failures.
    tree = dataclasses.replace(
        real_tree,
        topology=real_tree.topology._replace(
            node_ranges=jnp.asarray(node_ranges, dtype=jnp.int32)
        ),
    )
    assert isinstance(tree, Tree)
    assert int(tree.num_internal_nodes) == num_internal

    accelerations = jnp.asarray(rng.normal(size=(n, 3)) + 3.0, dtype=jnp.float64)
    scales = np.asarray(
        compute_node_force_scale_from_sorted_acc(
            tree=tree, accelerations_sorted=accelerations, reduction="min"
        )
    )

    magnitudes = np.linalg.norm(np.asarray(accelerations), axis=1)
    remaining = np.concatenate([magnitudes[:blank_lo], magnitudes[blank_hi + 1 :]])

    assert np.all(np.isfinite(scales)), "force scales must stay finite"
    # The empty leaf must be skipped, not treated as a zero-acceleration leaf:
    # the root min is the min over the particles that remain.
    assert np.isclose(scales[0], remaining.min()), (
        f"root min {scales[0]} should be the min over populated leaves "
        f"{remaining.min()}, not 0"
    )
    populated = [
        node
        for node in range(node_ranges.shape[0])
        if node_ranges[node, 1] >= node_ranges[node, 0]
    ]
    assert np.all(scales[populated] > 0.0), (
        f"{int(np.sum(scales[populated] <= 0.0))} populated nodes have a zero min "
        "force scale, so their whole ancestor chain would refuse every far pair"
    )


def test_eq16a_is_invariant_under_G_mass_rescaling():
    """eq (16a) is homogeneous under ``(G, m) -> (G/c, c*m)``.

    Both sides are invariant: the left-hand side is ``G Etilde M_A / r^2`` with
    ``Etilde`` depending on masses only through the ratio ``P_k / M_A``, so it
    picks up one factor of ``G * m``; the right-hand side is ``eps min_b |a_b|``
    and the accelerations themselves are unchanged because ``a ~ G m``. The
    decision therefore cannot move. It does move if ``G`` is dropped from the
    left-hand side, which is what made the criterion run at an effective
    tolerance of ``eps * G``.
    """

    rng = np.random.default_rng(23)
    num_nodes = 5
    base_power = rng.uniform(0.05, 1.0, size=(num_nodes, ORDER + 1))
    base_power[:, 0] = rng.uniform(0.5, 2.0, size=num_nodes)
    base_masses = base_power[:, 0].copy()
    radii = np.asarray([0.30, 0.22, 0.18, 0.40, 0.26])
    centers = np.zeros((num_nodes, 3))
    for i in range(1, num_nodes):
        centers[i] = [0.7 + 0.45 * i, 0.0, 0.0]
    targets = np.asarray([0, 0, 1, 2, 3], dtype=np.int32)
    sources = np.asarray([1, 2, 3, 4, 4], dtype=np.int32)

    # Pick per-target thresholds straddling the actual eq-(16a) left-hand side at
    # G = 1 so the fixture provably exercises both outcomes.
    lhs = []
    for tgt, src in zip(targets, sources):
        r = float(np.linalg.norm(centers[src] - centers[tgt]))
        lhs.append(
            _reference_eq13_eq15(
                power=base_power[src],
                mass=base_masses[src],
                rho_z=radii[src],
                rho_s=radii[tgt],
                distance=r,
                orders=(ORDER,),
            )[0]
            * base_masses[src]
            / r**2
        )
    thresholds = np.full(num_nodes, float(np.median(lhs)))

    def accept_mask(c: float) -> np.ndarray:
        state = _paper_state(
            power=base_power * c,  # P_n is linear in mass
            masses=base_masses * c,
            centers=centers,
            radii=radii,
            thresholds=thresholds,  # a_b, hence the threshold, is invariant
        )._replace(gravitational_constant=1.0 / c)
        npairs = targets.shape[0]
        actions, _ = adaptive_pair_policy(
            state,
            valid_pairs=jnp.ones((npairs,), dtype=jnp.bool_),
            mac_ok=jnp.zeros((npairs,), dtype=jnp.bool_),
            different_nodes=jnp.ones((npairs,), dtype=jnp.bool_),
            target_leaf=jnp.zeros((npairs,), dtype=jnp.bool_),
            source_leaf=jnp.zeros((npairs,), dtype=jnp.bool_),
            same_node=jnp.zeros((npairs,), dtype=jnp.bool_),
            target_nodes=jnp.asarray(targets),
            source_nodes=jnp.asarray(sources),
            center_target=jnp.asarray(centers[targets], dtype=jnp.float64),
            center_source=jnp.asarray(centers[sources], dtype=jnp.float64),
            dist_sq=jnp.asarray(
                np.sum((centers[targets] - centers[sources]) ** 2, axis=1),
                dtype=jnp.float64,
            ),
            extent_target=jnp.asarray(radii[targets], dtype=jnp.float64),
            extent_source=jnp.asarray(radii[sources], dtype=jnp.float64),
        )
        return np.asarray(actions) == 0

    baseline = accept_mask(1.0)
    assert baseline.any() and not baseline.all(), "fixture must exercise both outcomes"
    assert baseline.tolist() == accept_mask(4.0).tolist()
    assert baseline.tolist() == accept_mask(0.25).tolist()
