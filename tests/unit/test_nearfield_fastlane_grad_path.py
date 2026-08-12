"""The leaf-major near-field fast lane on the differentiable grad path.

``differentiable_accelerations`` runs the near field through the bucketed
edge-list kernel, which an in-context ablation showed dominates the differentiable
path (~83% of the forward, ~91% of the reverse at N=1024/p4).
``JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`` re-expresses the same near field
leaf-major and routes it through the radix fast lane instead -- with
``use_pallas=True`` that is the fused Pallas kernel behind an analytic O(N)
leaf-pair reverse, otherwise the tiled pure-JAX kernel under ordinary autodiff.
The lane is opt-in: measured in-context it neither clearly wins nor clearly loses
(see ``docs/differentiable_fmm_design.md``, PR-4).

The contract these tests pin is that the fast lane is a different *traversal*,
not a different force: the leaf-major payload must enumerate exactly the edge set
the bucketed lane scans, so the two agree to round-off in BOTH the value and the
gradient. That is not automatic -- the CSR neighbour buffer can carry ``-1``
padding inside a leaf's slice, and dropping or keeping a padded slot silently
changes which pairs are summed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.nearfield._fast_lane import (
    compute_leaf_p2p_accelerations_radix_fast_lane,
)
from jaccpot.nearfield.near_field import compute_leaf_p2p_accelerations
from jaccpot.runtime._nearfield_fastlane import (
    NearfieldTopologyNotConcrete,
    build_leaf_major_nearfield_payload,
    clear_nearfield_fastlane_payload_cache,
    leaf_major_nearfield_payload_cached,
    nearfield_topology_arrays,
)
from jaccpot.runtime.kernels.core import _build_nearfield_interop_data

FAST_LANE_ENV = "JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE"


def _system(n, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    return positions, masses


def _prepared(n, *, leaf, order=3, theta=0.5, basis="complex", seed=0):
    positions, masses = _system(n, seed=seed)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)
    return fmm, state, positions, masses


def _nearfield_view(state):
    view = state.nearfield_interop
    if view is None:
        view = _build_nearfield_interop_data(state.tree, state.neighbor_list)
    return view


def _payload(state, n):
    return leaf_major_nearfield_payload_cached(
        num_particles=n,
        max_leaf_size=int(state.max_leaf_size),
        **nearfield_topology_arrays(
            state.tree, state.neighbor_list, state.nearfield_interop
        ),
    )


def _bucketed_near(fmm, state, view, positions_sorted, masses_sorted):
    impl = fmm._impl
    return compute_leaf_p2p_accelerations(
        state.tree,
        state.neighbor_list,
        positions_sorted,
        masses_sorted,
        G=impl.G,
        softening=impl.softening,
        max_leaf_size=state.max_leaf_size,
        nearfield_mode="bucketed",
        node_ranges_override=view.node_ranges,
        leaf_nodes_override=view.leaf_nodes,
        neighbor_offsets_override=view.offsets,
        neighbor_indices_override=view.neighbors,
        neighbor_counts_override=view.counts,
        leaf_particle_indices_override=view.leaf_particle_indices,
        leaf_particle_mask_override=view.leaf_particle_mask,
    )


def _fastlane_near(fmm, state, payload, positions_sorted, masses_sorted):
    impl = fmm._impl
    return compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=positions_sorted,
        masses_sorted=masses_sorted,
        payload=payload,
        G=impl.G,
        softening=float(impl.softening),
        use_pallas=False,
        differentiable=True,
    )


def _rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / (denom if denom else 1.0))


# ---------------------------------------------------------------------------
# The payload enumerates the same edge set as the bucketed lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,leaf", [(192, 16), (320, 8), (512, 32)])
def test_leaf_major_payload_matches_bucketed_near_field(n, leaf):
    fmm, state, _, _ = _prepared(n, leaf=leaf)
    view = _nearfield_view(state)
    payload = _payload(state, n)
    positions_sorted = state.positions_sorted
    masses_sorted = state.masses_sorted

    bucketed = _bucketed_near(fmm, state, view, positions_sorted, masses_sorted)
    fastlane = _fastlane_near(fmm, state, payload, positions_sorted, masses_sorted)
    assert _rel_l2(bucketed, fastlane) < 1e-13


@pytest.mark.parametrize("n,leaf", [(192, 16), (320, 8)])
def test_leaf_major_payload_near_field_gradients_match_bucketed(n, leaf):
    """Positions AND masses -- the analytic leaf-pair reverse must match autodiff."""
    fmm, state, _, _ = _prepared(n, leaf=leaf)
    view = _nearfield_view(state)
    payload = _payload(state, n)
    positions_sorted = state.positions_sorted
    masses_sorted = state.masses_sorted

    def loss_bucketed(p, m):
        return jnp.sum(_bucketed_near(fmm, state, view, p, m) ** 2)

    def loss_fastlane(p, m):
        return jnp.sum(_fastlane_near(fmm, state, payload, p, m) ** 2)

    g_bucketed = jax.grad(loss_bucketed, argnums=(0, 1))(
        positions_sorted, masses_sorted
    )
    g_fastlane = jax.grad(loss_fastlane, argnums=(0, 1))(
        positions_sorted, masses_sorted
    )

    assert _rel_l2(g_bucketed[0], g_fastlane[0]) < 1e-12
    assert _rel_l2(g_bucketed[1], g_fastlane[1]) < 1e-12
    assert bool(jnp.all(jnp.isfinite(g_fastlane[0])))
    assert bool(jnp.all(jnp.isfinite(g_fastlane[1])))


def test_payload_covers_every_valid_neighbour_edge():
    """Edge-count identity: padded slots dropped, real neighbours all present."""
    _, state, _, _ = _prepared(320, leaf=8)
    view = _nearfield_view(state)
    payload = _payload(state, 320)
    offsets = np.asarray(view.offsets)
    neighbors = np.asarray(view.neighbors)
    leaf_nodes = np.asarray(view.leaf_nodes)
    num_leaves = int(leaf_nodes.shape[0])

    is_leaf = np.zeros((int(np.asarray(view.node_ranges).shape[0]),), dtype=bool)
    is_leaf[leaf_nodes] = True
    expected = 0
    for target in range(num_leaves):
        slice_ = neighbors[offsets[target] : offsets[target + 1]]
        expected += int(
            np.count_nonzero((slice_ >= 0) & is_leaf[np.clip(slice_, 0, None)])
        )

    assert int(np.asarray(payload.source_leaf_valid_mask).sum()) == expected
    # Prepacked layout (empty per-particle sources) is what selects the lane
    # carrying the analytic O(N) reverse.
    assert int(np.asarray(payload.source_particle_ids).size) == 0
    assert np.asarray(payload.source_leaf_ids).ndim == 3


def test_payload_handles_a_tree_with_no_cross_leaf_neighbours():
    """A single-leaf tree still has to yield a usable rank-3 prepacked payload."""
    positions, masses = _system(8, seed=5)
    payload = build_leaf_major_nearfield_payload(
        node_ranges=jnp.asarray([[0, 7]], dtype=jnp.int32),
        leaf_nodes=jnp.zeros((1,), dtype=jnp.int32),
        offsets=jnp.zeros((2,), dtype=jnp.int32),
        neighbors=jnp.zeros((0,), dtype=jnp.int32),
        num_particles=8,
        max_leaf_size=8,
    )
    assert np.asarray(payload.source_leaf_ids).ndim == 3
    assert not bool(np.asarray(payload.source_leaf_valid_mask).any())

    fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.5, G=1.0, softening=1e-2
    )
    near = compute_leaf_p2p_accelerations_radix_fast_lane(
        positions_sorted=positions,
        masses_sorted=masses,
        payload=payload,
        G=fmm._impl.G,
        softening=float(fmm._impl.softening),
        use_pallas=False,
        differentiable=True,
    )
    # Only the intra-leaf self term survives, and it is the full direct sum here.
    direct = np.zeros((8, 3))
    pos = np.asarray(positions)
    mass = np.asarray(masses)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            d = pos[i] - pos[j]
            r2 = float(d @ d) + 1e-4
            direct[i] -= mass[j] * d / r2**1.5
    assert _rel_l2(direct, near) < 1e-12


def test_payload_is_memoized_per_topology():
    clear_nearfield_fastlane_payload_cache()
    _, state, _, _ = _prepared(192, leaf=16)
    first = _payload(state, 192)
    second = _payload(state, 192)
    assert first is second


def test_payload_builder_rejects_traced_topology():
    """A payload built from tracers would silently mis-size the padded block.

    Under an outer ``jax.jit`` every ``jnp`` op is staged out even on concrete
    constants, so the max-neighbour reduction that sizes the block cannot be read
    back. That has to be a loud error, not a fallback.
    """
    _, state, _, _ = _prepared(192, leaf=16)
    args = nearfield_topology_arrays(
        state.tree, state.neighbor_list, state.nearfield_interop
    )

    def traced(neighbors):
        return build_leaf_major_nearfield_payload(
            **{**args, "neighbors": neighbors},
            num_particles=192,
            max_leaf_size=int(state.max_leaf_size),
        )

    with pytest.raises(NearfieldTopologyNotConcrete):
        jax.jit(traced)(jnp.asarray(args["neighbors"]))


# ---------------------------------------------------------------------------
# The analytic reverse's occupancy sort
# ---------------------------------------------------------------------------


def _skewed_leafpair_case(num_leaves=8, width=4, blocks=2, lanes=4, seed=11):
    """One leaf neighbours everything, the rest neighbour almost nothing.

    This is the production pathology in miniature: measured on the canonical
    galaxy config, ``max neighbours == leaves - 1`` at every N and every geometry,
    so the padded rectangle is sized by a single leaf while the mean leaf uses a
    fraction of it. The occupancy sort exists for exactly this shape, so a flat
    random occupancy would not exercise it.
    """
    rng = np.random.default_rng(seed)
    slots = blocks * lanes
    num_particles = num_leaves * width
    leaf_particle_idx = jnp.asarray(
        np.arange(num_particles).reshape(num_leaves, width), dtype=jnp.int32
    )
    positions = jnp.asarray(rng.normal(size=(num_particles, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=num_particles), dtype=jnp.float64)

    valid = np.zeros((num_leaves, slots), dtype=bool)
    valid[0, :] = True  # the pathological leaf: neighbours every slot
    for leaf in range(1, num_leaves):
        valid[leaf, : (leaf % 2) + 1] = True  # everyone else: 1-2 neighbours
    ids = rng.integers(0, num_leaves, size=(num_leaves, slots))

    return dict(
        leaf_positions=positions[leaf_particle_idx],
        leaf_masses=masses[leaf_particle_idx],
        leaf_mask=jnp.ones((num_leaves, width), dtype=bool),
        leaf_particle_idx=leaf_particle_idx,
        source_leaf_ids=jnp.asarray(
            ids.reshape(num_leaves, blocks, lanes), dtype=jnp.int32
        ),
        source_valid=jnp.asarray(valid.reshape(num_leaves, blocks, lanes)),
        cotangent=jnp.asarray(rng.normal(size=(num_particles, 3)), dtype=jnp.float64),
    )


def _run_analytic_reverse(case, *, skip_empty_tiles):
    from jaccpot.nearfield import near_field as nf

    return nf._leafpair_accel_analytic_vjp(
        case["leaf_positions"],
        case["leaf_masses"],
        case["leaf_mask"],
        case["leaf_particle_idx"],
        case["source_leaf_ids"],
        case["source_valid"],
        case["cotangent"],
        softening_sq=jnp.asarray(1e-2, dtype=jnp.float64),
        G=jnp.asarray(1.0, dtype=jnp.float64),
        leaf_batch=2,
        slot_tile=2,
        skip_empty_tiles=skip_empty_tiles,
    )


def test_analytic_reverse_empty_tile_skip_is_bit_exact():
    """Skipping an all-invalid tile must change nothing at all.

    Every term in the skipped branch is masked by ``valid_slot``, so the tile
    contributes exact zeros; dropping it cannot even reassociate the sum. Anything
    less than bit-equality here means the mask is not doing what it claims.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")

    case = _skewed_leafpair_case()
    base_pos, base_mass = _run_analytic_reverse(case, skip_empty_tiles=False)
    pos, mass = _run_analytic_reverse(case, skip_empty_tiles=True)
    np.testing.assert_array_equal(np.asarray(pos), np.asarray(base_pos))
    np.testing.assert_array_equal(np.asarray(mass), np.asarray(base_mass))
    # Guard against a degenerate case that would make the comparison vacuous.
    assert float(np.abs(np.asarray(base_pos)).max()) > 0.0


def test_reverse_tiers_partition_every_leaf_exactly_once():
    """A dropped or duplicated leaf would silently lose or double its gradient."""
    from jaccpot.nearfield import near_field as nf

    case = _skewed_leafpair_case(num_leaves=16, blocks=4, lanes=4)
    tiers = nf.build_leafpair_reverse_tiers(case["source_valid"], slot_tile=2)
    assert tiers is not None, "skewed occupancy should be worth tiering"
    seen = [leaf for members, _ in tiers for leaf in members]
    assert sorted(seen) == list(range(16))
    # Morton order preserved inside each tier -- that is what keeps the source
    # gather coherent, and it is the whole reason this beats a global sort.
    for members, _ in tiers:
        assert list(members) == sorted(members)
    # Every leaf's own occupancy must fit inside the width it was assigned.
    valid = np.asarray(case["source_valid"]).reshape(16, -1)
    for members, width in tiers:
        for leaf in members:
            assert int(valid[leaf].sum()) <= width


def test_reverse_tiers_match_the_untiered_reverse():
    """Tiering changes only which pass visits a leaf, never the gradient."""
    from jaccpot.nearfield import near_field as nf

    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")

    case = _skewed_leafpair_case(num_leaves=16, blocks=4, lanes=4)
    tiers = nf.build_leafpair_reverse_tiers(case["source_valid"], slot_tile=2)
    assert tiers is not None and len(tiers) > 1

    def run(t):
        return nf._leafpair_accel_analytic_vjp(
            case["leaf_positions"],
            case["leaf_masses"],
            case["leaf_mask"],
            case["leaf_particle_idx"],
            case["source_leaf_ids"],
            case["source_valid"],
            case["cotangent"],
            softening_sq=jnp.asarray(1e-2, dtype=jnp.float64),
            G=jnp.asarray(1.0, dtype=jnp.float64),
            leaf_batch=2,
            slot_tile=2,
            skip_empty_tiles=True,
            tiers=t,
        )

    base_pos, base_mass = run(None)
    pos, mass = run(tiers)
    # Round-off, not bit-exact: tiers reorder the scatter-add accumulation.
    np.testing.assert_allclose(
        np.asarray(pos), np.asarray(base_pos), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(mass), np.asarray(base_mass), rtol=1e-11, atol=1e-12
    )
    assert float(np.abs(np.asarray(base_pos)).max()) > 0.0


def test_reverse_tiers_declined_when_occupancy_is_uniform():
    """No split to make => stay on the single full-width pass, byte-identical."""
    from jaccpot.nearfield import near_field as nf

    valid = jnp.ones((8, 2, 4), dtype=bool)
    assert nf.build_leafpair_reverse_tiers(valid, slot_tile=2) is None


def test_analytic_reverse_matches_autodiff_under_skewed_occupancy():
    """The analytic reverse equals autodiff of the tiled twin at skewed occupancy."""
    from jaccpot.nearfield import near_field as nf

    if not jax.config.jax_enable_x64:
        pytest.skip("requires x64 for a tight tolerance")

    case = _skewed_leafpair_case()
    soft = jnp.asarray(1e-2, dtype=jnp.float64)
    G = jnp.asarray(1.0, dtype=jnp.float64)
    positions = jnp.zeros((int(case["leaf_particle_idx"].size), 3), dtype=jnp.float64)

    def twin(leaf_pos, leaf_mass):
        return nf._compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
            positions,
            case["source_leaf_ids"],
            case["source_valid"],
            leaf_pos,
            leaf_mass,
            case["leaf_mask"],
            case["leaf_particle_idx"],
            G=G,
            softening_sq=soft,
            target_leaf_batch_size=2,
            target_block_tile_size=2,
            target_block_tile_scan_unroll=1,
            target_block_batch_scan_unroll=1,
            occupancy_sort=False,
            skip_empty_tiles=False,
        )

    _, vjp_fn = jax.vjp(twin, case["leaf_positions"], case["leaf_masses"])
    ref_pos, ref_mass = vjp_fn(case["cotangent"])

    pos, mass = nf._leafpair_accel_analytic_vjp(
        case["leaf_positions"],
        case["leaf_masses"],
        case["leaf_mask"],
        case["leaf_particle_idx"],
        case["source_leaf_ids"],
        case["source_valid"],
        case["cotangent"],
        softening_sq=soft,
        G=G,
        leaf_batch=2,
        slot_tile=2,
        skip_empty_tiles=True,
    )
    np.testing.assert_allclose(
        np.asarray(pos), np.asarray(ref_pos), rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(mass), np.asarray(ref_mass), rtol=1e-11, atol=1e-11
    )


# ---------------------------------------------------------------------------
# End-to-end: the grad path with the fast lane engaged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["complex", "real"])
def test_differentiable_accelerations_fast_lane_matches_bucketed(basis, monkeypatch):
    fmm, state, positions, masses = _prepared(256, leaf=16, order=4, basis=basis)

    def loss(p, m):
        return jnp.sum(fmm.differentiable_accelerations(state, p, m) ** 2)

    monkeypatch.delenv(FAST_LANE_ENV, raising=False)
    acc_bucketed = fmm.differentiable_accelerations(state, positions, masses)
    grad_bucketed = jax.grad(loss, argnums=(0, 1))(positions, masses)

    monkeypatch.setenv(FAST_LANE_ENV, "1")
    acc_fastlane = fmm.differentiable_accelerations(state, positions, masses)
    grad_fastlane = jax.grad(loss, argnums=(0, 1))(positions, masses)

    assert _rel_l2(acc_bucketed, acc_fastlane) < 1e-13
    assert _rel_l2(grad_bucketed[0], grad_fastlane[0]) < 1e-12
    assert _rel_l2(grad_bucketed[1], grad_fastlane[1]) < 1e-12
    assert bool(jnp.all(jnp.isfinite(acc_fastlane)))


def test_fast_lane_is_opt_in(monkeypatch):
    """Flag off => the shipped bucketed near field, unchanged."""
    calls = []
    from jaccpot.runtime import fmm_evaluate

    original = fmm_evaluate.EvaluateMixin._evaluate_leaf_major_nearfield

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        fmm_evaluate.EvaluateMixin, "_evaluate_leaf_major_nearfield", spy
    )
    fmm, state, positions, masses = _prepared(128, leaf=16)

    monkeypatch.delenv(FAST_LANE_ENV, raising=False)
    fmm.differentiable_accelerations(state, positions, masses)
    assert calls == []

    monkeypatch.setenv(FAST_LANE_ENV, "1")
    fmm.differentiable_accelerations(state, positions, masses)
    assert calls == [1]


def test_fused_pallas_fast_lane_matches_bucketed_end_to_end(monkeypatch):
    """The configuration PR-4 actually recommends: fast lane + ``use_pallas=True``.

    This is the only path that reaches ``_radix_fast_lane_prepacked_accel_cvjp`` and
    therefore the analytic O(N) leaf-pair reverse; with ``use_pallas=False`` the same
    flag runs the tiled pure-JAX kernel under ordinary autodiff. Needs a real
    Ampere+ GPU -- the lane gates itself off elsewhere and would silently degrade
    to the case already covered above.
    """
    from jaccpot.pallas.nearfield_fused_leaf import pallas_nearfield_fused_supported

    if not pallas_nearfield_fused_supported():
        pytest.skip("fused near-field Pallas requires an Ampere+ (sm_80) GPU")

    positions, masses = _system(256, seed=0)
    reference_fmm = FastMultipoleMethod(
        basis="complex", use_pallas=False, theta=0.5, G=1.0, softening=1e-2
    )
    pallas_fmm = FastMultipoleMethod(
        basis="complex", use_pallas=True, theta=0.5, G=1.0, softening=1e-2
    )
    ref_state = reference_fmm.prepare_state(
        positions, masses, max_order=4, leaf_size=16
    )
    pallas_state = pallas_fmm.prepare_state(
        positions, masses, max_order=4, leaf_size=16
    )

    monkeypatch.delenv(FAST_LANE_ENV, raising=False)
    acc_ref = reference_fmm.differentiable_accelerations(ref_state, positions, masses)
    grad_ref = jax.grad(
        lambda p, m: jnp.sum(
            reference_fmm.differentiable_accelerations(ref_state, p, m) ** 2
        ),
        argnums=(0, 1),
    )(positions, masses)

    monkeypatch.setenv(FAST_LANE_ENV, "1")
    acc_pallas = pallas_fmm.differentiable_accelerations(
        pallas_state, positions, masses
    )
    grad_pallas = jax.grad(
        lambda p, m: jnp.sum(
            pallas_fmm.differentiable_accelerations(pallas_state, p, m) ** 2
        ),
        argnums=(0, 1),
    )(positions, masses)

    # Looser than the pure-JAX comparison on purpose: the forward is a different
    # kernel (summation order differs) and the reverse is the analytic rule rather
    # than autodiff of that kernel, so agreement is to force accuracy, not to bits.
    assert _rel_l2(acc_ref, acc_pallas) < 1e-9
    assert _rel_l2(grad_ref[0], grad_pallas[0]) < 1e-8
    assert _rel_l2(grad_ref[1], grad_pallas[1]) < 1e-8
    assert bool(jnp.all(jnp.isfinite(grad_pallas[0])))
    assert bool(jnp.all(jnp.isfinite(grad_pallas[1])))


def test_fast_lane_survives_an_outer_jit(monkeypatch):
    """Regression: the payload must be host-built, not staged into the jaxpr.

    Wrapping the whole call in ``jax.jit`` turns every ``jnp`` op on the frozen
    topology into a tracer, which is what broke the first cut of this lane (the
    max-neighbour reduction that sizes the padded source block).
    """
    fmm, state, positions, masses = _prepared(256, leaf=16, order=4)

    monkeypatch.delenv(FAST_LANE_ENV, raising=False)
    reference = fmm.differentiable_accelerations(state, positions, masses)

    monkeypatch.setenv(FAST_LANE_ENV, "1")
    jitted = jax.jit(lambda p, m: fmm.differentiable_accelerations(state, p, m))
    assert _rel_l2(reference, jitted(positions, masses)) < 1e-13

    jitted_grad = jax.jit(
        jax.grad(
            lambda p, m: jnp.sum(fmm.differentiable_accelerations(state, p, m) ** 2),
            argnums=(0, 1),
        )
    )
    g_pos, g_mass = jitted_grad(positions, masses)
    assert bool(jnp.all(jnp.isfinite(g_pos)))
    assert bool(jnp.all(jnp.isfinite(g_mass)))


def test_fast_lane_rejects_potentials():
    """Acceleration-only lane: the potential half of the cvjp is not wired."""
    fmm, state, _, _ = _prepared(128, leaf=16)
    with pytest.raises(NotImplementedError, match="acceleration-only"):
        fmm._impl._evaluate_leaf_major_nearfield(
            state.tree,
            state.neighbor_list,
            state.nearfield_interop,
            state.positions_sorted,
            state.masses_sorted,
            max_leaf_size=int(state.max_leaf_size),
            return_potential=True,
        )
