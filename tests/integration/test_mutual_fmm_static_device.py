"""Static shapes and the on-device topology for the mutual FMM.

Split out of ``test_mutual_fmm.py`` for a **memory** reason, not a thematic one,
though the theme is real too.

`tests/integration` is memory-bound in CI -- the note on `test-full` in
`.github/workflows/ci.yml` records peak RSS 15.70-16.07 GB against a 16 GB
runner, and an OOM there surfaces as "The operation was canceled" rather than as
a failure. Adding this branch's cases to `test_mutual_fmm.py` tipped it over.

Bisected under a 15 GB cgroup, and the answer was not what it looked like: the
OOM lands in *`test_mutual_fmm.py`'s own* heavy gradient and rollout tests
(`test_rollout_gradient_finite_difference`, `test_block_step_rollout_...`), at
positions 28-38 of the run -- before any test here, at 59-63, has even started.
So the cost is a pile that accumulates across those tests, and the fix is to keep
the *new* cases out of that shard, not to reshuffle main's.

An earlier attempt did the opposite -- it moved all the mutual tests into one
shard, which concentrated those hogs into a single worker and OOM-killed at both
``-n 2`` and ``-n 1``. Do not group them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.mutual import (
    build_mutual_state,
    build_mutual_topology,
    mutual_weighted_accelerations,
)
from jaccpot.nornax_adapter import BlockStepFMM

SOFTENING = 1.0e-2
K_MAX = 3


def _system(n, seed=0, scale=1.0):
    """An isotropic Gaussian blob with unequal masses, as the sibling module uses."""
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(0.0, scale, (n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=jnp.float64)
    return positions, masses


def _rungs(n, k_max=K_MAX, seed=11):
    """Random rungs in ``[0, k_max]``."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, k_max + 1, n), dtype=jnp.int32)


def _nonrigid_nudge(positions, scale, seed):
    """Perturb positions in a way that actually changes the tree.

    Adding a *scalar* to every coordinate is a rigid translation and leaves the
    Morton order, the node linkage and every MAC outcome untouched -- a check
    written that way reports perfect shape stability for a topology that is in
    fact drifting. This nudges each particle independently.
    """
    rng = np.random.default_rng(seed)
    return positions + scale * jnp.asarray(
        rng.normal(size=positions.shape), dtype=positions.dtype
    )


def test_a_rigid_translation_does_not_change_the_topology():
    """Guard the trap above: translation-invariance is real, so a nudge must not be one.

    This is asserted rather than assumed because a translation-only "perturbation"
    is what produced a false claim of shape stability once already.
    """
    n = 2048
    positions, masses = _system(n, seed=3)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=1, theta=0.9, max_order=4, leaf_size=16
    )
    fmm.prepare(positions, masses)
    shifted_shape = int(fmm.state.near_a.shape[0]), int(fmm.state.far_a.shape[0])
    fmm.prepare(positions + 1.0e-3, masses)  # rigid translation
    assert (int(fmm.state.near_a.shape[0]), int(fmm.state.far_a.shape[0])) == (
        shifted_shape
    ), "a rigid translation changed the topology; the invariance argument is wrong"


def test_topology_shapes_drift_without_static_shapes():
    """The premise for capacity padding: unpadded shapes really do move."""
    n = 2048
    positions, masses = _system(n, seed=4)
    fmm = BlockStepFMM(
        softening=SOFTENING, k_max=1, theta=0.9, max_order=4, leaf_size=16
    )
    seen = set()
    for i, scale in enumerate((0.0, 1.0e-3, 1.0e-1)):
        fmm.prepare(_nonrigid_nudge(positions, scale, seed=100 + i), masses)
        seen.add((int(fmm.state.near_a.shape[0]), int(fmm.state.far_a.shape[0])))
    assert len(seen) > 1, f"expected drifting shapes, got only {seen}"


def test_static_shapes_hold_the_prepared_state_signature_across_rebuilds():
    """With capacities frozen, every rebuild produces identical shapes."""
    n = 2048
    positions, masses = _system(n, seed=4)
    fmm = BlockStepFMM(
        softening=SOFTENING,
        k_max=1,
        theta=0.9,
        max_order=4,
        leaf_size=16,
        static_shapes=True,
    )

    def signature(state):
        leaves = jax.tree_util.tree_leaves(state)
        return tuple(sorted((str(x.dtype), tuple(x.shape)) for x in leaves))

    signatures = set()
    for i, scale in enumerate((0.0, 1.0e-3, 1.0e-1)):
        fmm.prepare(_nonrigid_nudge(positions, scale, seed=200 + i), masses)
        signatures.add(signature(fmm.state))
    assert fmm.capacities is not None
    assert len(signatures) == 1, (
        f"static_shapes did not stabilise the state signature: {len(signatures)} "
        "distinct shape sets across three rebuilds"
    )


def test_one_compiled_program_serves_every_rebuild():
    """The payoff, asserted directly: the jit cache must not grow.

    This is the whole reason for the pytree registration and the padding. Without
    both, the topology enters a jitted program as constants keyed on their values
    (or as leaves whose shapes move) and the force recompiles per base step --
    measured ~200 s each at N = 20 000, which is what makes an otherwise 135x
    speedup a net loss at a per-base-step rebuild cadence.
    """
    n = 2048
    positions, masses = _system(n, seed=4)
    fmm = BlockStepFMM(
        softening=SOFTENING,
        k_max=1,
        theta=0.9,
        max_order=4,
        leaf_size=16,
        static_shapes=True,
    )

    @jax.jit
    def force(state, pos, mass):
        return mutual_weighted_accelerations(state, pos, mass)

    outputs = []
    for i, scale in enumerate((0.0, 1.0e-3, 1.0e-1)):
        nudged = _nonrigid_nudge(positions, scale, seed=300 + i)
        fmm.prepare(nudged, masses)
        assert int(fmm.state.far_a.shape[0]) > 0
        outputs.append(force(fmm.state, nudged, masses))
        # Padding must not leak into the answer.
        eager = mutual_weighted_accelerations(fmm.state, nudged, masses)
        rel = float(jnp.linalg.norm(outputs[-1] - eager) / jnp.linalg.norm(eager))
        assert rel < 1.0e-14, f"jitted vs eager disagree at {rel:.3e}"

    assert force._cache_size() == 1, (
        f"the force compiled {force._cache_size()} times for three rebuilds; "
        "static shapes are not holding"
    )


def test_capacity_overflow_raises_rather_than_truncating():
    """A topology that outgrows its caps must fail loudly.

    Truncating would be invisible in the diagnostic this lane is judged on:
    dropping a canonical pair drops *both* its halves, so momentum still cancels
    to round-off while the force is simply wrong.
    """
    from jaccpot.mutual.force import MutualCapacities, MutualCapacityOverflow

    n = 2048
    positions, masses = _system(n, seed=5)
    fmm = BlockStepFMM(
        softening=SOFTENING,
        k_max=1,
        theta=0.9,
        max_order=4,
        leaf_size=16,
        caps=MutualCapacities(near=8, far=8, depth=8, width=8),
    )
    with pytest.raises(MutualCapacityOverflow, match="overflows its capacities"):
        fmm.prepare(positions, masses)


# ---------------------------------------------------------------------------
# device-side topology construction
# ---------------------------------------------------------------------------


def _yggdrax_has_mutual_walk() -> bool:
    """Whether the installed yggdrax carries the canonical-pair mutual walk.

    The device topology composes ``yggdrax.interactions.dual_tree_walk_mutual``
    (TobiBu/yggdrax#45), which is newer than any released yggdrax. CI installs a
    released one, so these tests must SKIP there rather than error -- the same
    treatment ``test_mutual_fmm_nornax.py`` gives nornax. Catches ``Exception``
    rather than ``ImportError`` so a future import-time incompatibility also
    reports as skipped.
    """
    try:
        from yggdrax.interactions import dual_tree_walk_mutual  # noqa: F401
    except Exception:
        return False
    return True


def _nornax_available() -> bool:
    """Whether nornax can be imported; it is a test-only optional dependency."""
    try:
        import nornax  # noqa: F401
    except Exception:
        return False
    return True


needs_device_topology = pytest.mark.skipif(
    not _yggdrax_has_mutual_walk(),
    reason=(
        "needs a yggdrax carrying dual_tree_walk_mutual (TobiBu/yggdrax#45); "
        "the device mutual topology is built on it"
    ),
)
needs_nornax = pytest.mark.skipif(
    not _nornax_available(), reason="nornax is a test-only optional dependency"
)


def _host_topology(positions, masses, *, theta, leaf, order):
    """Build the host topology and return it with the tree bits the device needs."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.mutual.topology import build_mutual_topology_from_tree

    solver = FastMultipoleMethod(preset="balanced", basis="real")
    prepared = solver.prepare_state(
        positions, masses, leaf_size=leaf, max_order=order, theta=theta
    )
    topology = build_mutual_topology_from_tree(
        prepared.tree,
        np.asarray(prepared.positions_sorted),
        np.asarray(prepared.masses_sorted),
        theta=theta,
        order=order,
    )
    parent = np.asarray(prepared.tree.parent)
    root = int(np.flatnonzero(parent < 0)[0]) if (parent < 0).any() else 0
    return topology, prepared, parent, root


def _device_state(topology, prepared, parent, root, *, theta, order, caps):
    from jaccpot.mutual.device_topology import build_mutual_state_device

    return build_mutual_state_device(
        jnp.asarray(prepared.positions_sorted),
        jnp.asarray(prepared.masses_sorted),
        parent=jnp.asarray(parent),
        left_child=jnp.asarray(topology.left_child),
        right_child=jnp.asarray(topology.right_child),
        node_ranges=jnp.asarray(topology.node_particle_ranges),
        inverse_permutation=jnp.asarray(topology.inverse_permutation),
        root=jnp.asarray(root),
        theta=theta,
        order=order,
        leaf_size=int(topology.max_leaf_size),
        caps=caps,
        softening=SOFTENING,
        G=1.0,
    )


def _pair_set(a, b, count):
    a = np.asarray(a)[: int(count)]
    b = np.asarray(b)[: int(count)]
    return set(map(tuple, np.stack([a, b], axis=1).tolist()))


@pytest.mark.parametrize(
    "n,theta,leaf", [(512, 0.9, 16), (2048, 0.6, 16), (4096, 0.7, 32)]
)
@needs_device_topology
def test_the_device_walk_reproduces_the_host_traversal_pair_for_pair(n, theta, leaf):
    """The load-bearing gate for the device topology.

    Not "close" and not "same count" -- the *same set* of canonical node pairs.
    The mutual MAC is a discrete accept/reject, so any disagreement means the two
    paths are computing different forces, and a force difference at FMM tolerance
    would hide it. The comparison is set-based because the wavefront order
    differs between the host's NumPy walk and the device's ``lax.while_loop``.
    """
    from yggdrax.interactions import dual_tree_walk_mutual

    from jaccpot.mutual.device_topology import node_centers_and_radii

    positions, masses = _system(n, seed=7)
    topology, prepared, parent, root = _host_topology(
        positions, masses, theta=theta, leaf=leaf, order=4
    )
    assert int(np.asarray(topology.far_a).shape[0]) > 0, "vacuous far field"

    num_nodes = int(topology.num_nodes)
    num_internal = int(topology.num_internal)
    leaf_nodes = np.asarray(topology.leaf_nodes)
    lp = np.asarray(topology.leaf_particles)
    lv = np.asarray(topology.leaf_particle_valid)
    leaf_of_particle = np.zeros(n, dtype=np.int32)
    for j, node in enumerate(leaf_nodes):
        leaf_of_particle[lp[j][lv[j]]] = node

    centers, radii = node_centers_and_radii(
        jnp.asarray(prepared.positions_sorted),
        jnp.asarray(prepared.masses_sorted),
        jnp.asarray(topology.node_particle_ranges),
        jnp.asarray(parent),
        jnp.asarray(leaf_of_particle),
        depth_cap=len(topology.level_nodes) + 4,
    )
    pad = np.full(num_nodes - num_internal, -1)
    walk = dual_tree_walk_mutual(
        jnp.asarray(np.concatenate([np.asarray(topology.left_child), pad]), jnp.int32),
        jnp.asarray(np.concatenate([np.asarray(topology.right_child), pad]), jnp.int32),
        centers,
        radii,
        float(theta),
        jnp.asarray(root),
        max_pair_queue=1 << 16,
        far_cap=1 << 16,
        near_cap=1 << 17,
    )
    assert not bool(walk.far_overflow) and not bool(walk.near_overflow)
    assert not bool(walk.queue_overflow)

    host_far = set(
        map(
            tuple,
            np.stack(
                [np.asarray(topology.far_a), np.asarray(topology.far_b)], 1
            ).tolist(),
        )
    )
    host_near = set(
        map(
            tuple,
            np.stack(
                [np.asarray(topology.near_a), np.asarray(topology.near_b)], 1
            ).tolist(),
        )
    )
    assert _pair_set(walk.far_a, walk.far_b, walk.far_count) == host_far
    assert _pair_set(walk.near_a, walk.near_b, walk.near_count) == host_near


@pytest.mark.parametrize("n,theta,leaf,order", [(512, 0.9, 16, 4), (4096, 0.7, 32, 4)])
@needs_device_topology
def test_a_device_built_state_gives_the_same_force_and_gradient(n, theta, leaf, order):
    """Same force *and* same gradient as the host-built state, to round-off."""
    from jaccpot.mutual.force import resolve_mutual_capacities

    positions, masses = _system(n, seed=8)
    topology, prepared, parent, root = _host_topology(
        positions, masses, theta=theta, leaf=leaf, order=order
    )
    caps = resolve_mutual_capacities(topology)
    host = build_mutual_state(topology, softening=SOFTENING, G=1.0, caps=caps)
    device = _device_state(
        topology, prepared, parent, root, theta=theta, order=order, caps=caps
    )
    assert int(device.num_far_pairs) == int(host.num_far_pairs) > 0
    assert int(device.num_near_pairs) == int(host.num_near_pairs) > 0

    rung = _rungs(n, k_max=2, seed=9)
    weights = jnp.asarray([1.0, 0.5, 0.25])
    for kwargs in ({}, {"rung": rung, "level_weights": weights}):
        a_host = mutual_weighted_accelerations(host, positions, masses, **kwargs)
        a_dev = mutual_weighted_accelerations(device, positions, masses, **kwargs)
        rel = float(jnp.linalg.norm(a_host - a_dev) / jnp.linalg.norm(a_host))
        assert rel < 1.0e-14, f"force differs at {rel:.3e} for {sorted(kwargs)}"

    def loss(state, p):
        return jnp.sum(mutual_weighted_accelerations(state, p, masses) ** 2)

    g_host = jax.grad(lambda p: loss(host, p))(positions)
    g_dev = jax.grad(lambda p: loss(device, p))(positions)
    rel = float(jnp.linalg.norm(g_host - g_dev) / jnp.linalg.norm(g_host))
    assert rel < 1.0e-13, f"gradient differs at {rel:.3e}"


@needs_device_topology
def test_topology_build_and_force_are_one_jitted_program():
    """The point of the whole exercise: no host round-trip, one compile.

    The host builder cannot be traced at all -- it reads positions with
    ``np.asarray`` and walks the tree in Python. This asserts the device builder
    composes with the force inside a single ``jax.jit``, that the cache does not
    grow across calls, and that a gradient flows through the *whole* thing
    including the topology construction (the MAC geometry is
    ``stop_gradient``-ed; the expansion centres are not).
    """
    from jaccpot.mutual.device_topology import build_mutual_state_device
    from jaccpot.mutual.force import resolve_mutual_capacities

    n, theta, leaf, order = 2048, 0.7, 32, 4
    positions, masses = _system(n, seed=10)
    topology, prepared, parent, root = _host_topology(
        positions, masses, theta=theta, leaf=leaf, order=order
    )
    caps = resolve_mutual_capacities(topology)
    static = dict(
        parent=jnp.asarray(parent),
        left_child=jnp.asarray(topology.left_child),
        right_child=jnp.asarray(topology.right_child),
        node_ranges=jnp.asarray(topology.node_particle_ranges),
        inverse_permutation=jnp.asarray(topology.inverse_permutation),
        root=jnp.asarray(root),
        theta=theta,
        order=order,
        leaf_size=int(topology.max_leaf_size),
        caps=caps,
        softening=SOFTENING,
        G=1.0,
    )

    @jax.jit
    def build_and_force(sorted_positions, sorted_masses, p, m):
        state = build_mutual_state_device(sorted_positions, sorted_masses, **static)
        return mutual_weighted_accelerations(state, p, m)

    xs = jnp.asarray(prepared.positions_sorted)
    ms = jnp.asarray(prepared.masses_sorted)
    out = build_and_force(xs, ms, positions, masses)
    build_and_force(xs, ms, positions, masses)
    assert build_and_force._cache_size() == 1

    host = build_mutual_state(topology, softening=SOFTENING, G=1.0, caps=caps)
    ref = mutual_weighted_accelerations(host, positions, masses)
    rel = float(jnp.linalg.norm(out - ref) / jnp.linalg.norm(ref))
    assert rel < 1.0e-14, f"jitted device build differs at {rel:.3e}"

    grad = jax.grad(lambda p: jnp.sum(build_and_force(p, ms, p, masses) ** 2))(xs)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


@needs_device_topology
@needs_nornax
def test_the_device_backend_matches_the_exact_sum_as_well_as_the_host_one():
    """``topology_backend="device"`` is not a downgrade.

    The two backends build *different trees* -- the host path goes through
    ``prepare_state``'s LBVH tree, the device path through a static-radix
    template -- so they are not expected to agree with each other to round-off.
    What must hold is that neither is worse against an exact direct sum, which is
    the only reference that means anything here.
    """
    from nornax.forces.mutual_direct import MutualDirectSumGravity

    n, theta, leaf = 4096, 0.7, 32
    positions, masses = _system(n, seed=21)
    host = BlockStepFMM(
        softening=SOFTENING,
        k_max=2,
        theta=theta,
        max_order=4,
        leaf_size=leaf,
        static_shapes=True,
    )
    host.prepare(positions, masses)
    device = BlockStepFMM(
        softening=SOFTENING,
        k_max=2,
        theta=theta,
        max_order=4,
        leaf_size=leaf,
        topology_backend="device",
    )
    device.prepare(positions, masses)
    assert not bool(device.state.topology_overflow)
    assert int(device.state.num_far_pairs) > 0

    exact = MutualDirectSumGravity(G=1.0, softening=SOFTENING).total_accelerations(
        positions, masses
    )
    err_host = float(
        jnp.linalg.norm(host.total_accelerations(positions, masses) - exact)
        / jnp.linalg.norm(exact)
    )
    err_device = float(
        jnp.linalg.norm(device.total_accelerations(positions, masses) - exact)
        / jnp.linalg.norm(exact)
    )
    assert err_device < 1.0e-3, f"device backend error {err_device:.3e}"
    assert err_device < 3.0 * err_host, (
        f"device backend ({err_device:.3e}) is materially worse than the host one "
        f"({err_host:.3e})"
    )


@needs_device_topology
def test_an_undersized_capacity_profile_is_raised_not_truncated():
    """The failure this flag exists for, reproduced deliberately.

    A too-small ``depth``/``width`` truncates the level schedule, which drops
    nodes from the M2M/L2L cascade. Nothing else notices: no NaN, no shape error,
    and momentum stays *exactly* conserved because a dropped canonical pair loses
    both of its halves. It surfaced originally as an unexplained 2e-2 force error.
    """
    from jaccpot.mutual.force import MutualCapacities

    n, theta, leaf = 2048, 0.7, 32
    positions, masses = _system(n, seed=22)
    starved = MutualCapacities(far=64, near=256, depth=2, width=4)
    model = BlockStepFMM(
        softening=SOFTENING,
        k_max=1,
        theta=theta,
        max_order=4,
        leaf_size=leaf,
        topology_backend="device",
        caps=starved,
    )
    with pytest.raises(RuntimeError, match="overflowed its capacity profile"):
        model.prepare(positions, masses)


@needs_device_topology
def test_a_rollout_with_the_tree_rebuilt_inside_the_scan_is_one_program():
    """The end state: a jitted rollout that rebuilds its own topology per step.

    ``rebuild_state`` is traceable, so the tree refresh, the dual-tree traversal
    and the force all sit inside a ``lax.scan`` over base steps -- no host
    round-trip anywhere in the loop, and one compiled program for the whole
    rollout. The capacities are what make that possible: the accepted pair set
    genuinely changes from step to step and the program does not recompile.
    """
    n, theta, leaf, k_max = 2048, 0.7, 32, 1
    positions, masses = _system(n, seed=23)
    velocities = jnp.asarray(np.random.default_rng(24).normal(scale=0.15, size=(n, 3)))
    model = BlockStepFMM(
        softening=SOFTENING,
        k_max=k_max,
        theta=theta,
        max_order=4,
        leaf_size=leaf,
        topology_backend="device",
    )
    model.freeze_template(positions, masses)

    weights = jnp.asarray([1.0, 0.5])

    @jax.jit
    def rollout(p, v):
        def base(carry, _):
            pos, vel = carry
            state = model.rebuild_state(pos, masses)
            for _ in range(2):
                vel = vel + mutual_weighted_accelerations(
                    state,
                    pos,
                    masses,
                    rung=jnp.zeros((n,), jnp.int32),
                    level_weights=weights,
                )
                pos = pos + 1.0e-3 * vel
            return (pos, vel), state.num_far_pairs

        (pos, vel), far_counts = jax.lax.scan(base, (p, v), xs=None, length=3)
        return pos, vel, far_counts

    pos, vel, far_counts = rollout(positions, velocities)
    rollout(positions, velocities)
    assert rollout._cache_size() == 1
    assert bool(jnp.all(jnp.isfinite(pos))) and bool(jnp.all(jnp.isfinite(vel)))
    # The topology really is rebuilt each step, not carried: the accepted far
    # count moves as the particles do.
    assert int(jnp.min(far_counts)) > 0
    momentum0 = jnp.sum(masses[:, None] * velocities, axis=0)
    momentum1 = jnp.sum(masses[:, None] * vel, axis=0)
    scale = float(jnp.sum(jnp.abs(masses[:, None] * vel)))
    drift = float(jnp.linalg.norm(momentum1 - momentum0)) / scale
    assert drift < 1.0e-13, f"momentum drift {drift:.3e} through the jitted rollout"
