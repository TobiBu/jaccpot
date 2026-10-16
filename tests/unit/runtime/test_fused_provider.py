"""``FusedForceProvider``: the fused lane's per-step force as an object (S2, C1).

The provider names the two halves of a fused-scan step -- ``rebuild`` (tree, P2M/M2M,
walk, M2L, L2L) and ``accelerations`` (near P2P, L2P, scatter) -- and must compute
exactly what the lane computes today:

* its evaluation IS the eval seam's (``strict_fused_prepared_eval_fn``): the same
  prepared state, the same program text, the same bits;
* a rebuild reads only the shapes of the state it is given (jax's dead-code
  elimination keeps none of its leaves), so a caller can carry a template;
* the rebuild's capacity verdict and walk needs move into the topology, and the
  engine's side channels are left empty (no tracer outlives the trace);
* the overflow bits agree with the scan's verdict, a cell-leaf overflow included.

CPU, N = 512, the lane goldens' operating point on the production leaf partition
(cells), with ``jax.default_backend`` reporting ``"gpu"`` as
``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.fused_topology import NEEDS_FIELDS

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING = 1e-2

#: the fused lane's switches and sizing knobs: unset, the library defaults run
_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
)
#: the CSR near-field lane (production's) on CPU: Pallas interpret mode
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(scope="module")
def lane():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        for key in _ENV_UNSET:
            mp.delenv(key, raising=False)
        yield mp


def _solver(*, leaf_capacity: int = 1024, use_pallas: bool = False):
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _blobs():
    """Four tight blobs in [-1, 1]^3 (provider_hlo_identity.py's CPU draw)."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3))
    pos = np.clip(pos, -1.0, 1.0)
    mass = np.abs(rng.normal(size=N)) + 0.5
    return jnp.asarray(pos, jnp.float32), jnp.asarray(mass, jnp.float32)


def _plummer():
    """A clipped Plummer sphere: more far and near pairs than the blobs."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 20.0)
    mu = rng.uniform(-1.0, 1.0, N)
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return jnp.asarray(pos, jnp.float32)


def _probe(prov):
    """One jitted step: rebuild, evaluate, both verdicts and the walk's needs."""

    @jax.jit
    def run(pos, masses):
        topo = prov.rebuild(pos, masses)
        return (
            prov.accelerations(topo),
            prov.capacity_ok(topo),
            topo.needs(),
            topo.overflow_bits(),
        )

    return run


@pytest.fixture(scope="module")
def base(lane):
    from jaccpot import FusedForceProvider

    pos, masses = _blobs()
    solver = _solver()
    prov, topo0 = FusedForceProvider.prepare(
        solver, pos, masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    return dict(
        solver=solver,
        pos=pos,
        masses=masses,
        prov=prov,
        topo0=topo0,
        probe=_probe(prov),
    )


@pytest.mark.usefixtures("lane")
def test_the_provider_prepares_and_evaluates_as_the_eval_seam():
    """Its prepare is the seam's prepare; its evaluation, the seam's program."""
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_topology import FusedTopology

    pos, masses = _blobs()
    seam = _solver()
    prepared, eval_fn = seam.strict_fused_prepared_eval_fn(
        positions=pos, masses=masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    prov, topo = FusedForceProvider.prepare(
        _solver(), pos, masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    ref_leaves, ref_def = jax.tree_util.tree_flatten(prepared)
    got_leaves, got_def = jax.tree_util.tree_flatten(topo.prepared)
    assert got_def == ref_def
    assert all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(got_leaves, ref_leaves)
    )

    def _eval(prepared_in):
        return jnp.asarray(prov.accelerations(FusedTopology.from_prepared(prepared_in)))

    # the same program: equal jaxprs (``lower`` cannot run under the runtime type
    # checker, which sees jax's abstract leaves in the payload's typed fields)
    assert str(jax.make_jaxpr(_eval)(topo.prepared)) == str(
        jax.make_jaxpr(eval_fn.__wrapped__)(prepared)
    )
    got = jax.jit(_eval)(topo.prepared)
    assert np.array_equal(np.asarray(got), np.asarray(eval_fn(prepared)))


def test_a_rebuild_reads_only_the_shapes_of_the_state_it_is_given(base):
    """The premise of carrying a template: no leaf of ``like`` reaches an output."""
    from dataclasses import replace

    from jax._src.interpreters import partial_eval as pe

    prov = base["prov"]
    like = replace(base["topo0"].prepared, compact_far_pairs=None)

    def step(like_in, pos, masses):
        topo = prov._rebuild_like(like_in, pos, masses)
        return (
            prov.accelerations(topo),
            prov.capacity_ok(topo),
            topo.needs(),
            topo.overflow_bits(),
        )

    closed = jax.make_jaxpr(step)(like, base["pos"], base["masses"])
    n_like = len(jax.tree_util.tree_leaves(like))
    _, used = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    assert n_like > 20  # non-vacuous: the state is a big pytree
    assert not any(used[:n_like])
    assert all(used[n_like:])  # positions and masses are read


def test_the_rebuild_moves_its_side_channels_into_the_topology(base):
    prov = base["prov"]
    impl = base["solver"]._impl
    seen = {}

    def step(pos, masses):
        topo = prov.rebuild(pos, masses)
        seen["topo"] = topo
        seen["after"] = (
            impl._last_refresh_capacity_ok,
            impl._last_refresh_walk_needs,
            impl._last_refresh_tree_overflow,
            impl._last_refresh_tree_needs,
        )
        return prov.accelerations(topo)

    jax.make_jaxpr(step)(base["pos"], base["masses"])
    topo = seen["topo"]
    # the flat walk reported its needs, the rebuild its far-list verdict, the cell
    # partition its overflow flag and what its tree needed -- all on the topology,
    # as this trace's tracers
    assert topo.fresh
    for value in (
        topo.refresh_ok,
        topo.walk_needs,
        topo.tree_overflow,
        topo.tree_needs,
    ):
        assert isinstance(value, jax.core.Tracer)
    # and none of them is left on the engine
    assert seen["after"] == (None, None, None, None)


def test_a_rebuild_at_the_prepare_positions_is_the_prepares_force(base):
    prov, topo0 = base["prov"], base["topo0"]
    eager = np.asarray(jax.jit(lambda t: prov.accelerations(t))(topo0))
    acc, ok, needs, bits = base["probe"](base["pos"], base["masses"])
    assert np.array_equal(np.asarray(acc), eager)
    assert bool(ok) and int(bits) == 0
    assert int(needs[0]) > 0 and int(needs[1]) > 0  # the walk reported counts
    # the tree's needs are the prepare's own tree: its live leaves, depth and
    # widest level (the same partition at the same positions)
    tree = topo0.prepared.tree
    ranges = np.asarray(tree.node_ranges)[int(tree.left_child.shape[0]) :]
    offsets = np.asarray(tree.level_offsets)
    want = {
        "leaf_needed": int(np.sum(ranges[:, 1] >= ranges[:, 0])),
        "depth_needed": int(np.max(np.asarray(tree.node_level))) + 1,
        "widest_needed": int(np.max(np.diff(offsets))),
    }
    got = {k: int(needs[NEEDS_FIELDS.index(k)]) for k in want}
    assert got == want
    assert int(needs[NEEDS_FIELDS.index("tree_overflow")]) == 0
    # one step is the same rebuild and evaluation, its aux the same verdict
    acc_s, (f_b, bits_s, needs_s) = jax.jit(prov.step)(base["pos"], base["masses"])
    assert f_b is None  # no criterion, no force scale
    assert np.array_equal(np.asarray(acc_s), eager)
    assert int(bits_s) == 0
    assert np.array_equal(np.asarray(needs_s), np.asarray(needs))


def test_the_overflow_bits_name_what_the_verdict_refuses(base):
    """Caps sized on the blobs, rebuilt on a Plummer sphere: the walk overflows."""
    from jaccpot.runtime import fused_topology as ft

    _, ok, needs, bits = base["probe"](_plummer(), base["masses"])
    bits = int(bits)
    assert not bool(ok)
    assert not bool(ft.overflow_ok(jnp.asarray(bits, jnp.uint32)))
    assert bits & ft.OVERFLOW_SATURATED  # the flat walk saturates its far list
    walk = ft.OVERFLOW_WALK_FAR | ft.OVERFLOW_WALK_NEAR | ft.OVERFLOW_WALK_QUEUE
    assert bits & walk
    assert bool(needs[3] or needs[4] or needs[5])
    # the fitting rebuild: no bit, and the verdict agrees
    _, ok_fit, _, bits_fit = base["probe"](base["pos"], base["masses"])
    assert bool(ok_fit) and bool(ft.overflow_ok(bits_fit))


@pytest.fixture(scope="module")
def cell_overflow(lane):
    """Cell leaves sized on a uniform draw (64 live leaves), rebuilt on the blobs (105).

    On the CSR near-field lane (production's): the rectangle guard is off there,
    so nothing else catches the overflow.
    """
    from jaccpot import FusedForceProvider

    for key, val in _CSR_ON_CPU.items():
        lane.setenv(key, val)
    try:
        pos, masses = _blobs()
        uniform = jnp.asarray(
            np.random.default_rng(2).uniform(-1.0, 1.0, (N, 3)), jnp.float32
        )
        solver = _solver(leaf_capacity=66, use_pallas=True)
        prov, _ = FusedForceProvider.prepare(
            solver, uniform, masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
        )
        acc, ok, needs, bits = _probe(prov)(pos, masses)
        ref_prov, ref_topo = FusedForceProvider.prepare(
            _solver(use_pallas=True),
            pos,
            masses,
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
        )
        ref = jax.jit(lambda t: ref_prov.accelerations(t))(ref_topo)
        yield dict(acc=acc, ok=ok, needs=needs, bits=int(bits), ref=ref)
    finally:
        for key in _CSR_ON_CPU:
            lane.delenv(key, raising=False)


def test_a_cell_leaf_overflow_fails_the_overflow_bits(cell_overflow):
    from jaccpot.runtime import fused_topology as ft

    got, ref = np.asarray(cell_overflow["acc"]), np.asarray(cell_overflow["ref"])
    # non-vacuous: the overflowing rebuild's force is wrong
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) > 0.1
    assert cell_overflow["bits"] & ft.OVERFLOW_TREE
    assert not bool(ft.overflow_ok(jnp.asarray(cell_overflow["bits"], jnp.uint32)))


def test_the_scans_verdict_sees_a_cell_leaf_overflow(cell_overflow):
    """The provider's verdict folds the tree's own flag: the traced refresh never
    lets the walk see a cell-leaf overflow, so its far list does not saturate and
    nothing else on the CSR lane fails. Missed, the verdict was True (S2 C1)."""
    assert not bool(cell_overflow["ok"])
    needs = cell_overflow["needs"]
    # named in the segment's needs, with the leaves the blobs need: the count is
    # the partition's before its cut to the capacity (66), what a re-plan sizes from
    assert int(needs[NEEDS_FIELDS.index("tree_overflow")]) == 1
    assert int(needs[NEEDS_FIELDS.index("leaf_needed")]) == 105


def test_the_key_holds_the_walk_caps_and_the_template(base):
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_provider import _walk_caps_key

    prov, impl = base["prov"], base["solver"]._impl
    key = prov.key()
    assert key[-1] == prov.template.key()
    assert _walk_caps_key(impl._strict_fused_validated_caps) in key
    # a provider frozen from the same state keys the same programs
    again = FusedForceProvider.freeze_template(
        base["solver"],
        base["topo0"].prepared,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    assert again.key() == key
    # rebuilding from a given state (the state carry) leaves the template out
    assert prov.key(like=base["topo0"].prepared) == key[:-1]
    # a re-plan that grew the walk caps keys a new program
    saved = dict(impl._strict_fused_validated_caps)
    try:
        grown = dict(saved)
        grown["compact_far_pair_capacity"] = 2 * int(grown["compact_far_pair_capacity"])
        impl._strict_fused_validated_caps = grown
        assert prov.key() != key
    finally:
        impl._strict_fused_validated_caps = saved


def test_force_parameters_resolve_to_literals_or_working_dtype_arrays(base):
    """The engine's own floats are the default program (``None``); other floats stay
    literals; arrays are cast to the working dtype (S2 C4,
    ``tests/integration/strict/test_traced_params.py`` runs them)."""
    from jaccpot.runtime.fused_topology import ForceParams, StepParams

    prov, topo0 = base["prov"], base["topo0"]
    own = prov.params()
    assert own == ForceParams(G=1.0, eps2=SOFTENING**2)
    assert prov.resolve_params(own) is None
    assert prov.resolve_params(StepParams(own, 1e-3)) is None
    other = prov.resolve_params(ForceParams(G=2.0, eps2=own.eps2))
    assert other == ForceParams(G=2.0, eps2=own.eps2)
    traced = prov.resolve_params(
        ForceParams(G=jnp.asarray(1.0, jnp.float64), eps2=own.eps2)
    )
    assert traced.G.dtype == jnp.float32 and traced.eps2 == own.eps2
    shape = jax.eval_shape(lambda t: prov.accelerations(t, own), topo0)
    assert shape.shape == (N, 3)
    shape = jax.eval_shape(lambda t, p: prov.accelerations(t, p), topo0, traced)
    assert shape.shape == (N, 3)
    with pytest.raises(ValueError, match="order"):
        prov.accelerations(topo0, order="morton")


def test_a_criterion_rebuild_needs_the_previous_force_scale():
    from jaccpot.runtime.fused_provider import FusedForceProvider

    class _CriterionEngine:
        def _flat_walk_criterion_active(self) -> bool:
            return True

    prov = FusedForceProvider(
        _CriterionEngine(), leaf_size=LEAF, max_order=ORDER, template=None
    )
    with pytest.raises(ValueError, match="force scale"):
        prov.rebuild(jnp.zeros((N, 3)), jnp.ones((N,)))


def test_a_template_rebuild_refuses_bucket_leaves():
    """Bucket leaves keep the eager tree's node ranges and children: values a shape
    template does not hold (``require_cell_leaves``)."""
    from jaccpot.runtime.fused_provider import FusedForceProvider
    from jaccpot.runtime.strict_carry import shape_template

    class _BucketEngine:
        _tree_leaf_partition = "buckets"

        def _flat_walk_criterion_active(self) -> bool:
            return False

    prov = FusedForceProvider(
        _BucketEngine(),
        leaf_size=LEAF,
        max_order=ORDER,
        template=shape_template({"x": jnp.zeros((4,))}),
    )
    with pytest.raises(ValueError, match="needs cell leaves"):
        prov.rebuild(jnp.zeros((N, 3)), jnp.ones((N,)))
