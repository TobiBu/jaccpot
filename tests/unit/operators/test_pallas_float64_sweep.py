"""The fused lane's Pallas kernels at float64, against their float64 XLA lanes.

The large-N lane's float64 force (``precision="fp64"`` with ``preset="large_n_gpu"``)
is a validation lane: it runs the fused lane's Pallas kernels in float64 wherever
they are dtype-generic. Each case here runs one kernel the lane launches at float64
-- ``[interpret]`` on CPU, ``[native]`` (the Triton lowering, an Ampere+ card only)
-- against the float64 XLA code the lane runs without Pallas, and asserts the
largest difference is at most ``1e-12`` of the reference's largest value: float64
round-off of a few hundred operations in another order, 1e4x below anything a
float32 operation in the chain would leave (``6e-8``).

====================  ===============================  ============================
kernel                float64 reference                interpret case
====================  ===============================  ============================
walk (``soa`` nodes)  ``dual_tree_walk_mutual``        here (theta MAC + floor);
                      (the same pair sets)             eq (16a): test_mutual_walk_pallas
COM radii table       the XLA level passes             test_mac_geometry_com
P2M leaf (blocked)    ``_p2m_leaves_real``             test_p2m_real_leaf
M2M / L2L lanes       the level loops                  here
M2L CSR lanes         ``m2l_real_csr_jax``             here
L2P particle-major    the XLA ``vmap``                 test_l2p_particle_major
CSR near, table       ``nearfield_leafpair_csr_jax``   here (``"input"``, ``"wide"``)
CSR near, direct      ``nearfield_leafpair_csr_jax``   here (+ potential, force scale)
====================  ===============================  ============================

Every kernel has a ``[native]`` case here; they are the only place the float64
Triton lowering is exercised, run under the ordinary GPU suite and skip on CPU.
Measured 2026-10-15 (A100, jax 0.11.2): every one lowers in float64 and lands within
6.4e-15 of its reference (the walk's pair sets equal), so the lane routes no kernel
to XLA for float64. ``csr_place`` is not here: it reads node ids only (its float32
cursors count the entries of a row, exact below ``2**24``), so the working dtype
never reaches it; ``test_near_csr_from_canonical.py`` pins it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import dual_tree_walk_mutual
from yggdrax.tree import Tree
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.pallas.m2l_real_csr import (
    m2l_real_csr_jax,
    pallas_m2l_real_csr_supported,
)

#: Largest difference allowed, as a fraction of the reference's largest value.
_TOL = 1e-12

_MODES = [
    pytest.param(True, id="interpret"),
    pytest.param(
        False,
        id="native",
        marks=pytest.mark.skipif(
            not pallas_m2l_real_csr_supported(),
            reason="the Triton lowering needs an Ampere+ (sm_80) GPU",
        ),
    ),
]
_NATIVE_ONLY = [_MODES[1]]
#: Expansion orders: the CPU goldens' 4 and production's 6. Not 12: two workers
#: running the interpreter's p = 12 cases of the M2M/L2L/M2L lanes held 25 and
#: 44 GB, and the Triton compile of the M2M lanes kernel grows with the order in
#: either dtype (A100 host, 2026-10-15: 140 s float32 / 136 s float64 at p = 8;
#: at p = 12 in float64 still compiling after 12 minutes).
_ORDER_CASES = [
    pytest.param(m.values[0], p, id=f"{m.id}-p{p}", marks=m.marks)
    for m in _MODES
    for p in (4, 6)
]
_NATIVE_ORDER_CASES = [c for c in _ORDER_CASES if c.id.startswith("native")]


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _assert_float64_close(got, ref, *, tol=_TOL):
    got, ref = np.asarray(got), np.asarray(ref)
    assert got.dtype == ref.dtype == np.float64
    assert got.shape == ref.shape
    scale = float(np.abs(ref).max())
    assert scale > 0.0, "vacuous: the reference is zero"
    err = float(np.abs(got - ref).max()) / scale
    assert err <= tol, f"max |got - ref| / max |ref| = {err:.3e} > {tol:g}"


@pytest.fixture(scope="module")
def tree_data():
    """A float64 static-radix tree (Plummer, leaf 16) and its COM centres."""
    n, leaf = 1500, 16
    P = jnp.asarray(_plummer(n, 1), jnp.float64)
    M = jnp.asarray(np.random.default_rng(2).uniform(0.5, 1.5, n), jnp.float64)
    tree = Tree.from_particles(
        P, M, tree_type="radix", build_mode="static_radix", leaf_size=leaf
    )
    topo = tree.topology
    ps = jnp.asarray(tree.positions_sorted, jnp.float64)
    ms = jnp.asarray(tree.masses_sorted, jnp.float64)
    com = jnp.asarray(
        compute_tree_mass_moments(topo, ps, ms).center_of_mass, jnp.float64
    )
    return topo, ps, ms, com, leaf


# ------------------------------------------------------------------ the walk


@pytest.mark.parametrize("interpret", _MODES)
@pytest.mark.parametrize("separation_floor", [0.0, 0.05])
def test_walk_soa_float64_lists_equal_the_flat_walk(
    tree_data, separation_floor, interpret
):
    """Float64 centres force the ``soa`` node layout; the opening-angle MAC (and
    the compact kernels' separation floor) give the flat walk's pair sets."""
    from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas
    from jaccpot.runtime._mac_geometry import com_mac_geometry

    topo, ps, _ms, com, leaf = tree_data
    geom = com_mac_geometry(topo, ps, com, leaf_cap=leaf)
    assert geom.center.dtype == geom.radius.dtype == jnp.float64
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    ranges = np.asarray(topo.node_ranges)
    active = jnp.asarray(ranges[:, 1] >= ranges[:, 0])
    root = jnp.argmin(topo.parent).astype(idx)
    kw = dict(
        max_pair_queue=1 << 14, far_cap=1 << 16, near_cap=1 << 16, node_active=active
    )
    ref = dual_tree_walk_mutual(
        left,
        right,
        geom.center,
        geom.radius,
        0.5,
        root,
        mac_type="dehnen",
        **({"separation_floor": separation_floor} if separation_floor else {}),
        **kw,
    )
    got = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        0.5,
        root,
        block=64,
        interpret=interpret,
        node_layout="record",  # overridden: float64 centres run the soa layout
        separation_floor=separation_floor,
        **kw,
    )
    for res in (ref, got):
        assert not (
            bool(res.queue_overflow)
            or bool(res.far_overflow)
            or bool(res.near_overflow)
        )
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"

    def _sets(res):
        nf, nn = int(res.far_count), int(res.near_count)
        far = zip(
            np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist()
        )
        near = zip(
            np.asarray(res.near_a)[:nn].tolist(),
            np.asarray(res.near_b)[:nn].tolist(),
        )
        return set(far), set(near)

    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert len(far_g) == int(got.far_count), "duplicate emission"
    assert far_g == far_r and near_g == near_r


# -------------------------------------------------------- upward / downward


@pytest.mark.parametrize("interpret", _NATIVE_ONLY)
def test_com_radii_table_float64_equals_the_level_passes(tree_data, interpret):
    """Native twin of ``test_mac_geometry_com``'s interpret case: same squared
    distances, maxima in any order, so a few ulp at most."""
    from yggdrax.tree import get_node_levels

    from jaccpot.runtime._mac_geometry import _com_radii

    topo, ps, _ms, com, leaf = tree_data
    args = (
        jnp.asarray(topo.node_ranges, jnp.int32),
        jnp.asarray(topo.left_child, jnp.int32),
        jnp.asarray(topo.right_child, jnp.int32),
        jnp.asarray(topo.parent, jnp.int32),
        ps,
        com,
    )
    depth = int(np.asarray(get_node_levels(topo)).max()) + 1
    kw = dict(leaf_cap=leaf, num_levels=depth)
    ref = _com_radii(*args, kernel="xla", **kw)
    got = _com_radii(*args, kernel="interpret" if interpret else "pallas", **kw)
    _assert_float64_close(got, ref)
    assert np.all(np.asarray(got) > 0)


@pytest.mark.parametrize("interpret, order", _NATIVE_ORDER_CASES)
def test_p2m_leaf_float64_equals_the_batched_reference(tree_data, order, interpret):
    """Native twin of ``test_p2m_real_leaf``'s float64 interpret case."""
    from jaccpot.pallas.p2m_real_leaf import p2m_real_leaves_pallas
    from jaccpot.upward.real_tree_expansions import _p2m_leaves_real

    topo, ps, ms, com, leaf = tree_data
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    ref = _p2m_leaves_real(
        topo.node_ranges,
        ps,
        ms,
        com,
        order=order,
        max_leaf_size=leaf,
        num_internal=ni,
        total_nodes=tot,
        leaf_batch_size=256,
    )
    got = p2m_real_leaves_pallas(
        ps,
        ms,
        com[ni:],
        topo.node_ranges[ni:],
        order=order,
        num_internal=ni,
        total_nodes=tot,
        leaf_width=leaf,
        interpret=interpret,
    )
    _assert_float64_close(got, ref)


def _level_args(topo):
    num_levels = int(jnp.max(topo.node_level)) + 1
    offs = topo.level_offsets
    width = int(jnp.max(offs[1:] - offs[:-1]))
    return num_levels, offs, width


@pytest.mark.parametrize("interpret, order", _ORDER_CASES)
def test_m2m_lanes_float64_equal_the_level_loop(tree_data, order, interpret):
    from jaccpot.pallas.cascade_real_lanes import m2m_real_levels_lanes_pallas
    from jaccpot.upward.real_tree_expansions import aggregate_m2m_real_by_level

    topo, _ps, _ms, com, _leaf = tree_data
    num_levels, offs, width = _level_args(topo)
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(4)
    leaves = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    leaves = leaves.at[:num_internal].set(0.0)
    kw = dict(
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=width,
    )
    args = (leaves, com, topo.left_child, topo.right_child, topo.nodes_by_level, offs)
    ref = aggregate_m2m_real_by_level(*args, **kw)
    got = m2m_real_levels_lanes_pallas(*args, interpret=interpret, **kw)
    _assert_float64_close(got, ref)
    assert np.abs(np.asarray(got)[0]).max() > 0  # the root received


@pytest.mark.parametrize("interpret, order", _ORDER_CASES)
def test_l2l_lanes_float64_equal_the_level_loop(tree_data, order, interpret):
    from jaccpot.pallas.cascade_real_lanes import l2l_real_levels_lanes_pallas
    from jaccpot.runtime.kernels._l2l import _propagate_solidfmm_locals_by_level

    topo, _ps, _ms, com, _leaf = tree_data
    num_levels, offs, width = _level_args(topo)
    total = int(topo.parent.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(5)
    locals0 = jnp.asarray(rng.standard_normal((total, C)), jnp.float64)
    ref = _propagate_solidfmm_locals_by_level(
        locals0 + 0.0,  # donated by its jit
        com,
        topo.left_child,
        topo.right_child,
        topo.node_level,
        order=order,
        total_nodes=total,
        num_levels=num_levels - 1,
    )
    got = l2l_real_levels_lanes_pallas(
        locals0,
        com,
        topo.parent,
        topo.nodes_by_level,
        offs,
        order=order,
        num_levels=num_levels,
        level_batch_width=width,
        interpret=interpret,
    )
    _assert_float64_close(got, ref)
    assert not np.allclose(np.asarray(got), np.asarray(locals0))


@pytest.mark.parametrize("interpret, order", _ORDER_CASES)
def test_m2l_csr_lanes_float64_equal_the_reference(order, interpret):
    """Ragged rows (an empty one, one of exactly ``k_lanes``, one past two tiles)
    with well-separated centres, as the walk's far pairs are."""
    from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas

    rng = np.random.default_rng(order)
    n, k_lanes = 40, 8
    C = (order + 1) ** 2
    rows = rng.integers(0, 20, n)
    rows[3], rows[5], rows[7] = 0, k_lanes, 2 * k_lanes + 1
    mult = jnp.asarray(rng.standard_normal((n, C)), jnp.float64)
    # centres on a coarse grid: every pair is at least 2 apart
    cent = jnp.asarray(
        2.0 * rng.permutation(np.stack(np.indices((4, 4, 4)), -1).reshape(-1, 3))[:n]
        + rng.uniform(-0.1, 0.1, (n, 3)),
        jnp.float64,
    )
    tgt = np.repeat(np.arange(n), rows)
    src = rng.integers(0, n, tgt.size)
    src = np.where(src == tgt, (src + 1) % n, src)
    src, tgt = jnp.asarray(src, jnp.int32), jnp.asarray(tgt, jnp.int32)
    ref = m2l_real_csr_jax(mult, cent, src, tgt, order=order)
    got = m2l_real_csr_lanes_pallas(
        mult, cent, src, tgt, order=order, k_lanes=k_lanes, interpret=interpret
    )
    _assert_float64_close(got, ref)
    assert np.all(np.asarray(got)[3] == 0.0)


@pytest.mark.parametrize("interpret, order", _NATIVE_ORDER_CASES)
def test_l2p_particle_major_float64_equals_the_xla_vmap(tree_data, order, interpret):
    """Native twin of ``test_l2p_particle_major``'s float64 interpret case."""
    from jaccpot.downward.local_expansions import LocalExpansionData
    from jaccpot.runtime.kernels._evaluate import (
        _evaluate_local_expansions_particle_major,
    )

    topo, ps, _ms, com, _leaf = tree_data
    total = int(topo.parent.shape[0])
    ni = int(topo.left_child.shape[0])
    C = (order + 1) ** 2
    rng = np.random.default_rng(6)
    # decaying coefficients, as a converged local expansion's are
    degree = np.floor(np.sqrt(np.arange(C)))
    coeff = rng.standard_normal((total, C)) * 0.5**degree
    local = LocalExpansionData(
        order=order, centers=com, coefficients=jnp.asarray(coeff, jnp.float64)
    )
    kw = dict(
        leaf_nodes=jnp.arange(ni, total, dtype=jnp.int32),
        node_ranges=jnp.asarray(topo.node_ranges, jnp.int32),
        order=order,
        chunk=256,
    )
    ref = _evaluate_local_expansions_particle_major(local, ps, kernel="xla", **kw)
    got = _evaluate_local_expansions_particle_major(
        local, ps, kernel="interpret" if interpret else "pallas", **kw
    )
    _assert_float64_close(got, ref)


# --------------------------------------------------------------- near field


def _near_case(seed, *, num_live, num_pad, W, n_dead, max_row):
    """Sorted particles in leaves of every occupancy (one full), empty padding
    leaves, dead rows past them, and neighbour rows that never hold their own leaf.
    Returns the particle-order arrays and their ``(L, W)`` tables."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, W + 1, size=num_live)
    counts[0] = W
    counts = np.concatenate([counts, np.zeros(num_pad, np.int64)])
    L = counts.size
    n_live = int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    starts[num_live:] = n_live
    pos = rng.standard_normal((n_live + n_dead, 3))
    mass = np.abs(rng.standard_normal(n_live + n_dead)) + 0.1
    slot = np.arange(W)[None, :]
    mask = slot < counts[:, None]
    ids = np.where(mask, starts[:, None] + slot, 0)
    row_counts = rng.integers(0, max_row + 1, size=L)
    row_counts[num_live:] = 0
    row_counts[2] = 0  # an empty row: the self term only
    rows = [
        rng.choice(np.setdiff1d(np.arange(num_live), [leaf]), size=row_counts[leaf])
        for leaf in range(L)
    ]
    nbr = np.concatenate([np.concatenate(rows), np.zeros(5)]).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(row_counts)]).astype(np.int32)
    f64 = jnp.float64
    return dict(
        pos=jnp.asarray(pos, f64),
        mass=jnp.asarray(mass, f64),
        leaf_pos=jnp.asarray(pos[ids], f64),
        leaf_mass=jnp.asarray(mass[ids], f64),
        mask=jnp.asarray(mask),
        starts=jnp.asarray(starts, jnp.int32),
        counts=jnp.asarray(counts, jnp.int32),
        nbr=jnp.asarray(nbr),
        offsets=jnp.asarray(offsets),
        row_counts=jnp.asarray(row_counts, jnp.int32),
        L=L,
        W=W,
    )


_SOFT_SQ = 0.05**2
_G = 1.3


@pytest.mark.parametrize("interpret", _MODES)
@pytest.mark.parametrize("accum", ["input", "wide"])
@pytest.mark.parametrize("softening_kernel", ["plummer", "ferrers3"])
def test_csr_table_kernel_float64_equals_the_dense_twin(
    softening_kernel, accum, interpret
):
    """``accum="wide"`` widens a float32 lane to float64; at float64 it is the
    input accumulator (``_resolve_accum_dtype``), so both give the reference."""
    from jaccpot.pallas.nearfield_leafpair_csr import (
        build_leafpair_chunk_table,
        leafpair_chunk_capacity,
        nearfield_leafpair_csr_jax,
        nearfield_leafpair_csr_pallas,
    )

    c = _near_case(11, num_live=9, num_pad=3, W=8, n_dead=0, max_row=7)
    chunk = 2
    cap = leafpair_chunk_capacity(int(c["nbr"].shape[0]), c["L"], chunk)
    tab = build_leafpair_chunk_table(
        c["offsets"], c["row_counts"], chunk=chunk, capacity=cap
    )
    common = dict(
        softening_sq=jnp.float64(_SOFT_SQ),
        G=jnp.float64(_G),
        softening_kernel=softening_kernel,
    )
    args = (c["leaf_pos"], c["leaf_mass"], c["mask"], c["nbr"])
    got = nearfield_leafpair_csr_pallas(
        *args, tab, chunk=chunk, interpret=interpret, accum=accum, **common
    )
    ref = nearfield_leafpair_csr_jax(*args, c["offsets"], c["row_counts"], **common)
    _assert_float64_close(got, ref)


def _force_scale_reference(c, softening_kernel):
    """Eq (16b)'s near force scale ``sum G m / (r^2 + eps^2)`` (Plummer) over the
    CSR rows and the own leaf, self excluded, in numpy float64."""
    assert softening_kernel == "plummer"
    pos, mass = np.asarray(c["pos"]), np.asarray(c["mass"])
    starts, counts = np.asarray(c["starts"]), np.asarray(c["counts"])
    off, rc, nbr = (np.asarray(c[k]) for k in ("offsets", "row_counts", "nbr"))
    out = np.zeros(pos.shape[0])
    for leaf in range(c["L"]):
        tgt = np.arange(starts[leaf], starts[leaf] + counts[leaf])
        srcs = [leaf, *nbr[off[leaf] : off[leaf] + rc[leaf]].tolist()]
        for s in srcs:
            src = np.arange(starts[s], starts[s] + counts[s])
            r2 = np.sum((pos[tgt, None] - pos[None, src]) ** 2, axis=-1)
            term = _G * mass[None, src] / (r2 + _SOFT_SQ)
            term[tgt[:, None] == src[None, :]] = 0.0
            out[tgt] += term.sum(axis=1)
    return out


@pytest.mark.parametrize("interpret", _MODES)
@pytest.mark.parametrize("fourth", ["potential", "force_scale"])
@pytest.mark.parametrize("source_tile", [None, 0])  # the default vector tiles, scalar
@pytest.mark.parametrize("softening_kernel", ["plummer", "ferrers3"])
def test_csr_direct_kernel_float64_equals_the_dense_twin(
    softening_kernel, source_tile, fourth, interpret
):
    """The production near-field kernel (sorted particle ranges in, particle order
    out) with each fourth lane: the potential against the dense twin, the
    ``dehnen_error`` force scale against a numpy sum (Plummer: the force scale is
    defined there)."""
    from jaccpot.pallas.nearfield_leafpair_csr import (
        nearfield_leafpair_csr_jax,
        nearfield_leafpair_csr_sorted_direct_pallas,
    )

    if fourth == "force_scale" and softening_kernel != "plummer":
        pytest.skip("the numpy force-scale reference is the Plummer one")
    c = _near_case(7, num_live=9, num_pad=3, W=16, n_dead=3, max_row=6)
    acc, lane4 = nearfield_leafpair_csr_sorted_direct_pallas(
        c["pos"],
        c["mass"],
        c["starts"],
        c["counts"],
        c["nbr"],
        c["offsets"],
        c["row_counts"],
        leaf_width=c["W"],
        softening_sq=jnp.float64(_SOFT_SQ),
        G=jnp.float64(_G),
        chunk=2,
        interpret=interpret,
        with_potential=fourth == "potential",
        with_force_scale=fourth == "force_scale",
        source_tile=source_tile,
        softening_kernel=softening_kernel,
    )
    table = np.asarray(
        nearfield_leafpair_csr_jax(
            c["leaf_pos"],
            c["leaf_mass"],
            c["mask"],
            c["nbr"],
            c["offsets"],
            c["row_counts"],
            softening_sq=jnp.float64(_SOFT_SQ),
            G=jnp.float64(_G),
            softening_kernel=softening_kernel,
        )
    )
    want = np.zeros((int(c["pos"].shape[0]), 4))
    starts, counts = np.asarray(c["starts"]), np.asarray(c["counts"])
    for leaf in range(c["L"]):
        want[starts[leaf] : starts[leaf] + counts[leaf]] = table[leaf, : counts[leaf]]
    _assert_float64_close(acc, want[:, :3])
    assert not np.any(np.asarray(acc)[-3:])  # the dead rows
    if fourth == "potential":
        _assert_float64_close(lane4, want[:, 3])
    else:
        _assert_float64_close(lane4, _force_scale_reference(c, softening_kernel))
