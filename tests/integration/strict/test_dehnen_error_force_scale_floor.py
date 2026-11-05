"""``mac_force_scale_floor`` on the fused lane: eq (16a) with a certified floor (#487).

At 1e8 particles under ``preset="galaxy"`` one leaf's near row held 70 % of all
leaves: eq (16a) accepts a pair only below ``eps * min_b f_b``, so a leaf whose
carried f_b is far too small accepts almost nothing and pairs with nearly every
leaf directly. The floor raises each node's ``min_b f_b`` to a fraction of a
bound that a correct estimate always meets
(:func:`~jaccpot.runtime._force_scale_levels.force_scale_floor_nodes`). Pinned
here, on CPU (the CSR near field in interpret mode, ``jax.default_backend``
reporting ``"gpu"`` as the lane goldens do):

* on the lane's own f_b the floor never binds: the near CSR, the far-pair count
  and the accelerations are the unfloored lane's, bit for bit, at the eager
  prepare and at a traced rebuild;
* a rebuild handed an f_b a billion times too small at the particles of a few
  leaves gives those leaves near rows of most of the tree; with the floor the
  rows stay within twice what the lane's own f_b gives, and the force keeps its
  accuracy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.slow

N = 4096
LEAF, ORDER = 16, 4
EPS, SOFTENING = 1e-2, 1e-2
FLOOR = 0.5

_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(scope="module", autouse=True)
def _lane():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        for key, value in _CSR_ON_CPU.items():
            patch.setenv(key, value)
        for key in (
            "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
            "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
            "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
        ):
            patch.delenv(key, raising=False)
        yield


def _solver(floor):
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        adaptive_eps=EPS,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
        mac_force_scale_floor=floor,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
    )


def _plummer():
    rng = np.random.default_rng(3)
    u = rng.uniform(1e-3, 0.97, N)
    r = 0.2 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(N, 3))
    pos = r[:, None] * d / np.linalg.norm(d, axis=1, keepdims=True)
    return jnp.asarray(pos, jnp.float32), jnp.full((N,), 1.0 / N, jnp.float32)


def _lists(prepared):
    nl = prepared.neighbor_list
    return (
        np.asarray(nl.counts),
        np.asarray(nl.offsets),
        np.asarray(nl.neighbors),
    )


@pytest.fixture(scope="module")
def lanes():
    from jaccpot.runtime.fused_provider import FusedForceProvider

    pos, mass = _plummer()
    out = {}
    for floor in (None, FLOOR):
        prov, topo = FusedForceProvider.prepare(
            _solver(floor), pos, mass, leaf_size=LEAF, max_order=ORDER
        )
        acc, f_b = prov.accelerations(topo, force_scale=True)

        @jax.jit
        def rebuild(x, m, fb, prov=prov):
            t = prov.rebuild(x, m, force_scale=fb)
            a = prov.accelerations(t)
            nl = t.prepared.neighbor_list
            return a, nl.counts, nl.offsets, nl.neighbors

        out[floor] = dict(prov=prov, topo=topo, acc=acc, f_b=f_b, rebuild=rebuild)
    return pos, mass, out


def test_the_floor_does_not_bind_on_the_lanes_own_force_scale(lanes):
    pos, mass, out = lanes
    plain, floored = out[None], out[FLOOR]
    for x, y in zip(_lists(plain["topo"].prepared), _lists(floored["topo"].prepared)):
        assert np.array_equal(x, y)
    assert np.array_equal(np.asarray(plain["acc"]), np.asarray(floored["acc"]))
    assert np.array_equal(np.asarray(plain["f_b"]), np.asarray(floored["f_b"]))
    a0 = plain["rebuild"](pos, mass, plain["f_b"])
    a1 = floored["rebuild"](pos, mass, plain["f_b"])
    for x, y in zip(a0, a1):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def _direct(pos, mass):
    x = np.asarray(pos, np.float64)
    m = np.asarray(mass, np.float64)
    d = x[None, :, :] - x[:, None, :]
    r2 = (d * d).sum(-1) + SOFTENING**2
    np.fill_diagonal(r2, np.inf)
    return (m[None, :, None] * d / r2[..., None] ** 1.5).sum(1)


def test_a_lost_force_scale_floods_near_rows_and_the_floor_stops_it(lanes):
    pos, mass, out = lanes
    good = np.asarray(out[None]["f_b"])
    # the particles of a few central leaves lose their f_b (1e-9 of it)
    r = np.linalg.norm(np.asarray(pos), axis=1)
    lost = r < np.sort(r)[40]
    bad = jnp.asarray(np.where(lost, good * 1e-9, good), good.dtype)
    counts = np.asarray(out[None]["rebuild"](pos, mass, out[None]["f_b"])[1])
    leaves = int(np.sum(counts > 0))  # live leaves (the rows are the capacity's)
    normal = int(counts.max())
    flooded = out[None]["rebuild"](pos, mass, bad)
    held = out[FLOOR]["rebuild"](pos, mass, bad)
    flooded_row = int(np.asarray(flooded[1]).max())
    held_row = int(np.asarray(held[1]).max())
    seen = f"rows: normal {normal}, flooded {flooded_row}, held {held_row}; {leaves} leaves"
    assert flooded_row > 0.5 * leaves, seen  # the #487 symptom
    assert flooded_row > 4 * normal, seen
    assert held_row <= 2 * normal, seen  # measured: 67 -> 878 flooded, 116 floored

    ref = _direct(pos, mass)
    err = np.linalg.norm(np.asarray(held[0], np.float64) - ref, axis=1)
    rel = np.sqrt((err**2).sum() / (np.linalg.norm(ref, axis=1) ** 2).sum())
    good_acc = np.asarray(out[None]["acc"], np.float64)
    err0 = np.linalg.norm(good_acc - ref, axis=1)
    rel0 = np.sqrt((err0**2).sum() / (np.linalg.norm(ref, axis=1) ** 2).sum())
    assert rel <= 2.0 * rel0 + 1e-6
