"""The LET exchange's open test is conservative (track M4).

:func:`jaccpot.distributed.let.let_open` decides which of a device's summary nodes a
peer's cross walk could split, from the peer's :func:`let_descriptor` alone. The LET
exchange's pair sets equal the all_gather path's exactly when that decision never
misses a split (``let.py``'s module docstring). These tests check the bound against
brute force on random node populations: for every (A, B), whenever the walk's MAC
refuses the pair AND its split rule picks A (A internal; B a leaf, or ``rho_A >=
rho_B``), ``let_open`` must have opened A against B's device. Both MACs: the opening
angle (``dehnen``) and eq (16a) (``dehnen_error``, through yggdrax's own
``dehnen_pair_accept``), with and without a separation floor.

And the bound is not vacuous: far nodes stay closed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax._dehnen_accept import dehnen_pair_accept

from jaccpot.distributed.let import let_descriptor, let_descriptor_width, let_open

ORDER = 3
THETA = 0.7
THETA_MAX = 1.0


def _rows(rng, n, centre, *, order, spread=1.0):
    """``n`` random summary rows around ``centre``: radii over 4 decades, half of
    them leaves, eq (16a) columns with ``s_n`` in [0, 1]."""
    cen = centre + spread * rng.uniform(-1.0, 1.0, (n, 3))
    rad = 10.0 ** rng.uniform(-3.0, 0.3, n)
    leaf = rng.uniform(size=n) < 0.5
    left = np.where(leaf, -1.0, 1.0)
    cols = [cen, rad[:, None], left[:, None], left[:, None], np.ones((n, 1)),
            np.full((n, 1), 8.0)]
    if order:
        gm = 10.0 ** rng.uniform(-6.0, -1.0, n)
        thr = 10.0 ** rng.uniform(-4.0, 0.0, n)
        s = rng.uniform(0.0, 1.0, (n, order))
        cols += [gm[:, None], thr[:, None], s]
    return np.concatenate(cols, axis=1)


def _refused_and_split_a(a, b, *, order, floor):
    """(n_a, n_b): the MAC refuses (A, B) and the walk would split A."""
    ca, cb = a[:, None, 0:3], b[None, :, 0:3]
    d2 = np.sum((ca - cb) ** 2, axis=-1)
    ra, rb = a[:, 3][:, None], b[:, 3][None, :]
    if order:
        ta = [jnp.asarray(np.broadcast_to(a[:, 8 + k][:, None], d2.shape)) for k in range(order + 2)]
        tb = [jnp.asarray(np.broadcast_to(b[:, 8 + k][None, :], d2.shape)) for k in range(order + 2)]
        ok = np.asarray(
            dehnen_pair_accept(
                row_a=ta,
                row_b=tb,
                radius_a=jnp.asarray(np.broadcast_to(ra, d2.shape)),
                radius_b=jnp.asarray(np.broadcast_to(rb, d2.shape)),
                dist_sq=jnp.asarray(d2),
                order=order,
                theta_max=THETA_MAX,
            )
        )
    else:
        ok = (ra + rb) ** 2 <= THETA**2 * d2
    if floor > 0.0:
        ok = ok & ((ra + rb + floor) ** 2 <= d2)
    ok = ok & (d2 > 0.0)
    a_internal = (a[:, 4] >= 0)[:, None]
    b_leaf = (b[:, 4] < 0)[None, :]
    split_a = a_internal & (b_leaf | (ra >= rb))
    return ~ok & split_a


@pytest.mark.parametrize("order", [0, ORDER], ids=["dehnen", "dehnen_error"])
@pytest.mark.parametrize("floor", [0.0, 0.05], ids=["no-floor", "floor"])
@pytest.mark.parametrize("offset", [0.0, 2.5, 6.0], ids=["overlap", "near", "far"])
def test_every_node_the_walk_could_split_is_opened(order, floor, offset):
    rng = np.random.default_rng(int(10 * offset) + order + int(100 * floor))
    a = _rows(rng, 400, np.zeros(3), order=order)
    b = _rows(rng, 300, np.array([offset, 0.3 * offset, 0.0]), order=order)
    desc = let_descriptor(jnp.asarray(b), jnp.ones(b.shape[0], bool), error_order=order)
    assert desc.shape == (let_descriptor_width(order),)
    opened = np.asarray(
        let_open(
            jnp.asarray(a),
            desc,
            theta=THETA,
            separation_floor=floor,
            error_order=order,
            theta_max=THETA_MAX,
        )
    )
    need = np.any(_refused_and_split_a(a, b, order=order, floor=floor), axis=1)
    missed = need & ~opened
    assert not np.any(missed), f"{int(missed.sum())} nodes the walk splits stay closed"
    if offset == 0.0:
        assert np.all(opened[a[:, 4] >= 0]), "an overlapping peer opens every internal node"


def test_the_bound_is_not_vacuous():
    """A peer far away (in units of the radii) opens nothing; a leaf never opens."""
    rng = np.random.default_rng(7)
    a = _rows(rng, 200, np.zeros(3), order=0, spread=0.5)
    a[:, 3] = 0.01
    b = _rows(rng, 200, np.array([50.0, 0.0, 0.0]), order=0, spread=0.5)
    b[:, 3] = 0.01
    desc = let_descriptor(jnp.asarray(b), jnp.ones(b.shape[0], bool))
    opened = np.asarray(let_open(jnp.asarray(a), desc, theta=THETA))
    assert not np.any(opened)
    near = b.copy()
    near[:, 0:3] -= np.array([49.5, 0.0, 0.0])
    desc = let_descriptor(jnp.asarray(near), jnp.ones(b.shape[0], bool))
    opened = np.asarray(let_open(jnp.asarray(a), desc, theta=THETA))
    assert np.any(opened)
    assert not np.any(opened[a[:, 4] < 0])


def test_an_empty_peer_opens_nothing():
    rng = np.random.default_rng(3)
    a = _rows(rng, 50, np.zeros(3), order=ORDER)
    b = _rows(rng, 50, np.zeros(3), order=ORDER)
    desc = let_descriptor(jnp.asarray(b), jnp.zeros(b.shape[0], bool), error_order=ORDER)
    opened = np.asarray(
        let_open(jnp.asarray(a), desc, theta=THETA, error_order=ORDER)
    )
    assert not np.any(opened)
