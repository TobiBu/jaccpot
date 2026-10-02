"""The near import's compact wire format delivers exactly what the tile format did.

The tile format ships every exported leaf as a W-wide particle tile (~70 % zeros at
~18 particles per 64-slot leaf); the compact one ships the live particles flat plus a
count per leaf, and the receiver rebuilds the tiles. Same tiles, same geometry rows,
same CSR, bit for bit -- on forced CPU devices (the all_gather exchange path).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/unit/distributed/test_near_payload_compaction.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

pytest.importorskip("yggdrax.distributed.export")
from yggdrax.distributed.export import build_send_buffers  # noqa: E402

from jaccpot.distributed.cross import (  # noqa: E402
    _near_exchange_compact,
    _near_exchange_tiles,
    _row_of_slot,
)

AXIS = "gpus"
W = 16
MAX_CELLS = 32
LEAVES = 40
N_COEFF = 9


def _mesh(n):
    devices = jax.devices()
    if len(devices) < n:
        pytest.skip(f"needs {n} devices, have {len(devices)}")
    return jax.sharding.Mesh(
        np.asarray(devices[:n]), (AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
    )


def test_row_of_slot_inverts_the_layout():
    counts = jnp.asarray([3, 0, 2, 0, 0, 4, 1], jnp.int32)
    row, within, live, total = _row_of_slot(counts, 16)
    assert int(total) == 10
    want = [
        (0, 0),
        (0, 1),
        (0, 2),
        (2, 0),
        (2, 1),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (6, 0),
    ]
    got = list(zip(np.asarray(row)[:10].tolist(), np.asarray(within)[:10].tolist()))
    assert got == want
    assert not np.asarray(live)[10:].any() and np.asarray(live)[:10].all()


def _device_data(seed):
    """One device: Morton-sorted particles in LEAVES leaves of 1..W particles."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, W + 1, size=LEAVES)
    sizes[3] = W  # a full leaf
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ends = starts + sizes - 1
    n = int(sizes.sum())
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    # the leaves this device exports to the OTHER device's cells (some twice)
    k = 60
    node = rng.integers(0, LEAVES, size=k)
    return pos, mass, starts, ends, node


def _run(fn, **extra):
    mesh = _mesh(2)
    data = [_device_data(s) for s in (0, 1)]
    n_max = max(len(d[0]) for d in data)
    pos = np.zeros((2, n_max, 3), np.float32)
    mass = np.zeros((2, n_max), np.float32)
    for d, (p, m, *_rest) in enumerate(data):
        pos[d, : len(p)] = p
        mass[d, : len(m)] = m
    starts = np.stack([d[2] for d in data])
    ends = np.stack([d[3] for d in data])
    nodes = np.stack([d[4] for d in data])
    k = nodes.shape[1]
    cells = np.zeros((2, k), np.int64)
    rng = np.random.default_rng(9)
    for d in (0, 1):  # device d exports to the other device's cells only
        other = 1 - d
        cells[d] = other * MAX_CELLS + rng.integers(0, MAX_CELLS, size=k)

    def body(pos, mass, starts, ends, nodes, cells):
        pos, mass, starts, ends, nodes, cells = (
            pos[0],
            mass[0],
            starts[0],
            ends[0],
            nodes[0],
            cells[0],
        )
        sb = build_send_buffers(
            jnp.asarray(cells),
            jnp.asarray(nodes),
            jnp.asarray(nodes.shape[0]),
            ndev=2,
            max_cells=MAX_CELLS,
            num_nodes=LEAVES,
            node_capacity=64,
            csr_capacity=128,
        )
        lrow = jnp.clip(sb.node_rows, 0, LEAVES - 1)
        okrow = (sb.node_rows >= 0)[:, None]
        geo = jnp.stack([lrow.astype(jnp.float32)] * 3, axis=1)  # a recognisable row
        leaf_rows = [
            jnp.where(okrow, geo, 0.0),
            jnp.where(okrow, lrow[:, None].astype(jnp.float32) + 0.5, 0.0),
            jnp.where(okrow, jnp.ones((lrow.shape[0], N_COEFF)) * lrow[:, None], 0.0),
            jnp.where(okrow, geo * 2.0, 0.0),
        ]
        got, ip, im, ovf = fn(
            sb,
            leaf_rows,
            starts[lrow],
            ends[lrow],
            pos,
            mass,
            W=W,
            payload_capacity=64,
            csr_capacity=128,
            ndev=2,
            axis_name=AXIS,
            **extra,
        )
        head = got.payload[:, : 7 + N_COEFF]
        return (
            head[None],
            ip[None],
            im[None],
            got.csr_cell[None],
            got.csr_row[None],
            got.num_csr[None],
            ovf[None],
        )

    f = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),) * 6,
            out_specs=(P(AXIS),) * 7,
            check_vma=False,
        )
    )
    out = f(
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(starts),
        jnp.asarray(ends),
        jnp.asarray(nodes),
        jnp.asarray(cells),
    )
    return [np.asarray(o) for o in out]


def test_the_compact_format_delivers_the_same_tiles_rows_and_csr():
    tiles = _run(_near_exchange_tiles)
    compact = _run(
        _near_exchange_compact, send_particle_cap=1024, recv_particle_cap=1024
    )
    names = ("geometry rows", "positions", "masses", "csr_cell", "csr_row", "num_csr")
    for name, a, b in zip(names, tiles[:6], compact[:6]):
        np.testing.assert_array_equal(a, b, err_msg=name)
    assert not tiles[6].any() and not compact[6].any()
    # not vacuous: particles actually arrived on both devices, in partly full tiles
    assert (tiles[2] != 0).sum(axis=(1, 2)).min() > 0
    assert (tiles[2] == 0).sum() > 0


def test_an_undersized_particle_buffer_raises_the_flag():
    """CONTROL: the compact format adds two capacities, and they must report."""
    out = _run(_near_exchange_compact, send_particle_cap=32, recv_particle_cap=1024)
    assert out[6].any()
    out = _run(_near_exchange_compact, send_particle_cap=1024, recv_particle_cap=32)
    assert out[6].any()
