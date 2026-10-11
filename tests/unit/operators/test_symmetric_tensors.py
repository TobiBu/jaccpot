import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.operators.symmetric_tensors import (
    component_lift_index_map_3d,
    contract_symmetric_one_axis_3d,
    symmetric_component_count,
    symmetric_multi_indices_3d,
    symmetric_order_offsets_3d,
)
from jaccpot.runtime.fmm import FMMEngine


def test_symmetric_component_count_3d_matches_closed_form() -> None:
    expected = [1, 3, 6, 10, 15, 21]
    got = [symmetric_component_count(order, dim=3) for order in range(6)]
    assert got == expected


def test_symmetric_multi_indices_3d_order_two_layout() -> None:
    combos = symmetric_multi_indices_3d(2)
    assert combos == (
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )


def test_symmetric_order_offsets_3d() -> None:
    offsets = symmetric_order_offsets_3d(4)
    assert offsets == (0, 1, 4, 10, 20, 35)


def test_contract_symmetric_one_axis_3d_matches_manual_hessian_vector() -> None:
    # Packed order-2 layout:
    # (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)
    h_packed = jnp.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0], dtype=jnp.float64)
    vec = jnp.array([17.0, 19.0, 23.0], dtype=jnp.float64)
    got = contract_symmetric_one_axis_3d(h_packed, vec, order=2)
    expected = jnp.array(
        [
            2.0 * 17.0 + 3.0 * 19.0 + 5.0 * 23.0,
            3.0 * 17.0 + 7.0 * 19.0 + 11.0 * 23.0,
            5.0 * 17.0 + 11.0 * 19.0 + 13.0 * 23.0,
        ],
        dtype=jnp.float64,
    )
    assert jnp.allclose(got, expected, rtol=0.0, atol=0.0)


def test_contract_symmetric_one_axis_3d_jit_shape_and_value() -> None:
    order = 3
    n = symmetric_component_count(order, dim=3)
    key = jax.random.PRNGKey(0)
    packed = jax.random.normal(key, (n,))
    vec = jnp.array([0.5, -1.0, 2.0], dtype=packed.dtype)

    compiled = jax.jit(lambda p, v: contract_symmetric_one_axis_3d(p, v, order=order))
    eager = contract_symmetric_one_axis_3d(packed, vec, order=order)
    jitted = compiled(packed, vec)

    assert jitted.shape == (symmetric_component_count(order - 1, dim=3),)
    assert np.allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12, atol=1e-12)


def test_component_lift_index_map_3d_order_one() -> None:
    # Order-1 base tuples are: (1,0,0), (0,1,0), (0,0,1)
    # Order-2 packed indices are:
    # 0:(2,0,0), 1:(1,1,0), 2:(1,0,1), 3:(0,2,0), 4:(0,1,1), 5:(0,0,2)
    got = component_lift_index_map_3d(1)
    assert got == (
        (0, 1, 2),
        (1, 3, 4),
        (2, 4, 5),
    )


# ---------------------------------------------------------------------------
# From tests/integration/test_fmm.py (2026-10 cleanup, Z-a)
# ---------------------------------------------------------------------------


def test_stf_symmetry_and_trace_free():
    """Quadrupole, octupole, hexadecapole are symmetric and trace-free."""
    key = jax.random.PRNGKey(1)
    n = 10
    pos = 0.2 * jax.random.normal(key, (n, 3))
    mass = jax.random.uniform(key, (n,), minval=0.1, maxval=2.0)

    exp = FMMEngine.compute_expansion(pos, mass, order=4)

    # Quadrupole symmetry and trace-free
    Q = exp.quadrupole
    assert jnp.allclose(Q, jnp.swapaxes(Q, 0, 1), atol=1e-6)
    tr_Q = jnp.trace(Q)
    assert jnp.allclose(tr_Q, 0.0, atol=1e-5)

    # Octupole: symmetric under pairwise swaps, single traces vanish
    oct_t = exp.octupole
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (1, 0, 2)), atol=1e-6)
    assert jnp.allclose(oct_t, jnp.transpose(oct_t, (2, 1, 0)), atol=1e-6)
    # single traces
    tr1 = jnp.trace(oct_t, axis1=0, axis2=1)  # shape (3,)
    tr2 = jnp.trace(oct_t, axis1=0, axis2=2)  # shape (3,)
    tr3 = jnp.trace(oct_t, axis1=1, axis2=2)  # shape (3,)
    assert jnp.allclose(tr1, 0.0, atol=1e-5)
    assert jnp.allclose(tr2, 0.0, atol=1e-5)
    assert jnp.allclose(tr3, 0.0, atol=1e-5)

    # Hexadecapole: symmetric under swaps, single traces vanish
    hex_t = exp.hexadecapole
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (1, 0, 2, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 2, 1, 3)), atol=1e-6)
    assert jnp.allclose(hex_t, jnp.transpose(hex_t, (0, 1, 3, 2)), atol=1e-6)
    # single traces along different pairs
    tr_01 = jnp.trace(hex_t, axis1=0, axis2=1)  # (3,3)
    tr_02 = jnp.trace(hex_t, axis1=0, axis2=2)  # (3,3)
    tr_03 = jnp.trace(hex_t, axis1=0, axis2=3)  # (3,3)
    assert jnp.allclose(tr_01, 0.0, atol=1e-4)
    assert jnp.allclose(tr_02, 0.0, atol=1e-4)
    assert jnp.allclose(tr_03, 0.0, atol=1e-4)
    # double trace ~ 0
    dt = jnp.trace(tr_01)
    assert jnp.allclose(dt, 0.0, atol=1e-4)
