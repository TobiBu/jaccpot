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
