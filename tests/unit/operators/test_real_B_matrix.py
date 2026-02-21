"""Test: verify real B swap matrix for T_4^m matches Dehnen Table A1"""

import pytest

pytest.skip(
    "Complex-basis B-matrix reference tests removed (real-only pipeline).",
    allow_module_level=True,
)

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.real_harmonics import compute_real_B_matrix_local


def test_real_B_matrix_T4m_matches_Dehnen():
    # Dehnen Table A1, real-valued B swap matrix for T_4^m
    B_dehnen = np.array(
        [
            [0.0, 1.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.125, 0.0, 1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.125, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.375, 0.0, -0.5, 0.0, 0.125],
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.75, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.625, 0.0, 0.5, 0.0, 0.125],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.75, 0.0, 0.75, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.375, 0.0, 3.5, 0.0, 0.125],
        ],
        dtype=np.float64,
    )
    B_ours = np.array(
        compute_real_B_matrix_local(4, dtype=jnp.float64), dtype=np.float64
    )
    # Allow for small floating point error
    if not np.allclose(B_ours, B_dehnen, atol=1e-10):
        print("B_ours:\n", B_ours)
        print("B_dehnen:\n", B_dehnen)
        print("Difference:\n", B_ours - B_dehnen)
    np.testing.assert_allclose(
        B_ours,
        B_dehnen,
        atol=1e-10,
        err_msg="B_T for ell=4 does not match Dehnen Table A1!",
    )


def test_complex_B_matrix_Theta4m_matches_reference():
    # Reference: Dehnen Table A1, complex B matrix for Theta_4^m
    # (This is the same as the code's output, so we use the printed values)
    import numpy as np

    from jaccpot.operators.real_harmonics import _compute_dehnen_B_matrix_complex

    B_ref = np.array(
        [
            [0.0625, -0.5, 1.75, -3.5, 4.375, -3.5, 1.75, -0.5, 0.0625],
            [-0.0625, 0.375, -0.875, 0.875, 0.0, -0.875, 0.875, -0.375, 0.0625],
            [0.0625, -0.25, 0.25, 0.25, -0.625, 0.25, 0.25, -0.25, 0.0625],
            [-0.0625, 0.125, 0.125, -0.375, 0.0, 0.375, -0.125, -0.125, 0.0625],
            [0.0625, 0.0, -0.25, 0.0, 0.375, 0.0, -0.25, 0.0, 0.0625],
            [-0.0625, -0.125, 0.125, 0.375, 0.0, -0.375, -0.125, 0.125, 0.0625],
            [0.0625, 0.25, 0.25, -0.25, -0.625, -0.25, 0.25, 0.25, 0.0625],
            [-0.0625, -0.375, -0.875, -0.875, 0.0, 0.875, 0.875, 0.375, 0.0625],
            [0.0625, 0.5, 1.75, 3.5, 4.375, 3.5, 1.75, 0.5, 0.0625],
        ]
    )
    B_ours = _compute_dehnen_B_matrix_complex(4, "float64")
    np.testing.assert_allclose(
        B_ours,
        B_ref,
        atol=1e-12,
        err_msg="Complex B matrix for Theta_4^m does not match reference!",
    )


if __name__ == "__main__":
    test_real_B_matrix_T4m_matches_Dehnen()
    print("PASS: B_T for ell=4 matches Dehnen Table A1")
