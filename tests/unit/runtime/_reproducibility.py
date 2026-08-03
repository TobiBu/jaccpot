"""Reproducibility assertions that survive GPU atomic reductions.

Several tests here check that *repeating* a computation reproduces it: that a
cached force scale reads back what writing it produced, that no history leaks
between solves, that a frozen MAC input is genuinely unread. The natural way to
write that is ``assert_array_equal``, and on CPU it holds exactly.

On GPU it does not, and not because anything is wrong. The near-field and M2L
accumulations scatter-add, which XLA lowers to atomics; atomics commit in
whatever order the hardware schedules them, and float addition is not
associative. So the same executable on the same inputs returns results that
differ in the last bits, run to run.

Measured on an A100, at the configuration these tests use (N=512, leaf=16,
p=4, float64), across 8 runs with identical inputs: **0 of 7** later runs were
bit-identical to the first, and the worst deviation was **3.8 eps** elementwise,
or 8.1e-17 relative to the rms ``|a|``. Forcing ``--xla_gpu_deterministic_ops``
does make them bit-identical, but it is far too slow to run a suite under.

So: exact equality where it is meaningful (CPU), and a tolerance on GPU that is
three orders of magnitude above the measured noise and many orders below any
regression these tests exist to catch. For calibration, the two defects this
area has actually produced were a 23% mass loss and a 15-41x error-tail blowup;
a 1e-13 relative band cannot hide either.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["assert_reproducible", "reduction_noise_tolerance"]

# Comfortably above the measured 3.8 eps (~8.4e-16 elementwise) reduction noise,
# and far below anything that would constitute a regression.
GPU_RTOL = 1.0e-13
GPU_ATOL_SCALE = 1.0e-13


def _on_gpu() -> bool:
    try:
        import jax

        return jax.default_backend() != "cpu"
    except Exception:  # pragma: no cover
        return False


def reduction_noise_tolerance(reference: Any) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for comparing two runs of the same computation.

    ``atol`` is scaled by the magnitude of ``reference`` so that near-cancelling
    components -- where a relative tolerance is meaningless -- are still judged
    against the size of the field rather than against their own tiny value.
    """

    if not _on_gpu():
        return (0.0, 0.0)
    ref = np.asarray(reference, dtype=np.float64)
    scale = float(np.abs(ref).max()) if ref.size else 1.0
    return (GPU_RTOL, GPU_ATOL_SCALE * max(scale, 1.0))


def assert_reproducible(actual: Any, desired: Any, *, err_msg: str = "") -> None:
    """Assert two runs of the same computation agree.

    Exact on CPU. On GPU, within the measured atomic-reduction noise band -- see
    the module docstring for the measurement and why a looser bound here cannot
    mask the regressions these tests guard.
    """

    a = np.asarray(actual)
    d = np.asarray(desired)
    rtol, atol = reduction_noise_tolerance(d)
    if rtol == 0.0 and atol == 0.0:
        np.testing.assert_array_equal(a, d, err_msg=err_msg)
        return
    np.testing.assert_allclose(a, d, rtol=rtol, atol=atol, err_msg=err_msg)
