"""Runtime switches shared by both fused M2L kernels.

``JACCPOT_FUSED_M2L_VJP`` selects the reverse-mode implementation for *both* the
real and the complex fused M2L, so it needs exactly one reader. It had two --
byte-identical bodies in ``m2l_real_fused`` and ``m2l_complex_fused`` (audit
**F13**) -- and they had already begun to drift: the docstrings disagreed about
what the switch does while the code still agreed. That is the cheap half of the
divergence; the expensive half is a fix applied to one copy and not the other, on
a knob that selects which VJP kernel runs.

This module holds the one definition. The two kernels import it under their old
private name, so ``module._fused_m2l_vjp_enabled`` still resolves in both -- the
same "thin alias" arrangement ``_env`` already documents for the readers it
consolidated in audit item 2.2.
"""

from __future__ import annotations

from jaccpot._env import env_flag

__all__ = [
    "fused_m2l_vjp_enabled",
]


def fused_m2l_vjp_enabled() -> bool:
    """Whether the M2L ``custom_vjp`` reverse uses the fused Pallas VJP kernel.

    Default ON: the reverse runs as a single fused Pallas launch instead of
    pure-JAX autodiff of the twin, so the reverse pass also gets the fused
    speedup. Set ``JACCPOT_FUSED_M2L_VJP=0`` to fall back to autodiff of the
    pure-jnp twin -- the correctness reference, identical to round-off, and
    useful for debugging.

    Read at call time, never captured at import: a knob captured into a
    module-level constant cannot be changed after ``import jaccpot`` and silently
    does nothing, which ``_env`` documents as worse than not having the knob.

    Returns
    -------
    bool
        ``True`` when the fused reverse kernel should be used.
    """
    return env_flag("JACCPOT_FUSED_M2L_VJP", True)
