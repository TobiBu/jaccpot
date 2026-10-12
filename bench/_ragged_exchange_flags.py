"""The NCCL ragged-exchange XLA flags, for bench scripts to set before jax starts.

Importing jaccpot starts the XLA backend, and XLA reads ``XLA_FLAGS`` only then, so a
bench cannot take :data:`jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS` before it
sets the flags. This module imports nothing but ``os``. ``bench/tests`` checks that the
copy below matches the library's, since a stale copy (the one-shot flag alone) is how the
fused two-card lane hung on jax 0.11.2.
"""

from __future__ import annotations

import os

#: Same value as ``jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS``; see there.
RAGGED_EXCHANGE_XLA_FLAGS = (
    "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false "
    "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true "
    "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
)


def add_ragged_exchange_flags(env: dict[str, str] | None = None) -> None:
    """Append each flag whose name ``XLA_FLAGS`` does not mention yet.

    A flag the caller already set, to any value, is left alone, so an A/B can override
    one (e.g. ``...one_shot_kernel=true`` for XLA's default kernel).

    Parameters
    ----------
    env : dict[str, str] or None
        The environment to edit; ``None`` edits ``os.environ``.
    """
    env = os.environ if env is None else env
    flags = env.get("XLA_FLAGS", "")
    for flag in RAGGED_EXCHANGE_XLA_FLAGS.split():
        if flag.split("=")[0] not in flags:
            flags = f"{flags} {flag}".strip()
    env["XLA_FLAGS"] = flags
