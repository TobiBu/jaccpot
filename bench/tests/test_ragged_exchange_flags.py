"""The bench scripts' copy of the NCCL ragged-exchange flags matches the library's.

The scripts set ``XLA_FLAGS`` before jax starts, so they cannot import the flags from
jaccpot (that import starts the backend). A drifted copy is how the fused two-card lane
hung on jax 0.11.2: the scripts kept the one-shot flag alone after XLA turned
``use_barrier_with_nccl`` on by default.
"""

from __future__ import annotations

import pathlib

from bench import _ragged_exchange_flags as bench_flags
from jaccpot.distributed import fused

_BENCH = pathlib.Path(__file__).resolve().parents[1]


def test_bench_copy_matches_the_library():
    assert bench_flags.RAGGED_EXCHANGE_XLA_FLAGS == fused.RAGGED_EXCHANGE_XLA_FLAGS


def test_add_appends_only_what_is_missing():
    flags = bench_flags.RAGGED_EXCHANGE_XLA_FLAGS.split()
    env = {"XLA_FLAGS": "--xla_gpu_enable_latency_hiding_scheduler=true"}
    bench_flags.add_ragged_exchange_flags(env)
    assert env["XLA_FLAGS"].split() == [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        *flags,
    ]
    assert not fused._missing_ragged_exchange_flags(env["XLA_FLAGS"])
    # a flag the caller set, to any value, is kept: that is how an A/B overrides one
    one_shot_name = flags[0].split("=")[0]
    env = {"XLA_FLAGS": f"{one_shot_name}=true"}
    bench_flags.add_ragged_exchange_flags(env)
    assert env["XLA_FLAGS"].split() == [f"{one_shot_name}=true", *flags[1:]]
    env = {}
    bench_flags.add_ragged_exchange_flags(env)
    assert env["XLA_FLAGS"] == bench_flags.RAGGED_EXCHANGE_XLA_FLAGS


def test_no_bench_script_hardcodes_the_one_shot_flag_alone():
    """Scripts take the flags from the shared module, not a literal of their own.

    The ``repro_*`` scripts are exempt: they are standalone upstream reproductions
    that import neither jaccpot nor this package, and they spell each flag out.
    """
    one_shot = bench_flags.RAGGED_EXCHANGE_XLA_FLAGS.split()[0]
    offenders = [
        path.name
        for path in _BENCH.glob("*.py")
        if path.name != "_ragged_exchange_flags.py"
        and not path.name.startswith("repro_")
        and f'"{one_shot}"' in path.read_text()
    ]
    assert not offenders, offenders
