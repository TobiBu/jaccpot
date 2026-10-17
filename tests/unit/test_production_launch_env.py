"""The production launch environment: XLA's fusion autotuner off (S3 session 2).

XLA's fusion autotuner picks some fusions' emitters by timing them on the card, so on
a shared GPU a fresh process could compile other code and a resumed run was not
bitwise the uninterrupted one. Production runs set
``--xla_gpu_experimental_enable_fusion_autotuner=false``; the library cannot (XLA
reads ``XLA_FLAGS`` once, at backend start), so ``ProductionRunner.run`` warns once
per process on a GPU when it is missing (docs/production.md, "Launch environment").

On CPU, with the platform name faked: the flag is read as XLA reads it, the warning
fires on a GPU only, without the flag only, and once; ``run`` asks with its state's
platform and the process's ``XLA_FLAGS``. Nothing compiles.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from jaccpot import FastMultipoleMethod, production
from jaccpot.production import (
    ProductionRunner,
    SegmentConfig,
    _fusion_autotuner_off,
    _warn_fusion_autotuner,
)

FLAG = "--xla_gpu_experimental_enable_fusion_autotuner"
OFF = f"{FLAG}=false"
CB = "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"


@pytest.fixture(autouse=True)
def _not_warned_yet(monkeypatch):
    # the warning is once per PROCESS; each test starts from a process that has not
    monkeypatch.setattr(production, "_FUSION_AUTOTUNER_WARNED", False)


@pytest.mark.parametrize(
    "xla_flags, off",
    [
        (None, False),
        ("", False),
        (OFF, True),
        (f"{CB} {OFF} --xla_gpu_graph_min_graph_size=2", True),
        # XLA's boolean spellings: false / 0 in any case, true / 1, bare is true
        (f"{FLAG}=False", True),
        (f"{FLAG}=0", True),
        (f"{FLAG}=true", False),
        (f"{FLAG}=1", False),
        (FLAG, False),
        # XLA parses left to right: the last setting wins
        (f"{OFF} {FLAG}=true", False),
        (f"{FLAG}=true {OFF}", True),
        # another flag that starts with the name, or holds it in its value
        (f"{FLAG}_top_k=false", False),
        (f"--xla_dump_to=/tmp/{OFF}", False),
    ],
)
def test_the_flag_is_read_as_xla_reads_it(xla_flags, off):
    assert _fusion_autotuner_off(xla_flags) is off


def test_a_gpu_run_without_the_flag_warns_once():
    with pytest.warns(UserWarning, match=f"{OFF}: XLA's fusion autotuner"):
        assert _warn_fusion_autotuner("gpu", CB)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _warn_fusion_autotuner("gpu", CB)
        assert not _warn_fusion_autotuner("gpu", None)


def test_no_warning_off_a_gpu_or_with_the_flag():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _warn_fusion_autotuner("cpu", None)
        assert not _warn_fusion_autotuner("gpu", f"{CB} {OFF}")
    # neither used up the one warning
    with pytest.warns(UserWarning, match="not be bitwise"):
        assert _warn_fusion_autotuner("gpu", f"{OFF} {FLAG}=true")


def test_run_asks_with_its_states_platform_and_the_processes_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(
        production, "_warn_fusion_autotuner", lambda *args: calls.append(args)
    )
    monkeypatch.setenv("XLA_FLAGS", f"{CB} {FLAG}=true")
    # no step and no diagnostics: nothing compiles
    runner = ProductionRunner(
        FastMultipoleMethod(preset="galaxy", softening=1e-2),
        segment=SegmentConfig(diagnostics_every=0),
    )
    state = jnp.zeros((8, 2, 3), jnp.float32)
    res = runner.run(state, jnp.ones(8, jnp.float32), dt=1e-3, num_steps=0)
    assert res.step == 0
    assert calls == [("cpu", f"{CB} {FLAG}=true")]
