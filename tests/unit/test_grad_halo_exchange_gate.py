"""The grad-path halo-exchange gate, tested where it can actually run.

``resolve_grad_halo_exchange("auto")`` decides whether the distributed reverse
pass uses ``jax.lax.ragged_all_to_all`` (``"native"``) or the ``all_gather``-based
``"buf"``. Choosing wrong is not a performance matter in either direction: too
eager and it is ``UNIMPLEMENTED`` on CPU, too eager on an old JAX and every later
ragged exchange in the process silently returns an unwritten buffer, dropping the
whole cross-domain near field.

WHY THIS FILE EXISTS SEPARATELY. These three are pure resolver checks -- a
monkeypatch and a function call, no devices, no compile. They lived in
``tests/distributed/test_distributed_grad_correctness.py``, which carries a
file-level ``skipif(device_count() < 2)`` and a ``slow`` mark, so they ran on a
two-device box and nowhere else. Since ``e98340d`` CI does not collect
``tests/distributed`` at all, which meant the gate guarding the distributed
reverse pass was itself guarded by nothing that runs.

That is not a hypothetical cost. The gate checked the JAX version and not the
backend, resolved to ``"native"`` on jax 0.10.2 regardless, and took out six tests
in that same file -- and because the file no longer ran anywhere, the breakage sat
there. A device-gated test cannot catch a bug in the decision about which device
you are on.

The device-requiring tests stay where they are. Only the ones that never needed a
device move.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yggdrax")

import jaccpot.distributed.fmm as dfmm  # noqa: E402


def test_auto_halo_exchange_version_gate(monkeypatch):
    """``auto`` must pick the ragged exchange only on a JAX where it is safe.

    Pure resolver test -- no devices, no compile. The boundary is measured, not
    read off a changelog: 0.9.0.1 corrupts (6/6 in the reproducer), 0.9.1 does not
    (4/4 clean, and ``test_native_halo_exchange_is_fixed_upstream`` passes there).

    The backend is pinned to ``"gpu"`` so this asserts the VERSION axis alone.
    ``"auto"`` now gates on both, and the two are deliberately tested in separate
    functions: run together, a change that broke one could be made green by
    weakening the other, and on a CPU runner -- which is every runner this file
    has -- an unpinned backend would let the version cases pass for the CPU
    reason instead of the version one.
    """
    monkeypatch.setattr(dfmm.jax, "default_backend", lambda: "gpu")

    cases = {
        "0.8.0": "buf",
        "0.9.0": "buf",
        "0.9.0.1": "buf",  # the version the bug was found on
        "0.9.1": "native",  # first fixed release
        "0.10.2": "native",
        "0.11.0": "native",
        "1.0.0": "native",
    }
    for version, expected in cases.items():
        monkeypatch.setattr(dfmm.jax, "__version__", version, raising=False)
        assert (
            dfmm.resolve_grad_halo_exchange("auto") == expected
        ), f"jax {version} should resolve to {expected!r}"
        # an explicit choice is never overridden by the gate
        assert dfmm.resolve_grad_halo_exchange("buf") == "buf"
        assert dfmm.resolve_grad_halo_exchange("native") == "native"

    # non-numeric suffixes (dev/rc builds) must not crash the parse
    for version in ("0.9.1.dev20260301", "0.10.0rc1", "0.9"):
        monkeypatch.setattr(dfmm.jax, "__version__", version, raising=False)
        assert dfmm.resolve_grad_halo_exchange("auto") in ("buf", "native")


def test_auto_halo_exchange_backend_gate(monkeypatch):
    """``auto`` must not pick the ragged exchange on a backend that cannot lower it.

    ``jax.lax.ragged_all_to_all`` has no XLA:CPU lowering at all, so on CPU
    ``"native"`` is not a slower-or-riskier choice, it is a hard
    ``UNIMPLEMENTED``. The version gate alone resolved to ``"native"`` on jax
    0.10.2 regardless of backend, which took out six tests in this very file:

        UNIMPLEMENTED: HLO opcode `ragged-all-to-all` is not supported by
        XLA:CPU ThunkEmitter

    Since `e98340d` CI does not collect ``tests/distributed``, so CPU was the only
    place these ran -- meaning the only reachable guard on the distributed reverse
    pass was dead, for a reason having nothing to do with the code it guards.

    The version is pinned to a fixed one so this asserts the BACKEND axis alone;
    the companion above pins the backend and varies the version.
    """
    monkeypatch.setattr(dfmm.jax, "__version__", "0.10.2", raising=False)

    monkeypatch.setattr(dfmm.jax, "default_backend", lambda: "cpu")
    assert dfmm.resolve_grad_halo_exchange("auto") == "buf"

    monkeypatch.setattr(dfmm.jax, "default_backend", lambda: "gpu")
    assert dfmm.resolve_grad_halo_exchange("auto") == "native"

    # An explicit choice still overrides the gate on every backend -- that is how
    # the corruption reproducer selects "native" on purpose.
    for backend in ("cpu", "gpu"):
        monkeypatch.setattr(dfmm.jax, "default_backend", lambda b=backend: b)
        assert dfmm.resolve_grad_halo_exchange("buf") == "buf"
        assert dfmm.resolve_grad_halo_exchange("native") == "native"


def test_the_two_halo_exchange_gates_are_independent(monkeypatch):
    """Neither gate alone decides: ``native`` needs a fixed JAX *and* a GPU.

    Written as a truth table because the failure this file suffered was one
    condition being checked and the other assumed. A future edit that drops
    either condition turns one of these four rows red.
    """
    for version, backend, expected in (
        ("0.10.2", "gpu", "native"),
        ("0.10.2", "cpu", "buf"),
        ("0.9.0.1", "gpu", "buf"),
        ("0.9.0.1", "cpu", "buf"),
    ):
        monkeypatch.setattr(dfmm.jax, "__version__", version, raising=False)
        monkeypatch.setattr(dfmm.jax, "default_backend", lambda b=backend: b)
        assert (
            dfmm.resolve_grad_halo_exchange("auto") == expected
        ), f"jax {version} on {backend} should resolve to {expected!r}"
