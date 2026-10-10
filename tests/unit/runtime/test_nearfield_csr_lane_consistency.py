"""The near-field rectangle is shrunk only when the CSR lane will really run.

The large-N lane's near field has two forms: the CSR row-chunk Pallas kernel,
which reads the neighbour CSR directly, and the rectangle / target-block route
(pure JAX, or the rectangle Pallas kernel). When the CSR lane runs, the prepare
shrinks the rectangle payload to a one-block placeholder and the strict runner
drops the rectangle's capacity guard.

That decision used to come from the hardware alone
(``_nearfield_csr_lane_enabled``), while the evaluation also needs the near
field on Pallas (``use_pallas``). With ``use_pallas=False`` on an Ampere card --
``ODISSEO_FMM_USE_PALLAS=0``, or any caller passing it -- the prepare built the
placeholder, the guard was off, and the pure-JAX route evaluated the near field
from one block per leaf: rel-L2 0.113 against direct summation at 5e4 on an
A100 (identical lists; 1.6e-3 with the Pallas near field). Reproduced here on
CPU by forcing the CSR lane on: 0.53 instead of 3.9e-4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.characterization.test_fmm_golden import (
    _direct_sum_accelerations,
    _make_inputs,
)
from tests.characterization.test_lane_goldens import (
    _FUSED_ENV,
    _large_n_solver,
    _rel_l2,
)

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the direct-sum anchor needs float64"
)

N = 512


def _forces(monkeypatch, csr_flag: str) -> np.ndarray:
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in _FUSED_ENV.items():
        monkeypatch.setenv(key, value)
    # Odisseo's block size: a leaf's neighbours overflow a single block
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "4")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS", "1")
    monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", csr_flag)
    positions, masses = _make_inputs("clustered", N)
    solver = _large_n_solver()
    assert not bool(getattr(solver._impl, "use_pallas", False))  # CPU: no Pallas
    prepared = solver.prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=16,
        max_order=4,
    )
    return np.asarray(solver.evaluate_prepared_state(prepared))


def test_csr_lane_enabled_without_pallas_keeps_the_full_rectangle(monkeypatch):
    rectangle = _forces(monkeypatch, "0")
    enabled_but_not_used = _forces(monkeypatch, "1")
    positions, masses = _make_inputs("clustered", N)
    reference = _direct_sum_accelerations(positions, masses)
    assert _rel_l2(rectangle, reference) < 1e-2
    # The CSR lane cannot run without Pallas, so this is the same rectangle run.
    np.testing.assert_array_equal(enabled_but_not_used, rectangle)


def test_the_csr_lane_predicate_needs_the_near_field_on_pallas(monkeypatch):
    from jaccpot.nearfield import _fast_lane

    monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1")
    assert _fast_lane._nearfield_csr_lane_enabled()
    assert not _fast_lane._nearfield_csr_lane_active(use_pallas=False)
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    assert _fast_lane._nearfield_csr_lane_active(use_pallas=True)
    monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_FOLD_SELF", "0")
    assert not _fast_lane._nearfield_csr_lane_active(use_pallas=True)


# The CSR lane in Pallas interpret mode, on the prepacked payload it reads.
_CSR_INTERPRET_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _csr_lane_forces(monkeypatch, **env) -> np.ndarray:
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state
    from tests.characterization.test_fmm_golden import G_CONST, SOFTENING
    from tests.characterization.test_lane_goldens import LEAF, ORDER, THETA

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in {**_CSR_INTERPRET_ENV, **env}.items():
        monkeypatch.setenv(key, value)
    positions, masses = _make_inputs("clustered", N)
    # the lane golden's solver with the near field on Pallas; a fresh one per
    # call, because the compiled evaluation is cached on the solver
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=THETA,
        G=G_CONST,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
        softening_kernel="plummer",
    )
    prepared = solver.prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=LEAF,
        max_order=ORDER,
    )
    assert prepared.neighbor_list is not None
    acc = evaluate_large_n_state(
        solver._impl,
        prepared,
        target_indices=None,
        return_potential=False,
        max_acc_derivative_order=0,
    )
    return np.asarray(acc, np.float64)


def test_wide_accumulation_runs_the_table_layout(monkeypatch):
    """``JACCPOT_NEARFIELD_ACCUM=wide`` still evaluates, through the table kernel.

    The direct kernel (the default layout) accumulates in the input dtype, so a
    wide request used to reroute to the ``sorted`` layout; that layout was
    removed in the 2026-10 cleanup (X5), and ``wide`` now runs the table kernel,
    which carries the two-level accumulator (as the multi-GPU cross term, which
    reads the same variable). Both entries are counted, so a silent fallback to
    the direct kernel cannot pass.
    """
    import jaccpot.pallas.nearfield_leafpair_csr as csr

    calls = {"direct": 0, "table": 0}
    direct_fn = csr.nearfield_leafpair_csr_sorted_direct_pallas
    table_fn = csr.nearfield_leafpair_csr_pallas

    def counted(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            if name == "table":
                assert kwargs.get("accum") == "wide"
            return fn(*args, **kwargs)

        return wrapped

    # the lane imports both from the module at call time
    monkeypatch.setattr(
        csr, "nearfield_leafpair_csr_sorted_direct_pallas", counted("direct", direct_fn)
    )
    monkeypatch.setattr(
        csr, "nearfield_leafpair_csr_pallas", counted("table", table_fn)
    )

    monkeypatch.delenv("JACCPOT_NEARFIELD_ACCUM", raising=False)
    monkeypatch.delenv("JACCPOT_NEARFIELD_LAYOUT", raising=False)
    default = _csr_lane_forces(monkeypatch)
    assert calls == {"direct": 1, "table": 0}
    wide = _csr_lane_forces(monkeypatch, JACCPOT_NEARFIELD_ACCUM="wide")
    assert calls == {"direct": 1, "table": 1}

    positions, masses = _make_inputs("clustered", N)
    reference = _direct_sum_accelerations(positions, masses)
    assert _rel_l2(wide, reference) < 1e-2
    # the same pairs and far field; only the near sums' rounding differs
    assert _rel_l2(wide, default) < 1e-5

    monkeypatch.setenv("JACCPOT_NEARFIELD_LAYOUT", "sorted")
    with pytest.raises(ValueError, match="JACCPOT_NEARFIELD_LAYOUT"):
        _csr_lane_forces(monkeypatch)


def _csr_lane_forces_at_budget(monkeypatch, payload_budget_mb: str):
    """The CSR lane on Pallas (interpret mode on CPU), at a given payload budget."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in _FUSED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "4")
    monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1")
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    monkeypatch.setenv("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", payload_budget_mb)
    positions, masses = _make_inputs("clustered", N)
    solver = _large_n_solver()
    solver._impl.use_pallas = True  # what an Ampere card resolves
    prepared = solver.prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=16,
        max_order=4,
    )
    materialised = int(prepared.radix_fast_payload.source_particle_ids.size) > 0
    return np.asarray(solver.evaluate_prepared_state(prepared)), materialised


def test_the_csr_lane_ignores_the_payload_budget(monkeypatch):
    """The per-particle source payload is never materialised from the placeholder.

    With the CSR lane active the rectangle is a one-block placeholder, so the
    payload-size estimate is tiny and fit any budget. With the library's default
    budget (1024 MB) the prepare materialised a payload from that one block, the
    evaluation took the pairs kernel instead of the CSR lane, and each leaf saw
    only its first block of neighbours: rel-L2 0.070 at 2e5 and 0.113 at 5e4 on an
    A100 (2026-10-09). The benches set the budget to 0 and never saw it.
    """
    no_payload, materialised_at_zero = _csr_lane_forces_at_budget(monkeypatch, "0")
    default_budget, materialised_by_default = _csr_lane_forces_at_budget(
        monkeypatch, "1024"
    )
    assert not materialised_at_zero
    assert not materialised_by_default
    np.testing.assert_array_equal(default_budget, no_payload)
    positions, masses = _make_inputs("clustered", N)
    assert _rel_l2(no_payload, _direct_sum_accelerations(positions, masses)) < 1e-2
