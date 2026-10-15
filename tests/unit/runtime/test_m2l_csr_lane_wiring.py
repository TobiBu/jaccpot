"""The CSR M2L lane (default on where it lowers) is (a) taken and (b) force-neutral.

``JACCPOT_STATIC_STRICT_FUSED_M2L_CSR`` (default ``1`` since 2026-09-10; ``0``
restores the chunked pure-JAX lanes) routes the flat real-basis M2L of
``_solidfmm_downward_accumulate_from_multipoles`` to the pair-per-lane kernel's
custom VJP,
:func:`jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`. On CPU
the kernel runs in interpret mode (``JACCPOT_M2L_CSR_INTERPRET=1``). A lane that
is silently not reached would pass any parity check, so the kernel entry is
counted. The per-target and tiled kernels ``JACCPOT_M2L_CSR_KERNEL`` used to
select were removed in the 2026-10 cleanup (X5); since Z-b the variable is read
nowhere.
"""

from __future__ import annotations

import numpy as np

import jaccpot.pallas.m2l_real_csr as csr_mod
import jaccpot.pallas.m2l_real_csr_lanes as lanes_mod
from jaccpot.runtime.kernels._downward_prep import _m2l_csr_pallas_active


def test_default_on_exactly_where_the_kernel_lowers(monkeypatch):
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", raising=False)
    monkeypatch.delenv("JACCPOT_M2L_CSR_INTERPRET", raising=False)
    assert _m2l_csr_pallas_active() is csr_mod.pallas_m2l_real_csr_supported()
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    assert _m2l_csr_pallas_active() is True


def test_flag_zero_switches_the_lane_off(monkeypatch):
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    assert _m2l_csr_pallas_active() is False


def test_flag_on_cpu_needs_interpret(monkeypatch):
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.delenv("JACCPOT_M2L_CSR_INTERPRET", raising=False)
    import jax

    if jax.default_backend() != "gpu":
        assert _m2l_csr_pallas_active() is False
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    assert _m2l_csr_pallas_active() is True


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def test_csr_lane_is_taken_and_matches_the_chunked_lane(monkeypatch):
    import jax.numpy as jnp

    from jaccpot import (
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    pos, mass = _plummer(3000)

    def solve():
        # the default (non-strict) runtime: CPU-friendly, real basis, same downward
        solver = FastMultipoleMethod(
            basis="real",
            theta=0.6,
            G=1.0,
            softening=1e-3,
            working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=32),
                nearfield=NearFieldConfig(mode="auto"),
                mac_type="dehnen",
            ),
            fixed_order=3,
        )
        acc = solver.compute_accelerations(
            jnp.asarray(pos), jnp.asarray(mass), leaf_size=32, max_order=3, theta=0.6
        )
        return np.asarray(acc, np.float64)

    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
    a_ref = solve()

    calls = {"n": 0}
    real_kernel = lanes_mod.m2l_real_csr_lanes_pallas_cvjp

    def counting(*a, **k):
        calls["n"] += 1
        return real_kernel(*a, **k)

    # the lane reads the entry from its module at call time (a local import)
    monkeypatch.setattr(lanes_mod, "m2l_real_csr_lanes_pallas_cvjp", counting)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    a_csr = solve()
    assert calls["n"] >= 1, "the CSR lane was never entered"
    rel = np.linalg.norm(a_csr - a_ref) / np.linalg.norm(a_ref)
    assert np.all(np.isfinite(a_csr))
    assert rel < 2e-5, rel


def _merged_cross_list(seed=0, n_local=40, n_import=24, order=4):
    """A local far list with cross-domain pairs behind it, as the multi-GPU lane
    hands it to the M2L (``_prepare_solidfmm_downward_sweep``): the local half is a
    live prefix padded with -1, the cross half has -1 holes, and the imported
    sources sit at ``[n_local, n_local + n_import)``. No count marks a live prefix.
    """
    import jax.numpy as jnp

    from jaccpot.operators.real_harmonics import sh_size
    from jaccpot.runtime.dtypes import INDEX_DTYPE

    rng = np.random.default_rng(seed)
    n = n_local + n_import
    # unit-lattice centres, so no pair is closer than 1 and neither half dominates
    lattice = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), -1)
    lattice = lattice.reshape(-1, 3)
    centers = np.concatenate(
        [
            lattice[rng.permutation(64)[:n_local]],
            lattice[rng.permutation(64)[:n_import]] + np.array([2.0, 0.5, 0.5]),
        ]
    )
    multipoles = rng.normal(size=(n, sh_size(order)))
    local_live, local_cap, cross_cap = 90, 128, 96
    loc_t = rng.integers(0, n_local, local_live)
    loc_s = (loc_t + rng.integers(1, n_local, local_live)) % n_local
    pad = np.full(local_cap - local_live, -1)
    cross_t = rng.integers(0, n_local, cross_cap)
    cross_s = rng.integers(n_local, n, cross_cap)
    hole = rng.uniform(size=cross_cap) < 0.3
    src = np.concatenate([loc_s, pad, np.where(hole, -1, cross_s)])
    tgt = np.concatenate([loc_t, pad, np.where(hole, -1, cross_t)])
    return (
        jnp.asarray(multipoles, jnp.float32),
        jnp.asarray(centers, jnp.float32),
        jnp.asarray(src, INDEX_DTYPE),
        jnp.asarray(tgt, INDEX_DTYPE),
        n_local,
        local_cap,
    )


def _accumulate(multipoles, centers, src, tgt, n_local, *, chunk_size, order=4):
    import jax.numpy as jnp

    from jaccpot.runtime.kernels._downward_prep import (
        _solidfmm_downward_accumulate_from_multipoles,
    )

    out = _solidfmm_downward_accumulate_from_multipoles(
        jnp.zeros((n_local, multipoles.shape[1]), multipoles.dtype),
        multipoles,
        n_targets=n_local,
        centers=centers,
        src=src,
        tgt=tgt,
        pair_count=int(src.shape[0]),
        active_pair_count=None,
        order=order,
        total_nodes=n_local,
        chunk_size=chunk_size,
    )
    return np.asarray(out, np.float64)


def test_flat_lanes_take_a_merged_cross_domain_list_like_the_csr_lane(monkeypatch):
    """Review G8: with the CSR lane off (pre-Ampere, ``..._FUSED_M2L_CSR=0``, or a
    CPU) the multi-GPU lane's merged list reached the flat lanes with more source
    rows than targets (a jaxtyping ``nodes`` mismatch) and no live-prefix count
    (``start_idx < None``). Both flat lanes must now give the CSR lane's field."""
    multipoles, centers, src, tgt, n_local, local_cap = _merged_cross_list()

    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    csr = _accumulate(multipoles, centers, src, tgt, n_local, chunk_size=4096)

    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
    full = _accumulate(multipoles, centers, src, tgt, n_local, chunk_size=4096)
    chunked = _accumulate(multipoles, centers, src, tgt, n_local, chunk_size=16)
    local_only = _accumulate(
        multipoles,
        centers,
        src[:local_cap],
        tgt[:local_cap],
        n_local,
        chunk_size=4096,
    )

    scale = np.linalg.norm(csr)
    # the cross half is a real part of the field, not lost behind the padding
    assert np.linalg.norm(csr - local_only) > 0.1 * scale
    for name, got in (("full-batch", full), ("chunked", chunked)):
        rel = np.linalg.norm(got - csr) / scale
        assert rel < 1e-5, f"{name} lane vs CSR lane: rel {rel:.3e}"
