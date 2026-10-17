"""``FusedForceProvider.potential``: the fused lane's O(N) potential (S3 session 2).

The potential of the self-gravity per particle, ``phi_i = -G sum_{j != i} m_j
psi(r_ij)``, from the near field's fourth lane and the far field's L2P value
(``evaluate_large_n_potential``), against an fp64 direct sum with the solver's own
softening kernel, on CPU at N = 512 (four blobs, the production leaf partition):

* on the pure-JAX near field (``use_pallas=False``: the generic near potential)
  with Plummer softening, and on the CSR near field in interpret mode (the
  production direct kernel) with ``ferrers3``;
* from the eager prepare's topology and from a rebuild inside a jit (the runner's
  energy program), equal;
* with traced force parameters on the CSR lane: ``G`` scales it.

The error bound is the far field's at this operating point (p = 4, theta 0.6):
measured 3.4e-4 (Plummer, whose far field is Newtonian) and 7.7e-5 (ferrers3),
against the forces' 7.1e-3 and 2.2e-3.

``jax.default_backend`` reports ``"gpu"``, as ``tests/characterization/
test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.softening import pair_factors, softening_params_np

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING = 1e-2

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
)
#: the CSR near field in interpret mode, with the prepacked payload it reads
_CSR_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _solver(*, use_pallas, kernel):
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        softening_kernel=kernel,
        working_dtype=jnp.float32,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _blobs():
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    masses = np.abs(rng.normal(size=N)) + 0.5
    return jnp.asarray(pos, jnp.float32), jnp.asarray(masses, jnp.float32)


def direct_potential(positions, masses, *, kernel, softening, G=1.0):
    """fp64 ``-G sum_{j != i} m_j psi(r_ij)`` with the solver's pair kernel."""
    x = np.asarray(positions, np.float64)
    m = np.asarray(masses, np.float64)
    d = x[:, None, :] - x[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    _, psi, _ = pair_factors(
        r2, softening_params_np(kernel, softening), kernel, potential=True, xp=np
    )
    np.fill_diagonal(psi, 0.0)
    return -G * (psi * m[None, :]).sum(axis=1)


@pytest.mark.parametrize(
    "route, kernel, bound",
    [("purejax", "plummer", 1e-3), ("csr", "ferrers3", 3e-4)],
)
def test_the_potential_is_the_direct_sum(monkeypatch, route, kernel, bound):
    """The fast lane's potential against the fp64 direct sum: eager, rebuilt
    inside a jit (bitwise the eager one) and, on the CSR lane, with traced
    parameters."""
    from jaccpot import ForceParams
    from jaccpot.runtime.fused_provider import FusedForceProvider

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)
    if route == "csr":
        for key, value in _CSR_ENV.items():
            monkeypatch.setenv(key, value)
    x, m = _blobs()
    ref = direct_potential(x, m, kernel=kernel, softening=SOFTENING)
    prov, topology = FusedForceProvider.prepare(
        _solver(use_pallas=route == "csr", kernel=kernel),
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
    )
    eager = np.asarray(prov.potential(topology), np.float64)
    rel = np.linalg.norm(eager - ref) / np.linalg.norm(ref)
    assert rel < bound, f"potential rel-L2 {rel:.3e}"

    @jax.jit
    def rebuilt(xi, mi, params):
        topo = prov.rebuild(xi, mi)
        return prov.potential(topo, params), prov.capacity_ok(topo)

    phi, ok = rebuilt(x, m, None)
    assert bool(ok)
    assert np.array_equal(np.asarray(phi), np.asarray(eager, np.float32))
    if route == "csr":
        # traced G and eps2 reach the near kernel and the far field
        traced = ForceParams(
            G=jnp.asarray(2.0, jnp.float32),
            eps2=jnp.asarray(SOFTENING**2, jnp.float32),
        )
        phi2, _ = rebuilt(x, m, traced)
        np.testing.assert_allclose(
            np.asarray(phi2, np.float64), 2.0 * eager, rtol=1e-5, atol=0.0
        )
