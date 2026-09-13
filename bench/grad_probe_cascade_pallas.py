"""Does jax.grad still work with the sub-10ms defaults on? Which lanes does it touch?"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from jaccpot import FastMultipoleMethod
from jaccpot.autodiff import direct_sum_gravitational_acceleration

n = int(os.environ.get("PROBE_N", 512))
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, n)
r = 1.0 / np.sqrt(x ** (-2 / 3) - 1)
mu = rng.uniform(-1, 1, n)
ph = rng.uniform(0, 2 * np.pi, n)
s = np.sqrt(1 - mu * mu)
pos = jnp.asarray(
    np.stack([r * s * np.cos(ph), r * s * np.sin(ph), r * mu], 1), jnp.float64
)
mass = jnp.asarray(np.full(n, 1.0 / n), jnp.float64)
soft = 1e-2


def report(label, fn):
    try:
        g = fn()
        print(f"{label}: OK  |grad| = {float(jnp.linalg.norm(g)):.6e}")
        return np.asarray(g)
    except Exception as e:
        print(f"{label}: FAILED  {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return None


loss = lambda a: jnp.sum(a**2)
ref = report(
    "direct sum (oracle)",
    lambda: jax.grad(
        lambda p: loss(
            direct_sum_gravitational_acceleration(p, mass, G=1.0, softening=soft)
        )
    )(pos),
)
fmm = FastMultipoleMethod(theta=0.5, softening=soft, G=1.0)
print(
    "  nearfield grad lane:",
    __import__("jaccpot.runtime.grad_options", fromlist=["x"])
    .resolve_grad_options(None, num_particles=n, supports_fast_lane=True)
    .nearfield_lane,
)
state = fmm.prepare_state(pos, mass, max_order=4, leaf_size=32)
print(
    "  CSR near lane enabled:",
    __import__(
        "jaccpot.nearfield._fast_lane", fromlist=["x"]
    )._nearfield_csr_lane_enabled(),
)
got = report(
    f"FMM differentiable_accelerations (CASCADE_PALLAS={os.environ.get('JACCPOT_CASCADE_PALLAS','unset')})",
    lambda: jax.grad(lambda p: loss(fmm.differentiable_accelerations(state, p, mass)))(
        pos
    ),
)
if ref is not None and got is not None:
    num = np.linalg.norm(got - ref)
    den = np.linalg.norm(ref)
    print(f"  grad rel-L2 vs oracle: {num/den:.3e}")
