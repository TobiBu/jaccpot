"""fp64 direct-sum references (4096 targets, seed 12345) for the jz-fmm vs jaccpot comparison.

Writes the harness's cache files, which both codes/jzfmm_force_eval.py and
jaccpot's bench/fused_memory_budget.py read. First checks that the harness's
plummer_clipped draws the same particles as the jaccpot bench's inline copy.
"""

import os
import sys
import time

import numpy as np

BENCH = os.environ["BENCH_DIR"]
sys.path.insert(0, BENCH)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from common.ic import plummer_clipped  # noqa: E402
from common.reference import direct_accelerations  # noqa: E402


def bench_inline(n, seed=0, rmax=20.0):
    rng = np.random.default_rng(seed)
    x_max = rmax**3 / (rmax**2 + 1.0) ** 1.5
    x = rng.uniform(0.0, x_max, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1).astype(
        np.float32
    )
    return pos, np.full(n, 1.0 / n, np.float32)


p1, m1 = plummer_clipped(1_000_003, seed=0)
p2, m2 = bench_inline(1_000_003)
assert np.array_equal(p1, p2) and np.array_equal(m1, m2), "IC generators differ"
print("IC generators identical (1e6+3 particles)", flush=True)

K, soft = 4096, 1e-7
for n in [int(x) for x in sys.argv[1:]]:
    path = os.path.join(
        BENCH,
        "artifacts",
        "reference",
        f"direct_fp64_plummer_clipped{n}_soft{soft:g}_ref{K}_seed12345.npy",
    )
    if os.path.exists(path):
        print(f"{n}: cached {path}", flush=True)
        continue
    pos, mass = plummer_clipped(n, seed=0)
    idx = np.sort(np.random.default_rng(12345).choice(n, K, replace=False))
    t0 = time.perf_counter()
    ref = direct_accelerations(
        pos, mass, G=1.0, softening=soft, block_size=0, target_indices=idx
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, ref)
    print(f"{n}: computed in {time.perf_counter() - t0:.1f} s -> {path}", flush=True)
