"""A tiny random subsample of a big ``state0`` / ``mass`` IC, for the bundle's CPU smoke and preflight.

The escaper rows (jobs ``escaper``, ``mesc_*``) read a whole IC file (the 1e8
unclipped Sgr merger, 2.8 GB). The smoke and the preflight run their exact commands at
a tiny N on a file this script writes: ``n`` rows drawn without replacement (seeded),
masses times ``N_all / n`` (the same total mass), ``state0`` and ``mass`` only. A random
draw keeps the far tail in proportion (~0.2 % of the Sgr IC lies beyond r_escape = 80),
so the escaper path still sees escapers at 2e5 rows.

    python bench/slurm/h0/tiny_ic.py IN.npz OUT.npz N [--seed 0]
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with np.load(args.src) as z:
        state = z["state0"]
        mass = z["mass"]
        n_all = int(state.shape[0])
        n = min(args.n, n_all)
        idx = np.sort(np.random.default_rng(args.seed).choice(n_all, n, replace=False))
        out_state = np.asarray(state[idx], np.float32)
        out_mass = (np.asarray(mass[idx], np.float64) * (n_all / n)).astype(np.float32)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, state0=out_state, mass=out_mass)
    os.replace(tmp, args.out)
    r = np.linalg.norm(out_state[:, 0].astype(np.float64), axis=1)
    print(
        f"tiny_ic: {n} of {n_all} rows -> {args.out} ({int((r > 80).sum())} beyond r = 80)"
    )


if __name__ == "__main__":
    main()
