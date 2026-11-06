"""The escaper path's cost (wave 3, ESC): step time and peak memory with and without.

Runs ``ProductionRunner`` on a 1e8 Sgr IC for ``--segments`` segments of ``--steps``
steps and records every edge (its segment's wall time, retries, the diagnostics with
the escaper fields) and the device's peak memory, in one JSON row. Arms:

``esc``       escapers ON (``EscaperPolicy(r_escape=--r-escape)``) on the IC as given:
              the UNCLIPPED IC (a 46 Mpc box, a tree 68 levels deep without the
              escapers);
``off``       escapers OFF on the IC with every particle beyond ``--clip`` pulled
              radially onto it (same N; the box then fits 64 levels): the baseline;
``esc_clip``  escapers ON on that clipped IC with ``r_escape`` past the clip (nobody
              escapes: the default programs, the guard's own control).

The first segment holds the compiles; time per step is the median of the later
segments' ``wall_s / steps``, and ``edge_s`` the run's wall time outside the
segments (diagnostics, energies, re-splits) divided by the edges after the first.

Usage (one card)::

    python bench/escaper_timing.py --ic sgr_light_1e8_noclip.npz --arm esc \\
        --steps 10 --segments 6 --out row.json

On a shared card the times are contended (functional only); timings come from a
dedicated node (``bench/slurm/escaper_timing.sbatch``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time

import numpy as np


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ic", required=True, help="an .npz with state0 [N,2,3] and mass")
    ap.add_argument("--arm", choices=("esc", "off", "esc_clip"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--r-escape", type=float, default=80.0, help="code units (10 kpc)")
    ap.add_argument("--clip", type=float, default=80.0)
    ap.add_argument("--mac", choices=("dehnen", "dehnen_error"), default="dehnen")
    ap.add_argument("--energy-every", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    a = _args()
    import jax
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod
    from jaccpot.escapers import EscaperPolicy
    from jaccpot.production import GalaxySettings, ProductionRunner, SegmentConfig

    z = np.load(a.ic)
    state = np.asarray(z["state0"], np.float32)
    mass = np.asarray(z["mass"], np.float32)
    del z
    row: dict = {"args": vars(a), "n": int(state.shape[0])}
    if a.arm in ("off", "esc_clip"):
        r = np.linalg.norm(state[:, 0].astype(np.float64), axis=1)
        far = r > a.clip
        state[far, 0] = (state[far, 0] * (a.clip / r[far])[:, None]).astype(np.float32)
        row["n_clipped"] = int(far.sum())
    policy = None
    if a.arm == "esc":
        policy = EscaperPolicy(r_escape=a.r_escape)
    elif a.arm == "esc_clip":
        policy = EscaperPolicy(r_escape=2.0 * a.clip)
    base = FastMultipoleMethod(preset="galaxy").advanced
    if a.mac == "dehnen":
        settings = GalaxySettings(mac_type="dehnen", adaptive_eps=None, theta=0.7)
        solver = FastMultipoleMethod(
            preset="galaxy",
            softening=1.5e-3,
            theta=0.7,
            advanced=dataclasses.replace(base, mac_type="dehnen"),
        )
    else:
        settings = GalaxySettings()
        solver = FastMultipoleMethod(preset="galaxy", softening=1.5e-3)
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(
            steps=a.steps, donate_input=True, diagnostics_every_energy=a.energy_every
        ),
        escapers=policy,
    )
    state_d = jnp.asarray(state)
    masses_d = jnp.asarray(mass)
    del state, mass
    t0 = time.perf_counter()
    res = runner.run(state_d, masses_d, dt=1e-3, num_steps=a.steps * a.segments)
    row["wall_s"] = time.perf_counter() - t0
    seg = [e.wall_s for e in res.edges[1:]]
    row["segment_wall_s"] = seg
    row["step_s"] = statistics.median(seg[1:]) / a.steps if len(seg) > 1 else None
    row["edge_s"] = (row["wall_s"] - sum(seg)) / max(1, len(seg)) if seg else None
    row["edges"] = [
        {
            "step": e.step,
            "retries": e.retries,
            "wall_s": e.wall_s,
            "diag": e.diagnostics,
        }
        for e in res.edges
    ]
    row["escapers"] = (
        None
        if res.escapers is None
        else {k: v for k, v in res.escapers.items() if k != "order"}
    )
    stats = jax.devices()[0].memory_stats() or {}
    row["peak_bytes"] = int(stats.get("peak_bytes_in_use", 0))
    row["peak_bytes_per_particle"] = row["peak_bytes"] / row["n"]
    row["device"] = str(jax.devices()[0].device_kind)
    with open(a.out, "w") as fh:
        json.dump(row, fh, indent=1, default=str)
    print(
        f"[done] {a.arm}: step {row['step_s']} s, edge {row['edge_s']} s, peak "
        f"{row['peak_bytes'] / 2**30:.2f} GiB ({row['peak_bytes_per_particle']:.0f} B/p)",
        flush=True,
    )


if __name__ == "__main__":
    main()
