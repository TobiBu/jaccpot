"""Score the 25M production runs (c1_25m_run.py) against the dt 2.5e-4 reference, with the C1 metrics.

usage: python c1_25m_score.py OUT.json REF_NAME RUN_NAME... (names of <name>.json in this
directory; snapshots and particle energies under /export/scratch/tbuck/c1_criterion_2026-10/25m/<name>/)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = "/export/scratch/tbuck/c1_criterion_2026-10/25m"
IC = np.load("/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz", mmap_mode="r")
RD = float(IC["rdisk_code"])
R_EDGES = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6)
LAGR = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)


def load(name):
    with open(os.path.join(HERE, name + ".json")) as fh:
        run = json.load(fh)
    snaps = sorted(glob.glob(os.path.join(SNAP, name, "snapshots", "*.npz")))
    final = np.load(snaps[-1])
    run["final_state"] = np.asarray(final["state"])
    run["final_time"] = float(final["time"])
    pe = np.load(os.path.join(SNAP, name, "particle_energy.npz"))
    run["e0"], run["e_end"], run["comp"] = pe["e0"], pe["e_end"], pe["comp"]
    run["x0"] = np.asarray(np.load(snaps[0])["state"])[:, 0]
    return run


def profiles(state, comp):
    disc = comp == 0
    x, v = state[disc, 0].astype(np.float64), state[disc, 1].astype(np.float64)
    rc = np.hypot(x[:, 0], x[:, 1])
    v_r = (x[:, 0] * v[:, 0] + x[:, 1] * v[:, 1]) / rc
    v_p = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rc
    b = np.digitize(rc, np.asarray(R_EDGES) * RD) - 1
    out = {k: [] for k in ("count", "Sigma", "sigma_z", "sigma_R", "v_phi")}
    for i, (lo, hi) in enumerate(zip(R_EDGES[:-1], R_EDGES[1:])):
        s = b == i
        out["count"].append(int(s.sum()))
        out["Sigma"].append(float(s.sum() / (np.pi * RD * RD * (hi * hi - lo * lo))))
        out["sigma_z"].append(float(np.std(v[s, 2])))
        out["sigma_R"].append(float(np.std(v_r[s])))
        out["v_phi"].append(float(np.mean(v_p[s])))
    return out


def diffusion(run):
    de = run["e_end"].astype(np.float64) - run["e0"].astype(np.float64)
    r0 = np.linalg.norm(run["x0"], axis=1)
    inner = r0 < np.percentile(r0, 2.0)
    comp = run["comp"]
    return dict(
        disc=float(np.sqrt(np.mean(de[comp == 0] ** 2))),
        bulge=float(np.sqrt(np.mean(de[comp == 1] ** 2))),
        inner_2pct=float(np.sqrt(np.mean(de[inner] ** 2))),
        disc_mean=float(np.mean(de[comp == 0])),
    )


def score(run, ref, p_ref, d_ref):
    rows = run["rows"]
    e = [
        abs(r["diagnostics"]["energy_drift"])
        for r in rows
        if r["diagnostics"] and r["diagnostics"].get("energy_drift") is not None
    ]
    dl = [r["diagnostics"]["angular_momentum_drift"] for r in rows if r["diagnostics"]]
    p = profiles(run["final_state"], run["comp"])
    good = np.asarray(p_ref["count"]) >= 1000
    prof = {}
    for k in ("Sigma", "sigma_z", "sigma_R", "v_phi"):
        d = (np.asarray(p[k]) - np.asarray(p_ref[k])) / np.abs(np.asarray(p_ref[k]))
        prof[k] = dict(
            rms=float(np.sqrt(np.mean(d[good] ** 2))),
            mean=float(np.mean(d[good])),
            max=float(np.max(np.abs(d[good]))),
        )
    t = [r["time"] for r in rows]
    t_ref = [r["time"] for r in ref["rows"]]
    a2 = {}
    for band in ("0-3", "0-1", "1-2", "2-3"):
        v = np.asarray([r["A2"][band] for r in rows])
        vr = np.interp(t, t_ref, [r["A2"][band] for r in ref["rows"]])
        a2[band] = dict(
            max_abs_diff=float(np.max(np.abs(v - vr))),
            end=float(v[-1]),
            ref_end=float(vr[-1]),
            ref_max=float(np.max(vr)),
        )
    lag = {
        k: (
            (
                np.asarray(rows[-1]["lagrange"][k])
                - np.asarray(ref["rows"][-1]["lagrange"][k])
            )
            / np.asarray(ref["rows"][-1]["lagrange"][k])
        ).tolist()
        for k in ("all", "disc", "bulge")
    }
    d = diffusion(run)
    return dict(
        dt=run["dt"],
        steps=run["num_steps"],
        steps_per_s_hint=run["steps_per_s_hint"],
        mem_peak_gib=run["mem_peak_gib"],
        device=run["device"],
        final_time=run["final_time"],
        max_abs_energy_drift=float(max(e)),
        final_energy_drift=float(rows[-1]["diagnostics"]["energy_drift"]),
        max_angular_momentum_drift=float(max(dl)),
        energy_diffusion=d,
        energy_diffusion_ratio={
            k: d[k] / d_ref[k] for k in ("disc", "bulge", "inner_2pct")
        },
        profiles=prof,
        A2=a2,
        lagrange_rel=lag,
        order_check=run["order_check"],
    )


def main():
    out_path, ref_name, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref = load(ref_name)
    p_ref = profiles(ref["final_state"], ref["comp"])
    p0 = profiles(
        np.stack(
            [
                ref["x0"],
                np.asarray(
                    np.load(
                        sorted(
                            glob.glob(
                                os.path.join(SNAP, ref_name, "snapshots", "*.npz")
                            )
                        )[0]
                    )["state"]
                )[:, 1],
            ],
            1,
        ),
        ref["comp"],
    )
    d_ref = diffusion(ref)
    out = dict(
        reference=ref_name,
        reference_self=score(ref, ref, p_ref, d_ref),
        reference_profiles=dict(t0=p0, t_end=p_ref, R_edges_over_Rd=list(R_EDGES)),
        reference_lagrange_t0=ref["rows"][0]["lagrange"],
        reference_lagrange_end=ref["rows"][-1]["lagrange"],
        runs={},
    )
    for name in names:
        run = load(name)
        out["runs"][name] = score(run, ref, p_ref, d_ref)
        print(name, json.dumps(out["runs"][name]), flush=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
