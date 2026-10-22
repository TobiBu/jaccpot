"""C1: which timestep criterion a galaxy disc should use, shared steps against HOLD, at equal cost.

usage: python bench/c1_criterion_calibration.py probe OUT.json [--n 100000] [--offset 0]
       python bench/c1_criterion_calibration.py run OUT.npz --scheme shared --dt DT --t-end T
       python bench/c1_criterion_calibration.py run OUT.npz --scheme hold --criterion acc|dyn|dynfloor
           --eta ETA --kmax K --t-end T [--floor-c C]
       python bench/c1_criterion_calibration.py score OUT.json --ref A.npz --ref2 B.npz RUN.npz ...

The IC is a random subsample of the 25M disc+bulge production IC (the file is shuffled, so
``[offset, offset + n)`` is an independent draw from the same agama distribution function;
masses rescaled by ``n_all / n``, mass-weighted centre and drift removed) in its analytic NFW
halo, with the production softening (``ferrers3``, Plummer-equivalent 1.5e-3, G = 1). The
self-gravity is a direct sum on the GPU (one Pallas kernel): fp32 pair arithmetic on fp32
positions, the per-tile sums carried in fp64, so a force is good to ~2e-7 and the pair force
of ``(i, j)`` is bitwise the negative of ``(j, i)``'s. The integrator state is fp64.

Schemes:

* ``shared``: KDK leapfrog, one ``dt`` for every particle. Both criteria give a shared run
  the same family: its step is ``eta * min_i tau_i`` for either, so a shared run is labelled
  by its ``dt`` only.
* ``hold``: the HOLD map (Pelupessy et al. 2012; GADGET-4's hierarchical gravity), the map S6
  builds. Rungs ``0..k_max`` (``dt_m = dt_max / 2**m``), assigned at every base step from the
  total acceleration as nornax ``block_kdk`` does (``rung = clip(ceil(log2(dt_max / dt_i)), 0,
  k_max)``, the fastest particle of t = 0 on rung ``k_max``). A pair is kicked at its slower
  end's level ``min(rung_i, rung_j)``, both ends with one weight, so momentum is exact. At an
  interior boundary only the synchronised set ``S_l = {rung >= l}`` is touched: one pass
  over its pairs with weight ``dt_max * max(2**-rung_i, 2**-rung_j)`` (the closing and opening
  kicks of every level ``>= l`` merged). A base boundary pays two full passes: the closing
  kick (which also gives the total force for the new rungs) and the opening kick on the new
  rungs (nornax's ``n_sub + 1`` with S6's merge).

Criteria (``dt_i = eta * tau_i``):

* ``acc``: ``tau = sqrt(eps / |a|)`` (nornax ``acceleration_timestep``, GADGET);
* ``dyn``: ``tau = sqrt(sqrt(r^2 + eps^2) / |a|)``, ``r`` from the halo centre: about the
  orbital time over 2 pi;
* ``dynfloor``: ``dyn`` with no step below ``floor_c * sqrt(eps^3 / (G M(<eps)))`` (the
  dynamical time at the softening length; see the record for why).

Costs, per run (both are counted, not timed):

* pair work: the ordered pairs a direct sum evaluates (``N^2`` per shared step);
* particle evaluations: the targets the FMM evaluates, summed over force evaluations
  (``N`` per shared step). HOLD on the fused lane evaluates the padded subset ``S_m`` of each
  active level ``m`` (the level-``m`` kick needs ``F(S_m) - F(S_{m+1})``): ``sum_{m >= l}
  |S_m|`` at a boundary of floor ``l`` (``evals``). ``evals_weighted`` is one weighted
  evaluation of ``S_l`` per boundary (S0's model; needs per-rung source multipoles), and
  ``evals_padded`` pads each level to its largest subset of the run (a static capacity plan).

``--static`` replaces the self-gravity by a fixed field (``make_static``): the IC's particles
become test particles, so encounters and chaos are gone and every scheme pays O(N) per
boundary, while the criteria keep the production timestep spread. ``--rungs-once`` keeps
the rungs of t = 0 (the diagnostic for the base-step reassignment).

``score`` compares each run's end state with the reference ``--ref`` (a shared run at a much
smaller step) and gives the noise floor from ``--ref2`` (the same at half that step): the
energy error, angular and linear momentum (the pair impulse balance; the halo changes the
total momentum), Sigma(R), sigma_z(R), sigma_R(R), mean v_phi(R) of the disc, A2(t), the bulge
and central Lagrangian radii, and the per-particle energy diffusion. Particle-wise
differences are chaos at this N, so every structural score is a profile. Recorded in
``docs/production_2026-10.md`` (C1); rows in ``bench/results/c1_criterion_2026-10/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

IC_PATH = "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
# Pallas tiles: 32 targets x 64 sources, 16 source tiles summed in fp32 per fp64 add
_BT, _BS, _CH = 32, 64, 16
_TILE = _BS * _CH


def _args(argv: Optional[list] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("probe", "run"):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("--ic", default=IC_PATH)
        p.add_argument("--n", type=int, default=100_000)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--softening", type=float, default=1.5e-3)
        p.add_argument("--static", action="store_true")
        if name == "probe":
            p.add_argument("--kmax", default="3,4,5,6,7,8")
            p.add_argument("--no-energy", action="store_true")
        else:
            p.add_argument("--scheme", required=True, choices=("shared", "hold"))
            p.add_argument(
                "--criterion", default="acc", choices=("acc", "dyn", "dynfloor")
            )
            p.add_argument("--eta", type=float, default=None)
            p.add_argument("--dt", type=float, default=None)
            p.add_argument("--kmax", type=int, default=4)
            p.add_argument("--floor-c", type=float, default=0.0)
            p.add_argument("--t-end", type=float, required=True)
            p.add_argument("--n-out", type=int, default=16)
            p.add_argument("--refine", action="store_true")
            p.add_argument("--rungs-once", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("out")
    s.add_argument("runs", nargs="+")
    s.add_argument("--ref", required=True)
    s.add_argument("--ref2", required=True)
    s.add_argument("--ic", default=IC_PATH)
    t = sub.add_parser("table")
    t.add_argument("scores")
    return ap.parse_args(argv)


ARGS = argparse.Namespace()  # set by main()

# ----------------------------------------------------------------------------- the IC


def load_ic(n: int, offset: int) -> dict:
    """``n`` particles of the shuffled 25M IC from ``offset``, masses rescaled, recentred."""
    ic = np.load(ARGS.ic, mmap_mode="r")
    n_all = int(ic["state0"].shape[0])
    st = np.asarray(ic["state0"][offset : offset + n], np.float64)
    m = np.asarray(ic["mass"][offset : offset + n], np.float64) * (n_all / n)
    x, v = st[:, 0].copy(), st[:, 1].copy()
    mt = m.sum()
    x -= (m[:, None] * x).sum(0) / mt
    v -= (m[:, None] * v).sum(0) / mt
    return dict(
        x=x,
        v=v,
        m=m,
        comp=np.asarray(ic["component"][offset : offset + n], np.int8),
        G=float(ic["G_code"]) if "G_code" in ic.files else 1.0,
        halo_m=float(ic["halo_mass_code"]),
        halo_rs=float(ic["halo_rs_code"]),
        rdisk=float(ic["rdisk_code"]),
        bulge_m=float(ic["bulge_mass_code"]),
        bulge_a=float(ic["bulge_scale_code"]),
        disk_m=float(ic["disk_mass_code"]),
        disk_h=float(ic["hdisk_code"]) if "hdisk_code" in ic.files else None,
    )


# ----------------------------------------------------------------------- direct sum


def _interpret() -> bool:
    """Pallas interpret mode off the GPU; only with ``C1_ALLOW_CPU=1`` (tests, checks)."""
    if jax.devices()[0].platform == "gpu":
        return False
    if os.environ.get("C1_ALLOW_CPU") != "1":
        raise SystemExit(
            f"no GPU ({jax.devices()}); set C1_ALLOW_CPU=1 for interpret mode"
        )
    return True


def make_pair_pass(n: int, eps: float, g_const: float):
    """The direct-sum pass ``(W, U) = pass(x, m, p, q, n_active)``.

    Targets and sources are the prefix ``[0, n_active)`` (the caller sorts by rung,
    fastest first, so a HOLD subset is a prefix). ``U_i = G sum_j m_j g(r_ij) d_ij``
    (the ferrers3 pair force, ``d_ij = x_j - x_i``) and ``W_i`` the same with each pair
    weighted by ``max(p_i, p_j) + max(q_i, q_j)``. Rows past ``n_active`` are zero.
    """
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    from jaccpot.softening import pair_factors, softening_params_np

    params = [float(v) for v in softening_params_np("ferrers3", eps)]
    pad = -(-n // _TILE) * _TILE

    def kernel(n_ref, xt_ref, xs_ref, out_ref):
        na = n_ref[0]
        i0 = pl.program_id(0) * _BT
        tx, ty, tz, tp, tq = (xt_ref[:, c] for c in (0, 1, 2, 4, 5))
        lane = lax.broadcasted_iota(jnp.int32, (_BS,), 0)
        chunks = jnp.where(i0 < na, (na + _TILE - 1) // _TILE, 0)

        def tile(jb, acc):
            j0 = jb * _BS
            sx, sy, sz, sm, sp, sq = (xs_ref[pl.ds(j0, _BS), c] for c in range(6))
            sm = jnp.where(j0 + lane < na, sm, 0.0)
            dx = sx[None, :] - tx[:, None]
            dy = sy[None, :] - ty[:, None]
            dz = sz[None, :] - tz[:, None]
            g, _, _ = pair_factors(dx * dx + dy * dy + dz * dz, params, "ferrers3")
            cm = g * sm[None, :]
            cw = cm * (
                jnp.maximum(tp[:, None], sp[None, :])
                + jnp.maximum(tq[:, None], sq[None, :])
            )
            terms = (cw * dx, cw * dy, cw * dz, cm * dx, cm * dy, cm * dz)
            return tuple(a + b for a, b in zip(acc, terms))

        def chunk(cb, acc64):
            init = tuple(jnp.zeros((_BT, _BS), jnp.float32) for _ in range(6))
            a32 = lax.fori_loop(cb * _CH, cb * _CH + _CH, tile, init)
            return tuple(
                a + jnp.sum(b, axis=1).astype(jnp.float64) for a, b in zip(acc64, a32)
            )

        acc = lax.fori_loop(
            0, chunks, chunk, tuple(jnp.zeros((_BT,), jnp.float64) for _ in range(6))
        )
        for k in range(6):
            out_ref[:, k] = acc[k]
        for k in (6, 7):
            out_ref[:, k] = jnp.zeros((_BT,), jnp.float64)

    call = pl.pallas_call(
        kernel,
        grid=(pad // _BT,),
        in_specs=[
            pl.BlockSpec((1,), lambda i: (0,)),
            pl.BlockSpec((_BT, 8), lambda i: (i, 0)),
            pl.BlockSpec((pad, 8), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((_BT, 8), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((pad, 8), jnp.float64),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=_interpret(),
        name="c1_direct_sum",
    )

    def pair_pass(x, m, p, q, n_active):
        packed = jnp.zeros((pad, 8), jnp.float32)
        packed = packed.at[:n, 0:3].set(x.astype(jnp.float32))
        packed = packed.at[:n, 3].set(m.astype(jnp.float32))
        packed = packed.at[:n, 4].set(p.astype(jnp.float32))
        packed = packed.at[:n, 5].set(q.astype(jnp.float32))
        na = jnp.reshape(jnp.asarray(n_active, jnp.int32), (1,))
        out = call(na, packed, packed)[:n]
        # a target tile that straddles n_active computes its rows past it too
        out = jnp.where((jnp.arange(n) < na[0])[:, None], out, 0.0)
        return g_const * out[:, 0:3], g_const * out[:, 3:6]

    return pair_pass


def make_potential(n: int, eps: float, g_const: float, block: int = 256):
    """fp64 per-particle self-gravity potential ``phi_i`` (ferrers3, self pair excluded)."""
    from jaccpot.softening import pair_factors, softening_params

    params = softening_params("ferrers3", eps, jnp.float64)
    nb = -(-n // block)
    pad = nb * block

    @jax.jit
    def phi(x, m):
        xp = jnp.zeros((pad, 3)).at[:n].set(x)

        def blk(b):
            xt = jax.lax.dynamic_slice_in_dim(xp, b * block, block)
            d = x[None, :, :] - xt[:, None, :]
            _, psi, _ = pair_factors(
                jnp.sum(d * d, -1), params, "ferrers3", potential=True
            )
            it = b * block + jnp.arange(block)
            psi = jnp.where(it[:, None] == jnp.arange(n)[None, :], 0.0, psi)
            return -g_const * jnp.sum(psi * m[None, :], axis=1)

        return jax.lax.map(blk, jnp.arange(nb)).reshape(pad)[:n]

    return phi


# ------------------------------------------------------------------------ the halo


def make_halo(g_const: float, hm: float, rs: float):
    """AGAMA's NFW (the production IC's ``ic_velocity_potential``), as the other benches."""

    def acc(x):
        r2 = jnp.sum(x * x, 1)
        r = jnp.sqrt(jnp.maximum(r2, 1e-30))
        q = r / rs
        big = jnp.log1p(q) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
            r2 * (rs + r), 1e-30
        )
        ser = (1.0 / (2.0 * rs**3)) * (1.0 - (4.0 / 3.0) * q)
        return -(g_const * hm) * jnp.where(q < 1e-3, ser, big)[:, None] * x

    def pot(x):
        r = jnp.sqrt(jnp.maximum(jnp.sum(x * x, 1), 1e-30))
        return -g_const * hm * jnp.log1p(r / rs) / r

    return acc, pot


def make_static(ic: dict, eps: float):
    """``--static``: the galaxy as a fixed field, for test particles (no self-gravity).

    The NFW halo, a Hernquist bulge softened at ``eps`` (``r -> sqrt(r^2 + eps^2)``, a
    smooth core like the N-body centre's) and a Miyamoto-Nagai disc of the disc's mass
    and height whose rotation curve peaks where the exponential disc's does (``a + b =
    2.15 R_d / sqrt(2)``). The IC is not exactly in equilibrium in it, which does not
    matter: the scores compare runs with a converged run in the same field.
    """
    g = ic["G"]
    halo_acc, halo_pot = make_halo(g, ic["halo_m"], ic["halo_rs"])
    mb, ab, md, bd = ic["bulge_m"], ic["bulge_a"], ic["disk_m"], ic["disk_h"]
    ad = 2.15 * ic["rdisk"] / math.sqrt(2.0) - bd

    def parts(x):
        s = jnp.sqrt(jnp.sum(x * x, 1) + eps * eps)
        zeta = jnp.sqrt(x[:, 2] ** 2 + bd * bd)
        dd = x[:, 0] ** 2 + x[:, 1] ** 2 + (ad + zeta) ** 2
        return s, zeta, dd

    def acc(x):
        s, zeta, dd = parts(x)
        a_b = -(g * mb) / ((s + ab) ** 2 * s)
        c = -(g * md) / dd**1.5
        a_d = jnp.stack([c * x[:, 0], c * x[:, 1], c * x[:, 2] * (ad + zeta) / zeta], 1)
        return halo_acc(x) + a_b[:, None] * x + a_d

    def pot(x):
        s, _, dd = parts(x)
        return halo_pot(x) - g * mb / (s + ab) - g * md / jnp.sqrt(dd)

    return acc, pot


def _no_pairs(x, m, p, q, n_active):
    """``--static``: no self-gravity."""
    return jnp.zeros_like(x), jnp.zeros_like(x)


# -------------------------------------------------------------------- the criteria


def floor_time(ic: dict, eps: float) -> float:
    """``sqrt(eps^3 / (G M(<eps)))``: the analytic Hernquist bulge, disc and halo inside eps."""
    g = ic["G"]
    m_b = ic["bulge_m"] * eps**2 / (eps + ic["bulge_a"]) ** 2
    q = eps / ic["halo_rs"]
    m_h = ic["halo_m"] * (math.log1p(q) - q / (1.0 + q))
    y = eps / ic["rdisk"]
    m_d = ic["disk_m"] * (1.0 - (1.0 + y) * math.exp(-y))  # in the cylinder R < eps
    return math.sqrt(eps**3 / (g * (m_b + m_h + m_d)))


def make_tau(criterion: str, eps: float):
    """``tau_i(x, |a|)`` with ``dt_i = eta * tau_i`` (``dynfloor``'s floor is applied to dt)."""

    def tau(x, a_norm):
        an = a_norm + 1e-30
        if criterion == "acc":
            return jnp.sqrt(eps / an)
        return jnp.sqrt(jnp.sqrt(jnp.sum(x * x, 1) + eps * eps) / an)

    return tau


def rungs_of(dt_i, dt_max: float, k: int):
    """nornax ``timestep_to_rung``: ``clip(ceil(log2(dt_max / dt_i)), 0, k)``."""
    return jnp.clip(jnp.ceil(jnp.log2(dt_max / dt_i)), 0, k).astype(jnp.int32)


def _bounds(rung: np.ndarray, k: int) -> dict:
    """Speed-up bounds of HOLD over shared ``dt_min`` for fixed rungs (per base step)."""
    n = rung.size
    hist = np.bincount(rung, minlength=k + 1)
    n_ge = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    floors = [k - ((s & -s).bit_length() - 1) for s in range(1, 1 << k)]
    evals = sum(n_ge[f:].sum() for f in floors) + n_ge.sum() + n_ge[1:].sum()
    weighted = sum(n_ge[f] for f in floors) + 2.0 * n
    pairs = sum(n_ge[f] ** 2 for f in floors) + 2.0 * n * n
    shared = float(1 << k)
    return dict(
        rung_hist=hist.tolist(),
        particle_step_bound=n * shared / float(np.sum(2.0**rung)),
        evals_speedup=n * shared / evals,
        evals_weighted_speedup=n * shared / weighted,
        pair_speedup=n * n * shared / pairs,
    )


def probe() -> None:
    """Criterion spreads, rung histograms and bounds, the rotation curve and energies at t = 0."""
    ic = load_ic(ARGS.n, ARGS.offset)
    n, eps = ARGS.n, ARGS.softening
    x, m = jnp.asarray(ic["x"]), jnp.asarray(ic["m"])
    halo_acc, halo_pot = make_halo(ic["G"], ic["halo_m"], ic["halo_rs"])
    pair_pass = jax.jit(make_pair_pass(n, eps, ic["G"]))
    if ARGS.static:
        halo_acc, halo_pot = make_static(ic, eps)
        pair_pass = _no_pairs
    zeros = jnp.zeros(n)
    t0 = time.perf_counter()
    _, u = pair_pass(x, m, zeros, zeros, n)
    a = np.asarray(u + halo_acc(x))
    t_pass = time.perf_counter() - t0
    xs, comp = ic["x"], ic["comp"]
    r = np.linalg.norm(xs, axis=1)
    rc = np.hypot(xs[:, 0], xs[:, 1])
    rd = ic["rdisk"]
    t_eps = floor_time(ic, eps)
    out: dict = dict(
        n=n,
        offset=ARGS.offset,
        softening=eps,
        first_pass_seconds=t_pass,
        t_eps=t_eps,
        rdisk=rd,
        particle_mass=float(ic["m"][0]),
        encounter_acc_at_eps=ic["G"] * float(ic["m"][0]) / eps**2,
        median_abs_a=float(np.median(np.linalg.norm(a, axis=1))),
        max_abs_a=float(np.max(np.linalg.norm(a, axis=1))),
    )
    # rotation curve from the midplane radial force of the disc particles
    sel = (comp == 0) & (np.abs(xs[:, 2]) < 0.5 * 0.03)
    a_r = -(a[sel, 0] * xs[sel, 0] + a[sel, 1] * xs[sel, 1]) / rc[sel]
    vphi = (xs[:, 0] * ic["v"][:, 1] - xs[:, 1] * ic["v"][:, 0]) / np.maximum(rc, 1e-12)
    curve = []
    for lo in np.arange(0.0, 6.0, 0.5):
        b = (rc[sel] >= lo * rd) & (rc[sel] < (lo + 0.5) * rd)
        bd = (comp == 0) & (rc >= lo * rd) & (rc < (lo + 0.5) * rd)
        if b.sum() < 20:
            continue
        rmid = float(np.median(rc[sel][b]))
        vc = float(np.sqrt(max(np.median(a_r[b]) * rmid, 0.0)))
        curve.append(
            dict(
                R_over_Rd=lo + 0.25,
                v_c=vc,
                T_rot=2 * math.pi * rmid / max(vc, 1e-12),
                mean_vphi_disc=float(np.mean(vphi[bd])),
            )
        )
    out["rotation_curve"] = curve
    if not ARGS.no_energy:
        phi = np.asarray(make_potential(n, eps, ic["G"])(x, m)) * (not ARGS.static)
        k_e = 0.5 * float(np.sum(ic["m"][:, None] * ic["v"] ** 2))
        w_self = 0.5 * float(np.sum(ic["m"] * phi))
        w_ext = float(np.sum(ic["m"] * np.asarray(halo_pot(x))))
        out["energy"] = dict(K=k_e, W_self=w_self, W_ext=w_ext, E=k_e + w_self + w_ext)
    crit = {}
    for name in ("acc", "dyn"):
        tau = np.asarray(
            make_tau(name, eps)(x, jnp.linalg.norm(jnp.asarray(a), axis=1))
        )
        i_min = int(np.argmin(tau))
        ratio = tau / tau.min()
        row = dict(
            tau_min=float(tau.min()),
            argmin_r=float(r[i_min]),
            argmin_component=int(comp[i_min]),
            ratio_percentiles={
                str(p): float(np.percentile(ratio, p)) for p in (0.1, 1, 10, 50, 90, 99)
            },
            ratio_max=float(ratio.max()),
            n_below_2x_min=int(np.sum(ratio < 2.0)),
            n_below_4x_min=int(np.sum(ratio < 4.0)),
        )
        prof = []
        for lo, hi in (
            (0, 0.25),
            (0.25, 0.5),
            (0.5, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 6),
            (6, 10),
        ):
            b = (rc >= lo * rd) & (rc < hi * rd)
            if b.sum() >= 20:
                prof.append(
                    dict(
                        R_over_Rd=[lo, hi],
                        fraction=float(b.mean()),
                        median_ratio=float(np.median(ratio[b])),
                    )
                )
        row["radial"] = prof
        row["kmax"] = {}
        for k in [int(s) for s in ARGS.kmax.split(",")]:
            rung = np.clip(np.ceil(k - np.log2(ratio)), 0, k).astype(np.int64)
            row["kmax"][k] = _bounds(rung, k)
        crit[name] = row
        print(f"[probe] {name}: {json.dumps(row)}", flush=True)
    out["criteria"] = crit
    print(f"[probe] {json.dumps({k: v for k, v in out.items() if k != 'criteria'})}")
    with open(ARGS.out, "w") as fh:
        json.dump(out, fh, indent=1)


# --------------------------------------------------------------------- diagnostics


def _lagrange(r: np.ndarray, m: np.ndarray, fracs) -> list:
    o = np.argsort(r)
    cm = np.cumsum(m[o]) / m.sum()
    return [float(r[o][np.searchsorted(cm, f)]) for f in fracs]


_LAGR_ALL = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)
_LAGR_BULGE = (0.1, 0.25, 0.5, 0.75, 0.9)


class Diagnostics:
    """Energy (fp64 potential), momenta, A2 and Lagrangian radii at synchronised times."""

    def __init__(self, ic: dict, eps: float, static: bool = False):
        self.ic = ic
        n = ic["x"].shape[0]
        self.m = jnp.asarray(ic["m"])
        self.phi = make_potential(n, eps, ic["G"])
        _, self.halo_pot = make_halo(ic["G"], ic["halo_m"], ic["halo_rs"])
        if static:
            self.phi = lambda x, m: jnp.zeros(n)
            _, self.halo_pot = make_static(ic, eps)
        self.rows: list = []
        self.p0 = None

    def __call__(self, t: float, x, v, j_ext, *, keep_particles: bool = False) -> dict:
        m = np.asarray(self.ic["m"])
        comp = self.ic["comp"]
        phi_self = np.asarray(self.phi(x, self.m))
        phi_ext = np.asarray(self.halo_pot(x))
        xs, vs = np.asarray(x), np.asarray(v)
        k_e = 0.5 * float(np.sum(m[:, None] * vs * vs))
        w_self = 0.5 * float(np.sum(m * phi_self))
        w_ext = float(np.sum(m * phi_ext))
        mom = np.sum(m[:, None] * vs, 0)
        if self.p0 is None:
            self.p0 = mom.copy()
        ang = np.sum(m[:, None] * np.cross(xs, vs), 0)
        rc = np.hypot(xs[:, 0], xs[:, 1])
        ph = np.arctan2(xs[:, 1], xs[:, 0])
        rd = self.ic["rdisk"]
        a2 = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (0, 3)):
            b = (comp == 0) & (rc >= lo * rd) & (rc < hi * rd)
            z = np.sum(m[b] * np.exp(2j * ph[b])) / np.sum(m[b])
            a2[f"{lo}-{hi}"] = float(abs(z))
        r = np.linalg.norm(xs, axis=1)
        row = dict(
            t=t,
            K=k_e,
            W_self=w_self,
            W_ext=w_ext,
            E=k_e + w_self + w_ext,
            L=ang.tolist(),
            P=mom.tolist(),
            pair_impulse=(mom - self.p0 - np.asarray(j_ext)).tolist(),
            sum_m_abs_v=float(np.sum(m * np.linalg.norm(vs, axis=1))),
            A2=a2,
            lagrange_all=_lagrange(r, m, _LAGR_ALL),
            lagrange_bulge=_lagrange(r[comp == 1], m[comp == 1], _LAGR_BULGE),
        )
        self.rows.append(row)
        if keep_particles:
            row["_e_i"] = 0.5 * np.sum(vs * vs, 1) + phi_self + phi_ext
        return row


def _record(path: str, cfg: dict, diag: Diagnostics, x0, v0, x, v, e0, e1, costs: dict):
    rows = [
        {k: val for k, val in r.items() if not k.startswith("_")} for r in diag.rows
    ]
    np.savez(
        path,
        x0=np.asarray(x0),
        v0=np.asarray(v0),
        x=np.asarray(x),
        v=np.asarray(v),
        e_i0=e0,
        e_i=e1,
        comp=diag.ic["comp"],
        m=np.asarray(diag.ic["m"]),
        meta=json.dumps(dict(config=cfg, costs=costs, series=rows)),
    )
    summary = dict(
        config=cfg,
        costs={k: c for k, c in costs.items() if k != "rung_hist_per_base_step"},
        final=rows[-1],
    )
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump(dict(config=cfg, costs=costs, series=rows), fh, indent=1)
    print(f"[run] {json.dumps(summary)}", flush=True)


# ------------------------------------------------------------------------- the runs


def run() -> None:
    """One run: a shared-step KDK or a HOLD block-step run to ``--t-end``."""
    wall0 = time.perf_counter()
    ic = load_ic(ARGS.n, ARGS.offset)
    n, eps, k = ARGS.n, ARGS.softening, ARGS.kmax
    m = jnp.asarray(ic["m"])
    halo_acc, _ = make_halo(ic["G"], ic["halo_m"], ic["halo_rs"])
    pair_pass = make_pair_pass(n, eps, ic["G"])
    if ARGS.static:
        halo_acc, _ = make_static(ic, eps)
        pair_pass = _no_pairs
    if (ARGS.criterion == "dynfloor") != (ARGS.floor_c > 0):
        raise SystemExit(
            "--floor-c > 0 goes with --criterion dynfloor, and only with it"
        )
    t_floor = ARGS.floor_c * floor_time(ic, eps)
    tau_fn = make_tau(ARGS.criterion, eps)
    diag = Diagnostics(ic, eps, ARGS.static)
    x0, v0 = jnp.asarray(ic["x"]), jnp.asarray(ic["v"])
    zeros = jnp.zeros(n)

    def total_acc(x):
        return pair_pass(x, m, zeros, zeros, n)[1] + halo_acc(x)

    def dt_of(x, a_norm):
        """The criterion's step per particle, with the optional floor in time units."""
        return jnp.maximum(ARGS.eta * tau_fn(x, a_norm), t_floor)

    def norm(a):
        return jnp.linalg.norm(a, axis=1)

    a0 = jax.jit(total_acc)(x0)
    cfg = dict(
        vars(ARGS),
        n_all_ic=int(np.load(ARGS.ic, mmap_mode="r")["state0"].shape[0]),
        t_floor=t_floor,
        device=str(jax.devices()[0]),
    )
    n_out = ARGS.n_out
    e0 = diag(0.0, x0, v0, np.zeros(3), keep_particles=True)["_e_i"]
    j_ext = jnp.zeros(3)
    if ARGS.scheme == "shared":
        dt_target = (
            ARGS.dt if ARGS.dt is not None else float(jnp.min(dt_of(x0, norm(a0))))
        )
        per_out = max(1, math.ceil(ARGS.t_end / (n_out * dt_target)))
        steps = per_out * n_out
        dt = ARGS.t_end / steps
        cfg.update(dt_target=dt_target, dt=dt, steps=steps)

        @jax.jit
        def segment(x, v, a, ext, j_ext):
            def body(_, c):
                x, v, a, ext, j = c
                v = v + 0.5 * dt * a
                j = j + 0.5 * dt * jnp.sum(m[:, None] * ext, 0)
                x = x + dt * v
                ext = halo_acc(x)
                a = pair_pass(x, m, zeros, zeros, n)[1] + ext
                j = j + 0.5 * dt * jnp.sum(m[:, None] * ext, 0)
                return x, v + 0.5 * dt * a, a, ext, j

            return jax.lax.fori_loop(0, per_out, body, (x, v, a, ext, j_ext))

        x, v, a, ext = x0, v0, a0, halo_acc(x0)
        for j in range(1, n_out + 1):
            x, v, a, ext, j_ext = segment(x, v, a, ext, j_ext)
            row = diag(j * ARGS.t_end / n_out, x, v, j_ext, keep_particles=j == n_out)
            print(
                f"[run] t {row['t']:.4f} dE/E0 {abs(row['E'] / diag.rows[0]['E'] - 1):.3e}",
                flush=True,
            )
        costs = dict(
            force_evaluations=steps + 1,
            evals=float(n) * (steps + 1),
            evals_weighted=float(n) * (steps + 1),
            evals_padded=float(n) * (steps + 1),
            pairs=float(n) * (n - 1) * (steps + 1),
        )
        cfg["wall_seconds"] = time.perf_counter() - wall0
        _record(ARGS.out, cfg, diag, x0, v0, x, v, e0, row["_e_i"], costs)
        return

    # ---- HOLD
    dt_fine_target = float(jnp.min(dt_of(x0, norm(a0))))
    per_out = max(1, math.ceil(ARGS.t_end / (n_out * dt_fine_target * (1 << k))))
    n_base = per_out * n_out
    dt_max = ARGS.t_end / n_base
    cfg.update(
        dt_fine_target=dt_fine_target,
        dt_max=dt_max,
        dt_fine=dt_max / (1 << k),
        n_base=n_base,
    )
    floors = jnp.asarray(
        [0] + [k - ((s & -s).bit_length() - 1) for s in range(1, 1 << k)], jnp.int32
    )
    ar = jnp.arange(n)
    levels = jnp.arange(k + 1)

    def suffix_counts(rung):
        """``n_ge[m] = #(rung >= m)``: the size of the level-``m`` subset."""
        h = jnp.bincount(rung, length=k + 1)
        return jnp.cumsum(h[::-1])[::-1].astype(jnp.float64)

    def tally(acc, ng_close, close, ng_open, open_):
        """Add one boundary's subset evaluations to ``(evals, uses, caps)``."""
        evals, uses, caps = acc
        evals = (
            evals
            + jnp.sum(jnp.where(close, ng_close, 0.0))
            + jnp.sum(jnp.where(open_, ng_open, 0.0))
        )
        uses = uses + close + open_
        caps = jnp.maximum(
            caps,
            jnp.maximum(
                jnp.where(close, ng_close, 0.0), jnp.where(open_, ng_open, 0.0)
            ),
        )
        return evals, uses, caps

    @jax.jit
    def closing(x, v, rung, mm, j_ext, acc):
        """The last kick of every level (all synchronised): old rungs; also the total force."""
        p = 2.0 ** -rung.astype(jnp.float64)
        w, u = pair_pass(x, mm, p, zeros, n)
        ext = halo_acc(x)
        h = 0.5 * dt_max
        kick = h * (w + p[:, None] * ext)
        j_ext = j_ext + jnp.sum(mm[:, None] * h * p[:, None] * ext, 0)
        acc = tally(acc, suffix_counts(rung), levels >= 0, zeros[: k + 1], levels < 0)
        return v + kick, u + ext, j_ext, acc

    @jax.jit
    def opening(x, v, a, pid, mm, j_ext, acc, rung_old):
        """New rungs from the total force, sort fastest first, open every level, drift.

        ``--rungs-once`` keeps the rungs of t = 0 (``rung_old``) for the whole run.
        """
        if ARGS.rungs_once:
            rung = jnp.where(
                rung_old < 0, rungs_of(dt_of(x, norm(a)), dt_max, k), rung_old
            )
        else:
            rung = rungs_of(dt_of(x, norm(a)), dt_max, k)
        order = jnp.argsort(-rung, stable=True)
        x, v, rung, pid, mm, a = (
            x[order],
            v[order],
            rung[order],
            pid[order],
            mm[order],
            a[order],
        )
        p = 2.0 ** -rung.astype(jnp.float64)
        w, _ = pair_pass(x, mm, p, zeros, n)
        ext = halo_acc(x)
        h = 0.5 * dt_max
        v = v + h * (w + p[:, None] * ext)
        j_ext = j_ext + jnp.sum(mm[:, None] * h * p[:, None] * ext, 0)
        x = x + dt_max * p[:, None] * v
        # F(S_0) is the closing pass's total force; only the levels >= 1 are new
        acc = tally(acc, zeros[: k + 1], levels < 0, suffix_counts(rung), levels >= 1)
        return x, v, rung, pid, mm, a, j_ext, acc

    @jax.jit
    def interior(x, v, rung, mm, pid, a_base, j_ext, acc, work):
        """Boundaries 1 .. 2**k - 1: kick and drift the synchronised prefix S_l only.

        With ``--refine`` a particle of the synchronised set may move to a finer rung
        at its own boundary (nornax ``apply_rung_change``: refine directly, coarsen only
        at the base step). Only the subset force is fresh there, so the criterion takes
        ``|a| = max(|a_base|, |a_sub|)``: the total force of the base step and the subset
        force plus halo of the particle's previous kick (the interior mass that dominates
        a plunge into the centre is on the fast rungs). Deciding from the previous kick
        lets the closing (old rungs) and opening (new rungs) kicks share one pass.
        """
        a_sub = jnp.zeros((n,))

        def body(s, c):
            x, v, rung, mm, pid, a_base, a_sub, j, acc, work = c
            f = floors[s]
            n_l = jnp.sum(rung >= f)
            live = ar < n_l
            if ARGS.refine:
                a_est = jnp.maximum(norm(a_base), a_sub)
                target = rungs_of(dt_of(x, a_est), dt_max, k)
                new = jnp.where(live, jnp.maximum(rung, target), rung)
            else:
                new = rung
            p = 2.0 ** -rung.astype(jnp.float64)
            q = 2.0 ** -new.astype(jnp.float64)
            w, u = pair_pass(x, mm, p, q, n_l)
            ext = halo_acc(x) * live[:, None]
            h = 0.5 * dt_max
            v = v + h * (w + (p + q)[:, None] * ext)
            j = j + jnp.sum(mm[:, None] * h * (p + q)[:, None] * ext, 0)
            a_sub = jnp.where(live, norm(u + ext), a_sub)
            x = x + live[:, None] * dt_max * q[:, None] * v
            ng_old, ng_new = suffix_counts(rung), suffix_counts(new)
            acc = tally(
                acc, ng_old, levels >= f, ng_new, (levels > f) & (ng_new != ng_old)
            )
            nl = n_l.astype(jnp.float64)
            moved = jnp.sum(new != rung).astype(jnp.float64)
            work = work + jnp.asarray([nl, nl * (nl - 1.0), moved])
            if ARGS.refine:
                order = jnp.argsort(-new, stable=True)
                x, v, new, mm, pid, a_base, a_sub = (
                    t[order] for t in (x, v, new, mm, pid, a_base, a_sub)
                )
            return x, v, new, mm, pid, a_base, a_sub, j, acc, work

        x, v, rung, mm, pid, a_base, _, j_ext, acc, work = jax.lax.fori_loop(
            1, 1 << k, body, (x, v, rung, mm, pid, a_base, a_sub, j_ext, acc, work)
        )
        return x, v, rung, mm, pid, a_base, j_ext, acc, work

    pid = jnp.arange(n)
    x, v, a, mm = x0, v0, a0, m
    hists = []
    # (evals, per-level evaluation counts, per-level largest subset); the t = 0 total force
    acc = (
        jnp.asarray(float(n)),
        jnp.zeros(k + 1).at[0].set(1.0),
        jnp.zeros(k + 1).at[0].set(float(n)),
    )
    work = jnp.asarray(
        [float(n), float(n) * (n - 1), 0.0]
    )  # evals_weighted, pairs, refinements
    full = jnp.asarray([float(n), float(n) * (n - 1), 0.0])
    rung = None
    for b in range(n_base):
        if b > 0:
            v, a, j_ext, acc = closing(x, v, rung, mm, j_ext, acc)
            work = work + full
            if b % per_out == 0:
                xs_, vs_ = _unsort(x, v, pid)
                row = diag(b * dt_max, xs_, vs_, j_ext)
                print(
                    f"[run] t {row['t']:.4f} dE/E0 {abs(row['E'] / diag.rows[0]['E'] - 1):.3e} rungs {hists[-1]}",
                    flush=True,
                )
        rung_old = jnp.full((n,), -1, jnp.int32) if rung is None else rung
        x, v, rung, pid, mm, a, j_ext, acc = opening(
            x, v, a, pid, mm, j_ext, acc, rung_old
        )
        work = work + full
        h0 = np.bincount(np.asarray(rung), minlength=k + 1)
        hists.append(h0.tolist())
        x, v, rung, mm, pid, a, j_ext, acc, work = interior(
            x, v, rung, mm, pid, a, j_ext, acc, work
        )
    v, a, j_ext, acc = closing(x, v, rung, mm, j_ext, acc)
    work = work + full
    evals, uses, caps = (np.asarray(t) for t in acc)
    work = np.asarray(work)
    costs = dict(
        force_evaluations=2 * n_base + n_base * ((1 << k) - 1) + 1,
        evals=float(evals),
        evals_weighted=float(work[0]),
        pairs=float(work[1]),
        # a static capacity plan: each level's subset padded to its largest of the run
        evals_padded=float(np.sum(uses * caps)),
        level_capacity=caps.tolist(),
        level_evaluations=uses.tolist(),
        refinements=float(work[2]),
        rung_hist_per_base_step=hists,
    )
    xs_, vs_ = _unsort(x, v, pid)
    row = diag(ARGS.t_end, xs_, vs_, j_ext, keep_particles=True)
    print(
        f"[run] t {row['t']:.4f} dE/E0 {abs(row['E'] / diag.rows[0]['E'] - 1):.3e}",
        flush=True,
    )
    cfg["wall_seconds"] = time.perf_counter() - wall0
    _record(ARGS.out, cfg, diag, x0, v0, xs_, vs_, e0, row["_e_i"], costs)


def _unsort(x, v, pid):
    inv = jnp.argsort(pid)
    return x[inv], v[inv]


# ------------------------------------------------------------------------ the scores

_R_EDGES = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6)


def _load(path: str) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != "meta"}
    out.update(json.loads(str(d["meta"])))
    return out


def _disc_profiles(run: dict, rd: float, key: str = "") -> dict:
    """Sigma, sigma_z, sigma_R and mean v_phi of the disc in cylindrical bins."""
    x, v = run["x" + key], run["v" + key]
    disc = run["comp"] == 0
    x, v, m = x[disc], v[disc], run["m"][disc]
    rc = np.hypot(x[:, 0], x[:, 1])
    v_r = (x[:, 0] * v[:, 0] + x[:, 1] * v[:, 1]) / rc
    v_p = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]) / rc
    prof = {k: [] for k in ("Sigma", "sigma_z", "sigma_R", "v_phi", "count")}
    for lo, hi in zip(_R_EDGES[:-1], _R_EDGES[1:]):
        b = (rc >= lo * rd) & (rc < hi * rd)
        prof["count"].append(int(b.sum()))
        prof["Sigma"].append(
            float(m[b].sum() / (math.pi * rd * rd * (hi * hi - lo * lo)))
        )
        prof["sigma_z"].append(float(np.std(v[b, 2])))
        prof["sigma_R"].append(float(np.std(v_r[b])))
        prof["v_phi"].append(float(np.mean(v_p[b])))
    return prof


def _rel(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a - b) / np.abs(b)


def _central(run: dict) -> dict:
    """Lagrangian radii of all baryons and of the bulge at the end, and their energy diffusion."""
    x0 = run["x0"]
    r0 = np.linalg.norm(x0, axis=1)
    de = run["e_i"] - run["e_i0"]
    inner = r0 < np.percentile(r0, 2.0)
    return dict(
        lagrange_all=run["series"][-1]["lagrange_all"],
        lagrange_bulge=run["series"][-1]["lagrange_bulge"],
        e_diffusion_rms=dict(
            disc=float(np.sqrt(np.mean(de[run["comp"] == 0] ** 2))),
            bulge=float(np.sqrt(np.mean(de[run["comp"] == 1] ** 2))),
            inner_2pct=float(np.sqrt(np.mean(de[inner] ** 2))),
        ),
    )


def _score_one(run: dict, ref: dict, rd: float) -> dict:
    ser = run["series"]
    e0, l0 = ser[0]["E"], np.asarray(ser[0]["L"])
    scale_p = ser[0]["sum_m_abs_v"]
    p_run, p_ref = _disc_profiles(run, rd), _disc_profiles(ref, rd)
    good = np.asarray(p_ref["count"]) >= 200
    prof = {}
    for key in ("Sigma", "sigma_z", "sigma_R", "v_phi"):
        d = _rel(p_run[key], p_ref[key])[good]
        prof[key] = dict(rms=float(np.sqrt(np.mean(d * d))), mean=float(np.mean(d)))
    a2 = np.asarray([r["A2"]["0-3"] for r in ser])
    a2_ref = np.asarray([r["A2"]["0-3"] for r in ref["series"]])
    t, t_ref = [r["t"] for r in ser], [r["t"] for r in ref["series"]]
    a2_ref_i = np.interp(t, t_ref, a2_ref)
    c_run, c_ref = _central(run), _central(ref)
    diff_ratio = {
        k: c_run["e_diffusion_rms"][k] / c_ref["e_diffusion_rms"][k]
        for k in c_run["e_diffusion_rms"]
    }
    out = {}
    if run["config"].get("static"):
        # a fixed field: E_i is an exact invariant (L_z,i too, but a leapfrog in an
        # axisymmetric field keeps it to round-off), and the orbits are regular, so
        # particle-wise errors against the reference are meaningful
        de = np.abs(run["e_i"] - run["e_i0"]) / np.abs(run["e_i0"])
        dx = np.linalg.norm(run["x"] - ref["x"], axis=1) / rd
        disc = run["comp"] == 0
        out["static"] = {
            name: {f"p{q}": float(np.percentile(val[sel], q)) for q in (50, 90, 99)}
            for name, val, sel in (
                ("dE_i_rel", de, slice(None)),
                ("dE_i_rel_bulge", de, ~disc),
                ("dx_vs_ref_over_Rd", dx, slice(None)),
            )
        }
    return out | dict(
        max_dE_over_E0=float(max(abs(r["E"] / e0 - 1.0) for r in ser)),
        final_dE_over_E0=float(ser[-1]["E"] / e0 - 1.0),
        max_dL_over_L0=float(
            max(np.linalg.norm(np.asarray(r["L"]) - l0) for r in ser)
            / np.linalg.norm(l0)
        ),
        max_pair_impulse_over_sum_m_v=float(
            max(np.linalg.norm(r["pair_impulse"]) for r in ser) / scale_p
        ),
        profiles=prof,
        A2_max_abs_diff=float(np.max(np.abs(a2 - a2_ref_i))),
        A2_end=float(a2[-1]),
        lagrange_all_rel=_rel(c_run["lagrange_all"], c_ref["lagrange_all"]).tolist(),
        lagrange_bulge_rel=_rel(
            c_run["lagrange_bulge"], c_ref["lagrange_bulge"]
        ).tolist(),
        e_diffusion_ratio=diff_ratio,
    )


def score() -> None:
    """Scores of each run against ``--ref``, and of ``--ref2`` against ``--ref`` (the noise floor)."""
    ref, ref2 = _load(ARGS.ref), _load(ARGS.ref2)
    rd = load_ic(8, 0)["rdisk"]
    p0, p_ref = _disc_profiles(ref, rd, "0"), _disc_profiles(ref, rd)
    out = dict(
        reference=ARGS.ref,
        reference2=ARGS.ref2,
        reference_profiles=dict(t0=p0, t_end=p_ref, R_edges_over_Rd=list(_R_EDGES)),
        reference_central=_central(ref),
        noise_floor=_score_one(ref2, ref, rd),
        runs={},
    )
    out["noise_floor"]["config"] = ref2["config"]
    for path in ARGS.runs:
        run = _load(path)
        cfg, costs = run["config"], run["costs"]
        n = cfg["n"]
        row = _score_one(run, ref, rd)
        row["config"] = {
            k: cfg.get(k)
            for k in (
                "scheme",
                "criterion",
                "eta",
                "kmax",
                "floor_c",
                "t_floor",
                "dt",
                "dt_max",
                "dt_fine",
                "n_base",
                "steps",
                "wall_seconds",
            )
        }
        # every proxy as equivalent shared steps: N targets (or N^2 pairs) per step
        row["cost_shared_steps"] = dict(
            evals=costs["evals"] / n,
            evals_weighted=costs["evals_weighted"] / n,
            evals_padded=costs["evals_padded"] / n,
            pairs=costs["pairs"] / (n * (n - 1.0)),
        )
        if "rung_hist_per_base_step" in costs:
            h = np.asarray(costs["rung_hist_per_base_step"], float)
            row["mean_rung_hist"] = (h.mean(0) / n).round(5).tolist()
        out["runs"][os.path.basename(path)] = row
        print(f"[score] {os.path.basename(path)}: {json.dumps(row)}", flush=True)
    with open(ARGS.out, "w") as fh:
        json.dump(out, fh, indent=1)


# ------------------------------------------------------------------------ the tables

_PROXIES = ("evals", "evals_weighted", "evals_padded", "pairs")


def _family(cfg: dict) -> str:
    if cfg["scheme"] == "shared":
        return "shared"
    crit = cfg["criterion"]
    if crit == "dynfloor":
        crit = f"dyn+floor {cfg['floor_c']:g}"
    return f"HOLD {crit} k{cfg['kmax']}" + (" +refine" if cfg.get("refine") else "")


def _scores_of(row: dict) -> dict:
    p = row["profiles"]
    if "static" in row:
        st = row["static"]
        return {
            "max_dE_over_E0": row["max_dE_over_E0"],
            "dE_i_rel_p50": st["dE_i_rel"]["p50"],
            "dE_i_rel_p99": st["dE_i_rel"]["p99"],
            "dE_i_rel_bulge_p99": st["dE_i_rel_bulge"]["p99"],
            "dx_vs_ref_p50": st["dx_vs_ref_over_Rd"]["p50"],
            "dx_vs_ref_p90": st["dx_vs_ref_over_Rd"]["p90"],
            "Sigma_rms": p["Sigma"]["rms"],
            "sigma_z_rms": p["sigma_z"]["rms"],
            "r_1pct_abs_rel": abs(row["lagrange_all_rel"][1]),
        }
    return {
        "max_dE_over_E0": row["max_dE_over_E0"],
        "Sigma_rms": p["Sigma"]["rms"],
        "sigma_z_rms": p["sigma_z"]["rms"],
        "sigma_R_rms": p["sigma_R"]["rms"],
        "v_phi_rms": p["v_phi"]["rms"],
        "A2_max_abs_diff": row["A2_max_abs_diff"],
        "r_1pct_abs_rel": abs(row["lagrange_all_rel"][1]),
        "bulge_r50_abs_rel": abs(row["lagrange_bulge_rel"][2]),
        "e_diffusion_inner": row["e_diffusion_ratio"]["inner_2pct"],
    }


def _shared_cost_at(shared: list, key: str, value: float, proxy: str):
    """The shared-step cost that reaches ``value`` of ``key`` (log-log interpolation).

    ``None`` outside the shared runs' range or where the shared curve is not monotone.
    """
    pts = sorted((r["cost_shared_steps"][proxy], _scores_of(r)[key]) for r in shared)
    if len(pts) < 2:
        return None
    c = np.log([q[0] for q in pts])
    v = np.log([max(q[1], 1e-300) for q in pts])
    if not np.all(np.diff(v) < 0):
        return None
    lv = math.log(max(value, 1e-300))
    if lv > v[0] or lv < v[-1]:
        return None
    return float(math.exp(np.interp(lv, v[::-1], c[::-1])))


def table() -> None:
    """Markdown tables of a ``score`` JSON: costs, scores, and HOLD's equal-accuracy gain."""
    with open(ARGS.scores) as fh:
        sc = json.load(fh)
    rows = sorted(
        sc["runs"].items(),
        key=lambda kv: (_family(kv[1]["config"]), -kv[1]["cost_shared_steps"]["pairs"]),
    )
    floor = _scores_of(sc["noise_floor"])
    print(
        "| run | family | eta / dt | dt_fine | dt_max | evals | weighted | padded | pairs |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, r in rows:
        c, k = r["config"], r["cost_shared_steps"]
        lab = f"dt {c['dt']:.4g}" if c["scheme"] == "shared" else f"eta {c['eta']:g}"
        dtf = c["dt"] if c["scheme"] == "shared" else c["dt_fine"]
        dtm = c["dt"] if c["scheme"] == "shared" else c["dt_max"]
        print(
            f"| {name[:-4]} | {_family(c)} | {lab} | {dtf:.3g} | {dtm:.3g} | "
            + " | ".join(f"{k[p]:.0f}" for p in _PROXIES)
            + " |"
        )
    keys = list(floor)
    print()
    print("| run | " + " | ".join(keys) + " |")
    print("| --- |" + " --- |" * len(keys))
    print(
        "| noise floor (ref2 vs ref) | "
        + " | ".join(f"{floor[q]:.3g}" for q in keys)
        + " |"
    )
    for name, r in rows:
        v = _scores_of(r)
        print(f"| {name[:-4]} | " + " | ".join(f"{v[q]:.3g}" for q in keys) + " |")
    shared = [r for _, r in rows if r["config"]["scheme"] == "shared"]
    print()
    print(
        "HOLD's gain at equal accuracy: the shared cost that reaches the run's score over the run's cost."
    )
    print()
    print("| run | score | value | " + " | ".join(_PROXIES) + " |")
    print("| --- | --- | --- |" + " --- |" * len(_PROXIES))
    for name, r in rows:
        if r["config"]["scheme"] == "shared":
            continue
        keys = ("max_dE_over_E0", "sigma_z_rms", "Sigma_rms")
        if "static" in r:
            keys = ("max_dE_over_E0", "dE_i_rel_p99", "dx_vs_ref_p90")
        for key in keys:
            val = _scores_of(r)[key]
            cells = []
            for proxy in _PROXIES:
                cs = _shared_cost_at(shared, key, val, proxy)
                cells.append(
                    "--" if cs is None else f"{cs / r['cost_shared_steps'][proxy]:.2f}"
                )
            print(f"| {name[:-4]} | {key} | {val:.3g} | " + " | ".join(cells) + " |")


def main(argv: Optional[list] = None) -> None:
    global ARGS
    ARGS = _args(argv)
    {"probe": probe, "run": run, "score": score, "table": table}[ARGS.mode]()


if __name__ == "__main__":
    main()
