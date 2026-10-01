"""Gate G2.2: a 100-step two-card rollout of the fused lane, repartitioned on device.

Two modes, each in its OWN process (the fast-lane environment is process-wide and the
fused profile gate keys on the exact array length):

* ``GATE_MODE=mesh`` (2 cards): ``decompose`` + ``setup_fused_force`` +
  ``FusedRollout`` for ``GATE_STEPS`` steps, repartitioning every ``GATE_EVERY``
  (``0`` = the control arm, never). Every step: flags clean, ids exactly once, count
  within the balance bound. At the probe steps: the rollout's force at 512 fixed ids
  vs an fp64 direct sum over all N, and the positions are dumped for the solo arm.
  Energies (fp64, exact potential) at a few steps.
* ``GATE_MODE=solo`` (1 card): the single-GPU fused lane scored on the SAME dumped
  positions (the ratio the gate needs), and a one-card rollout from the same IC for
  the energy comparison.

    GATE_MODE=mesh CUDA_VISIBLE_DEVICES=6,7 python bench/multigpu_rollout_gate.py
    GATE_MODE=solo CUDA_VISIBLE_DEVICES=5 GATE_DUMP=<mesh dump dir> python bench/...

Nothing here is a timing; it is a correctness gate.
"""

import json
import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
from codes.compare_force import apply_fast_lane_env, fast_lane_overrides_for_leaf

MODE = os.environ.get("GATE_MODE", "mesh")
NDEV = 2 if MODE == "mesh" else 1
N = int(os.environ.get("GATE_N", "200000"))
LEAF = int(os.environ.get("GATE_LEAF", "64"))
ORDER = int(os.environ.get("GATE_ORDER", "6"))
THETA = float(os.environ.get("GATE_THETA", "0.8"))
STEPS = int(os.environ.get("GATE_STEPS", "100"))
EVERY = int(os.environ.get("GATE_EVERY", "16"))
DT = float(os.environ.get("GATE_DT", "0.005"))
SOFT = float(os.environ.get("GATE_SOFT", "1e-7"))
OUT = os.environ.get("GATE_OUT", ".")
DUMP = os.environ.get("GATE_DUMP", OUT)
PROBE_STEPS = (1, 16, 17, 50, 96, 97, 100)
ENERGY_STEPS = (0, 25, 50, 75, 100)
NUM_SAMPLES = max(256, 64 * NDEV)
CAP = int(N / NDEV * 1.15)
apply_fast_lane_env(CAP, overrides=fast_lane_overrides_for_leaf(LEAF, CAP))
os.environ["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = f"{CAP}"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from common.ic import IC_GENERATORS
from common.reference import direct_accelerations

from jaccpot import FastMultipoleMethod
from jaccpot.config import (
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities
from jaccpot.distributed.rollout import (
    FusedRollout,
    RolloutConfig,
    decompose,
    setup_fused_force,
)
from yggdrax.distributed.sharding import make_mesh
import yggdrax._cell_partition as cp
from yggdrax.bounds import infer_bounds
from yggdrax.morton import morton_encode


def plummer_with_velocities(n, seed=0):
    """Plummer positions (the bench generator) and isotropic DF velocities (Aarseth
    et al. 1974), G = M = a = 1, so the system is in virial equilibrium."""
    pos, mass = IC_GENERATORS["plummer"](n, seed=seed)
    pos = np.asarray(pos, np.float64)
    rng = np.random.default_rng(seed + 1)
    r = np.linalg.norm(pos, axis=1)
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < x * x * (1.0 - x * x) ** 3.5
        q[todo[ok]] = x[ok]
        todo = todo[~ok]
    speed = q * np.sqrt(2.0) * (1.0 + r * r) ** -0.25
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - mu * mu)
    vel = speed[:, None] * np.stack([s * np.cos(phi), s * np.sin(phi), mu], axis=1)
    return pos.astype(np.float32), vel.astype(np.float32), np.asarray(mass, np.float32)


def potential_energy(pos, mass, block=512):
    """Exact fp64 potential energy, blocked over targets on the device."""
    p = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)
    n = p.shape[0]

    @jax.jit
    def blk(pt, mt, start):
        d = p[None, :, :] - pt[:, None, :]
        r2 = jnp.sum(d * d, axis=-1) + SOFT * SOFT
        rows = start + jnp.arange(pt.shape[0])
        self_ = rows[:, None] == jnp.arange(n)[None, :]
        inv = jnp.where(self_, 0.0, r2**-0.5)
        return -0.5 * jnp.sum(mt[:, None] * m[None, :] * inv)

    total = 0.0
    for s in range(0, n, block):
        e = min(s + block, n)
        pt = jnp.pad(p[s:e], ((0, block - (e - s)), (0, 0)))
        mt = jnp.pad(m[s:e], (0, block - (e - s)))
        total += float(blk(pt, mt, s))
    return total


def energy(pos, vel, mass):
    kin = 0.5 * float(
        np.sum(
            np.asarray(mass, np.float64) * np.sum(np.asarray(vel, np.float64) ** 2, 1)
        )
    )
    return kin + potential_energy(pos, mass)


def build_solver():
    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=LEAF_CAP,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


pos0, vel0, mass0 = plummer_with_velocities(N)
codes = np.sort(
    np.asarray(morton_encode(jnp.asarray(pos0), infer_bounds(jnp.asarray(pos0))))
)
leaves_total = int(cp.adaptive_cell_leaf_partition_numpy(codes, leaf_size=LEAF)[0].size)
# per-device leaf capacity with room for the cells of a drifting, repartitioned shard
LEAF_CAP = 1 << int(np.ceil(np.log2(1.4 * leaves_total / NDEV)))
rng = np.random.default_rng(12345)
PROBE_IDS = np.sort(rng.choice(N, 512, replace=False))
mesh = make_mesh(NDEV)
print(
    f"mode={MODE} N={N} ndev={NDEV} cap={CAP} leaf_cap={LEAF_CAP} every={EVERY}",
    flush=True,
)


def probe_error(acc_by_id, pos_by_id):
    ref = np.asarray(
        direct_accelerations(
            jnp.asarray(pos_by_id, jnp.float64),
            jnp.asarray(mass0, jnp.float64),
            G=1.0,
            softening=SOFT,
            target_indices=PROBE_IDS,
        ),
        np.float64,
    )
    got = np.asarray(acc_by_id, np.float64)[PROBE_IDS]
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


solver = build_solver()
state, decomp = decompose(mesh, pos0, vel0, mass0, cap=CAP, num_samples=NUM_SAMPLES)
caps = None
if NDEV > 1:
    per = leaves_total / NDEV
    pow2 = lambda x, floor: 1 << max(floor, int(np.ceil(np.log2(max(x, 1)))))
    caps = CrossCapacities(
        max_cells=pow2(2.5 * 0.29 * per, 13),
        export_far_cap=pow2(2.5 * 53 * per, 21),
        export_near_cap=pow2(2.5 * 10 * per, 21),
        send_node_cap=pow2(2.5 * 1.25 * per, 15),
        send_csr_cap=pow2(2.5 * 53 * per, 21),
        recv_node_cap=pow2(2.5 * 1.25 * per, 15),
        recv_csr_cap=pow2(2.5 * 53 * per, 19),
        walk_queue=2 * pow2(2.5 * 53 * per, 19),
        recv_far_cap=pow2(2.5 * 106 * per, 21),
        recv_near_cap=pow2(2.5 * 60 * per, 21),
        leaf_width=LEAF,
    )
force, plan, walk_caps = setup_fused_force(
    solver, mesh, state, leaf_size=LEAF, max_order=ORDER, theta=THETA, cross_caps=caps
)
result = dict(
    mode=MODE,
    n=N,
    ndev=NDEV,
    cap=CAP,
    leaf_cap=LEAF_CAP,
    every=EVERY,
    dt=DT,
    steps=STEPS,
    order=ORDER,
    theta=THETA,
    plan=str(plan),
    decompose_recv_counts=np.asarray(decomp["recv_counts"]).tolist(),
    probes={},
    energies={},
    steps_report=[],
)

if MODE == "solo" and os.path.isdir(DUMP):
    # the single-GPU lane on the SAME positions the mesh rollout had at each probe step
    flat_count = jnp.asarray([N], jnp.int32)
    for k in PROBE_STEPS:
        f = os.path.join(DUMP, f"positions_step{k}.npy")
        if not os.path.exists(f):
            continue
        p = np.load(f).astype(np.float32)
        x = np.concatenate([p, np.repeat(p[:1], CAP - N, 0)])
        m = np.concatenate([mass0, np.zeros(CAP - N, np.float32)])
        acc, flag = force(jnp.asarray(x), jnp.asarray(m), flat_count, None)
        acc = np.asarray(acc)[:N]
        result["probes"][str(k)] = dict(
            err=probe_error(acc, p), overflow=bool(np.asarray(flag))
        )
        print(
            f"  solo force on mesh positions, step {k}: rel-L2 {result['probes'][str(k)]['err']:.4e}",
            flush=True,
        )

roll = FusedRollout(
    mesh,
    state,
    RolloutConfig(cap=CAP, dt=DT, repartition_every=EVERY, num_samples=NUM_SAMPLES),
    force,
)
roll.start()
bound = int(np.ceil(N / NDEV)) * (1 + 2 * NDEV / NUM_SAMPLES)
if 0 in ENERGY_STEPS:
    g = roll.gather()
    result["energies"]["0"] = energy(g["positions"], g["velocities"], g["masses"])
for _ in range(STEPS):
    rep = roll.step()
    n = rep.step
    assert sum(rep.counts) == N, rep
    assert max(rep.counts) <= min(0.95 * CAP, bound), (rep.counts, bound)
    result["steps_report"].append(vars(rep))
    if rep.repartitioned or n in PROBE_STEPS or n in ENERGY_STEPS:
        g = roll.gather()  # ids exactly once, or this raises
    if MODE == "mesh" and n in PROBE_STEPS:
        np.save(os.path.join(OUT, f"positions_step{n}.npy"), g["positions"])
        e = probe_error(g["accel"], g["positions"])
        result["probes"][str(n)] = dict(err=e)
        print(
            f"  step {n}: rel-L2 vs fp64 direct {e:.4e}  counts {rep.counts}",
            flush=True,
        )
    if n in ENERGY_STEPS:
        result["energies"][str(n)] = energy(
            g["positions"], g["velocities"], g["masses"]
        )
        print(f"  step {n}: E = {result['energies'][str(n)]:.10f}", flush=True)

reps = [r for r in result["steps_report"] if r["repartitioned"]]
result["repartitions"] = len(reps)
result["repartitions_moved"] = sum(1 for r in reps if r["sent_off_device"] > 0)
result["moved_total"] = sum(r["sent_off_device"] for r in reps)
result["final_owner"] = roll.gather()["owner"].tolist() if MODE == "mesh" else None
e0 = result["energies"].get("0")
if e0:
    result["dE_over_E"] = {k: (v - e0) / abs(e0) for k, v in result["energies"].items()}
print(
    json.dumps(
        {
            k: result[k]
            for k in ("repartitions", "repartitions_moved", "moved_total", "dE_over_E")
            if k in result
        }
    ),
    flush=True,
)
with open(os.path.join(OUT, f"gate_{MODE}_every{EVERY}.json"), "w") as fh:
    json.dump(result, fh, indent=1, default=str)
