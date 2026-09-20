"""Gate G1, second half: ndev=2 local-only force == the single-GPU force per shard.

No cross-domain field yet. Each device computes the force of ITS OWN shard on
itself, and the reference is exactly that computation run singly on one device.
Any difference is the mesh plumbing -- stacking, the [0] slice, the specs, the
box all-reduce, plan installation -- and nothing else.

Both arms must see the SAME Morton box or they build different trees and the
comparison means nothing (learned the hard way at ndev=1), so the box is computed
on the host exactly as global_mesh_bounds computes it and handed to the reference.
"""

import os, sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
from codes.compare_force import (
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
    FAST_LANE_ENV_BY_LEAF,
)

NDEV = int(os.environ.get("PROBE_NDEV", "2"))
N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
CAP = int(N / NDEV * 1.15)
_ov = fast_lane_overrides_for_leaf(LEAF, CAP)
apply_fast_lane_env(CAP, overrides=_ov)
# BOTH lengths: the fused profile gate keys on the EXACT array length, so the
# per-shard arm (CAP rows) and the single-GPU comparison arm (N rows) each
# need their own entry or the second one refuses to run at all
os.environ["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = f"{CAP},{N}"
_TRAV = dict((FAST_LANE_ENV_BY_LEAF.get(LEAF) or {}).get("_traversal_overrides", {}))

import numpy as np, jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jaccpot import FastMultipoleMethod, TraversalOverrides
from jaccpot.config import (
    FMMAdvancedConfig,
    TreeConfig,
    FarFieldConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
)
from jaccpot.distributed.fused import (
    fused_force_step,
    stack_prepared_states,
    make_fused_force_evaluator,
)
from jaccpot.runtime.capacity_plan import (
    plan_from_registry,
    merge_plans,
    fused_capacity_plan_overrides,
)
from jaccpot.runtime import _level_shapes as LS
from yggdrax.distributed.sharding import make_mesh
from yggdrax.bounds import infer_bounds
from yggdrax.morton import morton_encode
import yggdrax._cell_partition as cp
from common.ic import IC_GENERATORS
from common.reference import direct_accelerations

ORDER, THETA, SOFT = 4, 0.8, 1e-7
print(f"devices visible: {jax.devices()}", flush=True)
assert len(jax.devices()) >= NDEV, f"need {NDEV} devices"

_ic = IC_GENERATORS["plummer"](N, seed=0)
pos = np.asarray(_ic[0], np.float32)
mass = np.asarray(_ic[1], np.float32)
P0 = jnp.asarray(pos)
codes = np.asarray(morton_encode(P0, infer_bounds(P0)))
order = np.argsort(codes)
shards = np.array_split(order, NDEV)
kk = int(cp.adaptive_cell_leaf_partition_numpy(np.sort(codes), leaf_size=LEAF)[0].size)
LEAF_CAP = 1 << int(np.ceil(np.log2(1.25 * kk / NDEV)))
print(
    f"N={N} ndev={NDEV} cap={CAP} leaf_capacity={LEAF_CAP} shards={[len(s) for s in shards]}",
    flush=True,
)


def build():
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
            runtime=(
                RuntimePolicyConfig(
                    traversal_config=TraversalOverrides(
                        **{k: int(v) for k, v in _TRAV.items()}
                    )
                )
                if _TRAV
                else RuntimePolicyConfig()
            ),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


# --- pad each shard; keep the per-device pieces
dev_pos, dev_mass, dev_live = [], [], []
for sel in shards:
    dev_pos.append(
        np.concatenate([pos[sel], np.repeat(pos[sel][:1], CAP - len(sel), 0)])
    )
    dev_mass.append(np.concatenate([mass[sel], np.zeros(CAP - len(sel), np.float32)]))
    dev_live.append(len(sel))

# --- the box, computed exactly as global_mesh_bounds does, so both arms agree
allpos = np.concatenate([dev_pos[d][: dev_live[d]] for d in range(NDEV)])
glo = allpos.min(0)
ghi = allpos.max(0)
span = np.maximum(ghi - glo, np.float32(1e-6))
slack = (span * np.float32(1e-6)).astype(np.float32)
BLO = jnp.asarray(glo - slack)
BHI = jnp.asarray(ghi + slack)
print(f"global box lo={np.asarray(BLO)} hi={np.asarray(BHI)}", flush=True)

# --- eager prepare per shard, collect plans (same discipline as the G1 probe:
# ONE solver instance, warmed by the eager prepares the traced body depends on)
solver = build()
preps, plans = [], []
for d in range(NDEV):
    LS._WIDTHS.clear()
    LS._LEVELS.clear()
    pr = solver.strict_fused_prepared_eval_fn(
        positions=jnp.asarray(dev_pos[d]),
        masses=jnp.asarray(dev_mass[d]),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )[0]
    TN = int(np.asarray(pr.tree.node_ranges).shape[0])
    NI = int(pr.tree.left_child.shape[0])
    plans.append(plan_from_registry(total_nodes=TN, num_internal=NI))
    preps.append(pr)
plan = merge_plans(plans)
print(f"merged plan: {plan}", flush=True)

from jaccpot.distributed.cross import CrossCapacities, make_cross_hook

mesh = make_mesh(NDEV)
stacked = stack_prepared_states(preps)
flat_pos = jnp.asarray(np.concatenate(dev_pos))
flat_mass = jnp.asarray(np.concatenate(dev_mass))
nv = jnp.asarray(np.asarray(dev_live, np.int32))


def run(hook):
    f = make_fused_force_evaluator(
        solver,
        stacked,
        mesh=mesh,
        plan=plan,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_hook=hook,
    )
    a, ovf = f(flat_pos, flat_mass, nv)
    return np.asarray(jax.block_until_ready(a), np.float64), bool(ovf)


caps = CrossCapacities(max_cells=1024, send_node_cap=4096, recv_node_cap=8192)
a_local, ovf_local = run(None)
a_cross, ovf_cross = run(make_cross_hook(ndev=NDEV, theta=THETA, caps=caps))
print(f"local-only overflow={ovf_local}   cross overflow={ovf_cross}", flush=True)

# --- the fp64 reference: every particle against ALL others, subsampled
rng = np.random.default_rng(12345)
allp = jnp.asarray(pos, jnp.float64)
allm = jnp.asarray(mass, jnp.float64)


def err_against_direct(accel_by_dev, tag):
    """rel-L2 of the distributed force against an fp64 direct sum over ALL particles."""
    num = den = 0.0
    for d in range(NDEV):
        sel = shards[d]
        k = min(512, len(sel))
        pick = rng.choice(len(sel), size=k, replace=False)
        # target_indices restricts the ROWS evaluated; every source still enters
        # every sum, which is what makes this the full-N reference and not a
        # shard-local one
        ref = np.asarray(
            direct_accelerations(
                allp, allm, G=1.0, softening=SOFT, target_indices=np.asarray(sel)[pick]
            ),
            np.float64,
        )
        got = accel_by_dev[d * CAP : (d + 1) * CAP][: dev_live[d]][pick]
        num += float(((got - ref) ** 2).sum())
        den += float((ref**2).sum())
    rel = float(np.sqrt(num / den))
    print(f"  {tag:<28} rel-L2 vs fp64 direct (ALL sources) = {rel:.4e}")
    return rel


print()
e_local = err_against_direct(a_local, "local-only (no cross)")
e_cross = err_against_direct(a_cross, "WITH cross field")

# --- the single-GPU comparison arm lives in its OWN process.
# `apply_fast_lane_env` is process-wide and was tuned for the SHARD length (CAP);
# re-tuning it here for N would change the env the mesh arm just ran under, and
# leaving it alone makes the solver refuse N outright ("could not fit N=20000").
# Run `PROBE_SOLO=1` for that arm; it configures itself for N and prints only it.
print()
print(
    f"C4  local-only / cross   = {e_local / e_cross:.2f}x  (CONTROL: dropping the "
    f"cross field must be clearly WORSE, or nothing is being added)"
)
print(
    "C4  NOTE: the FAR half only. `make_cross_hook` returns far pairs and discards "
    "the near list, so the residual is the missing cross NEAR field (phase C5)."
)
if ovf_cross:
    raise SystemExit("C4 FAILED: a capacity overflowed")
# A LIVENESS threshold, not an accuracy one. The question this control answers is
# whether the cross field reaches the force at all; how MUCH it should improve the
# error is not something to assert in advance, and the 3x first written here was
# picked out of the air and rejected a genuine 2.4x improvement.
if not (e_local > 1.2 * e_cross):
    raise SystemExit(
        f"C4 FAILED (control): local-only {e_local:.3e} is not materially worse than "
        f"cross {e_cross:.3e} -- the cross field is not reaching the force"
    )
print(
    "C4 numbers above; gate is judged against the single-GPU lane, and the "
    "per-order sweep is the coverage check (momentum is blind here)."
)
