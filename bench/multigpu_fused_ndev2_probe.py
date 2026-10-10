"""Gate G1, second half: ndev=2 local-only force == the single-GPU force per shard.

No cross-domain field yet. Each device computes the force of ITS OWN shard on
itself, and the reference is exactly that computation run singly on one device.
Any difference is the mesh plumbing -- stacking, the [0] slice, the specs, the
box all-reduce, plan installation -- and nothing else.

Both arms must see the SAME Morton box or they build different trees and the
comparison means nothing (learned the hard way at ndev=1), so the box is computed
on the host exactly as global_mesh_bounds computes it and handed to the reference.
"""

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
from codes.compare_force import (
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)

NDEV = int(os.environ.get("PROBE_NDEV", "2"))
N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
CAP = int(N / NDEV * 1.15)
_ov = fast_lane_overrides_for_leaf(LEAF, CAP)
apply_fast_lane_env(CAP, overrides=_ov)
os.environ["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = str(CAP)
_TRAV = dict((FAST_LANE_ENV_BY_LEAF.get(LEAF) or {}).get("_traversal_overrides", {}))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import yggdrax._cell_partition as cp
from common.ic import IC_GENERATORS
from common.reference import direct_accelerations
from yggdrax.bounds import infer_bounds
from yggdrax.distributed.sharding import make_mesh
from yggdrax.morton import morton_encode

from jaccpot import FastMultipoleMethod, TraversalOverrides
from jaccpot.config import (
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.distributed.fused import (
    fused_force_step,
    make_fused_force_evaluator,
    stack_prepared_states,
)
from jaccpot.runtime import _level_shapes as LS
from jaccpot.runtime.capacity_plan import (
    fused_capacity_plan_overrides,
    merge_plans,
    plan_from_registry,
)

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

# --- eager prepare per shard, collect plans
# ONE solver, and it must be the same instance the mesh run uses: the solver
# carries host-side state that an eager prepare fills and the traced body reads
# (the level-shape registry, the upward depth stash, and the fused mode the eval
# fn opens). The dual-downward planner hint, whose bool() made a cold solver die
# on a TracerBoolConversionError inside shard_map, went in the 2026-10 cleanup
# (X6); the assumption "an eager prepare always precedes the trace" did not.
solver = build()
preps, plans = [], []
for d in range(NDEV):
    LS._WIDTHS.clear()
    LS._LEVELS.clear()  # isolate each device's LEVEL plan
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
    print(f"  device {d}: live={dev_live[d]} plan={plans[d]}", flush=True)
plan = merge_plans(plans)
print(f"merged plan: {plan}", flush=True)

# --- reference: each shard alone on one device, same box, same plan
refs = []
for d in range(NDEV):
    with fused_capacity_plan_overrides(plan):
        _, a = fused_force_step(
            solver,
            preps[d],
            jnp.asarray(dev_pos[d]),
            jnp.asarray(dev_mass[d]),
            bounds=(BLO, BHI),
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
            num_valid=jnp.asarray(dev_live[d], jnp.int32),
        )
    refs.append(np.asarray(jax.block_until_ready(a), np.float64))
    print(f"  reference device {d} done", flush=True)

# --- the mesh arm
mesh = make_mesh(NDEV)
stacked = stack_prepared_states(preps)
force = make_fused_force_evaluator(
    solver, stacked, mesh=mesh, plan=plan, leaf_size=LEAF, max_order=ORDER, theta=THETA
)
flat_pos = jnp.asarray(np.concatenate(dev_pos))
flat_mass = jnp.asarray(np.concatenate(dev_mass))
nv = jnp.asarray(np.asarray(dev_live, np.int32))
a_mesh, ovf = force(flat_pos, flat_mass, nv)
a_mesh = np.asarray(jax.block_until_ready(a_mesh), np.float64)
print(f"mesh run done; overflow={bool(ovf)}  accel shape={a_mesh.shape}", flush=True)

print()
worst = 0.0
for d in range(NDEV):
    got = a_mesh[d * CAP : (d + 1) * CAP][: dev_live[d]]
    ref = refs[d][: dev_live[d]]
    dd = got - ref
    rel = float(np.sqrt((dd * dd).sum() / (ref * ref).sum()))
    worst = max(worst, rel)
    print(
        f"  device {d}: mesh vs single-device  max|da| = {np.abs(dd).max():.3e}  rel-L2 = {rel:.3e}"
    )

# --- and the physics is still right: local-only force vs an fp64 direct sum over the SAME shard
rng = np.random.default_rng(12345)
for d in range(NDEV):
    sel = shards[d]
    k = min(1024, len(sel))
    idx = rng.choice(len(sel), k, replace=False)
    ref64 = np.asarray(
        direct_accelerations(
            jnp.asarray(pos[sel], jnp.float64),
            jnp.asarray(mass[sel], jnp.float64),
            G=1.0,
            softening=SOFT,
            target_indices=jnp.asarray(idx),
        ),
        np.float64,
    )
    got = a_mesh[d * CAP : (d + 1) * CAP][: dev_live[d]][idx]
    dd = got - ref64
    print(
        f"  device {d}: local-only aggL2 vs fp64 direct over its own shard = "
        f"{float(np.sqrt((dd*dd).sum()/(ref64*ref64).sum())):.4e}"
    )

assert not bool(ovf), "a capacity saturated"
assert worst < 1e-5, f"mesh differs from single-device by {worst:.3e}"
print(
    f"\nGATE G1 (ndev={NDEV}) PASSED: worst mesh-vs-single-device rel-L2 = {worst:.3e}"
)
