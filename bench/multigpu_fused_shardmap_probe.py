"""Gate G1, first half: the fused pipeline under ``shard_map`` at one device.

GPU only, and not a unit test: the fused lane needs the large-N production
profile, which does not engage on CPU, so this is a bench probe in the pattern of
``bench/grad_step_profile.py``. Run it on an idle card::

    CUDA_VISIBLE_DEVICES=<idle> PYTHONPATH=<bench>/sitecustom_wt \
    JACCPOT_WORKTREE=$PWD YGGDRAX_WORKTREE=<ygg worktree> \
    XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/multigpu_fused_shardmap_probe.py

A one-device mesh, with the prepared state riding as a closure constant --
exactly right at one device, and the cheapest way to learn whether the Pallas
kernels, the walk's ``while_loop`` and the nested jits survive a manual-axis
trace at all.

**Two arms, and the control is the point.** The production arm takes its Morton
box from the all-reduce, which adds slack that ``infer_bounds`` does not, so it
builds a DIFFERENT (equally valid) tree and lands ~2.6e-3 away from the
reference -- the size of the FMM error itself. Comparing only that arm would
leave "the trace changed the answer" and "the box changed the tree"
indistinguishable. The same-box arm passes the reference's own box in and so
isolates the trace.

Measured 2026-09-15, one A100, N=2e5, cells64, theta 0.8, p4 (host load ~35, so
no timing is quoted -- this probe is about correctness):

    same-box inside vs outside   max|da| 6.985e-10   rel-L2 1.760e-11
    aggL2 vs an fp64 direct sum  2.2625e-03 outside, 2.2625e-03 same-box,
                                 2.2392e-03 with the reduced box
    capacity plan                level_batch_width 6788 against num_internal
                                 16383, i.e. the cold-registry fallback would
                                 have been 2.4x wider per level
"""

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
from codes.compare_force import (
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)

N = int(os.environ.get("PROBE_N", "200000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
_ov = fast_lane_overrides_for_leaf(LEAF, N)
apply_fast_lane_env(N, overrides=_ov)
_TRAV = dict((FAST_LANE_ENV_BY_LEAF.get(LEAF) or {}).get("_traversal_overrides", {}))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
import yggdrax._cell_partition as cp
from common.ic import IC_GENERATORS
from common.reference import direct_accelerations
from jax.sharding import PartitionSpec as P
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
    global_mesh_bounds,
    reduce_flag_across_mesh,
)
from jaccpot.runtime.capacity_plan import (
    fused_capacity_plan_overrides,
    plan_from_registry,
)

ORDER, THETA, SOFT = 4, 0.8, 1e-7
_ic = IC_GENERATORS["plummer"](N, seed=0)
pos = np.asarray(_ic[0], np.float32)
mass = np.asarray(_ic[1], np.float32)
P0 = jnp.asarray(pos)
codes = np.sort(np.asarray(morton_encode(P0, infer_bounds(P0))))
k = int(cp.adaptive_cell_leaf_partition_numpy(codes, leaf_size=LEAF)[0].size)
CAPACITY = 1 << int(np.ceil(np.log2(1.25 * k)))
print(f"N={N} cells={k} leaf_capacity={CAPACITY}", flush=True)

solver = FastMultipoleMethod(
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
            leaf_capacity=CAPACITY,
        ),
        farfield=FarFieldConfig(mode="auto"),
        nearfield=NearFieldConfig(mode="auto"),
        runtime=(
            RuntimePolicyConfig(
                traversal_config=TraversalOverrides(
                    **{k2: int(v) for k2, v in _TRAV.items()}
                )
            )
            if _TRAV
            else RuntimePolicyConfig()
        ),
        mac_type="dehnen",
    ),
    fixed_order=ORDER,
)
prepared = solver.strict_fused_prepared_eval_fn(
    positions=P0, masses=jnp.asarray(mass), leaf_size=LEAF, max_order=ORDER, theta=THETA
)[0]

tree = prepared.tree
TN = int(np.asarray(tree.node_ranges).shape[0])
NI = int(tree.left_child.shape[0])
plan = plan_from_registry(total_nodes=TN, num_internal=NI)
print(f"plan: {plan}", flush=True)

lo, hi = infer_bounds(P0)

# --- reference: the same step OUTSIDE shard_map
_, a_ref = fused_force_step(
    solver,
    prepared,
    P0,
    jnp.asarray(mass),
    bounds=(lo, hi),
    leaf_size=LEAF,
    max_order=ORDER,
    theta=THETA,
)
a_ref = np.asarray(jax.block_until_ready(a_ref), np.float64)
print("reference (no shard_map) done", flush=True)

# --- the same step INSIDE a 1-device shard_map
mesh = make_mesh(1)
print(f"mesh: {mesh}", flush=True)


def body_same_box(positions, masses, blo, bhi):
    # CONTROLLED arm: identical box to the reference, so any difference is the
    # shard_map trace itself and not a different Morton frame.
    _, accel = fused_force_step(
        solver,
        prepared,
        positions,
        masses,
        bounds=(blo, bhi),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    ok = reduce_flag_across_mesh(jnp.asarray(False))
    return accel, ok


def body_reduced_box(positions, masses):
    # PRODUCTION arm: the box comes from the all-reduce.
    bounds = global_mesh_bounds(positions)
    _, accel = fused_force_step(
        solver,
        prepared,
        positions,
        masses,
        bounds=bounds,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    return accel, reduce_flag_across_mesh(jnp.asarray(False))


with fused_capacity_plan_overrides(plan):
    fn_same = jax.jit(
        jax.shard_map(
            body_same_box,
            mesh=mesh,
            in_specs=(P("gpus"), P("gpus"), P(), P()),
            out_specs=(P("gpus"), P()),
            check_vma=False,
        )
    )
    a_same, ok = fn_same(P0, jnp.asarray(mass), lo, hi)
    a_same = np.asarray(jax.block_until_ready(a_same), np.float64)
    fn = jax.jit(
        jax.shard_map(
            body_reduced_box,
            mesh=mesh,
            in_specs=(P("gpus"), P("gpus")),
            out_specs=(P("gpus"), P()),
            check_vma=False,
        )
    )
    a_sm, _ok2 = fn(P0, jnp.asarray(mass))
    a_sm = np.asarray(jax.block_until_ready(a_sm), np.float64)
print(f"shard_map done; overflow flag = {bool(ok)}", flush=True)
ds = a_same - a_ref
print(
    f"  SAME-BOX inside vs outside: max|da| = {np.abs(ds).max():.3e}  "
    f"rel-L2 = {np.sqrt((ds*ds).sum()/(a_ref**2).sum()):.3e}",
    flush=True,
)

idx = np.random.default_rng(12345).choice(N, 2048, replace=False)
ref = np.asarray(
    direct_accelerations(
        jnp.asarray(pos, jnp.float64),
        jnp.asarray(mass, jnp.float64),
        G=1.0,
        softening=SOFT,
        target_indices=jnp.asarray(idx),
    ),
    np.float64,
)


def aggL2(a):
    d = a[idx] - ref
    return float(np.sqrt((d * d).sum() / (ref * ref).sum()))


d = a_sm - a_ref
print(f"  same-box   shard_map aggL2 = {aggL2(a_same):.4e}")
print(f"\n  outside shard_map aggL2 = {aggL2(a_ref):.4e}")
print(f"  inside  shard_map aggL2 = {aggL2(a_sm):.4e}")
print(
    f"  inside vs outside: max|da| = {np.abs(d).max():.3e}  rel-L2 = {np.sqrt((d*d).sum()/(a_ref**2).sum()):.3e}"
)
assert aggL2(a_sm) < 5e-3
print("\nGATE G1 (ndev=1): the fused pipeline traces and runs under shard_map.")
