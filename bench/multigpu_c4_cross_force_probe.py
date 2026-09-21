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
SOLO = os.environ.get("PROBE_SOLO") == "1"
# Working dtype of BOTH arms. The gate compares the distributed lane against the
# same lane on one device, so the two must run at the same width or the ratio
# means nothing; the SOLO process reads the same variable. `float64` is the
# instrument for the fp32-floor question (record: the ratio widened 1.26x -> 2.36x
# over p = 4 -> 6 at fp32, i.e. the distributed arm approached a floor near
# 1.9e-03 that the reference passed through).
DTYPE = os.environ.get("PROBE_DTYPE", "float32")
assert DTYPE in ("float32", "float64"), DTYPE
CAP = int(N / NDEV * 1.15)
# `apply_fast_lane_env` is process-wide and has to be tuned to the length the lane
# will actually see. The mesh arm sees CAP rows per device, the reference arm sees
# all N, and tuning for one makes the solver refuse the other outright -- which is
# why the reference arm is a separate PROCESS and not a third call here.
apply_fast_lane_env(
    N if SOLO else CAP, overrides=fast_lane_overrides_for_leaf(LEAF, N if SOLO else CAP)
)
# BOTH lengths: the fused profile gate keys on the EXACT array length, so the
# per-shard arm (CAP rows) and the single-GPU comparison arm (N rows) each
# need their own entry or the second one refuses to run at all
os.environ["JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"] = f"{CAP},{N}"
_TRAV = dict((FAST_LANE_ENV_BY_LEAF.get(LEAF) or {}).get("_traversal_overrides", {}))

import numpy as np, jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
WDT = getattr(np, DTYPE)
JDT = getattr(jnp, DTYPE)
ACCUM = os.environ.get("JACCPOT_NEARFIELD_ACCUM", "input")
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

ORDER = int(os.environ.get("PROBE_ORDER", "4"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
SOFT = float(os.environ.get("PROBE_SOFT", "1e-7"))
print(f"devices visible: {jax.devices()}", flush=True)
# The reference arm runs the lane on ONE device, but it still splits by NDEV --
# it has to sample the same particles the mesh arm does, and the shard split is
# what decides those.
if not SOLO:
    assert len(jax.devices()) >= NDEV, f"need {NDEV} devices"

_ic = IC_GENERATORS["plummer"](N, seed=0)
# The generator hands out float32; widening it changes no particle, so the fp64
# arm evaluates the SAME system as the fp32 one, just without fp32 arithmetic.
pos = np.asarray(_ic[0], WDT)
mass = np.asarray(_ic[1], WDT)
P0 = jnp.asarray(pos)
codes = np.asarray(morton_encode(P0, infer_bounds(P0)))
order = np.argsort(codes)
shards = np.array_split(order, NDEV)
kk = int(cp.adaptive_cell_leaf_partition_numpy(np.sort(codes), leaf_size=LEAF)[0].size)
# The leaf capacity is per TREE, and the reference arm trees all N while each mesh
# device trees a shard -- dividing by NDEV in the reference arm overflows the cut.
LEAF_CAP = 1 << int(np.ceil(np.log2(1.25 * kk / (1 if SOLO else NDEV))))
print(
    f"N={N} ndev={NDEV} cap={CAP} leaf_capacity={LEAF_CAP} shards={[len(s) for s in shards]} "
    f"dtype={DTYPE} nearfield_accum={ACCUM}",
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
        working_dtype=JDT,
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
    dev_mass.append(np.concatenate([mass[sel], np.zeros(CAP - len(sel), WDT)]))
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

if SOLO:
    # THE GATE'S REFERENCE. Not "is the distributed force exact" -- it cannot be,
    # the cross field is approximated -- but "is it as accurate as the SAME lane
    # run on one device at the same (p, theta, leaf)". That is the only comparison
    # the design can be held to.
    #
    # It samples the SAME particles as the mesh arm: same shard split, same seed,
    # same draw order. Comparing different target sets would swing the ratio on
    # its own (see the rel_l2-probe finding in the record).
    LS._WIDTHS.clear()
    LS._LEVELS.clear()
    solo = build()
    a_solo = np.asarray(
        jax.block_until_ready(
            solo.compute_accelerations(
                jnp.asarray(pos),
                jnp.asarray(mass),
                bounds=(BLO, BHI),  # the SAME Morton box, or the trees differ
                leaf_size=LEAF,
                max_order=ORDER,
                theta=THETA,
            )
        ),
        np.float64,
    )
    rng = np.random.default_rng(12345)
    allp = jnp.asarray(pos, jnp.float64)
    allm = jnp.asarray(mass, jnp.float64)
    num = den = 0.0
    for d in range(NDEV):
        sel = shards[d]
        k = min(512, len(sel))
        pick = rng.choice(len(sel), size=k, replace=False)
        ref = np.asarray(
            direct_accelerations(
                allp, allm, G=1.0, softening=SOFT, target_indices=np.asarray(sel)[pick]
            ),
            np.float64,
        )
        got = a_solo[np.asarray(sel)[pick]]
        num += float(((got - ref) ** 2).sum())
        den += float((ref**2).sum())
    print(
        f"SOLO order={ORDER} theta={THETA} leaf={LEAF} dtype={DTYPE} accum={ACCUM}  "
        f"rel-L2 vs fp64 direct = {float(np.sqrt(num / den)):.4e}"
    )
    raise SystemExit(0)


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


DIAG_KEYS = (
    "summary_cells",
    "summary_leaves",
    "export_far",
    "recv_nodes",
    # num_csr is the number of SEEDS the receiver walk starts from. If far_pairs
    # ~= recv_csr the sender's decisions survive the receiver's re-test and the
    # walk is a pass-through; if near_pairs is large instead, the receiver is
    # rejecting what the sender promised, and that is a design fault, not a cap.
    "recv_csr",
    "far_pairs",
    "near_pairs",
    "imported_rows",
    "imported_zero_radius",
    "near_recv_nodes",
    "near_list_pairs",
    "near_walk_far_pairs",
    "center_mismatch",
    "center_max_delta",
    "export_far_live",
    "export_mac_fail",
    "export_near_live",
    "export_near_mac_fail",
    "seed_live",
    "seed_mac_fail",
    "leaf_pool_mismatch",
    "leaf_pool_rows",
    "perm_nonidentity",
)


def run(hook, near_sink=None, record=None, keys=()):
    f = make_fused_force_evaluator(
        solver,
        stacked,
        mesh=mesh,
        plan=plan,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_hook=hook,
        cross_near_sink=near_sink,
        cross_record=record,
        cross_record_keys=keys,
    )
    out = f(flat_pos, flat_mass, nv)
    a, ovf = out[0], out[1]
    acc = np.asarray(jax.block_until_ready(a), np.float64)
    if keys:
        vals = {k: np.asarray(v) for k, v in zip(keys, out[2])}
        print("  diagnostics (per device):")
        for k in keys:
            print(f"    {k:<24} {vals[k]}")
    return acc, bool(ovf)


caps = CrossCapacities(
    # The cut needs about num_leaves / max_leaves_per_cell cells; at leaf 64 and
    # N/2 per device that is ~2000, so the 1024 first written here TRUNCATED the
    # summary and quietly removed half of each receiver from the exchange.
    max_cells=int(os.environ.get("PROBE_MAX_CELLS", "8192")),
    # Every one of these is over-allocated on purpose. The previous run saturated
    # export_far (69717 against a 65536 cap) and the summary cut, and NEITHER was
    # visible: the flags existed but were not OR-ed into anything the caller reads.
    # The rule the record already states -- over-allocate and read the flags -- only
    # works if the flags are wired, so both halves of that are now true.
    export_far_cap=1 << 21,
    export_near_cap=1 << 21,
    send_node_cap=1 << 15,
    send_csr_cap=1 << 21,
    recv_node_cap=1 << 15,
    # The receiver walk SEEDS from the received CSR, one pair per entry, so the
    # queue has to be able to hold that seed: walk_queue > recv_csr_cap is a hard
    # requirement, not a tuning choice.
    recv_csr_cap=1 << 19,
    walk_queue=1 << 20,
    recv_far_cap=1 << 21,
    recv_near_cap=1 << 21,
    leaf_width=LEAF,
)
a_local, ovf_local = run(None)
rec_far = {}
a_far, ovf_far = run(
    make_cross_hook(ndev=NDEV, theta=THETA, caps=caps, record=rec_far),
    record=rec_far,
    keys=("summary_cells", "export_far", "recv_nodes", "far_pairs", "near_pairs"),
)
# The near half rides the SAME hook -- one walk, one exchange. Handing it a sink
# is what makes it ship the leaf particles and return the near list; without one
# the hook does the far half alone, which is exactly the `a_far` arm above.
sink = {}
rec = {}
a_both, ovf_both = run(
    make_cross_hook(
        ndev=NDEV,
        theta=THETA,
        caps=caps,
        near_sink=sink,
        record=rec,
        near_theta=(
            float(os.environ["PROBE_NEAR_THETA"])
            if "PROBE_NEAR_THETA" in os.environ
            else None
        ),
    ),
    near_sink=sink,
    record=rec,
    keys=DIAG_KEYS,
)
print(
    f"local-only overflow={ovf_local}   far overflow={ovf_far}   "
    f"far+near overflow={ovf_both}",
    flush=True,
)

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
    print(
        f"  {tag:<28} rel-L2 vs fp64 direct (ALL sources) = {rel:.4e}"
        f"   [order={ORDER} dtype={DTYPE} accum={ACCUM}]"
    )
    return rel


print()
e_local = err_against_direct(a_local, "local-only (no cross)")
e_far = err_against_direct(a_far, "+ cross FAR")
e_cross = err_against_direct(a_both, "+ cross FAR and NEAR")

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
    f"C5  far-only / far+near = {e_far / e_cross:.2f}x  (the cross NEAR field; "
    f"Phase 3.1 measured it at 1.4-4.3% of the local near list, so a SMALL "
    f"improvement here is the expected size, not a weak result)"
)
if ovf_far or ovf_both:
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
