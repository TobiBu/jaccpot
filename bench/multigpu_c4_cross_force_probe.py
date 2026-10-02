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

# the Odisseo bench harness (codes/, common/); BENCH_DIR on any other machine
sys.path.insert(
    0,
    os.environ.get(
        "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
    ),
)
from codes.compare_force import (
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
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
# The NCCL ragged exchange, not XLA's one-shot kernel: 4-6x faster per round on PCIe
# (see `jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAG`). It has to be in XLA_FLAGS
# before the backend starts; PROBE_RAGGED_ONE_SHOT=1 keeps the old path for an A/B.
_RAGGED = "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false"
if os.environ.get("PROBE_RAGGED_ONE_SHOT") != "1" and _RAGGED.split("=")[0] not in (
    os.environ.get("XLA_FLAGS", "")
):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _RAGGED).strip()
# XLA's latency-hiding scheduler: lets the cross exchange overlap independent work.
# 2-card cross arm, cards 1+2: 90.0 / 89.5 -> 88.0 / 87.6 ms at 2e6 and 21.7 -> 21.0
# ms at 4e5, forces unchanged (2026-10-02). PROBE_LHS=0 leaves it out for an A/B.
_LHS = "--xla_gpu_enable_latency_hiding_scheduler=true"
if os.environ.get("PROBE_LHS") != "0" and _LHS.split("=")[0] not in (
    os.environ.get("XLA_FLAGS", "")
):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _LHS).strip()

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
WDT = getattr(np, DTYPE)
JDT = getattr(jnp, DTYPE)
ACCUM = os.environ.get("JACCPOT_NEARFIELD_ACCUM", "input")
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
    cube_bounds,
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
from jaccpot.runtime.kernels._evaluate import _infer_bounds as _infer_lane_bounds

ORDER = int(os.environ.get("PROBE_ORDER", "4"))
THETA = float(os.environ.get("PROBE_THETA", "0.8"))
SOFT = float(os.environ.get("PROBE_SOFT", "1e-7"))
print(f"devices visible: {jax.devices()}", flush=True)
# The reference arm runs the lane on ONE device, but it still splits by NDEV --
# it has to sample the same particles the mesh arm does, and the shard split is
# what decides those.
if not SOLO:
    assert len(jax.devices()) >= NDEV, f"need {NDEV} devices"

# PROBE_SEED: the Plummer draw. Gate rows on ONE seed conflate code and draw (the
# 2e6 seed-0 draw is a heavy geometry: walk peak 12.6 per leaf against 8.0 for the
# 1e6 and 8e6 draws), so the gate is measured over several.
SEED = int(os.environ.get("PROBE_SEED", "0"))
_ic = IC_GENERATORS["plummer"](N, seed=SEED)
# The generator hands out float32; widening it changes no particle, so the fp64
# arm evaluates the SAME system as the fp32 one, just without fp32 arithmetic.
pos = np.asarray(_ic[0], WDT)
mass = np.asarray(_ic[1], WDT)
P0 = jnp.asarray(pos)
# the partition and the leaf counts use the lane's own (cubic) box rule
codes = np.asarray(morton_encode(P0, _infer_lane_bounds(P0)))
order = np.argsort(codes)
shards = np.array_split(order, NDEV)
# no leaf coarser than this Morton level (TreeConfig.cell_min_level; 0 = unconstrained).
# Default 8: one A100 -4..-18 % from 2e5 to 8e6, two A100s -8..-16 %, forces unchanged.
CELL_MIN_LEVEL = int(os.environ.get("PROBE_CELL_MIN_LEVEL", "8"))
kk = int(
    cp.adaptive_cell_leaf_partition_numpy(
        np.sort(codes), leaf_size=LEAF, min_level=CELL_MIN_LEVEL
    )[0].size
)
# live cell leaves of the WORST shard: what every cross capacity scales with
SHARD_LEAVES = max(
    int(
        cp.adaptive_cell_leaf_partition_numpy(
            np.sort(codes[s]), leaf_size=LEAF, min_level=CELL_MIN_LEVEL
        )[0].size
    )
    for s in shards
)
# The leaf capacity is per TREE, and the reference arm trees all N while each mesh
# device trees a shard. It used to be the next power of two above 1.25x the leaves,
# which padded up to 2.4x -- and time is linear in that padding (4M on one card:
# 299 / 374 / 541 ms at 1.2 / 2.4 / 4.8x, same pairs to the digit), so crossing a
# power of two by a few hundred leaves doubled the cost and masqueraded as a
# regression of whatever had added them. Now 1.15x the live leaves in steps of 1024;
# nothing needs a power of two. PROBE_LEAF_CAP_RULE=pow2 restores the old rule.
_live_for_cap = kk if (SOLO or NDEV == 1) else SHARD_LEAVES
if os.environ.get("PROBE_LEAF_CAP_RULE") == "pow2":
    LEAF_CAP = 1 << int(np.ceil(np.log2(1.25 * kk / (1 if SOLO else NDEV))))
else:
    LEAF_CAP = int(-(-int(np.ceil(1.15 * _live_for_cap)) // 1024) * 1024)
if os.environ.get("PROBE_LEAF_CAP"):
    LEAF_CAP = int(os.environ["PROBE_LEAF_CAP"])
# Evaluate at a different theta from the prepare's. A DIAGNOSTIC knob only: a tighter
# theta than the capacities were sized for saturates the walk UNDER TRACE, which is
# how the capacity-flag gate makes the local guard fire without tripping the eager
# prepare's own raise.
EVAL_THETA = float(os.environ.get("PROBE_EVAL_THETA", THETA))
# timing-mode arms: "local" (no cross hook), "cross" (production cross field)
TIME_ARMS = tuple(
    a.strip() for a in os.environ.get("PROBE_TIME_ARMS", "local,cross").split(",")
)
print(
    f"N={N} ndev={NDEV} cap={CAP} leaf_capacity={LEAF_CAP} worst-shard live leaves={SHARD_LEAVES} "
    f"shards={[len(s) for s in shards]} "
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
                cell_min_level=CELL_MIN_LEVEL or None,
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


# --- the probe targets: ONE draw per device, used by every arm in this process
# and by the SOLO process. Before this the rng was advanced once per arm, so the
# reference (first draw) and the far+near arm (third draw) scored DIFFERENT
# particles -- the rel_l2-probe finding in the record says that alone can swing a
# ratio by tens of percent.
_pick_rng = np.random.default_rng(12345)
PICKS = [
    _pick_rng.choice(len(sel), size=min(512, len(sel)), replace=False) for sel in shards
]

_DUMP = {}


def _dump(key, p, got, ref):
    """Per-particle (position, lane force, fp64 direct force) for the probe targets,
    written to PROBE_DUMP (an .npz) at exit so the error can be located on the sphere.
    """
    if os.environ.get("PROBE_DUMP"):
        _DUMP[f"{key}_pos"] = np.asarray(p, np.float64)
        _DUMP[f"{key}_got"] = np.asarray(got, np.float64)
        _DUMP[f"{key}_ref"] = np.asarray(ref, np.float64)


import atexit


@atexit.register
def _write_dump():
    if os.environ.get("PROBE_DUMP") and _DUMP:
        np.savez(os.environ["PROBE_DUMP"], **_DUMP)


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
BLO, BHI = cube_bounds(jnp.asarray(glo), jnp.asarray(ghi), pad=1e-6)
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
    allp = jnp.asarray(pos, jnp.float64)
    allm = jnp.asarray(mass, jnp.float64)
    num = den = 0.0
    for d in range(NDEV):
        sel = shards[d]
        pick = PICKS[d]
        ref = np.asarray(
            direct_accelerations(
                allp, allm, G=1.0, softening=SOFT, target_indices=np.asarray(sel)[pick]
            ),
            np.float64,
        )
        got = a_solo[np.asarray(sel)[pick]]
        num += float(((got - ref) ** 2).sum())
        den += float((ref**2).sum())
        _dump(f"solo_d{d}", pos[np.asarray(sel)[pick]], got, ref)
    print(
        f"SOLO order={ORDER} theta={THETA} leaf={LEAF} dtype={DTYPE} accum={ACCUM}  "
        f"rel-L2 vs fp64 direct = {float(np.sqrt(num / den)):.4e}"
    )
    raise SystemExit(0)


# --- eager prepare per shard, collect plans (same discipline as the G1 probe:
# ONE solver instance, warmed by the eager prepares the traced body depends on)
solver = build()
from jaccpot.runtime.capacity_plan import (
    install_walk_caps,
    measure_shard_plan,
    merge_walk_caps,
)

preps, plans, walk_reports = [], [], []
for d in range(NDEV):
    # Prepare each shard ON ITS OWN device: at large N one card cannot hold every
    # shard's state, and the assembly below then needs no cross-device copy.
    with jax.default_device(jax.devices()[d if not SOLO else 0]):
        pr, shard_plan, shard_caps = measure_shard_plan(
            solver,
            jnp.asarray(dev_pos[d]),
            jnp.asarray(dev_mass[d]),
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
        )
    plans.append(shard_plan)
    walk_reports.append(shard_caps)
    preps.append(pr)
plan = merge_plans(plans)
# each eager prepare overwrote the engine's walk record with ITS shard's needs; the
# traced walk of every device must be sized for the worst one
walk_caps = merge_walk_caps(walk_reports)
install_walk_caps(solver, walk_caps)
print(
    "merged walk caps: "
    + str(
        {k: walk_caps.get(k) for k in ("queue_capacity", "peak_wavefront")}
        if walk_caps
        else None
    ),
    flush=True,
)
print(f"merged plan: {plan}", flush=True)

from jaccpot.distributed.cross import CrossCapacities, make_cross_hook

mesh = make_mesh(NDEV)
from jaccpot.distributed.fused import assemble_prepared_states

# sharded P(axis) from per-device pieces: not re-split on every call, and no device
# holds another's state
stacked = assemble_prepared_states(preps, mesh)
del preps
flat_pos = jnp.asarray(np.concatenate(dev_pos))
flat_mass = jnp.asarray(np.concatenate(dev_mass))
nv = jnp.asarray(np.asarray(dev_live, np.int32))
# Pre-shard the inputs on the mesh once. Handing the jitted shard_map unsharded
# arrays (committed to device 0) makes every call re-shard them first, which a
# timing would then charge to the force.
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as _P
from yggdrax.distributed.comm import AXIS_NAME as _AXIS

_shard = NamedSharding(mesh, _P(_AXIS))
flat_pos = jax.device_put(flat_pos, _shard)
flat_mass = jax.device_put(flat_mass, _shard)
nv = jax.device_put(nv, _shard)

# Timing mode (PROBE_TIME_REPS > 0): the production configuration only -- no
# diagnostics record (it adds in-trace MAC recomputations), no far-only arm --
# timed with the bench's own `timed_calls` under its contention monitor. Every
# call is a FULL force: tree rebuild, walk, exchange, evaluation; the same scope
# as the single-GPU record's `scan_full` step (11.45 ms at N = 2e5, p6, th0.8).
TIME_REPS = int(os.environ.get("PROBE_TIME_REPS", "0"))
TIME_WARMUP = int(os.environ.get("PROBE_TIME_WARMUP", "3"))
TIMINGS = {}


DIAG_KEYS = (
    "summary_cells",
    # two-sided export (JACCPOT_CROSS_TWO_SIDED=1): the published summary TREE's size
    "summary_nodes",
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
    "near_csr",
    "near_particles",
    "near_rows_needing_particles",
    "near_rows_far_only",
    "near_csr_max_per_cell",
    "near_csr_top100_cells",
    "near_csr_cells_over_1000",
    "worst_cell_is_leaf",
    "worst_cell_particles",
    "worst_cell_leaves",
    "worst_cell_radius",
    "worst_cell_edge",
    "worst_cell_r_center",
    "root_radius",
    "export_near",
    "near_walk_peak",
    "export_walk_peak",
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


def _physical_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        return [int(x) for x in cvd.split(",") if x.strip() != ""][:NDEV]
    except ValueError:
        return list(range(NDEV))


def run(hook, near_sink=None, record=None, keys=(), time_tag=None):
    f = make_fused_force_evaluator(
        solver,
        stacked,
        mesh=mesh,
        plan=plan,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=EVAL_THETA,
        cross_hook=hook,
        cross_near_sink=near_sink,
        cross_record=record,
        cross_record_keys=keys,
    )
    t0 = time.perf_counter()
    out = f(flat_pos, flat_mass, nv)
    a, ovf = out[0], out[1]
    acc = np.asarray(jax.block_until_ready(a), np.float64)
    first_call_s = time.perf_counter() - t0
    if time_tag is not None and TIME_REPS > 0:
        from common.gpu_guard import timed_calls

        _, timing, cont = timed_calls(
            lambda: f(flat_pos, flat_mass, nv),
            repeats=TIME_REPS,
            warmup=TIME_WARMUP,
            devices=_physical_devices(),
            block=jax.block_until_ready,
        )
        timing["first_call_s"] = first_call_s
        timing["contention"] = {
            k: getattr(cont, k)
            for k in (
                "loadavg1_max",
                "other_gpu_util_max",
                "foreign_pids_on_devices",
                "contaminated",
                "flags",
            )
            if hasattr(cont, k)
        }
        TIMINGS[time_tag] = timing
        if os.environ.get("PROBE_PROFILE_DIR"):
            # One trace of a few warm calls per timed arm; attribute it per DEVICE
            # (the bench's analyser sums every device pid together).
            _tdir = os.path.join(os.environ["PROBE_PROFILE_DIR"], time_tag)
            with jax.profiler.trace(_tdir, create_perfetto_trace=True):
                for _ in range(5):
                    jax.block_until_ready(f(flat_pos, flat_mass, nv))
            print(f"  TRACE {time_tag} -> {_tdir} (5 calls)", flush=True)
        print(
            f"  TIMING {time_tag:<10} min {1e3 * timing['min']:.2f} ms  median "
            f"{1e3 * timing['median']:.2f} ms  iqr {1e3 * timing['iqr']:.2f} ms  "
            f"(first call incl. compile {first_call_s:.1f} s; {timing['contention']})",
            flush=True,
        )
    if keys:
        vals = {k: np.asarray(v) for k, v in zip(keys, out[2])}
        print("  diagnostics (per device):")
        for k in keys:
            print(f"    {k:<24} {vals[k]}")
    return acc, bool(ovf)


def _cap_bits(name, default):
    """log2 of a capacity, from the environment. The all-direct bisection arm
    (PROBE_EXPORT_THETA=0) ships every sender leaf and pairs it with every local
    leaf, which is far past the defaults; every flag is still read."""
    return int(os.environ.get(name, default))


EXPORT_THETA = (
    float(os.environ["PROBE_EXPORT_THETA"])
    if "PROBE_EXPORT_THETA" in os.environ
    else None
)


def _auto_cap(env_bits, per_leaf, floor_bits, headroom=2.5):
    """A cross capacity sized from the worst shard's live leaf count.

    `per_leaf` is the occupancy per live leaf measured at 1e5 per device (record,
    Phase C / Task 1 under the COM geometry), times `headroom`, times the number of
    senders a receiver can hear from; rounded up to a power of two and never below
    `floor_bits`. An explicit PROBE_*_BITS still wins. The flags are real now, so an
    undersized cap reports itself rather than truncating quietly.
    """
    if os.environ.get(env_bits):
        return 1 << int(os.environ[env_bits])
    need = headroom * per_leaf * SHARD_LEAVES * max(1, NDEV - 1)
    return 1 << max(int(floor_bits), int(np.ceil(np.log2(max(need, 1.0)))))


# JACCPOT_CROSS_TWO_SIDED=1: the export walk refines the receiver's summary TREE too,
# so far pairs land on its internal nodes. Measured at 2e5 on two cards: export far
# 54k per device (8.7 per leaf) against 283-367k one-sided (45-59), walk peak 6.2k
# (1.0 per leaf) against 36-47k, seeded with ndev pairs instead of ndev x max_cells.
from jaccpot.distributed.cross import _cross_two_sided

TWO_SIDED = _cross_two_sided()
_far_per_leaf = 16 if TWO_SIDED else 53

# per live leaf at 1e5/dev, leaf 64, theta 0.8, COM geometry: summary cells ~0.29;
# export far pairs ~53 (293k / 5.5k leaves); export near ~10; sent / received
# nodes ~1.25 (far nodes 6.7k, near leaves 5.5k); received CSR ~53; receiver far
# pairs ~53 (+ near-walk far pairs); receiver near pairs ~60
_recv_csr = _auto_cap("PROBE_RECV_CSR_BITS", _far_per_leaf, 19)
# Measured at 1e6 per device (2 cards, 2026-10-02, before summary_cell_level = 8):
# near CSR 1.43-1.51M entries (~29 per leaf), near receiver walk peak 2.66M pairs
# (~52 per leaf), export walk peak 0.72-0.90M (~17 per leaf). The FAR receiver walk
# is skipped (a pass-through, see cross._direct_far_lists), so the queue no longer
# has to hold the far CSR.
# Re-measured 2026-10-03 with summary_cell_level = 8 (both export walks): near CSR
# 0.27-0.29M (5.5 per leaf), near walk peak 0.52M (9.9), near walk far pairs 0.50M
# (9.5), near list 0.15M (2.9). The old factors left every one of these 8-17x
# oversized, and each is padded work: the near-walk far list joins the merged M2L CSR
# sort (27M wide at 2e6) and two more 8M sorts, the queue sets the Pallas walk grid.
# Sized to 1e6 per device -> 2^20 / 2^20 / 2^20 / 2^19, 4e6 -> 2^22 / 2^22 / 2^22 /
# 2^21 (both measured clean): 2-card 2e6 -3.3 ms one-sided, -2.9 two-sided; 8e6
# -10 / -13 ms; forces identical.
_recv_near_csr = _auto_cap("PROBE_RECV_NEAR_CSR_BITS", 7.5, 18)
_far_walk = EXPORT_THETA is not None or os.environ.get(
    "JACCPOT_CROSS_FAR_RECEIVER_WALK"
) in ("1", "true", "on")
# Two-sided, the summary cut defaults to ONE leaf per cell (cross._max_leaves_per_cell:
# 2e6 on two cards 61.3 -> 57.3 ms), so max_cells must hold every live leaf: the
# shard's leaf capacity bounds that exactly, and 2 x it bounds the summary tree.
_ml_env = os.environ.get("PROBE_MAX_LEAVES_PER_CELL")
MAX_LEAVES = int(_ml_env) if _ml_env else (1 if TWO_SIDED else 4)
_max_cells = int(os.environ.get("PROBE_MAX_CELLS", 0)) or (
    LEAF_CAP if MAX_LEAVES == 1 else _auto_cap("PROBE_MAX_CELLS_BITS", 0.29, 13)
)
caps = CrossCapacities(
    max_leaves_per_cell=MAX_LEAVES,
    # a summary cell must fit in one Morton cell of this level (0: off). Default 8:
    # 2-card 2e6 75.0 -> 72.1 ms, 8e6 254.2 -> 231.6 ms, forces unchanged.
    summary_cell_level=int(os.environ.get("PROBE_SUMMARY_CELL_LEVEL", "8")) or None,
    max_cells=_max_cells,
    export_far_cap=_auto_cap("PROBE_EXPORT_FAR_BITS", _far_per_leaf, 21),
    export_near_cap=_auto_cap("PROBE_EXPORT_NEAR_BITS", 10, 21),
    send_node_cap=_auto_cap("PROBE_SEND_NODE_BITS", 1.25, 15),
    send_csr_cap=_auto_cap("PROBE_SEND_CSR_BITS", _far_per_leaf, 21),
    recv_node_cap=_auto_cap("PROBE_RECV_NODE_BITS", 1.25, 15),
    recv_csr_cap=_recv_csr,
    recv_near_csr_cap=_recv_near_csr,
    # the near import's live particles, flat: ~1 exported leaf per local leaf at
    # ~18 particles each (51k leaves, ~0.95M particles at 1e6 per device)
    send_particle_cap=_auto_cap("PROBE_SEND_PARTICLE_BITS", 20, 18, headroom=2.0),
    recv_particle_cap=_auto_cap("PROBE_RECV_PARTICLE_BITS", 20, 18, headroom=2.0),
    # export walk peak ~17 per leaf; its seed is ndev x max_cells pairs (one-sided)
    # or ndev pairs (two-sided, peak ~1 per leaf at 2e5)
    export_walk_queue=(
        _auto_cap("PROBE_EXPORT_WALK_QUEUE_BITS", 3, 16, headroom=1.5)
        if TWO_SIDED
        else max(
            _auto_cap("PROBE_EXPORT_WALK_QUEUE_BITS", 17, 18, headroom=1.5),
            1 << int(np.ceil(np.log2(NDEV * _max_cells))),
        )
    ),
    # A receiver walk SEEDS from its received CSR, one pair per entry, so the queue
    # has to hold that seed (and the walk's peak). Only the near walk runs now
    # unless the far one is forced; every queue overflow raises the cross flag.
    walk_queue=max(
        _auto_cap("PROBE_WALK_QUEUE_BITS", 13, 18, headroom=1.5),
        _recv_near_csr,
        2 * _recv_csr if _far_walk else 0,
    ),
    # Re-measured at 1e6 per device with near_theta = theta (the old 106 / 60 per leaf
    # were the near_theta = 0 counts): with the far receiver walk skipped these hold
    # only the NEAR walk's lists -- far 2.36-2.38M (~46 per leaf), near 1.0M (~20).
    # Both widths are padded work downstream: the far list joins the M2L CSR sort and
    # the near list sets the cross near-field kernel's grid.
    recv_far_cap=_auto_cap("PROBE_RECV_FAR_BITS", 9.5, 18, headroom=2.0),
    recv_near_cap=_auto_cap("PROBE_RECV_NEAR_BITS", 4.5, 17, headroom=2.0),
    leaf_width=LEAF,
)
print(f"cross caps: {vars(caps)}", flush=True)
# --- the fp64 reference: every particle against ALL others, subsampled
allp = jnp.asarray(pos, jnp.float64)
allm = jnp.asarray(mass, jnp.float64)


def err_against_direct(accel_by_dev, tag):
    """rel-L2 of the distributed force against an fp64 direct sum over ALL particles."""
    num = den = 0.0
    for d in range(NDEV):
        sel = shards[d]
        pick = PICKS[d]
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
        _key = {
            "local-only (no cross)": "local",
            "+ cross FAR": "far",
            "+ cross FAR and NEAR": "both",
        }[tag]
        _dump(f"{_key}_d{d}", pos[np.asarray(sel)[pick]], got, ref)
    rel = float(np.sqrt(num / den))
    print(
        f"  {tag:<28} rel-L2 vs fp64 direct (ALL sources) = {rel:.4e}"
        f"   [order={ORDER} dtype={DTYPE} accum={ACCUM} export_theta={EXPORT_THETA}]"
    )
    return rel


if TIME_REPS > 0:
    e_local = e_cross = None
    ovf_local = ovf_both = None
    if "local" in TIME_ARMS:
        a_local, ovf_local = run(None, time_tag="local")
        e_local = err_against_direct(a_local, "local-only (no cross)")
    if "cross" in TIME_ARMS:
        # the production cross configuration, no record: near_theta defaults to theta
        sink = {}
        a_both, ovf_both = run(
            make_cross_hook(ndev=NDEV, theta=THETA, caps=caps, near_sink=sink),
            near_sink=sink,
            time_tag="cross",
        )
        e_cross = err_against_direct(a_both, "+ cross FAR and NEAR")
    print(f"local-only overflow={ovf_local}   far+near overflow={ovf_both}", flush=True)
    result = dict(
        n=N,
        ndev=NDEV,
        n_per_dev=N // NDEV,
        leaf=LEAF,
        order=ORDER,
        theta=THETA,
        eval_theta=EVAL_THETA,
        arms=TIME_ARMS,
        leaf_capacity=LEAF_CAP,
        worst_shard_live_leaves=SHARD_LEAVES,
        cross_caps=vars(caps),
        dtype=DTYPE,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        err_local=e_local,
        err_cross=e_cross,
        overflow_local=ovf_local,
        overflow_cross=ovf_both,
        timings=TIMINGS,
        cross_mac_geometry=os.environ.get("JACCPOT_CROSS_MAC_GEOMETRY", "com"),
        cross_two_sided=os.environ.get("JACCPOT_CROSS_TWO_SIDED", "0"),
        seed=SEED,
    )
    if os.environ.get("PROBE_JSON"):
        import json

        with open(os.environ["PROBE_JSON"], "w") as fh:
            json.dump(result, fh, indent=1, default=str)
    if ovf_local or ovf_both:
        raise SystemExit("TIMING FAILED: a capacity overflowed")
    raise SystemExit(0)
a_local, ovf_local = run(None)
rec_far = {}
a_far, ovf_far = run(
    # the same export knob as the far+near arm, or under PROBE_EXPORT_THETA=0 this
    # arm would be the ordinary far import while the arm below has none
    make_cross_hook(
        ndev=NDEV, theta=THETA, caps=caps, record=rec_far, export_theta=EXPORT_THETA
    ),
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
        export_theta=EXPORT_THETA,
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
