"""Calibrate the traversal caps for the Dehnen mass-dependent MAC.

`_derive_walk_caps` was fitted against the GEOMETRIC MAC, and its wavefront rule scales
as `(0.4/theta)**1.5` because for a geometric walk theta is what sets the descent depth.
Under `mac_type="dehnen_error"` theta decides nothing -- `adaptive_pair_policy` deletes
`mac_ok` outright -- so that rule is being driven by a knob that gates nothing, and the
port currently floors the queues at theta 0.3 as a BOUND rather than a derivation. This
measures the derivation.

METHOD, following the rule's own provenance: hold every other cap generous and walk ONE
cap down by halving until the walk truncates. The floor is the smallest value that did
not. Two independent witnesses, because one of them can be wrong:

  * the overflow flag for that buffer, and
  * the far/near COUNTS against an untruncated reference run.

A truncated walk reads FASTER and only these say so (trap 6). Counts are checked as well
as flags because a flag is a guard someone had to remember to write; a count change is
the thing itself.

Reported as a COEFFICIENT on the structural quantity each cap is derived from, so the
result is a rule that extrapolates rather than a number for one problem size.
"""

import json
import math
import os
import time

import numpy as np
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import DistributedFMMConfig, distributed_fmm_accelerations

SP = "/tmp/claude-2701/-export-home-tbuck-jaccpot/dbf56a41-2e29-4b34-83a5-303b4a7e8552/scratchpad"
ndev = device_count()
mesh = make_mesh(ndev)
PER = int(os.environ.get("CAL_PER", 524288))
LEAF = int(os.environ.get("CAL_LEAF", 512))
EPS = float(os.environ.get("CAL_EPS", 1e-5))
ORDER, THETA, SOFT = 4, 0.5, 0.01
SEED = int(os.environ.get("CAL_SEED", 20260831))
TAG = f"ndev{ndev}_per{PER}_leaf{LEAF}_eps{EPS:g}"

rng = np.random.default_rng(SEED)
n = PER * ndev
r = (rng.random(n) ** (-2.0 / 3.0) - 1.0) ** -0.5
d = rng.normal(size=(n, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
pts = (d * np.clip(r, 0, 50.0)[:, None]).astype(np.float32)
mass = np.full(n, 1.0 / n, dtype=np.float32)

# structural quantities the geometric rule derives each cap from
num_leaves = max(1, -(-PER // LEAF))
wavefront = num_leaves * math.isqrt(num_leaves)
remote = max(1, ndev - 1)
STRUCT = {
    "max_pair_queue": wavefront,
    "max_interactions_per_node": num_leaves,
    "max_neighbors_per_leaf": num_leaves,
    "cross_max_pair_queue": math.sqrt(remote) * wavefront,
    "cross_max_interactions_per_node": remote * num_leaves,
    "cross_max_neighbors_per_leaf": remote * num_leaves,
}
FLAG = {
    "max_pair_queue": "self_queue_overflow",
    "max_interactions_per_node": "self_far_overflow",
    "max_neighbors_per_leaf": "self_near_overflow",
    "cross_max_pair_queue": "cross_queue_overflow",
    "cross_max_interactions_per_node": "cross_far_overflow",
    "cross_max_neighbors_per_leaf": "cross_near_overflow",
}
COUNTS = ("self_far_pairs", "self_near_pairs", "cross_far_pairs", "cross_near_pairs")

# Generous: enough headroom that no cap under test is limited by another. 4x the
# derived value, which the geometric rule already carries 2x margin on.
derived = DistributedFMMConfig(
    leaf_size=LEAF, theta=THETA, order=ORDER, mac_type="dehnen_error", adaptive_eps=EPS
).resolved_for(PER, ndev)
GENEROUS = {k: 4 * int(getattr(derived, k)) for k in STRUCT}
print(
    f"[{TAG}] num_leaves={num_leaves} wavefront={wavefront} remote={remote}", flush=True
)
print(
    "derived:  " + "  ".join(f"{k}={int(getattr(derived,k))}" for k in STRUCT),
    flush=True,
)


def _clear_queue_cache():
    from yggdrax._interactions_impl import _DUAL_TREE_QUEUE_CACHE

    _DUAL_TREE_QUEUE_CACHE.clear()


def run(caps):
    _clear_queue_cache()
    cfg = DistributedFMMConfig(
        leaf_size=LEAF,
        order=ORDER,
        theta=THETA,
        softening=SOFT,
        mac_type="dehnen_error",
        adaptive_eps=EPS,
        nearfield_accum="wide",
        m2l_chunk=131072,
        nearfield_chunk=512,
        **caps,
    )
    t0 = time.perf_counter()
    res = distributed_fmm_accelerations(pts, mass, config=cfg, mesh=mesh, jit=True)
    dg = {k: np.asarray(v) for k, v in res.diagnostics.items()}
    return {
        "seconds": round(time.perf_counter() - t0, 1),
        "flags": {k: float(dg[k].sum()) for k in FLAG.values()},
        "counts": {k: float(dg[k].sum()) for k in COUNTS},
        "overflow": bool(res.overflow),
    }


ref = run(dict(GENEROUS))
assert not ref["overflow"], f"the generous reference itself overflowed: {ref['flags']}"
print(f"reference (4x derived): counts={ref['counts']} t={ref['seconds']}s", flush=True)

rows = []
for cap in STRUCT:
    start = int(getattr(derived, cap))

    def probe(v):
        caps = dict(GENEROUS)
        caps[cap] = v
        out = run(caps)
        bad_flag = out["flags"][FLAG[cap]] > 0
        bad_count = any(out["counts"][c] != ref["counts"][c] for c in COUNTS)
        ok = not bad_flag and not bad_count
        print(
            f"  {cap:34s} {v:9d} ok={ok!s:5s} flag={bad_flag!s:5s} "
            f"count_moved={bad_count!s:5s} t={out['seconds']}s",
            flush=True,
        )
        return ok

    # Both directions. The derived value is a GUESS fitted against the geometric MAC,
    # so it can be too large (walk down to the floor) or too small (walk up until the
    # walk stops truncating). Assuming only the first is how a cap that is already
    # under-provisioned gets recorded as "no data" instead of as a defect.
    floor, failed_at = None, None
    if probe(start):
        floor, value = start, start // 2
        while value >= 2 and probe(value):
            floor, value = value, value // 2
        failed_at = value if value >= 2 else None
    else:
        failed_at = start
        value = start * 2
        while value <= 64 * start:
            if probe(value):
                floor = value
                break
            value *= 2
    rows.append(
        dict(
            cap=cap,
            derived=start,
            floor=floor,
            failed_at=failed_at,
            structural=STRUCT[cap],
            coeff_derived=start / STRUCT[cap],
            coeff_floor=(floor / STRUCT[cap]) if floor else None,
            headroom=(start / floor) if floor else None,
        )
    )
    json.dump(
        {
            "tag": TAG,
            "ndev": ndev,
            "per_device": PER,
            "leaf": LEAF,
            "eps": EPS,
            "num_leaves": num_leaves,
            "reference": ref,
            "rows": rows,
        },
        open(f"{SP}/capcal_{TAG}.json", "w"),
        indent=1,
    )

print()
print(f"{'cap':34s} {'derived':>9} {'floor':>9} {'headroom':>9} {'coeff@floor':>12}")
for r_ in rows:
    floor = r_["floor"]
    head = "n/a" if not r_["headroom"] else f"{r_['headroom']:.1f}x"
    coef = "n/a" if not r_["coeff_floor"] else f"{r_['coeff_floor']:.4g}"
    print(f"{r_['cap']:34s} {r_['derived']:9d} {(floor or -1):9d} {head:>9} {coef:>12}")
print("CAPCAL_DONE", flush=True)
