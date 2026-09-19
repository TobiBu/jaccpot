"""Phase C1/C0: the cross hook exists, fires, and changes nothing.

Plan `~/.claude/plans/phase-c-interleaved-cross-field.md`. The cross-domain exchange
has to happen between the upward and downward sweeps: the multipoles exist there and
the downward sweep has not consumed them, and doing it afterwards would mean a second
L2L cascade -- which Phase 3.4 ruled out by measuring the far half as the bigger one.

**C0, measured here and not assumed**: `LargeNPreparedState.upward` is ``None``. The
multipoles are an unretained intermediate, so no caller holding a prepared state can
reach them and the hook cannot live outside the prepare. The near payload, by
contrast, IS on the state (`nearfield_leaf_particle_indices` and its mask).

**C1**: with the hook installed but its result discarded, the force must be
BIT-IDENTICAL to the force without it. That is the first link in the chain of
bit-identity controls that runs to C3; only C4 is allowed to change an answer, so any
divergence is localised to exactly one change.

Run (pin the card -- `autocvd` is not on PATH here and an unpinned run lands on
device 0 where another user is):

    CUDA_VISIBLE_DEVICES=<idle> python bench/multigpu_cross_hook_probe.py
"""

import os
import sys

sys.path.insert(0, "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu")
from codes.compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)

N = int(os.environ.get("PROBE_N", "20000"))
LEAF = int(os.environ.get("PROBE_LEAF", "64"))
apply_fast_lane_env(N, overrides=fast_lane_overrides_for_leaf(LEAF, N))
_TRAV = dict((FAST_LANE_ENV_BY_LEAF.get(LEAF) or {}).get("_traversal_overrides", {}))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import yggdrax._cell_partition as cp  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402
from jaccpot import FastMultipoleMethod, TraversalOverrides  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.distributed.fused import fused_force_step  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.morton import morton_encode  # noqa: E402

ORDER, THETA, SOFT = 4, 0.8, 1e-7


def build():
    pos, mass = IC_GENERATORS["plummer"](N, seed=0)
    P0 = jnp.asarray(np.asarray(pos, np.float32))
    M0 = jnp.asarray(np.asarray(mass, np.float32))
    codes = np.sort(np.asarray(morton_encode(P0, infer_bounds(P0))))
    k = int(cp.adaptive_cell_leaf_partition_numpy(codes, leaf_size=LEAF)[0].size)
    cap = 1 << int(np.ceil(np.log2(1.25 * k)))
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
                leaf_capacity=cap,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            runtime=(
                RuntimePolicyConfig(
                    traversal_config=TraversalOverrides(
                        **{a: int(b) for a, b in _TRAV.items()}
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
        positions=P0, masses=M0, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )[0]
    return solver, prepared, P0, M0, mass


def main():
    solver, prepared, P0, M0, mass = build()
    print(f"N={N} leaf={LEAF} order={ORDER}")

    # ---- C0: the multipoles are NOT on the prepared state -------------------
    up = getattr(prepared, "upward", None)
    print(f"C0  prepared.upward is None: {up is None}")
    if up is not None:
        raise SystemExit("C0: upward IS retained -- the hook may not be needed")
    have_near = getattr(prepared, "nearfield_leaf_particle_indices", None)
    print(
        f"C0  near payload on the state: {None if have_near is None else have_near.shape}"
    )

    lo, hi = infer_bounds(P0)
    kw = dict(bounds=(lo, hi), leaf_size=LEAF, max_order=ORDER, theta=THETA)

    # ---- C1: the hook fires, and the force does not move --------------------
    seen = {}

    def hook(tree_artifacts):
        seen["called"] = seen.get("called", 0) + 1
        mp = tree_artifacts.upward.multipoles
        seen["packed"] = tuple(mp.packed.shape)
        seen["centers"] = tuple(mp.centers.shape)

    _, a_off = fused_force_step(solver, prepared, P0, M0, **kw)
    a_off = np.asarray(jax.block_until_ready(a_off))
    _, a_on = fused_force_step(solver, prepared, P0, M0, cross_hook=hook, **kw)
    a_on = np.asarray(jax.block_until_ready(a_on))

    print(f"C1  hook calls: {seen.get('called', 0)}")
    print(
        f"C1  multipoles seen: packed {seen.get('packed')} centers {seen.get('centers')}"
    )
    same = np.array_equal(a_off, a_on)
    print(f"C1  force BIT-IDENTICAL with the hook installed: {same}")
    if not seen.get("called"):
        raise SystemExit(
            "C1 FAILED: the hook never fired, so bit-identity proves nothing"
        )
    if not same:
        d = np.abs(a_off - a_on)
        raise SystemExit(f"C1 FAILED: force moved, max|d|={d.max():.3e}")
    print("C1 PASS")


if __name__ == "__main__":
    main()
