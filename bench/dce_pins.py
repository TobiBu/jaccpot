"""GPU pins of the protected lanes for the 2026-10 cleanup (``docs/cleanup_2026-10.md``).

A deletion phase must not move a number on a lane it is not meant to touch. The
CPU suite and its goldens cover the pure-JAX routes; these pins cover what only a
GPU runs (Pallas kernels, the fused scan as Odisseo drives it, the
differentiable lane's reverse kernels, the block-step lane on device).

Each pin runs in a FRESH process (``case``), with deterministic GPU ops, and saves
its arrays under ``--root/<tag>/<pin>.npz`` plus scalars in ``<pin>.json``:

=====  ===========================================================================
S1     fused ``strict_run_v2``, clipped Plummer 2e5 (``fused_memory_budget.py``
       defaults: leaf 64 cells, theta 0.8, p6): first force, final scan state,
       fp64 rel-L2 on 4096 targets
S2     the same lane at 5e4 with ``--use-pallas off`` (the near field's pure-JAX
       rectangle / target-block route)
S3     ``FastMultipoleMethod()`` defaults at 3e4 and 1e5 (a raise is recorded as
       the pin)
S4     Odisseo's differentiable lane below its large-N profile (preset "fast"):
       d/dpos and d/dmass of a weighted force sum, 2e4
S4b    the same through the large-N lane and its grad plan, with Odisseo's
       large-N env overrides, 2e4
S5     ``BlockStepFMM`` with Odisseo's ``BlockStepOptions`` defaults (jax
       backend, leaf 64, static shapes, device topology), 20 base steps at 2e4
S6     S1 with ``--clean-env``: no JACCPOT_* / YGGDRAX_* variables at all, so
       the fused lane runs on the library's own defaults
S1p    S1 on the particle carry (``JACCPOT_STRICT_CARRY=particles``): the start
       program, the shape template, the chain through the handle
S1t    S1p with ``JACCPOT_STRICT_CARRY_ORDER=tree``: the rows in Morton order
S1e    S1 with ``mac_type="dehnen_error"`` (eq 16a per pair inside the flat walk,
       ``adaptive_eps`` ``_S1E_EPS``, f_b carried through the scan), on the state
       carry (``force``, ``state``) and the particle carry (``*_particles``), with
       each arm's far-pair count
S1x    a 2e5 prefix of the 25M disc+bulge IC with its NFW halo as
       ``external_acceleration_fn`` (traced inside the scan), eps 1.5e-3, on the
       state and the particle carry
S1g    S1 at G = 4.3e-6 (``--G``), the masses divided by G: G m, the orbits and the
       forces are S1's, through a G that is not 1
M1     two cards: ``distributed/fmm.py`` built as Odisseo's mesh lane builds it
       (``MeshOptions`` defaults: leaf 512, theta 0.7, p6, rcb), one force at
       131,072, in input order
M2     two cards: ``bench/multigpu_rollout_gate.py`` (``FusedRollout``,
       repartitioned on device) at 1e5, 17 steps, repartition every 8: positions
       at steps 1, 16 and 17
M3     two cards: ``DistributedBlockStepFMM`` (jax backend, cross_theta 0.5,
       Plummer), 4 base steps at 8192
L1     TIMING: the fused production sequence (``--skip-eval``, particle carry,
       unnamed caps, arena 0.88), clipped Plummer 8e6, default kernel: step ms
       and B/particle
L2     TIMING: the same at 1e8 (the 1 s / step budget)
L3     TIMING: the 25M disc+bulge (ferrers3, eps 1.5e-3): step ms, B/particle,
       and the force's rel-L2 on 4096 targets
R1p    S1p through ``jaccpot.production.ProductionRunner`` (``fused_memory_budget.py
       --runner``): the same calls as runs continuing each other, one segment each;
       compared with S1p's arrays (the segments are the lane's own calls)
R1x    S1x's particle arm through the runner, chained (1, 4, 4, 4 steps) and in one
       segment of 13 (``--runner-once``): passes when the two states are bitwise --
       the external field across segment edges (S3 R2)
R1r    R1p with a restart file written at step 5, then the sequence continued from
       it in a FRESH process (``--runner-resume``, no eval prepare): passes when that
       state is the straight run's, bitwise, and the restore's digests matched
DET1   S1 recorded WITHOUT ``--xla_gpu_deterministic_ops`` (the production flags)
DET2   S4b's large-N gradient, recorded the same way
DET3   R1p recorded the same way (the runner on the production flags)
G1     the differentiable forward against the production forward on the fused
       lane, ``bench/grad_step_profile.py``'s operating point at 2e5 (cells of
       64, theta 0.8, p5, fp32, softening 1e-7): the production eval, then the
       differentiable forward on a FRESH solver (the bench's call order) and on
       the production solver after its strict call
=====  ===========================================================================

The M pins need two cards (``autocvd -n 2``) and are recorded separately:
``record --pins M1,M2,M3``.

The S1 variants (S1p, S1t, S1e, S1x, S1g) pin what S1 does not run: they were added
for the FusedForceProvider refactor (S2, 2026-10-15), which must keep every variant
of the scan bitwise, and are not in the default set yet:
``record --pins S1p,S1t,S1e,S1x,S1g``, ``compare ... --pins S1p,S1t,S1e,S1x,S1g``.
S2 gated every commit on nine of them, ``S1,S6,DET1,G1,S1p,S1t,S1e,S1x,S1g``.

**A change that must keep the fused lane's programs** (a refactor of
``strict_run_v2`` or ``FusedForceProvider``) runs ``bench/provider_hlo_identity.py``
first: it fails on one character of any program's text or one bit of any array,
on CPU (36 variants at N = 512, about 9 min at ``--jobs 4``) and on a GPU (18 at
2e5), and says which program moved, where a pin only says that a number did.
Record the base and the candidate from the SAME worktree path (the Triton IR embeds
the kernel source paths), the GPU base twice for the ``--control``, on a card booked
as under Usage below::

    cd <worktree>                      # at the base commit
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
        python bench/provider_hlo_identity.py record --tag base-cpu --jobs 4
    python bench/provider_hlo_identity.py record --device gpu --tag base-gpu
    python bench/provider_hlo_identity.py record --device gpu --tag base-gpu-b
    git checkout <candidate>           # the same worktree
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
        python bench/provider_hlo_identity.py record --tag cand-cpu --jobs 4
    python bench/provider_hlo_identity.py compare base-cpu cand-cpu
    python bench/provider_hlo_identity.py record --device gpu --tag cand-gpu
    python bench/provider_hlo_identity.py compare base-gpu cand-gpu --control base-gpu-b

Then the pins above (``record --pins ...`` on base and candidate, ``compare``).

The L pins time the lane, so they run without the deterministic-ops flag, on a
FREE card (``autocvd -n 1`` without ``-l``), and are compared on their scalars:
``compare <ref> <new> --pins L1,L2,L3 --envelope <ref>-b`` passes a pin when
neither row is contaminated, the step time is within max(3 %, twice the A-vs-A
spread) and the peak within max(2 %, twice the spread). L2 needs a 40 GB card for
about 10 minutes.

The runner pins (S3: R1p, R1x, R1r) are not in the default set either:
``record --pins R1p,R1x,R1r`` and ``compare <ref> <new> --pins R1p,R1x,R1r
--envelope <ref>-b``. R1p is compared with the reference's S1p when the reference has
no R1p of its own (``_REFERENCE_PIN``); R1x and R1r check their own two arms.

The DET pins answer one question: is the production path bitwise reproducible?
Record them twice under the same code (``record --tag X --pins DET1,DET2`` and
``--tag X-b``); ``compare X X-b --pins DET1,DET2`` passes only when every array
is bitwise equal.

G1 checks itself: ``compare <ref> X --pins G1`` passes when X's own record has the
differentiable forward within ``_G1_REL_L2_LIMIT`` (fp32 round-off) of the
production forward, in both call orders; the ref is only reported (``compare X X``
works). ``docs/fast_gradients_2026-09.md`` measured the two equal to 1.8e-11, on its
branch before bc62978; S0 found them 1.46e-3 apart (``docs/production_2026-10.md``),
which S4b and DET2 could not see because they compare the gradient lane only with
itself. G1 on 63587f2: 1.21e-3 on the fresh solver, 1.66e-7 on the production one.

Usage (book the card with autocvd first; the org rule)::

    export CUDA_VISIBLE_DEVICES=$(autocvd -n 1 -o -q -i 20)
    python bench/dce_pins.py record --tag 3a4bfc7 [--pins S1,S3]
    python bench/dce_pins.py record --tag 3a4bfc7-b       # A-vs-A control
    python bench/dce_pins.py compare 3a4bfc7 3a4bfc7-b    # the envelope
    python bench/dce_pins.py compare 3a4bfc7 <new> --envelope 3a4bfc7-b

``compare`` with ``--envelope`` passes a pin when it is bitwise equal to the
reference, or -- if the A-vs-A control itself was not bitwise -- within twice the
control's max difference with the rel-L2 scalars equal to two significant figures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = "/export/scratch/tbuck/dce_pins"
PINS = ("S1", "S2", "S3", "S4", "S4b", "S5", "S6")
#: the fused scan's variants beyond S1's state carry (S2 provider refactor); not in
#: the default set
PINS_S1_VARIANTS = ("S1p", "S1t", "S1e", "S1x", "S1g")
PINS_TWO_CARD = ("M1", "M2", "M3")
PINS_LARGE = ("L1", "L2", "L3")
PINS_DET = ("DET1", "DET2", "DET3")
#: the production runner's pins (S3); not in the default set
PINS_RUNNER = ("R1p", "R1x", "R1r")
#: pass or fail on their own record, whatever the reference
PINS_SELF = ("G1", "R1x", "R1r")
#: a pin compared with ANOTHER pin's arrays in the reference tag when the reference
#: has none of its own: the runner's R1p computes S1p's numbers
_REFERENCE_PIN = {"R1p": "S1p"}
#: recorded without --xla_gpu_deterministic_ops: they time the lane, or test whether
#: the production flags alone reproduce it
_PRODUCTION_FLAGS = frozenset(PINS_LARGE + PINS_DET)
#: the 25M disc+bulge production softening (docs/softening_2026-10.md)
_DISC_SOFTENING = "1.5e-3"
#: G1's limit on rel-L2(differentiable forward - production forward). On the same
#: lists the two lanes differ only in fp32 summation order. At the bench's operating
#: point with the COM MAC forced: 1.3e-11 on 0c36f36, 3.7e-9 from #363 (round 4),
#: 7.3e-8 from #367 (round 6), 1.7e-7 from #368 (round 7) on, every time with equal
#: far-pair and near-edge counts. A geometry split is 1.2e-3 to 1.5e-3.
_G1_REL_L2_LIMIT = 1e-5
#: S1e's eq (16) accuracy target: where the criterion, not theta 0.8, decides the
#: lists at S1's operating point. Eval-only rows on f967805 (A100, 2026-10-15): far
#: pairs 3,350,146 against the geometric MAC's 1,876,324, rel-L2 1.45e-4 against
#: 7.76e-4 (1e-3: 1,366,602 and 1.33e-3, looser than theta; 1e-5: 8,047,234 and
#: 1.98e-5). The pin's meta records each arm's far-pair count.
_S1E_EPS = "1e-4"
#: S1g's gravitational constant: kpc, km/s, Msun
_S1G_G = "4.3e-6"


# --------------------------------------------------------------------- cases


def _plummer(n: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    r = np.minimum(r, 20.0)
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (n, 3))
    mass = np.full(n, 1.0 / n)
    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32)


def _fused_bench(
    out: Path, n: int, extra: list[str], ic: str = "plummer_clipped"
) -> dict:
    """S1 / S2: the production bench, unchanged, with its bitwise save."""
    import numpy as np

    js = out.with_suffix(".bench.json")
    npz = out.with_suffix(".bench.npz")
    cmd = [
        sys.executable,
        str(HERE / "fused_memory_budget.py"),
        "--n",
        str(n),
        "--ic",
        ic,
        "--caps",
        "unnamed",
        "--steps",
        "4",
        "--reps",
        "2",
        "--eval-repeats",
        "2",
        "--accuracy-targets",
        "4096",
        "--no-analysis",
        "--save-forces",
        str(npz),
        "--out",
        str(js),
        *extra,
    ]
    subprocess.run(cmd, check=True)
    data = np.load(npz)
    np.savez(out, force=data["force"], state=data["state"])
    result = json.loads(js.read_text())
    return {"rel_l2": result["accuracy"]["rel_l2"], "n": n}


def case_s1(out: Path) -> dict:
    return _fused_bench(out, 200_000, [])


def case_s2(out: Path) -> dict:
    return _fused_bench(out, 50_000, ["--use-pallas", "off"])


def case_s6(out: Path) -> dict:
    """S1 with no JACCPOT_* / YGGDRAX_* environment at all: the library defaults.

    Every other fused pin inherits the bench harness's env, which hid the CSR
    payload bug (#384) behind its RADIX_FAST_PAYLOAD_MAX_MB=0.
    """
    return _fused_bench(out, 200_000, ["--clean-env"])


_PARTICLE_CARRY = ["--env", "JACCPOT_STRICT_CARRY=particles"]


def _fused_arms(
    out: Path, n: int, extra: list[str], ic: str = "plummer_clipped"
) -> dict:
    """``_fused_bench`` on the state carry and on the particle carry, one process
    each: ``force`` / ``state`` and ``force_particles`` / ``state_particles``."""
    import numpy as np

    arrays, meta = {}, {}
    for arm, more in (("", []), ("_particles", _PARTICLE_CARRY)):
        part = out.with_name(f"{out.stem}{arm or '_state'}.npz")
        row = _fused_bench(part, n, extra + more, ic=ic)
        counts = json.loads(part.with_suffix(".bench.json").read_text())["counts"]
        with np.load(part) as data:
            arrays.update({f"{k}{arm}": data[k] for k in data.files})
        meta[f"rel_l2{arm}"] = row["rel_l2"]
        meta[f"far_pair_count{arm}"] = counts.get("far_pair_count")
    np.savez(out, **arrays)
    return {"n": n, **meta}


def case_s1p(out: Path) -> dict:
    return _fused_bench(out, 200_000, _PARTICLE_CARRY)


def case_s1t(out: Path) -> dict:
    return _fused_bench(
        out,
        200_000,
        _PARTICLE_CARRY + ["JACCPOT_STRICT_CARRY_ORDER=tree"],
    )


def case_s1e(out: Path) -> dict:
    """Dehnen (2014) eq (16a) per pair, both carries; f_b rides the scan."""
    return _fused_arms(
        out, 200_000, ["--mac-type", "dehnen_error", "--adaptive-eps", _S1E_EPS]
    )


def case_s1x(out: Path) -> dict:
    """The disc+bulge with its NFW halo traced inside the scan, both carries."""
    return _fused_arms(out, 200_000, ["--softening", _DISC_SOFTENING], ic="disc")


def case_s1g(out: Path) -> dict:
    return _fused_bench(out, 200_000, ["--G", _S1G_G])


_RUNNER = ["--runner"]


def case_r1p(out: Path) -> dict:
    """S1p's calls through the ProductionRunner (S3 R1p)."""
    return _fused_bench(out, 200_000, _PARTICLE_CARRY + _RUNNER)


def case_r1x(out: Path) -> dict:
    """S1x's particle arm through the runner: chained calls against one segment."""
    import numpy as np

    arrays, meta = {}, {}
    for arm, more in (("", []), ("_once", ["--runner-once"])):
        part = out.with_name(f"{out.stem}{arm or '_chained'}.npz")
        row = _fused_bench(
            part,
            200_000,
            ["--softening", _DISC_SOFTENING] + _PARTICLE_CARRY + _RUNNER + more,
            ic="disc",
        )
        with np.load(part) as data:
            arrays.update({f"{k}{arm}": data[k] for k in data.files})
        meta[f"rel_l2{arm}"] = row["rel_l2"]
    np.savez(out, **arrays)
    differ = arrays["state"] != arrays["state_once"]
    return {
        "n": 200_000,
        **meta,
        "bitwise": bool(not differ.any()),
        "n_diff": int(np.count_nonzero(differ)),
        "max_abs": float(np.max(np.abs(arrays["state"] - arrays["state_once"]))),
    }


def case_r1r(out: Path) -> dict:
    """R1p with a restart at step 5, continued in a fresh process (S3 R5 on a card)."""
    import numpy as np

    restart = out.with_name(f"{out.stem}_restart.npz")
    straight = out.with_name(f"{out.stem}_straight.npz")
    _fused_bench(
        straight,
        200_000,
        _PARTICLE_CARRY
        + _RUNNER
        + ["--runner-restart", str(restart), "--runner-restart-at", "5"],
    )
    resumed = out.with_name(f"{out.stem}_resumed.npz")
    js = resumed.with_suffix(".bench.json")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "fused_memory_budget.py"),
            "--n",
            "200000",
            "--ic",
            "plummer_clipped",
            "--caps",
            "unnamed",
            "--steps",
            "4",
            "--reps",
            "2",
            "--no-analysis",
            "--skip-eval",
            "--save-forces",
            str(resumed),
            "--out",
            str(js),
            *_PARTICLE_CARRY,
            *_RUNNER,
            "--runner-resume",
            str(restart),
        ],
        check=True,
    )
    with np.load(straight) as a, np.load(resumed) as b:
        state, state_resumed = a["state"], b["state"]
    np.savez(out, state=state, state_resumed=state_resumed)
    restored = json.loads(js.read_text())["runner_restored"]
    differ = state != state_resumed
    return {
        "n": 200_000,
        "bitwise": bool(not differ.any()) and bool(restored["bitwise"]),
        "n_diff": int(np.count_nonzero(differ)),
        "restored": restored,
        "restart_bytes": restart.stat().st_size,
    }


def case_det3(out: Path) -> dict:
    return case_r1p(out)


def case_s3(out: Path) -> dict:
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import FastMultipoleMethod

    arrays, meta = {}, {}
    for n in (30_000, 100_000):
        pos, _, mass = _plummer(n)
        try:
            acc = FastMultipoleMethod().compute_accelerations(
                jnp.asarray(pos), jnp.asarray(mass)
            )
            arrays[f"acc_{n}"] = np.asarray(acc)
            meta[f"n{n}"] = "ok"
        except Exception as exc:  # noqa: BLE001 -- the raise IS the pin
            meta[f"n{n}"] = f"raises {type(exc).__name__}: {str(exc)[:300]}"
    np.savez(out, **arrays)
    return meta


def _grad_pin(out: Path, solver, n: int, plan_fn=None) -> dict:
    """d/dpos and d/dmass of a fixed weighting of the forces, through
    ``differentiable_accelerations`` (Odisseo's differentiable lane)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    pos, _, mass = _plummer(n)
    p = jnp.asarray(pos)
    m = jnp.asarray(mass)
    prepared = solver.prepare_state(p, m, leaf_size=32, max_order=4, theta=0.6)
    plan = None if plan_fn is None else plan_fn(solver, prepared)
    w = jnp.asarray(np.random.default_rng(2).normal(size=(n, 3)), jnp.float32)

    def loss(x, mm):
        acc = solver.differentiable_accelerations(prepared, x, mm, grad_plan=plan)
        return jnp.sum(acc * w)

    value, (g_pos, g_mass) = jax.value_and_grad(loss, argnums=(0, 1))(p, m)
    np.savez(out, g_pos=np.asarray(g_pos), g_mass=np.asarray(g_mass))
    return {"loss": float(value), "n": n}


def case_s4(out: Path) -> dict:
    """Odisseo's differentiable lane below its large-N profile: preset "fast"."""
    import jax.numpy as jnp

    from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig

    solver = FastMultipoleMethod(
        preset="fast",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
            mac_type="dehnen",
        ),
    )
    return _grad_pin(out, solver, 20_000)


def case_s4b(out: Path) -> dict:
    """The large-N differentiable lane, with Odisseo's large-N env overrides."""
    import os

    sys.path.insert(
        0,
        os.environ.get(
            "BENCH_DIR",
            "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu",
        ),
    )
    from codes.compare_force import apply_fast_lane_env

    n = 20_000
    apply_fast_lane_env(n)  # what Odisseo sets for its large_n_gpu profile
    import jax.numpy as jnp

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=32),
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
    )
    return _grad_pin(out, solver, n, plan_fn=prepare_large_n_grad_plan)


def case_g1(out: Path) -> dict:
    """The differentiable forward against the production forward, the bench's way.

    ``bench/grad_step_profile.py`` evaluates the production force on one solver and
    builds the differentiable state on a NEW one. Which MAC geometry a walk tests
    (box or centre of mass) followed the engine's ``_strict_fused_mode_active``,
    which only a strict call sets (bc62978), so the two lanes walked different
    lists. The second differentiable arm runs on the production solver after its
    strict call: equal there and not on the fresh solver means the geometry.
    """
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(
        0,
        os.environ.get(
            "BENCH_DIR",
            "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu",
        ),
    )
    from codes.compare_force import (
        FAST_LANE_ENV_BY_LEAF,
        apply_fast_lane_env,
        fast_lane_overrides_for_leaf,
    )

    n, leaf, order, theta, softening = 200_000, 64, 5, 0.8, 1e-7
    apply_fast_lane_env(n, overrides=fast_lane_overrides_for_leaf(leaf, n))
    trav = dict(FAST_LANE_ENV_BY_LEAF[leaf]["_traversal_overrides"])
    from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
    from yggdrax.bounds import infer_bounds
    from yggdrax.morton import morton_encode

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TraversalOverrides,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

    pos, _, mass = _plummer(n)
    p, m = jnp.asarray(pos), jnp.asarray(mass)
    codes = np.sort(np.asarray(morton_encode(p, infer_bounds(p))).astype(np.uint64))
    leaves = adaptive_cell_leaf_partition_numpy(codes, leaf_size=leaf)[0].size

    def solver():
        return FastMultipoleMethod(
            preset="large_n_gpu",
            basis="real",
            theta=theta,
            G=1.0,
            softening=softening,
            working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(
                    mode="static_radix",
                    leaf_target=leaf,
                    leaf_partition="cells",
                    leaf_capacity=1 << int(np.ceil(np.log2(1.25 * leaves))),
                ),
                farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
                nearfield=NearFieldConfig(mode="auto"),
                runtime=RuntimePolicyConfig(
                    traversal_config=TraversalOverrides(**trav)
                ),
                mac_type="dehnen",
            ),
            fixed_order=order,
        )

    def differentiable(fmm):
        state = fmm.prepare_state(p, m, leaf_size=leaf, max_order=order, theta=theta)
        plan = prepare_large_n_grad_plan(fmm, state)
        acc = fmm.differentiable_accelerations(state, p, m, grad_plan=plan)
        lists = {
            "far_pairs": int(plan.far_pair_active_count),
            "near_edges": int(np.asarray(state.neighbor_list.counts).sum()),
        }
        return np.asarray(jax.block_until_ready(acc)), lists

    production = solver()
    prepared, eval_fn = production.strict_fused_prepared_eval_fn(
        positions=p, masses=m, leaf_size=leaf, max_order=order, theta=theta
    )
    acc_prod = np.asarray(jax.block_until_ready(eval_fn(prepared)))
    lists_prod = {"far_pairs": int(prepared.compact_far_pairs.far_pair_count)}
    prepared = eval_fn = None
    acc_fresh, lists_fresh = differentiable(solver())
    acc_same, lists_same = differentiable(production)

    # fp64 direct sum on 1024 targets (informative; the pin is the difference)
    targets = np.random.default_rng(3).choice(n, 1024, replace=False)
    p64, m64 = pos.astype(np.float64), mass.astype(np.float64)
    ref = np.empty((targets.size, 3))
    for s in range(0, targets.size, 32):
        t = targets[s : s + 32]
        dx = p64[None, :, :] - p64[t, None, :]
        r2 = np.sum(dx * dx, -1)
        g = np.where(r2 > 0.0, (r2 + softening * softening) ** -1.5, 0.0)
        ref[s : s + 32] = np.sum(m64[None, :, None] * dx * g[..., None], 1)

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    np.savez(
        out,
        acc_production=acc_prod,
        acc_differentiable_fresh=acc_fresh,
        acc_differentiable_same_engine=acc_same,
    )
    return {
        "n": n,
        "limit": _G1_REL_L2_LIMIT,
        "rel_l2_fresh": rel(acc_fresh, acc_prod),
        "rel_l2_same_engine": rel(acc_same, acc_prod),
        "lists": {
            "production": lists_prod,
            "fresh": lists_fresh,
            "same_engine": lists_same,
        },
        "direct_rel_l2": {
            "production": rel(acc_prod[targets], ref),
            "fresh": rel(acc_fresh[targets], ref),
            "same_engine": rel(acc_same[targets], ref),
        },
    }


def _bitwise_pass(meta: dict) -> bool:
    """R1x / R1r: the record's own two arms agreed bitwise."""
    return bool(meta.get("bitwise"))


def _g1_pass(meta: dict) -> bool:
    """G1 passes when both differentiable arms equal production to round-off."""
    return all(
        meta.get(key, float("inf")) <= _G1_REL_L2_LIMIT
        for key in ("rel_l2_fresh", "rel_l2_same_engine")
    )


def case_s5(out: Path) -> dict:
    """``BlockStepFMM`` with Odisseo's ``BlockStepOptions`` defaults, on device."""
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import BlockStepFMM

    n = 20_000
    pos, vel, mass = _plummer(n)
    force = BlockStepFMM(
        softening=1e-3,
        k_max=2,
        theta=0.6,
        max_order=4,
        G=1.0,
        basis="real",
        backend="jax",
        leaf_size=64,
        static_shapes=True,
        topology_backend="device",
    )
    x, v, m = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(mass)
    rung = jnp.asarray(np.arange(n) % 3, jnp.int32)
    force.prepare(x, m)
    for _ in range(20):
        x, v, _ = force.advance_base_step(x, v, m, rung=rung, dt_max=1e-3)
    p0 = np.sum(mass[:, None] * vel, axis=0)
    p1 = np.sum(mass[:, None] * np.asarray(v, np.float64), axis=0)
    np.savez(out, positions=np.asarray(x), velocities=np.asarray(v))
    return {"momentum_drift": float(np.linalg.norm(p1 - p0)), "n": n}


def _need_two_devices() -> None:
    import jax

    if jax.local_device_count() < 2:
        raise RuntimeError("the M pins need two cards: book them with autocvd -n 2")


def case_m1(out: Path) -> dict:
    """``distributed/fmm.py`` as Odisseo's mesh lane builds it."""
    _need_two_devices()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from yggdrax.distributed import make_mesh

    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
        scatter_to_input_order,
    )

    ndev, leaf = 2, 512
    n = ndev * leaf * 128  # the mesh lane wants N = ndev * k * leaf: no padding rows
    pos, _, mass = _plummer(n)
    part = partition_for_devices(pos, mass, ndev, leaf_size=leaf, partitioner="rcb")
    cap = int(part["cap"])
    # odisseo.mesh_coupling.integrate_mesh with MeshOptions() defaults
    cfg = DistributedFMMConfig(
        leaf_size=leaf,
        theta=0.7,
        order=6,
        softening=1e-3,
        G=1.0,
        m2l_chunk=65_536,
        nearfield_chunk=512,
        nearfield_accum="wide",
        mac_type="dehnen",
        adaptive_eps=None,
        mac_cross_criterion=True,
    ).resolved_for(cap, ndev)
    mesh = make_mesh(ndev)
    evaluate = make_force_evaluator(
        cfg, ndev, cap, mesh, jit=True, halo_exchange="auto"
    )
    x = jax.device_put(
        jnp.asarray(part["pos_flat"]), NamedSharding(mesh, P("gpus", None))
    )
    m = jax.device_put(jnp.asarray(part["mass_flat"]), NamedSharding(mesh, P("gpus")))
    a_raw, gid, diag = evaluate(
        x, m, jnp.asarray(part["gid_flat"]), jnp.asarray(part["counts"])
    )
    acc = scatter_to_input_order(np.asarray(a_raw), np.asarray(gid), n)
    d = np.asarray(diag)
    overflow = {
        name: float(d[:, i].sum())
        for i, name in enumerate(DIAG_FIELDS)
        if name.endswith("overflow") and i < d.shape[1]
    }
    # fp64 direct sum on 1024 targets with the lane's own softening kernel
    # (informative; the pin is the array)
    try:
        from jaccpot.softening import (
            pair_factors,
            resolve_softening_kernel,
            softening_params_np,
        )

        kernel = resolve_softening_kernel(None)
    except ImportError:  # a tree from before the softening kernels: Plummer only
        kernel = "plummer"
    targets = np.random.default_rng(3).choice(n, 1024, replace=False)
    p64, m64 = pos.astype(np.float64), mass.astype(np.float64)
    ref = np.empty((targets.size, 3))
    for s in range(0, targets.size, 32):
        t = targets[s : s + 32]
        dx = p64[None, :, :] - p64[t, None, :]
        r2 = np.sum(dx * dx, -1)
        self_pair = r2 == 0.0
        r2 = np.where(self_pair, 1.0, r2)
        if kernel == "plummer":
            g = (r2 + 1e-6) ** -1.5
        else:
            g = pair_factors(r2, softening_params_np(kernel, 1e-3), kernel, xp=np)[0]
        g = np.where(self_pair, 0.0, g)
        ref[s : s + 32] = np.sum(m64[None, :, None] * dx * g[..., None], 1)
    rel = float(np.linalg.norm(acc[targets] - ref) / np.linalg.norm(ref))
    np.savez(out, acc=acc)
    return {
        "n": n,
        "cap": cap,
        "kernel": kernel,
        "rel_l2": rel,
        "overflow": overflow,
    }


def case_m2(out: Path) -> dict:
    """The fused two-card rollout gate, short: the forward, the cross field and the
    on-device repartition all feed the positions it dumps."""
    _need_two_devices()
    import numpy as np

    n, steps, every = 100_000, 17, 8
    gate_dir = out.parent / f"{out.stem}_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        GATE_MODE="mesh",
        GATE_N=str(n),
        GATE_STEPS=str(steps),
        GATE_EVERY=str(every),
        GATE_OUT=str(gate_dir),
    )
    # The gate sets XLA's NCCL ragged exchange itself (bench/_ragged_exchange_flags.py).
    subprocess.run(
        [sys.executable, str(HERE / "multigpu_rollout_gate.py")], env=env, check=True
    )
    arrays = {
        f"positions_step{k}": np.load(gate_dir / f"positions_step{k}.npy")
        for k in (1, 16, 17)
    }
    np.savez(out, **arrays)
    result = json.loads((gate_dir / f"gate_mesh_every{every}.json").read_text())
    return {
        "n": n,
        "probes": result["probes"],
        "repartitions": result["repartitions"],
        "moved_total": result["moved_total"],
    }


def case_m3(out: Path) -> dict:
    """``DistributedBlockStepFMM``: the distributed mutual lane, block steps."""
    _need_two_devices()
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import DistributedBlockStepFMM
    from jaccpot.mutual.force import MutualCapacities

    n = 8192
    pos, vel, mass = _plummer(n)
    # explicit: the heuristic's depth (16) is too shallow for the clipped Plummer core
    caps = MutualCapacities(near=16384, far=16384, depth=32, width=1024, queue=1 << 17)
    # Plummer, explicitly: cross_theta > 0 with a compact kernel raises (the cross
    # walk has no separation floor yet), and this pin covers the cross M2L path
    import inspect

    kw = {}
    if "softening_kernel" in inspect.signature(DistributedBlockStepFMM).parameters:
        kw["softening_kernel"] = "plummer"
    force = DistributedBlockStepFMM(
        softening=1e-3,
        k_max=2,
        theta=0.5,
        cross_theta=0.5,
        max_order=4,
        G=1.0,
        leaf_size=32,
        backend="jax",
        ndev=2,
        caps=caps,
        **kw,
    )
    x, v, m = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(mass)
    rung = jnp.asarray(np.arange(n) % 3, jnp.int32)
    force.prepare(x, m)
    for _ in range(4):
        x, v, _ = force.advance_base_step(x, v, m, rung=rung, dt_max=1e-3)
    p0 = np.sum(mass[:, None].astype(np.float64) * vel, axis=0)
    p1 = np.sum(mass[:, None].astype(np.float64) * np.asarray(v, np.float64), axis=0)
    np.savez(out, positions=np.asarray(x), velocities=np.asarray(v))
    return {"momentum_drift": float(np.linalg.norm(p1 - p0)), "n": n}


def _timed_row(js: Path, n: int, args: list[str]) -> dict:
    """One ``fused_memory_budget.py`` process; its scalars, as ``_compare_timed`` reads them."""
    cmd = [
        sys.executable,
        str(HERE / "fused_memory_budget.py"),
        "--n",
        str(n),
        "--caps",
        "unnamed",
        "--prealloc",
        "0.88",
        "--no-analysis",
        "--out",
        str(js),
        *args,
        "--env",
        "JACCPOT_STRICT_CARRY=particles",
    ]
    subprocess.run(cmd, check=True)
    d = json.loads(js.read_text())
    peak = (
        d.get("final_peak_gib")
        or d.get("peak_after_scan_gib")
        or d.get("peak_after_eval_gib")
        or 0.0
    )
    cont = d.get("scan_contention") or d.get("eval_contention") or {}
    row = {
        "peak_gib": float(peak),
        "bytes_per_particle": float(peak) * 2**30 / n,
        "far_pair_count": d.get("counts", {}).get("far_pair_count"),
        "contaminated": bool(cont.get("contaminated", True)),
        "foreign_pids": cont.get("foreign_pids_on_devices"),
        "loadavg_max": cont.get("loadavg1_max"),
    }
    if d.get("step_ms"):
        row["ms_min"] = float(d["step_ms"]["min"])
        row["ms_median"] = float(d["step_ms"]["median"])
    if d.get("eval_ms"):
        row["ms_min"] = float(d["eval_ms"]["min"])
        row["ms_median"] = float(d["eval_ms"]["median"])
    if d.get("accuracy"):
        row["rel_l2"] = float(d["accuracy"]["rel_l2"])
    return row


_STEP_ROW = ["--skip-eval", "--steps", "4", "--reps", "3"]
_FORCE_ROW = ["--no-scan", "--accuracy-targets", "4096", "--eval-repeats", "5"]


def _timed_pin(out: Path, n: int, ic: list[str], force_row: bool) -> dict:
    meta = {
        "timing": True,
        "n": n,
        "step": _timed_row(out.with_suffix(".step.json"), n, ic + _STEP_ROW),
    }
    if force_row:
        meta["force"] = _timed_row(out.with_suffix(".force.json"), n, ic + _FORCE_ROW)
    return meta


def case_l1(out: Path) -> dict:
    return _timed_pin(out, 8_000_000, ["--ic", "plummer_clipped"], force_row=False)


def case_l2(out: Path) -> dict:
    return _timed_pin(out, 100_000_000, ["--ic", "plummer_clipped"], force_row=False)


def case_l3(out: Path) -> dict:
    disc = [
        "--ic",
        "disc",
        "--softening",
        _DISC_SOFTENING,
        "--softening-kernel",
        "ferrers3",
    ]
    return _timed_pin(out, 25_000_000, disc, force_row=True)


def case_det1(out: Path) -> dict:
    return case_s1(out)


def case_det2(out: Path) -> dict:
    return case_s4b(out)


CASES = {
    "DET1": case_det1,
    "DET2": case_det2,
    "DET3": case_det3,
    "R1p": case_r1p,
    "R1r": case_r1r,
    "R1x": case_r1x,
    "G1": case_g1,
    "L1": case_l1,
    "L2": case_l2,
    "L3": case_l3,
    "M1": case_m1,
    "M2": case_m2,
    "M3": case_m3,
    "S1": case_s1,
    "S1e": case_s1e,
    "S1g": case_s1g,
    "S1p": case_s1p,
    "S1t": case_s1t,
    "S1x": case_s1x,
    "S2": case_s2,
    "S3": case_s3,
    "S4": case_s4,
    "S4b": case_s4b,
    "S5": case_s5,
    "S6": case_s6,
}


# --------------------------------------------------------------------- driver


def _record(args) -> int:
    out_dir = Path(args.root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    deterministic = dict(os.environ)
    deterministic["XLA_FLAGS"] = (
        deterministic.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()
    failed = []
    for pin in args.pins.split(","):
        print(f"[{args.tag}] {pin} ...", flush=True)
        rc = subprocess.run(
            [sys.executable, __file__, "case", pin, "--out", str(out_dir / pin)],
            env=dict(os.environ) if pin in _PRODUCTION_FLAGS else deterministic,
            check=False,
        ).returncode
        if rc != 0:
            failed.append(pin)
    print(f"[{args.tag}] done; failed: {failed or 'none'}")
    return 1 if failed else 0


def _case(args) -> int:
    import jax

    # jaccpot needs 64-bit types (README); neither it nor yggdrax turns them on. Each
    # case imports jaccpot before it makes an array, which is where yggdrax used to.
    jax.config.update("jax_enable_x64", True)
    out = Path(args.out).with_suffix(".npz")
    meta = CASES[args.pin](out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return 0


def _diff(a: Path, b: Path) -> dict:
    import numpy as np

    da, db = np.load(a), np.load(b)
    rows = {}
    for key in sorted(set(da.files) | set(db.files)):
        if key not in da.files or key not in db.files:
            rows[key] = {"missing": True}
            continue
        x, y = da[key], db[key]
        if x.shape != y.shape:
            rows[key] = {"shape": [list(x.shape), list(y.shape)]}
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        ref = np.linalg.norm(x.astype(np.float64))
        rows[key] = {
            "bitwise": bool(np.array_equal(x, y)),
            "n_diff": int(np.count_nonzero(x != y)),
            "size": int(x.size),
            "max_abs": float(d.max()) if d.size else 0.0,
            "rel_l2": float(np.linalg.norm(d) / ref) if ref else 0.0,
        }
    return rows


def _meta_matches(ref, new) -> bool:
    """Floats agree to 2 significant digits, at any depth; everything else exactly.

    Nested so that M2's per-step probe errors (a dict of dicts) get the same tolerance
    as a top-level float: M2's A-vs-A control is not bitwise (ulp-level, see the
    cleanup record), and an exact match on its probes would fail every comparison.
    """
    if isinstance(ref, float) and isinstance(new, float):
        return f"{ref:.2g}" == f"{new:.2g}"
    if isinstance(ref, dict) and isinstance(new, dict):
        return all(_meta_matches(val, new.get(key)) for key, val in ref.items())
    return ref == new


#: the timing pins' floors: a step-time change below 3 % or a peak change below 2 %
#: is never a failure, whatever the A-vs-A control's spread
_TIME_FLOOR, _PEAK_FLOOR = 0.03, 0.02


def _rel(a: float, b: float) -> float:
    return abs(b - a) / abs(a) if a else float("inf")


def _compare_timed(pin: str, ref: dict, new: dict, control: dict | None) -> dict:
    """An L pin passes on clean rows within the time and peak envelopes, rel-L2 at 2 digits."""
    entry: dict = {"meta": {"ref": ref, "new": new}, "rows": {}}
    passed = True
    for name in ("step", "force"):
        r, n = ref.get(name), new.get(name)
        if r is None:
            continue
        if n is None:
            entry["rows"][name] = {"missing": True}
            passed = False
            continue
        c = (control or {}).get(name) or {}
        t_lim = (
            max(_TIME_FLOOR, 2.0 * _rel(r["ms_min"], c["ms_min"])) if c else _TIME_FLOOR
        )
        p_lim = (
            max(_PEAK_FLOOR, 2.0 * _rel(r["peak_gib"], c["peak_gib"]))
            if c
            else _PEAK_FLOOR
        )
        row = {
            "ms_ref": r["ms_min"],
            "ms_new": n["ms_min"],
            "time_change": (n["ms_min"] - r["ms_min"]) / r["ms_min"],
            "time_limit": t_lim,
            "peak_change": (n["peak_gib"] - r["peak_gib"]) / r["peak_gib"],
            "peak_limit": p_lim,
            "lists_equal": r.get("far_pair_count") == n.get("far_pair_count"),
            "contaminated": bool(r.get("contaminated") or n.get("contaminated")),
        }
        ok_row = (
            not row["contaminated"]
            and abs(row["time_change"]) <= t_lim
            and abs(row["peak_change"]) <= p_lim
        )
        if "rel_l2" in r:
            row["rel_l2"] = [r["rel_l2"], n.get("rel_l2")]
            ok_row = (
                ok_row
                and f"{r['rel_l2']:.2g}" == f"{n.get('rel_l2', float('nan')):.2g}"
            )
        row["pass"] = ok_row
        passed = passed and ok_row
        entry["rows"][name] = row
    entry["pass"] = passed
    return entry


def _compare(args) -> int:
    root = Path(args.root)
    ref, new = root / args.ref, root / args.new
    env = root / args.envelope if args.envelope else None
    report, ok = {}, True
    wanted = args.pins.split(",")
    for pin in [p for p in wanted if p in PINS_LARGE]:
        if not (ref / f"{pin}.json").exists() or not (new / f"{pin}.json").exists():
            report[pin] = {"missing": True}
            ok = False
            continue
        control = (
            json.loads((env / f"{pin}.json").read_text())
            if env is not None and (env / f"{pin}.json").exists()
            else None
        )
        report[pin] = _compare_timed(
            pin,
            json.loads((ref / f"{pin}.json").read_text()),
            json.loads((new / f"{pin}.json").read_text()),
            control,
        )
        ok = ok and report[pin]["pass"]
    wanted = [p for p in wanted if p not in PINS_LARGE]
    # a pin's reference arrays: its own, else the pin it reproduces (R1p: S1p)
    refs: dict = {}
    for pin in wanted:
        own = ref / f"{pin}.npz"
        other = ref / f"{_REFERENCE_PIN.get(pin, pin)}.npz"
        if own.exists() or other.exists():
            refs[pin] = own if own.exists() else other
        elif pin in PINS_SELF and (new / f"{pin}.npz").exists():
            # it judges its own record; a reference older than the pin has none
            refs[pin] = new / f"{pin}.npz"
        else:
            report[pin] = {"missing_in_ref": True}
            ok = False
    for pin, npz in sorted(refs.items()):
        if not (new / f"{pin}.npz").exists():
            report[pin] = {"missing": True}
            ok = False
            continue
        rows = _diff(npz, new / f"{pin}.npz")
        meta_ref = json.loads(npz.with_suffix(".json").read_text())
        meta_new = json.loads((new / f"{pin}.npz").with_suffix(".json").read_text())
        entry = {"arrays": rows, "meta": {"ref": meta_ref, "new": meta_new}}
        if pin in PINS_DET:
            # no envelope: the question is whether the production flags reproduce
            entry["pass"] = all(row.get("bitwise") for row in rows.values())
            ok = ok and entry["pass"]
        elif pin in PINS_SELF:
            # the question is inside the new record: do the two lanes agree?
            entry["pass"] = (_g1_pass if pin == "G1" else _bitwise_pass)(meta_new)
            ok = ok and entry["pass"]
        elif env is not None:
            control = _diff(npz, env / npz.name)
            passed = True
            for key, row in rows.items():
                c = control.get(key, {})
                if row.get("bitwise"):
                    continue
                if c.get("bitwise", True) or "max_abs" not in row:
                    passed = False
                elif row["max_abs"] > 2.0 * c["max_abs"]:
                    passed = False
            if not _meta_matches(meta_ref, meta_new):
                passed = False
            entry["pass"] = passed
            ok = ok and passed
        report[pin] = entry
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--tag", required=True)
    rec.add_argument("--pins", default=",".join(PINS))
    rec.add_argument("--root", default=ROOT_DEFAULT)
    case = sub.add_parser("case")
    case.add_argument("pin", choices=sorted(CASES))
    case.add_argument("--out", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("ref")
    cmp_.add_argument("new")
    cmp_.add_argument("--envelope", default=None)
    cmp_.add_argument(
        "--pins",
        default=",".join(PINS),
        help=(
            "M1,M2,M3 for the two-card set; L1,L2,L3 timing; DET1,DET2,DET3 "
            "determinism; G1 differentiable == production; R1p,R1x,R1r the "
            "production runner"
        ),
    )
    cmp_.add_argument("--root", default=ROOT_DEFAULT)
    cmp_.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return {"record": _record, "case": _case, "compare": _compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
