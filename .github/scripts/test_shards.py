"""Which pytest selection each CI job runs -- the one place that says so.

The workflow asks this file for a job's arguments
(``python .github/scripts/test_shards.py args <shard>``) instead of spelling them
out, and the ``test-partition`` job runs ``check``: every shard is collected
separately, and the shards must be pairwise disjoint and together cover exactly
the default collection (``tests/perf`` excluded by ``addopts``). A test directory or file that appears, moves or disappears then
either lands in a shard or turns that job red -- before, the partition was
checked by hand-counted comments in ``ci.yml``, which drifted.

``EXTRA`` holds selections that are not part of the partition: one that
deliberately re-runs a subset (the Python-floor job, the ``smoke`` marker), and the
tests of the ``bench/`` scripts, which live outside ``tests/`` and run in the lint
job.

``GATED_SHARDS`` do not run on every pull request (``GATED_PATHS`` says when they
do). They stay in ``SHARDS``, so the partition check accounts for their tests like
any other's: skipping a job on a PR is a scheduling decision, not a gap.

``check`` also fails on a node id in ``tests/slow_tests.txt`` or
``tests/smoke_tests.txt`` that the default collection no longer contains: a moved
or renamed test silently drops out of either marker otherwise.

Usage::

    python .github/scripts/test_shards.py args unit      # one argument per line
    python .github/scripts/test_shards.py check          # exit 1 on any overlap or gap
    git diff --name-only BASE HEAD | python .github/scripts/test_shards.py touches
                                                         # "true" if a gated job must run
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

# The Dehnen-MAC / force-scale criterion family of tests/unit/runtime. Every case
# builds a solver and runs a full FMM solve (26-95 s each on CPU), which is why
# the directory used to have a job to itself; it is now three shards of test-full,
# this family, the gradient family below and the rest of the directory.
_RUNTIME_MAC_FILES = (
    "tests/unit/runtime/test_criterion_reaches_the_split_build.py",
    "tests/unit/runtime/test_dehnen_mac_gradients.py",
    "tests/unit/runtime/test_dehnen_mac_reference.py",
    "tests/unit/runtime/test_fb_force_scale_estimator.py",
    "tests/unit/runtime/test_force_scale_injection.py",
    "tests/unit/runtime/test_force_scale_prepass_cost.py",
    "tests/unit/runtime/test_interaction_cache_policy_key.py",
    "tests/unit/runtime/test_large_n_lane_carries_criterion.py",
    "tests/unit/runtime/test_mac_type_resolution.py",
    "tests/unit/runtime/test_refuted_dehnen_theta_mode.py",
    "tests/unit/runtime/test_split_build_carries_pair_policy.py",
    "tests/unit/runtime/test_split_build_default_predicate.py",
)

# The gradients on the production layout (plan S8): jax.grad through the large-N
# lane's reverse kernels, in Pallas interpret mode on CPU, against the direct sum or
# central differences. Every heavy case compiles a reverse program of the whole lane
# (minutes each on CPU); with them `unit-runtime` was cancelled at its cap or
# OOM-killed (2026-10-22, PR #443), so they are a shard of their own.
_RUNTIME_GRAD_FILES = (
    "tests/unit/runtime/test_forward_seams.py",
    "tests/unit/runtime/test_l2p_particle_major_reverse.py",
    "tests/unit/runtime/test_large_n_far_chain_grad.py",
    "tests/unit/runtime/test_large_n_grad_reverse_path.py",
    "tests/unit/runtime/test_mass_moments_vjp.py",
)

# The forward-mode cascade tangent ops (M2M/L2L, K up to 8 at p4, interpret mode):
# one process peaks at 11.5 GB even with each test's programs dropped (16.2 GB
# kept), so beside the `unit` shard's second worker it shut the 16 GB runner down
# (#500/#502, 2026-11-04). Its own shard, one worker, cold compile.
_UNIT_FORWARD_CASCADE = "tests/unit/operators/test_cascade_real_level_jvp.py"

# Single files that need their own job: forced host devices change the device
# topology of the whole process, and nornax has to be installed for the
# cross-repo files to collect at all.
_MUTUAL_DISTRIBUTED = "tests/integration/test_mutual_distributed.py"
_NORNAX_ADAPTER = "tests/integration/test_mutual_fmm_nornax.py"
_NORNAX_DISTRIBUTED = "tests/integration/test_mutual_distributed_nornax.py"
_SEGMENT_CRITERION = "tests/distributed/test_mesh_segment_dehnen_error.py"
_SEGMENT_MIGRATION = "tests/distributed/test_mesh_segment_migration_fused.py"
# the mesh potential and the production runner's multi-GPU mode on the fused force
# (S11 session 2, tracks E and D): ~11 min of compilation between them on CI
_SEGMENT_RUNNER_FILES = (
    "tests/distributed/test_mesh_potential.py",
    "tests/distributed/test_production_runner_mesh_fused.py",
)
_MUTUAL_STATIC_DEVICE = "tests/integration/test_mutual_fmm_static_device.py"
# The fused strict lane's CPU tests (every case a full fused-lane solve, 60-245 s on
# CPU) left the integration shard in 2026-10 (S3): with them it ran 42 of its 50
# minutes before the production runner's tests arrived, and was cancelled at the
# cap after. The provider-runner and production-runner files are the heaviest half.
_STRICT = "tests/integration/strict"
_STRICT_RUNNER_FILES: tuple[str, ...] = (
    "tests/integration/strict/test_strict_runners_on_provider.py",
    "tests/integration/strict/test_production_runner.py",
)
# The differentiable fused step and the rollout over it (S9): every Pallas kernel in
# interpret mode, and reverse programs of the whole lane (2-5 min of compilation
# each on CPU). With them `integration-strict` was cancelled at its cap on #490
# (2026-10-31: the three files alone ran 55 min at `-n 2`, 23 GB in one worker).
# The step's file and the forward-only rollout over the provider's step in one
# shard. The differentiable rollout's file is ~45 min of compilation and its
# gradients peak at 12 GB RSS in one process, so two of its workers do not fit the
# 16 GB runner (20.5 GB summed at `-n 2`): it runs at `-n 1` in two shards, the
# tests that read Plummer's "step" gradient (G2 Plummer, G4 -- every checkpoint
# mode, so a new mode lands here -- and G5) and the rest (G1, G2 under
# `dehnen_error`, G8).
_STRICT_GRAD_FILES: tuple[str, ...] = (
    "tests/integration/strict/test_fused_differentiable_step.py",
    "tests/integration/strict/test_grad_rollout_on_provider.py",
)
_STRICT_GRAD_ROLLOUT = "tests/integration/strict/test_grad_rollout_differentiable.py"
_STRICT_GRAD_ROLLOUT_MODES: tuple[str, ...] = tuple(
    f"{_STRICT_GRAD_ROLLOUT}::{test}"
    for test in (
        "test_the_rollout_gradient_matches_finite_differences[plummer]",
        "test_the_checkpoint_modes_agree_and_each_repeats_bitwise",
        "test_the_far_field_is_load_bearing_in_the_rollout_gradient",
    )
)
# The production runner's gradient mode (S9 session 2, track E): the runner's
# forward pre-pass, then one gradient of the chain of its segments (three segments,
# one re-planned). Its G1 (the chain's jitted forward against the runner, three
# lanes, 8-12.5 GB each in one process) runs here; the gradient's tests are
# collected and SKIPPED unless JACCPOT_RUN_MANUAL_TESTS=1: ONE gradient compile of
# the chain peaks at 18.7 GB (Plummer) / 20.9 GB (`dehnen_error`) RSS on CPU, more
# than the 16 GB runner holds (2026-11-01; the user's call: run them by hand).
_STRICT_GRAD_RUNNER = "tests/integration/strict/test_runner_gradient_mode.py"
# Its opt-ins (S10 follow-ups): the start force inside the adjoint
# (`start_force="adjoint"`) and the plan's arrays as program arguments. Their
# forwards against the runner run here (one pre-pass and one forward compile per
# lane, 8-12 GB each); their gradient comparisons are MANUAL like the file above's.
# One shard per file: both files keep their plans in module-level dicts that the
# per-module `jax.clear_caches()` does not free, so in one worker the arrays
# forward (12.2 GB alone) landed on the start-force file's 11.9 GB and the 16 GB
# runner was killed (#498, 2026-11-03). Alone, measured locally at `-n 0`:
# start-force 7.7 / 11.9 / 3.2 GB (plummer / dehnen_error_nofield / restart),
# arrays 12.2 / 4.6 GB (forward / restart).
_STRICT_GRAD_RUNNER_START_FORCE = (
    "tests/integration/strict/test_runner_gradient_start_force.py"
)
_STRICT_GRAD_RUNNER_ARRAYS = "tests/integration/strict/test_runner_gradient_arrays.py"
# Forward mode (ad_mode="forward") through the production lane: the F gates (jvp
# primal bitwise, the adjoint identity against the reverse path, finite
# differences, jacfwd == jacrev, NaN on overflow, linearize == jvp, the gradient
# plan's forward twins). Forward and reverse programs of the whole 3-step rollout
# in interpret mode, each test's programs dropped after it (kept, the file peaked
# at 18-20 GB RSS). In two shards of one worker, measured with the Pallas tangent
# kernels on a loaded 64-core host (2026-11-04): the adjoint identity and finite
# differences 50 min / 13.1 GB, the rest 57 min / 13.4 GB.
_STRICT_GRAD_FORWARD = "tests/integration/strict/test_forward_mode.py"
_STRICT_GRAD_FORWARD_ADJOINT: tuple[str, ...] = tuple(
    f"{_STRICT_GRAD_FORWARD}::{test}[{lane}]"
    for test in (
        "test_the_tangent_is_the_adjoints_transpose",
        "test_the_tangent_matches_finite_differences",
    )
    for lane in ("plummer", "dehnen_error")
)
# G-batch (forward mode round 2): jacfwd in chunks of directions_per_pass against the
# single pass, 8 and 11 columns -- three jacfwd programs per case (the single pass
# at K = 8 / 8 + 4, chunks of 1 and 3). Its own shard, one worker.
_STRICT_GRAD_FORWARD_BATCH: tuple[str, ...] = tuple(
    f"{_STRICT_GRAD_FORWARD}::test_jacfwd_in_chunks_is_the_single_pass[{num}]"
    for num in (8, 11)
)
_STRICT_GRAD_RUNNER_OPT: tuple[str, ...] = (
    _STRICT_GRAD_RUNNER_START_FORCE,
    _STRICT_GRAD_RUNNER_ARRAYS,
)

# Unit tests that need two or more devices and skip below them: in the unit shard
# (one device) they collected and skipped, so the repartition gate they pin (the
# rollout's bitwise ownership invariance) ran nowhere. They run with the
# forced-device tier instead; their four-device cases still skip there (two).
_UNIT_MULTI_DEVICE: tuple[str, ...] = ("tests/unit/distributed/test_fused_rollout.py",)

SHARDS: dict[str, list[str]] = {
    # test-full matrix
    "integration": [
        "tests/integration",
        f"--ignore={_STRICT}",
        f"--ignore={_MUTUAL_STATIC_DEVICE}",
        f"--ignore={_MUTUAL_DISTRIBUTED}",
        f"--ignore={_NORNAX_ADAPTER}",
        f"--ignore={_NORNAX_DISTRIBUTED}",
    ],
    "integration-strict": [
        _STRICT,
        *(
            f"--ignore={path}"
            for path in (
                *_STRICT_RUNNER_FILES,
                *_STRICT_GRAD_FILES,
                _STRICT_GRAD_ROLLOUT,
                _STRICT_GRAD_RUNNER,
                *_STRICT_GRAD_RUNNER_OPT,
                _STRICT_GRAD_FORWARD,
            )
        ),
    ],
    "integration-strict-runner": list(_STRICT_RUNNER_FILES),
    "integration-strict-grad": list(_STRICT_GRAD_FILES),
    "integration-strict-grad-rollout": [
        _STRICT_GRAD_ROLLOUT,
        *(f"--deselect={node}" for node in _STRICT_GRAD_ROLLOUT_MODES),
    ],
    "integration-strict-grad-modes": list(_STRICT_GRAD_ROLLOUT_MODES),
    "integration-strict-grad-runner": [_STRICT_GRAD_RUNNER],
    "integration-strict-grad-start-force": [_STRICT_GRAD_RUNNER_START_FORCE],
    "integration-strict-grad-arrays": [_STRICT_GRAD_RUNNER_ARRAYS],
    "integration-strict-grad-forward": [
        _STRICT_GRAD_FORWARD,
        *(
            f"--deselect={node}"
            for node in (*_STRICT_GRAD_FORWARD_ADJOINT, *_STRICT_GRAD_FORWARD_BATCH)
        ),
    ],
    "integration-strict-grad-forward-adjoint": list(_STRICT_GRAD_FORWARD_ADJOINT),
    "integration-strict-grad-forward-batch": list(_STRICT_GRAD_FORWARD_BATCH),
    "mutual-static-device": [_MUTUAL_STATIC_DEVICE],
    "unit": [
        "tests/unit",
        "--ignore=tests/unit/runtime",
        f"--ignore={_UNIT_FORWARD_CASCADE}",
        *(f"--ignore={path}" for path in _UNIT_MULTI_DEVICE),
    ],
    "unit-forward-cascade": [_UNIT_FORWARD_CASCADE],
    "unit-runtime-mac": list(_RUNTIME_MAC_FILES),
    "unit-runtime-grad": list(_RUNTIME_GRAD_FILES),
    "unit-runtime": [
        "tests/unit/runtime",
        *(f"--ignore={path}" for path in (*_RUNTIME_MAC_FILES, *_RUNTIME_GRAD_FILES)),
    ],
    "characterization": ["tests/characterization"],
    # single-purpose jobs
    "distributed-mutual": [_MUTUAL_DISTRIBUTED],
    "nornax-adapter": [_NORNAX_ADAPTER],
    "nornax-distributed": [_NORNAX_DISTRIBUTED],
    "distributed-tier": [
        "tests/distributed",
        *_UNIT_MULTI_DEVICE,
        "-m",
        "not distributed_criterion and not distributed_segment",
    ],
    "distributed-criterion": ["tests/distributed", "-m", "distributed_criterion"],
    # the fused mesh lane's segment runner (S11): minutes of compile per case
    "distributed-segment": [
        "tests/distributed",
        "-m",
        "distributed_segment",
        f"--ignore={_SEGMENT_CRITERION}",
        f"--ignore={_SEGMENT_MIGRATION}",
        *(f"--ignore={path}" for path in _SEGMENT_RUNNER_FILES),
    ],
    # the same runner under Dehnen's error criterion (S12), a job of its own: with it
    # the segment job took 27 of its 40 minutes on CI
    "distributed-segment-criterion": [_SEGMENT_CRITERION, "-m", "distributed_segment"],
    # the runner's migration to fixed cuts on the fused force (S11 session 2, track
    # I): ~5 min of compilation on a loaded host, kept out of the segment job
    "distributed-segment-migration": [_SEGMENT_MIGRATION, "-m", "distributed_segment"],
    # the mesh potential and the runner's multi-GPU mode on the fused force (S11
    # session 2, tracks E and D): with them the segment job took ~24 of its 40
    # minutes (1,390 s of calls on this host at -n 1, ~1.2x that on CI)
    "distributed-segment-runner": [*_SEGMENT_RUNNER_FILES, "-m", "distributed_segment"],
}

# The two jobs that do not run on every pull request. Each costs minutes of
# forced-device compilation per configuration, and only changes to the code below
# can move them. They run on pushes to main, on the nightly schedule, on a manual
# dispatch, and on a pull request whose diff touches one of `GATED_PATHS` (prefix
# match) -- the `changes` job in ci.yml asks `touches`.
GATED_SHARDS: tuple[str, ...] = (
    "distributed-criterion",
    "distributed-segment",
    "distributed-segment-criterion",
    "distributed-segment-migration",
    "distributed-segment-runner",
    "nornax-distributed",
)
GATED_PATHS: tuple[str, ...] = (
    "jaccpot/distributed/",
    # the production runner's multi-GPU mode dispatches from here
    # (tests/distributed/test_production_runner_mesh_fused.py)
    "jaccpot/production.py",
    # what the fused mesh lane (test-distributed-segment) takes its shapes, template,
    # verdict and plan from on the runtime side
    "jaccpot/runtime/fused_provider.py",
    "jaccpot/runtime/fused_topology.py",
    "jaccpot/runtime/capacity_plan.py",
    "jaccpot/runtime/strict_carry.py",
    "jaccpot/runtime/_level_shapes.py",
    "jaccpot/runtime/capacity_guard.py",
    # where the cross far pairs and their force scale enter the downward sweep
    # (S11's CSR merge, S12's f_b over imported nodes)
    "jaccpot/runtime/_force_scale_levels.py",
    "jaccpot/runtime/kernels/_l2l.py",
    "jaccpot/pallas/m2l_real_csr.py",
    "jaccpot/mutual/",
    "jaccpot/nornax_adapter.py",
    "jaccpot/pallas/nearfield_",
    # their own tests, the fixtures every test shares, and the CI definition
    "tests/distributed/",
    _NORNAX_DISTRIBUTED,
    "tests/conftest.py",
    ".github/",
    "pyproject.toml",
)

EXTRA: dict[str, list[str]] = {
    # The Python-floor job: version compatibility on the oldest supported
    # interpreter, not a second run of the suite. The curated `smoke` marker
    # (tests/smoke_tests.txt): one fast representative per lane plus the public
    # API, contract and guard tests.
    "py-floor": ["tests", "-m", "smoke"],
    # The lint job: the tests of the bench/ scripts (the GPU and two-card gates'
    # own checks, the annotation tools, a comparison benchmark's exit code). They
    # test no library code, so they sit outside `tests/` and outside the partition,
    # and run once per push (moved out of tests/unit in the 2026-10 cleanup, Z-a).
    "bench-tooling": ["bench/tests"],
}

_UNIVERSE = ["tests"]
_TESTS = pathlib.Path(__file__).resolve().parents[2] / "tests"
_MARKER_LISTS = ("slow_tests.txt", "smoke_tests.txt")


def _listed_node_ids(name: str) -> list[str]:
    path = _TESTS / name
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _selection(name: str) -> list[str]:
    if name in SHARDS:
        return SHARDS[name]
    if name in EXTRA:
        return EXTRA[name]
    known = ", ".join([*SHARDS, *EXTRA])
    raise SystemExit(f"unknown shard {name!r}; known: {known}")


def _collect(args: list[str]) -> set[str]:
    # `-n 0` overrides addopts' `-n auto`: collection must not fan out to workers.
    # No `-q` here: addopts already carries one, and a second (`-qq`) makes pytest
    # print per-file counts instead of node ids.
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-n", "0", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Exit 5 is "no tests collected": legitimate for an empty selection, and the
    # partition check below reports it as a gap if the universe disagrees.
    if proc.returncode not in (0, 5):
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"collection failed ({proc.returncode}): {' '.join(args)}")
    return {line.strip() for line in proc.stdout.splitlines() if "::" in line}


def check() -> int:
    """Collect every shard and the universe; report overlaps and gaps."""
    universe = _collect(_UNIVERSE)
    owner: dict[str, str] = {}
    problems = 0
    for name, args in SHARDS.items():
        items = _collect(args)
        print(f"{name:24s} {len(items):5d}")
        for node in sorted(items):
            if node in owner:
                problems += 1
                print(f"  OVERLAP {node}: {owner[node]} and {name}")
            owner[node] = name
    missing = sorted(universe - owner.keys())
    extra = sorted(owner.keys() - universe)
    for node in missing:
        print(f"  NOT IN ANY SHARD {node}")
    for node in extra:
        print(f"  NOT IN THE DEFAULT COLLECTION {node} ({owner[node]})")
    problems += len(missing) + len(extra)
    print(f"{'universe':24s} {len(universe):5d}; shards cover {len(owner)}")
    for name in GATED_SHARDS:
        if name not in SHARDS:
            problems += 1
            print(f"  GATED SHARD {name!r} IS NOT A SHARD")
    for name in _MARKER_LISTS:
        listed = _listed_node_ids(name)
        stale = sorted(set(listed) - universe)
        for node in stale:
            print(f"  STALE ID IN tests/{name}: {node}")
        problems += len(stale)
        print(f"{name:24s} {len(listed):5d} ids; {len(stale)} stale")
    return 1 if problems else 0


def touches(paths: list[str]) -> bool:
    """Whether any changed path is one the gated jobs must run for."""
    return any(path.startswith(GATED_PATHS) for path in paths)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "args":
        print("\n".join(_selection(argv[1])))
        return 0
    if argv == ["check"]:
        return check()
    if argv == ["touches"]:
        paths = [line.strip() for line in sys.stdin if line.strip()]
        print("true" if touches(paths) else "false")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
