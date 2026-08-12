import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

YGGDRAX_ROOT = REPO_ROOT.parent / "yggdrax"
if YGGDRAX_ROOT.exists() and str(YGGDRAX_ROOT) not in sys.path:
    sys.path.insert(0, str(YGGDRAX_ROOT))


# nornax is a TEST-ONLY dependency, used by the cross-repo momentum/block-step
# checks. The library never imports it -- the dependency graph is
# Jaccpot -> Yggdrax, Nornax standalone, ODISSEO -> both -- so it is put on the
# path here rather than declared as a package dependency. Absent, those tests skip.
#
# Searched across REPO_ROOT's ancestors rather than just `REPO_ROOT.parent`,
# because a git worktree checkout sits at `<repo>/.claude/worktrees/<name>`, so its
# parent is `worktrees/` and the plain sibling lookup finds nothing.
def _find_sibling_checkout(name: str) -> pathlib.Path | None:
    """Return a sibling source checkout of ``name``, searching upward."""
    for ancestor in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = ancestor.parent / name
        if (candidate / name / "__init__.py").exists():
            return candidate
    return None


NORNAX_ROOT = _find_sibling_checkout("nornax")
if NORNAX_ROOT is not None and str(NORNAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NORNAX_ROOT))

# --- Test-suite performance setup -------------------------------------------
# The FMM correctness tests assume float64; set it once here so individual
# tests/modules do not each have to depend on the ambient environment.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Opt-in persistent JAX compilation cache. JAX already caches compiled
# executables in-process (per jaxpr + shapes + static args), so within one
# xdist worker repeat FMM compiles are already free; the *disk* cache only helps
# ACROSS workers/runs. A cold disk cache adds serialization overhead to a single
# run without enough cross-worker hits to pay it back, so this is off by default
# and only activates when a cache dir is explicitly provided. CI sets
# JACCPOT_TEST_JAX_CACHE_DIR and persists it across runs (actions/cache), turning
# the expensive FMM compiles into warm-cache hits on subsequent runs.
_jax_cache_dir = os.environ.get("JACCPOT_TEST_JAX_CACHE_DIR")
if _jax_cache_dir and os.environ.get(
    "JACCPOT_TEST_NO_JAX_CACHE", "0"
).strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _jax_cache_dir)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")


# --- Data-driven slow-test marking ------------------------------------------
# The heavy, compile-bound tests (>~8s each) are listed by node id in
# tests/slow_tests.txt and auto-marked `slow` here, so CI can run the full suite
# on one Python version and a fast `-m "not slow"` smoke on the others without
# scattering @pytest.mark.slow across ~50 tests. Regenerate the list with:
#   pytest --durations=0 | awk '/call/ && $1+0>=8 {print $3}' | sort -u
_SLOW_LIST = pathlib.Path(__file__).parent / "slow_tests.txt"
_SLOW_NODE_IDS = (
    {
        line.strip()
        for line in _SLOW_LIST.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if _SLOW_LIST.exists()
    else set()
)


def pytest_collection_modifyitems(config, items):
    """Auto-apply the `slow` marker to node ids listed in slow_tests.txt."""
    if not _SLOW_NODE_IDS:
        return
    slow = pytest.mark.slow
    for item in items:
        if item.nodeid in _SLOW_NODE_IDS:
            item.add_marker(slow)


# --- Bound peak memory of the differentiable-FMM tests ----------------------
# `jax.grad` of the fixed-topology FMM (differentiable_accelerations) compiles a
# large reverse-mode XLA program, and each distinct (basis, order, N, tree)
# config leaves its executable in JAX's *in-process* compilation cache. Across a
# long-lived `pytest-xdist` worker these accumulate -- the three differentiable
# modules alone climb to ~4.6 GB, which (stacked on the ~50 `slow` FMM tests the
# `test-full` job also runs, times two workers) OOM-reclaims the ~7 GB CI runner
# ("The runner has received a shutdown signal"). Trimming N/order does NOT help:
# the peak is the retained-executable pile, not the per-test tape. Dropping the
# in-memory cache after each of these tests holds the added footprint to ~1 GB.
# The persistent on-disk cache survives `clear_caches()`, so the follow-up
# recompiles are warm-cache reads; scoped to just these modules so the rest of
# the suite keeps its warm in-process cache.
_DIFF_FMM_TEST_FILES = frozenset(
    {
        "test_grad_fmm_vs_directsum.py",
        "test_gradient_correctness.py",
        "test_custom_vjp_parity.py",
        "test_nearfield_fastlane_grad_path.py",
        # The gradient golden compiles one reverse program per (basis, order, N)
        # case and measures 1.1-1.5 GB peak each, so it belongs on this list for
        # exactly the reason above -- it is a characterization test, but the
        # footprint is a differentiable-FMM footprint.
        "test_fmm_grad_golden.py",
    }
)

# Cleared only when the on-disk JAX cache is available to serve the recompiles.
#
# The mutual FMM suite belongs in the set above for exactly the reason given
# there and was simply never added: it is the heaviest file in
# `tests/integration`, and its gradient tests (FD-vs-AD, vs-direct-sum, rollout,
# per-level, and the Pallas backend) each leave another reverse-mode executable
# behind. Omitting it is what let `test-full` drift back to the ceiling and
# start OOM-killing the runner. Measured on `tests/integration` at `-n 2 --cov`,
# clearing here moves peak RSS 12.68 GB -> 5.47 GB (-57%).
#
# It is gated because that trade is only good when the recompiles are warm -- the
# note above ("the follow-up recompiles are warm-cache reads") holds only where
# JACCPOT_TEST_JAX_CACHE_DIR is set. Ungated, this cost +66% wall on the mutual
# tests and pushed both test-smoke jobs from 27.9/29.9 min straight into their
# cap, trading one CI failure for another.
#
# The gate is on the CONDITION, not on a job name, which is what makes it still
# correct now that test-smoke has a JAX cache of its own: the clearing simply
# switches itself on there, because the premise it waits for is now true. Do not
# rewrite this as "test-full only" -- that was a symptom of which jobs had a
# cache, not the rule.
_DIFF_FMM_TEST_FILES_WARM_ONLY = frozenset(
    {
        "test_mutual_fmm.py",
        "test_mutual_fmm_nornax.py",
    }
)


# The same footprint, arrived at from the other direction. tests/unit/runtime is the
# Dehnen-MAC suite: ~94 cases that each build a solver and run a full FMM solve. None takes
# a gradient, so none qualifies for the list above, but the retained-executable pile is the
# same pile -- measured 8.20 GB peak RSS for `pytest -n 2 tests/unit/runtime`, against the
# ~7 GB the CI runner has. That is why its job (test-mac-runtime) OOM-kills an xdist worker
# intermittently, and why the crash lands on an arbitrary victim -- whichever test happened
# to be executing when the worker died, which is what made it look like a flake:
#
#     main, 2026-08-02   gw1  test_supplied_force_scale_rejects_a_mismatched_shape[two_dim]
#     branch             gw0  test_supplied_force_scale_rejects_a_mismatched_shape[short]
#     branch             gw0  test_far_near_partition_is_complete[dehnen-None-False]
#
# Three victims, one cause, and the first predates the branch that surfaced it. Listed as a
# directory rather than filenames because the whole suite has this shape.
#
# WHAT IT COSTS, because it is more than the note above implies. Clearing after each of 94
# solver tests takes that command from 269 s to 506 s, a 1.9x wall-clock cost, while peak
# RSS drops to 3.33 GB. The on-disk cache recovers only ~12% of the recompiles (577 s cold
# vs 506 s warm), NOT most of them: `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` is 1.0
# above, so the many sub-second compiles this suite churns through are never written to
# disk. The trade is taken anyway -- test-mac-runtime runs ~8 min against a 50 min cap, so
# there is room, and an intermittently-crashing job is worth more than the minutes. If that
# ever stops being true, the lever is a coarser trigger (clear every k tests, or only above
# an RSS watermark) rather than dropping this.
_SOLVER_HEAVY_TEST_DIRS = frozenset(
    {pathlib.Path(__file__).parent / "unit" / "runtime"}
)


def _retains_heavy_executables(node: pytest.Item) -> bool:
    """Whether this test leaves a compiled-executable pile worth dropping.

    Parameters
    ----------
    node : pytest.Item
        The test item that just finished.

    Returns
    -------
    bool
        True when the test is one of the differentiable-FMM modules, lives in a
        solver-heavy directory, or is one of the warm-only modules and the on-disk
        JAX cache is available to serve its recompiles.
    """
    return (
        node.path.name in _DIFF_FMM_TEST_FILES
        or node.path.parent in _SOLVER_HEAVY_TEST_DIRS
        or (bool(_jax_cache_dir) and node.path.name in _DIFF_FMM_TEST_FILES_WARM_ONLY)
    )


@pytest.fixture(autouse=True)
def _bound_diff_fmm_compile_cache(request):
    """Free JAX's in-process compiled-executable cache after each heavy
    differentiable-FMM or solver-heavy test (see the notes above)."""
    yield
    if _retains_heavy_executables(request.node):
        import gc

        import jax

        jax.clear_caches()
        gc.collect()


@pytest.fixture(autouse=True)
def _isolate_process_env(tmp_path):
    """Isolate ``os.environ`` per test to stop env-var leakage across tests.

    Under ``pytest -n auto`` each xdist worker runs many tests in one process,
    so any ``os.environ`` mutation that is not undone leaks into every later
    test on that worker. Two sources make strict / large-N tests order-dependent
    and flaky:

    * production code writes process-global flags directly, e.g. the strict
      fused lane sets ``YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_*`` in
      ``_fmm_impl.py`` (never cleaned up). A prior strict test then changes the
      dual-tree neighbour/count construction of a later test's *first* build,
      which can pin an undersized static neighbour-edge cap and blow up a
      subsequent step (e.g. ``test_strict_run_v2_api``).
    * the strict fused lane records/loads its traversal cap profile via
      ``JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH`` (default a single shared
      ``/tmp`` file), so a test only sees a recorded profile if another test on
      the same worker wrote it first, and concurrent workers race on the file.

    Snapshot the environment at test start and fully restore it at teardown so
    every test begins from the same baseline regardless of what earlier tests
    (or the code they exercised) wrote. Also point the strict cap-profile file
    at a per-test temp path so record/reload stays self-contained (still shared
    across FastMultipoleMethod instances *within* one test). Tests that set env
    vars themselves (via monkeypatch or directly) are unaffected -- their
    changes simply do not survive past their own teardown.

    **The restore touches only the keys that actually changed, and that is a
    correctness requirement, not an optimisation.** It used to be
    ``os.environ.clear(); os.environ.update(saved)``, which issues one ``unsetenv``
    and one ``putenv`` per variable -- roughly 200 C-level environment mutations per
    test, ~35k across the Dehnen-MAC job. ``setenv``/``unsetenv`` are **not
    thread-safe**: glibc's ``unsetenv`` frees entries in ``environ`` while any other
    thread calling ``getenv`` may be reading them, and both JAX and ``execnet``'s
    receiver thread are live during teardown. That is a segfault, and it was
    happening -- ``test-mac-runtime`` died with ``Fatal Python error: Segmentation
    fault`` whose faulting frame is this fixture's teardown inside
    ``os.environ.__setitem__`` -> ``os.encode``:

        File ".../tests/conftest.py", line 197 in _isolate_process_env
        File "<frozen _collections_abc>", line 991 in update
        File "<frozen os>", line 723 in __setitem__
        File "<frozen os>", line 875 in encode

    Restoring only the diff makes that one mutation for a typical test (the
    cap-profile path this fixture itself sets) instead of ~200, which shrinks the
    race window by the same factor. The final environment is identical either way.

    NOTE: the earlier ``worker 'gwN' crashed`` failures in ``tests/unit/runtime``
    were attributed to memory in the note above. At least one of them was this
    segfault instead, which is why the ``clear_caches`` mitigation did not stop them
    -- a compiled-executable pile has nothing to do with an ``environ`` race. The
    memory measurements in that note stand on their own; the crash attribution did
    not.
    """
    saved_environ = dict(os.environ)
    os.environ["JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH"] = str(
        tmp_path / "strict_caps.json"
    )
    try:
        yield
    finally:
        current_environ = dict(os.environ)
        for key in current_environ.keys() - saved_environ.keys():
            del os.environ[key]
        for key, value in saved_environ.items():
            if current_environ.get(key) != value:
                os.environ[key] = value
