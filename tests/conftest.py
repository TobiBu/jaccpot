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
    """
    saved_environ = dict(os.environ)
    os.environ["JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH"] = str(
        tmp_path / "strict_caps.json"
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)
