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
