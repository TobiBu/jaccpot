"""``experimental/`` must not be pulled in by importing production packages.

STYLE_GUIDE section 8 says ``experimental/`` is "not imported by production
paths". That was **false** until audit G.5's second edge was fixed:
``pallas/__init__.py`` re-exports ``treecode_walk_pallas``, which imported
``jaccpot.experimental.treecode_walk`` at module scope, so

    import jaccpot          ->  experimental NOT loaded
    import jaccpot.pallas   ->  loads jaccpot.experimental.treecode_walk

The claim was true for the entry point and false one package down, which is
exactly the shape of thing a prose guarantee hides. This test is the claim.

Why a subprocess: ``sys.modules`` is process-global and every other test in the
suite has already imported half the package, so an in-process check would pass or
fail depending on collection order. Each case gets a clean interpreter.

Scope note. This pins the EAGER import graph only. ``experimental/`` is still
reachable deliberately and lazily -- ``runtime/_interaction_cache.py`` imports
``treecode_far_near`` inside a function when ``local_walk="treecode"`` is
selected, which G.5 decided to leave as an accepted, bounded exposure. Nothing
here forbids that; a function-local import loads nothing until called.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Production packages a user could plausibly import directly. `jaccpot.pallas` is
# the one that regressed; the others are here so the next such edge is caught
# wherever it lands rather than only in the module that happened to have one.
PRODUCTION_IMPORTS = (
    "jaccpot",
    "jaccpot.pallas",
    "jaccpot.operators",
    "jaccpot.runtime",
    "jaccpot.nearfield",
    "jaccpot.distributed",
)

_PROBE = """
import sys
import {module}  # noqa: F401
leaked = sorted(m for m in sys.modules if m.startswith("jaccpot.experimental"))
print(";".join(leaked))
"""


@pytest.mark.parametrize("module", PRODUCTION_IMPORTS)
def test_importing_production_does_not_load_experimental(module: str) -> None:
    """Importing a production package must not load ``jaccpot.experimental``."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert (
        result.returncode == 0
    ), f"probe failed to import {module}:\n{result.stderr[-2000:]}"
    leaked = [m for m in result.stdout.strip().split(";") if m]
    assert not leaked, (
        f"`import {module}` eagerly loaded {leaked}. STYLE_GUIDE section 8 says "
        "`experimental/` is not imported by production paths. Make the offending "
        "import function-local (or TYPE_CHECKING-only if it is annotation-only) "
        "rather than relaxing this test -- see jaccpot/pallas/treecode_walk_pallas.py "
        "for the pattern."
    )
