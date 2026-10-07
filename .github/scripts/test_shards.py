"""Which pytest selection each CI job runs -- the one place that says so.

The workflow asks this file for a job's arguments
(``python .github/scripts/test_shards.py args <shard>``) instead of spelling them
out, and the ``test-partition`` job runs ``check``: every shard is collected
separately, and the shards must be pairwise disjoint and together cover exactly
the default collection (``tests/perf`` excluded by ``addopts``). A test directory or file that appears, moves or disappears then
either lands in a shard or turns that job red -- before, the partition was
checked by hand-counted comments in ``ci.yml``, which drifted.

``EXTRA`` holds selections that deliberately re-run a subset (the Python-floor
job) and are not part of the partition.

Usage::

    python .github/scripts/test_shards.py args unit      # one argument per line
    python .github/scripts/test_shards.py check          # exit 1 on any overlap or gap
"""

from __future__ import annotations

import subprocess
import sys

# The Dehnen-MAC / force-scale criterion family of tests/unit/runtime. Every case
# builds a solver and runs a full FMM solve (26-95 s each on CPU), which is why
# the directory used to have a job to itself; it is now two shards of test-full,
# this family and the rest of the directory.
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

# Single files that need their own job: forced host devices change the device
# topology of the whole process, and nornax has to be installed for the
# cross-repo files to collect at all.
_MUTUAL_DISTRIBUTED = "tests/integration/test_mutual_distributed.py"
_NORNAX_ADAPTER = "tests/integration/test_mutual_fmm_nornax.py"
_NORNAX_DISTRIBUTED = "tests/integration/test_mutual_distributed_nornax.py"
_MUTUAL_STATIC_DEVICE = "tests/integration/test_mutual_fmm_static_device.py"

SHARDS: dict[str, list[str]] = {
    # test-full matrix
    "integration": [
        "tests/integration",
        f"--ignore={_MUTUAL_STATIC_DEVICE}",
        f"--ignore={_MUTUAL_DISTRIBUTED}",
        f"--ignore={_NORNAX_ADAPTER}",
        f"--ignore={_NORNAX_DISTRIBUTED}",
    ],
    "mutual-static-device": [_MUTUAL_STATIC_DEVICE],
    "unit": ["tests/unit", "--ignore=tests/unit/runtime"],
    "unit-runtime-mac": list(_RUNTIME_MAC_FILES),
    "unit-runtime": [
        "tests/unit/runtime",
        *(f"--ignore={path}" for path in _RUNTIME_MAC_FILES),
    ],
    "characterization": ["tests/characterization"],
    # single-purpose jobs
    "distributed-mutual": [_MUTUAL_DISTRIBUTED],
    "nornax-adapter": [_NORNAX_ADAPTER],
    "nornax-distributed": [_NORNAX_DISTRIBUTED],
    "distributed-tier": [
        "tests/distributed",
        "-m",
        "not distributed_criterion",
    ],
    "distributed-criterion": ["tests/distributed", "-m", "distributed_criterion"],
}

EXTRA: dict[str, list[str]] = {
    # The Python-floor job: version compatibility on the oldest supported
    # interpreter, not a second run of the suite. The goldens plus the public
    # API surface and the downstream (Odisseo) coupling contract.
    "py-floor": [
        "tests/characterization",
        "tests/unit/test_public_api_surface.py",
        "tests/unit/test_odisseo_coupling.py",
    ],
}

_UNIVERSE = ["tests"]


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
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "args":
        print("\n".join(_selection(argv[1])))
        return 0
    if argv == ["check"]:
        return check()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
