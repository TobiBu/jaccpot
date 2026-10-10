"""Which tests execute the code each cleanup phase removes.

Input is a coverage data file recorded with per-test contexts::

    pytest tests/unit tests/integration tests/characterization -m "not experimental" \\
        --cov=jaccpot --cov-context=test --cov-report=

(``COVERAGE_FILE`` selects the file). Every removal family below is a set of
REGIONS: whole files, or the bodies of functions found by name in the AST (so the
inventory follows the code, not stale line numbers), or a span between two text
anchors. A family's phase removes its regions; a removed file, a function name
that no longer matches and a span whose anchor is gone contribute nothing, so the
families stay as the record of what each phase removed. A test belongs to a family when it executed at least one line inside one
of the family's regions; ``def`` lines and module-level code run at import and
carry no test context, so they never count. Neither do a function's leading guard
clauses (``if not use_dense: return None``), nor the ``gatekeepers`` a family
lists: functions every prepare calls that only decide the feature is off (the
dual-planner hint, the autotune entry under static sizing, the grouped budget
predicates). A test that only passed through those does not use the feature.

The output says, per family, which tests touch it and from which files -- the
list a phase has to settle: tests of the removed code go with it, tests that only
pass through it are re-targeted first. See ``docs/cleanup_2026-10.md``.

Usage::

    python bench/cleanup_inventory.py COVERAGE_FILE [--json OUT] [--family NAME]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "jaccpot"

# Each family: whole files, functions (regex on the qualified name, per file glob),
# and anchored spans (file, start text, end text).
FAMILIES: dict[str, dict] = {
    "complex": {
        "files": [
            "operators/complex_ops.py",
            "operators/complex_harmonics.py",
            "operators/solidfmm_reference.py",
            "pallas/m2l_complex_fused.py",
            "upward/solidfmm_complex_tree_expansions.py",
            "basis/complex_sh.py",
        ],
        "functions": [("**/*.py", r"complex")],
    },
    "cartesian": {
        "files": ["operators/multipole_utils.py"],
        "functions": [
            ("upward/tree_expansions.py", r"^[^.]+$"),
            ("downward/local_expansions.py", r"^[^.]+$"),
            ("**/*.py", r"cartesian"),
        ],
    },
    "dense": {"functions": [("runtime/**/*.py", r"dense")]},
    "octree": {
        "gatekeepers": [r"_prepared_state_octree_upward_payload$"],
        "files": [
            "runtime/_octree_fmm.py",
            "runtime/_octree_adapter.py",
            "experimental/octree_fmm_uvwx.py",
        ],
        "functions": [("**/*.py", r"octree")],
    },
    "grouped": {
        "gatekeepers": [
            r"_grouped_schedule_item_budget$",
            r"_should_precompute_grouped_class_segments$",
        ],
        "functions": [
            ("runtime/**/*.py", r"grouped|class_major|class_segment"),
        ],
    },
    "autotune": {
        "gatekeepers": [r"_prepare_state_autotune_downward_chunk_size$"],
        "files": ["runtime/fmm_autotune.py"],
        "functions": [("**/*.py", r"autotune")],
    },
    "treecode": {
        "files": [
            "experimental/treecode_far_near.py",
            "experimental/treecode_walk.py",
            "pallas/treecode_walk_pallas.py",
        ],
        "functions": [("**/*.py", r"treecode")],
    },
    "large_n_farfield": {"files": ["runtime/_large_n_farfield.py"]},
    "fixed_depth": {"functions": [("**/*.py", r"fixed_depth")]},
    "legacy_strict_api": {
        "functions": [
            (
                "**/*.py",
                r"(^|\.)(strict_run_segmented|update_multipoles_only"
                r"|rebuild_topology_in_place)$",
            )
        ],
    },
    "dual_planner": {
        "gatekeepers": [r"_resolve_dual_downward_planner_hint$"],
        "functions": [
            ("runtime/**/*.py", r"dual_downward_planner|refresh_dual_planner"),
        ],
    },
    "nonfused_strict_loop": {
        "spans": [
            (
                "runtime/fmm_strict_run.py",
                "            state_curr = state_arr\n            history_parts",
                "history_out = jnp.stack(history_parts",
            )
        ],
    },
    "kernel_variants": {
        "files": ["pallas/m2l_real_csr_tiled.py", "pallas/m2l_core_z_real.py"],
        "functions": [
            (
                "pallas/com_radii_leaf.py",
                r"^(_com_radii_chunk_kernel|com_radii_chunk_pallas)$",
            ),
            ("pallas/p2m_real_leaf.py", r"^_p2m_leaf_kernel$"),
            (
                "pallas/cascade_real_level.py",
                r"^(_m2m_level_kernel|_l2l_level_kernel|_level_call"
                r"|m2m_real_levels_pallas|l2l_real_levels_pallas)$",
            ),
            ("pallas/m2l_real_csr.py", r"^(_m2l_real_csr_kernel|m2l_real_csr_pallas)$"),
            (
                "operators/m2l_real_rot_scale.py",
                r"^(_centred_degree_maps|_padded_dz|_rotate_degree_batched)$",
            ),
            ("runtime/_mac_geometry.py", r"^_node_depths$"),
        ],
    },
}


def _is_guard(stmt: ast.stmt) -> bool:
    """``if <cond>: return ...`` with no else -- a gatekeeper's early exit."""
    return (
        isinstance(stmt, ast.If)
        and not stmt.orelse
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Return)
    )


def _function_spans(path: Path) -> list[tuple[str, int, int]]:
    """(qualified name, first feature line, last line) for every def in a file.

    The span starts after the docstring and after any LEADING guard clauses
    (``if not use_dense: return None``): many of the functions a removal targets
    are gatekeepers that every prepare calls and that return at once when the
    feature is off, and a test that only took that exit does not use the feature.
    """
    tree = ast.parse(path.read_text())
    out: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                body = list(child.body)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                while body and _is_guard(body[0]):
                    body = body[1:]
                if body:
                    # nested defs are covered by their parent's span too
                    out.append((name, body[0].lineno, child.end_lineno or 0))
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")

    visit(tree, "")
    return out


def _family_regions(spec: dict) -> dict[Path, list[tuple[int, int, str]]]:
    regions: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    for rel in spec.get("files", []):
        path = PKG / rel
        if path.exists():
            regions[path].append((1, 10**9, f"{rel} (file)"))
    skip = [re.compile(p) for p in spec.get("gatekeepers", [])]
    for glob, pattern in spec.get("functions", []):
        rx = re.compile(pattern)
        for path in sorted(PKG.glob(glob)):
            for name, first, last in _function_spans(path):
                if rx.search(name) and not any(g.search(name) for g in skip):
                    regions[path].append(
                        (first, last, f"{path.relative_to(PKG)}:{name}")
                    )
    for rel, start, end in spec.get("spans", []):
        path = PKG / rel
        text = path.read_text() if path.exists() else ""
        a = text.find(start)
        b = text.find(end, a) if a >= 0 else -1
        if b < 0:
            continue  # removed by its phase
        regions[path].append(
            (text.count("\n", 0, a) + 1, text.count("\n", 0, b) + 1, f"{rel} (span)")
        )
    return regions


def _test_file(context: str) -> str:
    return context.split("::", 1)[0]


def inventory(coverage_file: str, only: str | None = None) -> dict[str, dict]:
    from coverage import CoverageData

    data = CoverageData(basename=coverage_file)
    data.read()
    measured = {Path(f).resolve(): f for f in data.measured_files()}
    result: dict[str, dict] = {}
    for family, spec in FAMILIES.items():
        if only and family != only:
            continue
        tests: dict[str, set[str]] = defaultdict(set)
        for path, spans in _family_regions(spec).items():
            key = measured.get(path.resolve())
            if key is None:
                continue
            by_line = data.contexts_by_lineno(key)
            for lineno, contexts in by_line.items():
                for first, last, label in spans:
                    if first <= lineno <= last:
                        for ctx in contexts:
                            if ctx:
                                tests[ctx.split("|", 1)[0]].add(label)
        files: dict[str, int] = defaultdict(int)
        for ctx in tests:
            files[_test_file(ctx)] += 1
        result[family] = {
            "tests": {ctx: sorted(labels) for ctx, labels in sorted(tests.items())},
            "files": dict(sorted(files.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("coverage_file")
    parser.add_argument("--json", help="write the full inventory here")
    parser.add_argument("--family", help="only this family")
    args = parser.parse_args(argv)
    result = inventory(args.coverage_file, args.family)
    for family, entry in result.items():
        print(f"{family}: {len(entry['tests'])} tests in {len(entry['files'])} files")
        for name, count in entry["files"].items():
            print(f"    {count:4d}  {name}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
