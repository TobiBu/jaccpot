"""``git_meta`` must separate a dirty results file from dirty source.

This exists because conflating them made every section-7 figure artifact look
unfit for the manuscript. The committed figure artifacts are tracked files under
``bench/results/``, and several benches rewrite theirs after every point so an
interrupted multi-hour sweep still leaves something usable. Two benches running
concurrently therefore each see the other's output as a dirty tree and record
``git_dirty: true`` from a clean checkout -- which says nothing about whether the
recorded commit describes the code that ran.
"""

import subprocess
import sys

sys.path.insert(0, "examples/jaccpot_paper")

from common import runmeta  # noqa: E402


def test_git_meta_reports_both_kinds_of_dirty():
    """Both flags are present, and the source-only one is the narrower."""
    meta = runmeta.git_meta()
    for key in ("git_sha", "git_branch", "git_dirty", "git_dirty_sources"):
        assert key in meta, f"git_meta lost {key}"
    assert isinstance(meta["git_dirty"], bool)
    assert isinstance(meta["git_dirty_sources"], bool)
    # Source-dirty implies tree-dirty; the converse is exactly what we allow.
    if meta["git_dirty_sources"]:
        assert meta["git_dirty"], "source changes must also make the tree dirty"
    assert isinstance(meta["git_dirty_source_paths"], list)
    # No reported source path may live under bench/results/.
    for path in meta["git_dirty_source_paths"]:
        assert not path.startswith("bench/results/"), path


def test_a_results_only_change_is_not_source_dirty(tmp_path):
    """Touching a tracked results JSON must not flag the sources as dirty.

    Uses a throwaway clone so the real working tree is left alone, and asserts
    the property that matters: a modified ``bench/results/`` file makes
    ``git_dirty`` true and ``git_dirty_sources`` false.
    """
    repo = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--depth", "1", ".", str(repo)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-m",
            "base",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target = repo / "bench" / "results" / "density_reconstruction" / "probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-f", str(target)],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "add probe artifact",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    # Now dirty ONLY that tracked results file.
    target.write_text('{"changed": true}\n')

    script = (
        "import sys, json; sys.path.insert(0, 'examples/jaccpot_paper');"
        "from common import runmeta; print(json.dumps(runmeta.git_meta()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    meta = json.loads(done.stdout)
    assert meta["git_dirty"] is True, "a modified tracked file must show as dirty"
    assert meta["git_dirty_sources"] is False, (
        "a results-only change must NOT count as source dirt: "
        f"{meta['git_dirty_source_paths']}"
    )


def test_a_source_change_is_source_dirty(tmp_path):
    """A modified source file must flag both."""
    repo = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--depth", "1", ".", str(repo)],
        check=True,
        capture_output=True,
    )
    target = repo / "jaccpot" / "__init__.py"
    target.write_text(target.read_text() + "\n# probe\n")

    script = (
        "import sys, json; sys.path.insert(0, 'examples/jaccpot_paper');"
        "from common import runmeta; print(json.dumps(runmeta.git_meta()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    meta = json.loads(done.stdout)
    assert meta["git_dirty"] is True
    assert meta["git_dirty_sources"] is True
    assert any(
        p.endswith("jaccpot/__init__.py") for p in meta["git_dirty_source_paths"]
    ), meta["git_dirty_source_paths"]
