"""The yggdrax on the path must carry every symbol ``jaccpot`` imports.

`pyproject.toml` declares `yggdrax>=0.0.1,<0.1.0` and yggdrax's version has never
left 0.0.1, so the floor matches every yggdrax that has ever existed. That is not
a slack constraint, it is a vacuous one: until 2026-08-25 the venv held a
non-editable copy while `tests/conftest.py` put the sibling checkout on
`sys.path`, so pytest and every script imported different libraries, and the
script side could not import `jaccpot.mutual.distributed` at all while the pytest
side collected and passed.

`tests/conftest.py` closes that with a collection-time check derived from the
source rather than hand-listed. This file is the check's own test: the derivation
rule (module body only, production only) and the resolution rule (attribute *or*
submodule) are both easy to get subtly wrong in a direction that makes the guard
silently vacuous, which is the failure mode it exists to prevent.

`test_the_environment_satisfies_the_contract` is the invariant itself, stated as
a test so it also holds under a runner that never reaches `pytest_configure`.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from tests.conftest import (
    _JACCPOT_PACKAGE,
    _missing_yggdrax_symbols,
    _yggdrax_symbols_jaccpot_imports,
)


def test_the_environment_satisfies_the_contract() -> None:
    """Every module-level yggdrax name jaccpot imports must resolve."""
    required = _yggdrax_symbols_jaccpot_imports(_JACCPOT_PACKAGE)
    # Vacuity guard: an empty scan would satisfy the assertion below while
    # checking nothing, which is exactly how the version floor failed.
    assert len(required) > 5, f"scan found almost nothing: {sorted(required)}"
    assert _missing_yggdrax_symbols(required) == []


# The distributed mutual path is where jaccpot and yggdrax move together --
# yggdrax#46/#49/#50/#51 and jaccpot#197/#202/#203/#205 were paired merges -- so
# it is where a stale yggdrax bites first. These three are its load-bearing
# imports, pinned by name so a refactor that moves one out of a module body
# cannot quietly shrink the contract to something that still passes.
_LOAD_BEARING = (
    ("yggdrax.distributed.reverse_halo", "halo_return_addresses"),
    ("yggdrax.distributed.cross_walk", "single_owner_domain"),
    ("yggdrax.distributed.cross_walk", "dual_tree_walk_cross_mutual"),
    ("yggdrax.dtypes", "INDEX_DTYPE"),
)


@pytest.mark.parametrize(("module", "name"), _LOAD_BEARING)
def test_the_scan_finds_the_load_bearing_imports(module: str, name: str) -> None:
    """Each named symbol is a module-level import and must be collected."""
    required = _yggdrax_symbols_jaccpot_imports(_JACCPOT_PACKAGE)
    assert name in required.get(module, set())


def test_the_scan_ignores_imports_jaccpot_guards_itself(
    tmp_path: pathlib.Path,
) -> None:
    """Function-local, ``try``-guarded and TYPE_CHECKING imports are excluded."""
    package = tmp_path / "jaccpot"
    package.mkdir()
    (package / "mod.py").write_text(textwrap.dedent("""
            from typing import TYPE_CHECKING

            from yggdrax.dtypes import INDEX_DTYPE

            if TYPE_CHECKING:
                from yggdrax.tree import Tree

            try:
                from yggdrax.interactions import maybe_new
            except ImportError:
                maybe_new = None


            def build():
                from yggdrax.interactions import dual_tree_walk_mutual

                return dual_tree_walk_mutual
            """))
    assert _yggdrax_symbols_jaccpot_imports(package) == {
        "yggdrax.dtypes": {"INDEX_DTYPE"}
    }


def test_the_scan_skips_experimental(tmp_path: pathlib.Path) -> None:
    """`jaccpot/experimental/` is not production and is not a requirement."""
    package = tmp_path / "jaccpot"
    (package / "experimental").mkdir(parents=True)
    (package / "experimental" / "proto.py").write_text(
        "from yggdrax.octree_uvwx import something\n"
    )
    assert _yggdrax_symbols_jaccpot_imports(package) == {}


def test_a_missing_symbol_is_reported_by_name() -> None:
    """A name yggdrax does not define comes back as a dotted string."""
    missing = _missing_yggdrax_symbols({"yggdrax.dtypes": {"INDEX_DTYPE", "nope"}})
    assert missing == ["yggdrax.dtypes.nope"]


def test_a_missing_module_is_reported_with_its_error() -> None:
    """An unimportable module is reported once, not once per name."""
    missing = _missing_yggdrax_symbols({"yggdrax.no_such_module": {"a", "b"}})
    assert len(missing) == 1
    assert missing[0].startswith("yggdrax.no_such_module (")


def test_a_submodule_import_counts_as_present() -> None:
    """`from yggdrax.distributed import let` takes a submodule, not an attribute.

    A package need not bind its submodules as attributes, so an attribute-only
    check would report `yggdrax.distributed.let` missing on a perfectly good
    install -- a false alarm that would make the whole guard untrustworthy.
    """
    assert _missing_yggdrax_symbols({"yggdrax.distributed": {"let"}}) == []
