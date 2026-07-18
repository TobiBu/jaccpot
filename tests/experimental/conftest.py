"""Relegate the experimental octree/treecode prototype tests behind an opt-in marker.

The modules under ``tests/experimental/`` compile large, non-production FMM graphs
(uniform/adaptive octree U/V/W lists, per-leaf treecode walk) and dominate the
suite's wall clock. They are marked ``experimental`` here so the everyday run
(``-m "not experimental"`` in ``addopts``) skips them; run them explicitly with
``pytest -m experimental``. The production path is the radix real fast lane.
"""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/experimental/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.experimental)
