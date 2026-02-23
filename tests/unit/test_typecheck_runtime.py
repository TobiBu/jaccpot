"""Runtime type-checking configuration tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_typecheck_module():
    path = Path(__file__).resolve().parents[2] / "jaccpot" / "_typecheck.py"
    spec = importlib.util.spec_from_file_location("jaccpot_typecheck_test_module", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_typecheck_disabled_by_default(monkeypatch):
    monkeypatch.delenv("JACCPOT_RUNTIME_TYPECHECK", raising=False)
    module = _load_typecheck_module()

    assert module.enable_runtime_typecheck() is False


def test_runtime_typecheck_can_be_enabled(monkeypatch):
    monkeypatch.setenv("JACCPOT_RUNTIME_TYPECHECK", "1")
    module = _load_typecheck_module()

    assert module.enable_runtime_typecheck() is True


def test_runtime_typecheck_can_be_disabled(monkeypatch):
    monkeypatch.setenv("JACCPOT_RUNTIME_TYPECHECK", "0")
    module = _load_typecheck_module()

    assert module.enable_runtime_typecheck() is False
