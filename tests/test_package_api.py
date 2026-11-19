"""Tests for package root exports, exception hierarchy, and deprecation shims."""

from __future__ import annotations

import importlib
import sys

import pytest

import langcomp
from langcomp.errors import (
    ApprovalRejectedError,
    ConfigurationError,
    InvalidModelSpecError,
    LangcompError,
    MissingApiKeyError,
    ProviderNotInstalledError,
    ReducerConflictError,
    ScriptExhaustedError,
    SnapshotMismatchError,
    ToolExecutionError,
    UnknownProviderError,
    UnknownToolError,
)


def test_root_exports_match_dunder_all() -> None:
    assert langcomp.__version__
    for symbol in langcomp.__all__:
        assert hasattr(langcomp, symbol), f"Missing exported symbol: {symbol}"
    assert "agent" in langcomp.__all__
    assert "managed_checkpointer" in langcomp.__all__
    assert "async_checkpointer" in langcomp.__all__
    assert "arun_and_snapshot" in langcomp.__all__
    assert "ToolExecutionError" in langcomp.__all__


def test_error_hierarchy_inherits_from_langcomp_error() -> None:
    for err_cls in (
        ConfigurationError,
        InvalidModelSpecError,
        UnknownProviderError,
        ProviderNotInstalledError,
        MissingApiKeyError,
        ReducerConflictError,
        SnapshotMismatchError,
        ScriptExhaustedError,
        ApprovalRejectedError,
        UnknownToolError,
        ToolExecutionError,
    ):
        assert issubclass(err_cls, LangcompError)

    assert issubclass(ToolExecutionError, RuntimeError)


def test_devtools_testing_deprecation_shim() -> None:
    sys.modules.pop("langcomp.devtools.testing", None)
    with pytest.warns(DeprecationWarning, match="langcomp.devtools.testing is deprecated"):
        mod = importlib.import_module("langcomp.devtools.testing")
    assert mod.FakeModel is langcomp.FakeModel
    assert mod.run_and_snapshot is langcomp.run_and_snapshot
    assert mod.arun_and_snapshot is langcomp.arun_and_snapshot
