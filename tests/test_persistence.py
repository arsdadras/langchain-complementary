"""Unit tests for langcomp.persistence (checkpointer, managed_checkpointer, async_checkpointer)."""

from __future__ import annotations

import sys

import pytest

from langcomp import async_checkpointer, checkpointer, managed_checkpointer


def test_checkpointer_memory_warns_and_validation_errors() -> None:
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        saver = checkpointer("memory")
    assert saver is not None

    with pytest.raises(ValueError, match="does not accept"):
        checkpointer("memory", path="bad.db")

    with pytest.raises(ValueError, match="requires 'path'"):
        checkpointer("sqlite")

    with pytest.raises(ValueError, match="non-empty 'path'"):
        checkpointer("sqlite", path="   ")

    with pytest.raises(ValueError, match="does not accept 'conn_string'"):
        checkpointer("sqlite", path=":memory:", conn_string="postgresql://localhost/db")

    with pytest.raises(ValueError, match="requires 'conn_string'"):
        checkpointer("postgres")

    with pytest.raises(ValueError, match="does not accept 'path'"):
        checkpointer("postgres", path="x.db", conn_string="postgresql://localhost/db")

    with pytest.raises(ValueError, match="unknown checkpointer kind"):
        checkpointer("redis")


def test_managed_checkpointer_memory_and_missing_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        with managed_checkpointer("memory") as saver:
            assert saver is not None

    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="langcomp\\[sqlite\\]"):
        with managed_checkpointer("sqlite", path=":memory:"):
            pass

    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", None)  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match="sslmode"):
        with pytest.raises(ImportError, match="langcomp\\[postgres\\]"):
            with managed_checkpointer("postgres", conn_string="postgresql://user:pw@db.example.com/app"):
                pass


async def test_async_checkpointer_memory_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.warns(UserWarning, match="in-memory checkpointer"):
        async with async_checkpointer("memory") as saver:
            assert saver is not None

    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite.aio", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="langcomp\\[sqlite\\]"):
        async with async_checkpointer("sqlite", path=":memory:"):
            pass

    with pytest.raises(ValueError, match="unknown checkpointer kind"):
        async with async_checkpointer("unknown"):
            pass
