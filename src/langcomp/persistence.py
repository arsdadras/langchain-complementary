"""
Durability-aware checkpointer factories for LangGraph agents.
Provides sync, managed context-manager, and async presets for memory, SQLite, and Postgres.
"""

from __future__ import annotations

import sqlite3
import warnings
import weakref
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from langgraph.checkpoint.base import BaseCheckpointSaver

CheckpointerKind = Literal["memory", "sqlite", "postgres"]
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


def _warn_memory_durability() -> None:
    warnings.warn(
        "in-memory checkpointer: state is lost on process restart — "
        "suitable for quick scripts or unit tests, not for durable workflows.",
        UserWarning,
        stacklevel=3,
    )


def _normalize_sqlite_path(path: str | Path | None) -> str:
    if path is None:
        raise ValueError("checkpointer('sqlite', path=...) requires 'path'")
    raw = str(path).strip()
    if not raw:
        raise ValueError("checkpointer('sqlite', path=...) requires a non-empty 'path'")
    if raw != ":memory:" and not raw.startswith("file:"):
        Path(raw).parent.mkdir(parents=True, exist_ok=True)
    return raw


def _validate_postgres_conn_string(conn_string: str | None) -> str:
    if conn_string is None or not conn_string.strip():
        raise ValueError("checkpointer('postgres', conn_string=...) requires 'conn_string'")
    cleaned = conn_string.strip()
    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    qs = parse_qs(parsed.query)
    if host not in _LOCAL_HOSTS and "sslmode" not in qs:
        warnings.warn(
            "PostgreSQL checkpointer connection string targets a remote host without an explicit "
            "'sslmode' parameter; consider setting sslmode=require or sslmode=verify-full.",
            UserWarning,
            stacklevel=3,
        )
    return cleaned


def _open_sqlite_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _attach_close(saver: Any, close_fn: Any) -> Any:
    """Attach a deterministic ``close()`` helper and GC finalizer without mutating class API."""
    finalizer = weakref.finalize(saver, close_fn)
    try:
        object.__setattr__(saver, "close", finalizer)
    except (AttributeError, TypeError):
        pass
    return saver


def _validate_backend_args(
    caller: str,
    kind: CheckpointerKind | str,
    path: str | Path | None,
    conn_string: str | None,
) -> CheckpointerKind:
    if kind == "memory":
        if path is not None or conn_string is not None:
            raise ValueError(f"{caller}('memory') does not accept 'path' or 'conn_string'.")
        return "memory"
    if kind == "sqlite":
        if conn_string is not None:
            raise ValueError(f"{caller}('sqlite') does not accept 'conn_string'; use 'path'.")
        return "sqlite"
    if kind == "postgres":
        if path is not None:
            raise ValueError(f"{caller}('postgres') does not accept 'path'; use 'conn_string'.")
        return "postgres"
    raise ValueError(
        f"unknown checkpointer kind: {kind!r} (expected 'memory', 'sqlite', or 'postgres')"
    )


def _import_sqlite_saver() -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-sqlite is required for SQLite persistence. "
            "Install it with: pip install 'langcomp[sqlite]'"
        ) from exc
    return SqliteSaver


def checkpointer(
    kind: CheckpointerKind | str,
    *,
    path: str | Path | None = None,
    conn_string: str | None = None,
) -> BaseCheckpointSaver[Any]:
    """Return a configured LangGraph checkpointer for the requested backend.

    For deterministic DB connection teardown in long-lived applications, prefer
    :func:`managed_checkpointer` (sync) or :func:`async_checkpointer` (async).

    Args:
        kind: Backend preset — one of ``"memory"``, ``"sqlite"``, or ``"postgres"``.
        path: File path (or ``":memory:"``) required when ``kind="sqlite"``.
        conn_string: PostgreSQL connection URI required when ``kind="postgres"``.

    Returns:
        An initialized LangGraph checkpoint saver instance.
    """
    backend = _validate_backend_args("checkpointer", kind, path, conn_string)

    if backend == "memory":
        _warn_memory_durability()
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if backend == "sqlite":
        db_path = _normalize_sqlite_path(path)
        sqlite_saver_cls = _import_sqlite_saver()
        conn = _open_sqlite_connection(db_path)
        try:
            saver = sqlite_saver_cls(conn)
            saver.setup()
            return _attach_close(saver, conn.close)
        except Exception:
            conn.close()
            raise

    uri = _validate_postgres_conn_string(conn_string)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-postgres and psycopg are required for "
            "Postgres persistence. Install it with: pip install 'langcomp[postgres]'"
        ) from exc

    conn = Connection.connect(
        uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=10,
    )
    try:
        saver = PostgresSaver(conn)
        saver.setup()
        return _attach_close(saver, conn.close)
    except Exception:
        conn.close()
        raise


@contextmanager
def managed_checkpointer(
    kind: CheckpointerKind | str,
    *,
    path: str | Path | None = None,
    conn_string: str | None = None,
) -> Iterator[BaseCheckpointSaver[Any]]:
    """Context manager yielding a configured checkpointer and closing DB resources on exit."""
    backend = _validate_backend_args("managed_checkpointer", kind, path, conn_string)

    if backend == "memory":
        _warn_memory_durability()
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    if backend == "sqlite":
        db_path = _normalize_sqlite_path(path)
        sqlite_saver_cls = _import_sqlite_saver()
        conn = _open_sqlite_connection(db_path)
        try:
            saver = sqlite_saver_cls(conn)
            saver.setup()
            yield saver
        finally:
            conn.close()
        return

    uri = _validate_postgres_conn_string(conn_string)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-postgres and psycopg are required for "
            "Postgres persistence. Install it with: pip install 'langcomp[postgres]'"
        ) from exc

    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        with PostgresSaver.from_conn_string(uri) as saver:
            saver.setup()
            yield saver
        return

    with ConnectionPool(
        conninfo=uri,
        kwargs={"autocommit": True, "prepare_threshold": 0, "connect_timeout": 10},
        open=True,
    ) as pool:
        saver = PostgresSaver(pool)
        saver.setup()
        yield saver


@asynccontextmanager
async def async_checkpointer(
    kind: CheckpointerKind | str,
    *,
    path: str | Path | None = None,
    conn_string: str | None = None,
) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    """Async context manager yielding an async-compatible LangGraph checkpointer."""
    backend = _validate_backend_args("async_checkpointer", kind, path, conn_string)

    if backend == "memory":
        _warn_memory_durability()
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    if backend == "sqlite":
        db_path = _normalize_sqlite_path(path)
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-sqlite (and aiosqlite) is required for async SQLite "
                "persistence. Install it with: pip install 'langcomp[sqlite]'"
            ) from exc

        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()
            yield saver
        return

    uri = _validate_postgres_conn_string(conn_string)
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-postgres is required for async Postgres persistence. "
            "Install it with: pip install 'langcomp[postgres]'"
        ) from exc

    async with AsyncPostgresSaver.from_conn_string(uri) as saver:
        await saver.setup()
        yield saver


__all__ = ["CheckpointerKind", "async_checkpointer", "checkpointer", "managed_checkpointer"]
