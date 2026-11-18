"""
Durability-aware checkpointer factory for LangGraph agents.
Provides unified presets for memory (with explicit durability warning), SQLite, and Postgres.
"""

from __future__ import annotations

import sqlite3
import warnings
from typing import Any, Optional


def checkpointer(
    kind: str,
    *,
    path: Optional[str] = None,
    conn_string: Optional[str] = None,
) -> Any:
    """Return a configured LangGraph checkpointer for the requested backend.

    Args:
        kind: Backend preset — one of ``"memory"``, ``"sqlite"``, or ``"postgres"``.
        path: File path (or ``":memory:"``) required when ``kind="sqlite"``.
        conn_string: PostgreSQL connection URI required when ``kind="postgres"``.

    Returns:
        An initialized LangGraph checkpoint saver instance.
    """
    if kind == "memory":
        warnings.warn(
            "in-memory checkpointer: state is lost on process restart — "
            "suitable for quick scripts or unit tests, not for durable workflows.",
            UserWarning,
            stacklevel=2,
        )
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        if not path:
            raise ValueError("checkpointer('sqlite', path=...) requires 'path'")
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-sqlite is required for SQLite persistence. "
                "Install it with: pip install 'langcomp[sqlite]'"
            ) from exc

        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            saver.setup()
            return saver
        except Exception:
            conn.close()
            raise

    if kind == "postgres":
        if not conn_string:
            raise ValueError("checkpointer('postgres', conn_string=...) requires 'conn_string'")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "langgraph-checkpoint-postgres and psycopg[binary] are required for "
                "Postgres persistence. Install with: pip install 'langcomp[postgres]'"
            ) from exc

        conn = Connection.connect(
            conn_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        try:
            saver = PostgresSaver(conn)
            saver.setup()
            return saver
        except Exception:
            conn.close()
            raise

    raise ValueError(
        f"unknown checkpointer kind: {kind!r} (expected 'memory', 'sqlite', or 'postgres')"
    )
