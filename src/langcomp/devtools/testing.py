"""Deprecated re-export shim for :mod:`langcomp.testing`.

Import ``FakeModel``, ``ScriptTurn``, ``run_and_snapshot``, and ``arun_and_snapshot``
directly from :mod:`langcomp.testing` or top-level :mod:`langcomp`.
"""

from __future__ import annotations

import warnings

from langcomp.testing import FakeModel, ScriptTurn, arun_and_snapshot, run_and_snapshot

warnings.warn(
    "langcomp.devtools.testing is deprecated; import from langcomp.testing or langcomp instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["FakeModel", "ScriptTurn", "arun_and_snapshot", "run_and_snapshot"]
