"""
Typed exception hierarchy for the langcomp framework.
"""


class LangcompError(Exception):
    """Base class for all langcomp framework errors."""


class ReducerConflictError(LangcompError):
    """Raised when a State field's reducer configuration is contradictory."""


class SnapshotMismatchError(LangcompError):
    """Raised when a replayed run's output differs from its recorded snapshot."""


class ScriptExhaustedError(LangcompError):
    """Raised when FakeModel is invoked more times than its script provides."""


class ApprovalRejectedError(LangcompError):
    """Raised when an ApprovalGate rejects a tool or function execution."""
