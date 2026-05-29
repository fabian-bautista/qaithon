"""Qaithon-specific exception hierarchy.

All exceptions raised by Qaithon descend from :class:`QaithonError`, so callers
can catch *any* Qaithon failure with a single ``except QaithonError`` clause.
More specific subclasses are provided so callers can react to particular
failure modes when they want to.

Design rules every exception in this module follows:

* The message must tell the caller **what failed** and, when possible,
  **how to fix it**. Surface variable values, available options, and
  hints to relevant features in plain language.
* No traceback chaining is hidden — when wrapping a downstream exception,
  re-raise with ``from`` so the original is preserved.
* No I/O, logging, or side effects happen inside ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "BackendError",
    "BackendNotAvailableError",
    "BackendNotRegisteredError",
    "BackendUnreachableError",
    "CompileError",
    "IncompatibleHardwareError",
    "IncompatibleModelError",
    "QaithonError",
    "QuotaExceededError",
    "UnsupportedOperationError",
]


class QaithonError(Exception):
    """Base class for every exception raised by Qaithon."""


# ---------------------------------------------------------------------------
# Backend errors
# ---------------------------------------------------------------------------
class BackendError(QaithonError):
    """Generic failure inside a backend implementation."""


class BackendNotRegisteredError(BackendError):
    """Raised when ``backend`` argument refers to an unknown name.

    Includes the requested name and the registry's currently known names so
    the user can see exactly what they had available.
    """

    def __init__(self, name: str, available: Iterable[str]) -> None:
        available_list = sorted(set(available))
        available_str = ", ".join(available_list) if available_list else "<none>"
        super().__init__(
            f"Backend {name!r} is not registered. "
            f"Available backends: {available_str}. "
            "Register a custom backend with `qaithon.backends.register_backend(...)`."
        )
        self.name = name
        self.available = tuple(available_list)


class BackendUnreachableError(BackendError):
    """Raised when a network call to a cloud backend fails.

    Distinguishes transient connectivity issues (network glitch, queue
    timeout) from permanent ones (bad credentials, missing dependency).
    Callers can decide whether to retry, fallback, or surface.
    """

    def __init__(self, backend: str, underlying: Exception | str) -> None:
        message = (
            f"Backend {backend!r} is configured but unreachable: {underlying}. "
            "Check network connectivity and credentials, then retry."
        )
        super().__init__(message)
        self.backend = backend
        self.underlying = underlying


class QuotaExceededError(BackendError):
    """Raised when a cloud backend rejects a request due to billing/quota limits.

    Includes the backend, the limit that was hit, and the resource
    (shots, minutes, USD) so users can react programmatically.
    """

    def __init__(self, backend: str, resource: str, limit: float | str | None = None) -> None:
        message = (
            f"Backend {backend!r} rejected the request: {resource} quota exhausted"
            + (f" (limit {limit})." if limit is not None else ".")
            + " Top up your account or fall back to a local backend."
        )
        super().__init__(message)
        self.backend = backend
        self.resource = resource
        self.limit = limit


class BackendNotAvailableError(BackendError):
    """Raised when a backend is registered but its runtime requirements are missing.

    Typical cause: optional dependency not installed. The message points the user
    to the corresponding ``pip install`` extra.
    """

    def __init__(self, name: str, missing: str, extra: str | None = None) -> None:
        hint = (
            f"Install it with `pip install qaithon[{extra}]`."
            if extra is not None
            else "Install the missing dependency to enable this backend."
        )
        super().__init__(
            f"Backend {name!r} is registered but cannot be initialized: "
            f"missing dependency {missing!r}. {hint}"
        )
        self.name = name
        self.missing = missing
        self.extra = extra


# ---------------------------------------------------------------------------
# Compile / transformation errors
# ---------------------------------------------------------------------------
class CompileError(QaithonError):
    """Generic failure during :func:`qaithon.compile`."""


class IncompatibleModelError(CompileError):
    """The model cannot be compiled by Qaithon under the current configuration.

    Examples that should raise this:
        * The model is already quantized with ``bitsandbytes`` or ``gguf``.
        * The model contains layer types that the chosen backend cannot
          represent and ``strict=True`` was passed.
    """

    def __init__(self, reason: str, *, hint: str | None = None) -> None:
        msg = reason if hint is None else f"{reason} Hint: {hint}"
        super().__init__(msg)
        self.reason = reason
        self.hint = hint


class IncompatibleHardwareError(QaithonError):
    """The model is too large or too deep for the targeted hardware.

    Raised by :func:`qaithon.validate_for_hardware` (and ``train(...,
    target_hardware=...)``) when the model exceeds the qubit count or the
    coherent-depth budget of the target. The error message includes the
    exact bound that was breached and lists concrete fixes — smaller dim,
    fewer layers, switch to a future hardware target.
    """

    def __init__(self, reason: str, recommendations: list[str] | None = None) -> None:
        message = reason
        if recommendations:
            message += "\nSuggestions:\n  - " + "\n  - ".join(recommendations)
        super().__init__(message)
        self.reason = reason
        self.recommendations = recommendations or []


class UnsupportedOperationError(CompileError):
    """A specific tensor operation is not supported by the chosen backend.

    Raised at compile time (during analysis) rather than at forward time, so the
    user finds out before running inference.
    """

    def __init__(self, op: str, backend: str, *, hint: str | None = None) -> None:
        msg = f"Operation {op!r} is not supported by backend {backend!r}."
        if hint is not None:
            msg = f"{msg} {hint}"
        super().__init__(msg)
        self.op = op
        self.backend = backend
