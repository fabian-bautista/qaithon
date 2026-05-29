"""Graceful runtime fallback across multiple backends.

When a backend's :meth:`Backend.matmul` raises at inference time (network
glitch on a cloud QPU, queue timeout, hardware fault, OOM on a constrained
device), the user's model should NOT crash. The
:class:`FallbackBackend` wraps an ordered list of candidate backends and
retries through them in order until one succeeds.

Usage::

    from qaithon.backends import get_backend
    from qaithon.fallback import FallbackBackend

    backend = FallbackBackend(
        primary=get_backend("quandela.sim"),
        fallbacks=[get_backend("pennylane.sim"), get_backend("mock")],
    )
    # If primary raises, pennylane.sim is tried, then mock.

This is intentionally separate from caching and tracing — composable
wrappers, each owning one concern (Single Responsibility).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    pass

__all__ = ["FallbackBackend"]

logger = get_logger(__name__)


class FallbackBackend(Backend):
    """Try a primary backend, fall back through ordered alternates on failure.

    Failures are logged at WARNING level but never propagated unless every
    backend in the chain raises. In that case the last exception is
    re-raised so the user sees a real error rather than silently wrong
    results.

    Args:
        primary: Backend tried first.
        fallbacks: Ordered alternates tried in sequence if ``primary``
            raises. Empty list means "no fallback" (which makes this
            wrapper a no-op).
        retry_exceptions: Tuple of exception classes that trigger fallback.
            Defaults to ``(Exception,)`` (catch everything except
            ``BaseException``). Tighten for specific recoverable errors.

    Profile:
        Adopts the primary's profile, so the selector treats the
        fallback-wrapped backend the same as the primary. This is
        intentional — the wrapper exists for resilience, not to alter
        the selector's accounting.
    """

    def __init__(
        self,
        primary: Backend,
        fallbacks: Sequence[Backend] = (),
        retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        self._primary = primary
        self._fallbacks: tuple[Backend, ...] = tuple(fallbacks)
        self._retry_exceptions = retry_exceptions
        # Surface the primary's profile to the selector / report.
        self.profile = primary.profile

    def is_available(self) -> bool:
        return self._primary.is_available() or any(b.is_available() for b in self._fallbacks)

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last_exc: BaseException | None = None
        for i, backend in enumerate((self._primary, *self._fallbacks)):
            try:
                return backend.matmul(x, weight, bias)
            except self._retry_exceptions as exc:
                last_exc = exc
                next_name = (
                    self._fallbacks[i].profile.name
                    if i < len(self._fallbacks)
                    else "<none>"
                )
                logger.warning(
                    "Backend %r raised %s during matmul; falling back to %r.",
                    backend.profile.name,
                    type(exc).__name__,
                    next_name,
                )
        # All backends failed.
        assert last_exc is not None
        raise last_exc
