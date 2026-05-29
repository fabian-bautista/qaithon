"""Lightweight observability for Qaithon-compiled models.

Inspired by LangSmith and OpenTelemetry, but standalone and zero-dependency.
Wraps a backend in a tracer that records every matmul: backend name, input
shape, wall-clock latency, and estimated energy from the backend's profile.

Use cases:

* Confirm that the selector's decisions actually translate to real wall-clock
  savings (compile-time estimate vs runtime measurement).
* Debug "why is my pipeline slow?" — see the hot layer + its backend.
* Export a JSON trace to share with collaborators or attach to an arXiv
  paper as supplementary material.

API
---

* :func:`trace` — context manager that activates collection for any backend
  wrapped with :func:`traced`.
* :func:`traced` — opt-in wrapper around a :class:`Backend`.
* :class:`Trace` — the collected record. ``.to_json()`` for export.

Example:

    >>> from qaithon.backends import get_backend
    >>> from qaithon.tracing import trace, traced
    >>> backend = traced(get_backend("quandela.sim"))
    >>> with trace() as t:
    ...     # ... do inference with `backend` somewhere
    ...     pass
    >>> t.events  # doctest: +SKIP
    [TraceEvent(backend='quandela.sim', shape=(...), latency_us=...), ...]

The context manager is thread-local — concurrent traces in different threads
do not pollute each other.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Trace", "TraceEvent", "trace", "traced"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One matmul call recorded by the tracer.

    Attributes:
        backend: Profile name of the backend that executed the matmul.
        kind: Backend kind (``"photonic"``, ``"quantum"``, ``"classical"``,
            ``"mock"``).
        input_shape: Tuple representation of ``x.shape``.
        weight_shape: Tuple representation of ``weight.shape``.
        latency_us: Wall-clock latency in microseconds, measured here.
        estimated_energy_pj: Energy estimate from the backend's profile
            applied to the actual MAC count of this call.
    """

    backend: str
    kind: str
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    latency_us: float
    estimated_energy_pj: float


@dataclass
class Trace:
    """Container for events recorded during a trace session."""

    events: list[TraceEvent] = field(default_factory=list)

    @property
    def total_latency_us(self) -> float:
        return sum(e.latency_us for e in self.events)

    @property
    def total_energy_pj(self) -> float:
        return sum(e.estimated_energy_pj for e in self.events)

    def to_json(self) -> str:
        """Serialize the trace to JSON. Suitable for sharing or archiving."""
        return json.dumps(
            {
                "events": [asdict(e) for e in self.events],
                "summary": {
                    "n_events": len(self.events),
                    "total_latency_us": self.total_latency_us,
                    "total_energy_pj": self.total_energy_pj,
                },
            },
            indent=2,
        )


# Thread-local active trace. Each thread sees its own trace independently.
_local = threading.local()


def _active_trace() -> Trace | None:
    return getattr(_local, "trace", None)


@contextmanager
def trace() -> Iterator[Trace]:
    """Activate a fresh :class:`Trace` for the duration of the ``with`` block.

    Any backend wrapped with :func:`traced` will append its events to the
    active trace. After the block exits, the trace can be inspected.

    Example:
        >>> from qaithon.tracing import trace, traced
        >>> from qaithon.backends import get_backend
        >>> backend = traced(get_backend("mock"))
        >>> with trace() as t:
        ...     import torch
        ...     _ = backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        >>> len(t.events)
        1
    """
    new = Trace()
    previous = getattr(_local, "trace", None)
    _local.trace = new
    try:
        yield new
    finally:
        _local.trace = previous


class _TracedBackend(Backend):
    """Internal wrapper exposing the same :class:`Backend` interface."""

    def __init__(self, inner: Backend) -> None:
        self._inner = inner
        self.profile = inner.profile

    def is_available(self) -> bool:
        return self._inner.is_available()

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = _active_trace()
        if active is None:
            # Tracing not active: pure passthrough, no overhead.
            return self._inner.matmul(x, weight, bias)

        t0 = time.perf_counter()
        result = self._inner.matmul(x, weight, bias)
        latency_us = (time.perf_counter() - t0) * 1e6

        macs = int(x.shape[-1]) * int(weight.shape[0])
        energy_pj = self.profile.energy_pj_per_mac * macs

        active.events.append(
            TraceEvent(
                backend=self.profile.name,
                kind=self.profile.kind,
                input_shape=tuple(x.shape),
                weight_shape=tuple(weight.shape),
                latency_us=latency_us,
                estimated_energy_pj=energy_pj,
            )
        )
        return result


def traced(backend: Backend) -> Backend:
    """Wrap a backend so calls during an active :func:`trace` are recorded.

    Outside of an active trace, the wrapper is a passthrough with negligible
    overhead.
    """
    return _TracedBackend(backend)
