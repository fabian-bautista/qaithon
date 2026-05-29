"""Base contracts every Qaithon backend must satisfy.

This module defines the abstract surface that every backend implementation
needs to expose. The compiler, the analyzer and the layers depend **only**
on these abstractions (Dependency Inversion), so adding a new backend never
requires touching the core.

Architecture
------------

* :class:`Backend` — the minimal behavioral contract: given an input tensor
  and a weight, produce the linear projection result. Whether that's computed
  on a photonic chip, on a quantum cloud, or on a CPU mock is the backend's
  business.
* :class:`BackendProfile` — declarative metadata: energy per MAC, latency per
  op, queue time, whether the backend supports autograd. The compiler reads
  this when choosing which backend to use for which op.
* :class:`BackendRegistry` — a process-wide registry mapping a string name
  (e.g. ``"mock"``, ``"quandela"``, ``"pennylane"``) to a backend factory.
  Third parties can register their own backends without modifying Qaithon.

Invariant every implementation MUST respect
-------------------------------------------

``Backend.matmul(x, weight, bias=None)`` must return a tensor of shape
``(*x.shape[:-1], weight.shape[0])`` that, in the limit of perfect simulation,
equals ``x @ weight.T + bias``. Backends are allowed to introduce noise, but
they must not change the *semantics* (shape, dtype, gradient flow).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from qaithon._logging import get_logger
from qaithon.exceptions import BackendNotRegisteredError

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

__all__ = [
    "Backend",
    "BackendFactory",
    "BackendProfile",
    "BackendRegistry",
    "HealthStatus",
    "get_backend",
    "list_backends",
    "register_backend",
]


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Result of :meth:`Backend.health_check`.

    Attributes:
        backend: Backend profile name.
        online: ``True`` when the backend is ready to accept new jobs.
        message: Vendor-provided status string (or local description).
        pending_jobs: Number of jobs queued ahead of the user, when the
            vendor exposes it. ``None`` otherwise.
        latency_ms: Wall-clock time the health probe itself took.
    """

    backend: str
    online: bool
    message: str
    pending_jobs: int | None = None
    latency_ms: float | None = None

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BackendProfile — declarative cost model
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Declarative metadata about a backend's expected cost.

    These numbers are *estimates* used by the compiler to choose between
    backends for a given op (LightCode-style dual-objective compilation).
    Backends should set realistic values; mock backends can use zeros.

    Attributes:
        name: Stable identifier exposed to users (e.g. ``"quandela.merlin"``).
        kind: Coarse-grained category. One of ``"classical"``, ``"photonic"``,
            ``"quantum"``, ``"mock"``.
        energy_pj_per_mac: Estimated energy per multiply-accumulate, in picojoules.
        latency_us_per_op: Estimated wall-clock latency per matmul op, microseconds.
        queue_us: Estimated cloud queue wait per job, microseconds. Zero for local.
        supports_autograd: Whether the backend's forward propagates gradients.
        supports_batching: Whether the backend can run on tensors with leading batch dims.
        max_dim: Maximum supported feature dimension. ``None`` means unbounded.
        notes: Free-form notes (defaults, limitations) surfaced in documentation.
    """

    name: str
    kind: str
    energy_pj_per_mac: float = 0.0
    latency_us_per_op: float = 0.0
    queue_us: float = 0.0
    supports_autograd: bool = True
    supports_batching: bool = True
    max_dim: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        valid_kinds = {"classical", "photonic", "quantum", "mock"}
        if self.kind not in valid_kinds:
            raise ValueError(
                f"BackendProfile.kind must be one of {sorted(valid_kinds)}, "
                f"got {self.kind!r}."
            )
        if self.energy_pj_per_mac < 0 or self.latency_us_per_op < 0 or self.queue_us < 0:
            raise ValueError("Cost metrics must be non-negative.")


# ---------------------------------------------------------------------------
# Backend — minimal behavioral contract
# ---------------------------------------------------------------------------
class Backend(ABC):
    """Abstract base for every backend implementation.

    A backend is responsible for executing a single primitive: the linear
    projection ``y = x @ W^T + b`` (matching ``torch.nn.functional.linear``).
    Everything else in Qaithon — layer replacement, model walking,
    HuggingFace integration — relies only on this primitive.

    Subclasses must override:
        * :attr:`profile` — class attribute or property of type :class:`BackendProfile`.
        * :meth:`matmul` — the actual computation.

    Subclasses may optionally override:
        * :meth:`is_available` — runtime check (e.g. is the optional dep installed?).
        * :meth:`warmup` — pre-allocate resources, open connections, etc.
        * :meth:`teardown` — release resources.
    """

    #: Declarative cost model. Subclasses MUST override.
    profile: BackendProfile

    @abstractmethod
    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``x @ weight.T + bias``.

        Args:
            x: Input tensor with shape ``(..., in_features)``.
            weight: Weight tensor with shape ``(out_features, in_features)``.
                Matches the layout of ``torch.nn.Linear.weight``.
            bias: Optional bias of shape ``(out_features,)``.

        Returns:
            Tensor of shape ``(..., out_features)`` matching, semantically,
            ``torch.nn.functional.linear(x, weight, bias)``.
        """

    def is_available(self) -> bool:
        """Return ``True`` when the backend can actually run on this machine.

        Default implementation returns ``True``. Backends that depend on optional
        packages or external services should override this to perform a quick
        runtime check.
        """
        return True

    def warmup(self) -> None:
        """Optional pre-allocation hook. Default: no-op."""

    def teardown(self) -> None:
        """Optional cleanup hook. Default: no-op."""

    def health_check(self) -> "HealthStatus":
        """Probe whether the backend's hardware/service is reachable and operational.

        Default implementation reuses :meth:`is_available` (a cheap local
        check). Backends that talk to cloud services should override to
        contact the vendor's status endpoint via the library they wrap.

        Returns:
            :class:`HealthStatus` indicating whether the backend is ready
            for new jobs, plus a vendor-provided status message and any
            queue size data the upstream API exposes.
        """
        ok = bool(self.is_available())
        return HealthStatus(
            backend=self.profile.name,
            online=ok,
            message="local backend" if ok else "not available",
        )

    # Nice repr for logs and error messages.
    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.profile.name!r} kind={self.profile.kind!r}>"


# ---------------------------------------------------------------------------
# Registry — Open/Closed extension point
# ---------------------------------------------------------------------------
BackendFactory = Callable[[], Backend]
"""Type alias: a zero-argument callable returning a backend instance.

Factories are preferred over already-instantiated backends because some
backends are expensive to construct (e.g. open a network session). The
registry only instantiates on demand.
"""


@dataclass(slots=True)
class BackendRegistry:
    """Process-wide mapping ``name -> BackendFactory``.

    Implemented as a regular instance (not a singleton) so tests can
    instantiate a fresh registry to avoid global state pollution.
    The module-level helpers (:func:`register_backend`, :func:`get_backend`)
    operate on the module's default registry.
    """

    _factories: dict[str, BackendFactory] = field(default_factory=dict)

    def register(
        self,
        name: str,
        factory: BackendFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a backend factory under ``name``.

        Args:
            name: Stable identifier. Convention: lowercase, dotted
                (``"quandela.merlin"``, ``"pennylane.default_qubit"``).
            factory: Zero-argument callable returning a :class:`Backend`.
            overwrite: If ``True``, an existing entry with the same name is
                replaced. If ``False`` (default), re-registration raises.

        Raises:
            ValueError: If ``name`` is already registered and ``overwrite`` is False.
        """
        if not overwrite and name in self._factories:
            raise ValueError(
                f"Backend {name!r} is already registered. "
                "Pass `overwrite=True` to replace it explicitly."
            )
        self._factories[name] = factory
        logger.debug("Registered backend %r", name)

    def get(self, name: str) -> Backend:
        """Instantiate the backend registered under ``name``.

        Raises:
            BackendNotRegisteredError: if ``name`` is not registered.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            raise BackendNotRegisteredError(name, self._factories.keys()) from None
        return factory()

    def names(self) -> Iterable[str]:
        """Return an iterable over registered backend names."""
        return tuple(self._factories.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._factories


# Module-level default registry. Consumers can use the helpers below.
_DEFAULT_REGISTRY = BackendRegistry()


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a backend factory in the default registry.

    See :meth:`BackendRegistry.register` for argument details.

    Example:
        >>> from qaithon.backends.base import Backend, BackendProfile, register_backend
        >>> class MyBackend(Backend):
        ...     profile = BackendProfile(name="mine", kind="classical")
        ...     def matmul(self, x, w, bias=None):
        ...         return x @ w.T + (bias if bias is not None else 0)
        >>> register_backend("mine", MyBackend)
    """
    _DEFAULT_REGISTRY.register(name, factory, overwrite=overwrite)


def get_backend(name: str) -> Backend:
    """Instantiate the backend registered under ``name`` in the default registry.

    Raises:
        BackendNotRegisteredError: if ``name`` is not registered.
    """
    return _DEFAULT_REGISTRY.get(name)


def list_backends() -> tuple[str, ...]:
    """Return the tuple of currently registered backend names (sorted)."""
    return tuple(sorted(_DEFAULT_REGISTRY.names()))
