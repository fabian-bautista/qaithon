"""Hardware-specific metrics beyond the generic latency / energy / fidelity.

The ``BackendProfile`` carries declarative cost numbers but no per-call
telemetry. This module defines richer metric structures that backends
populate during execution and exposes a unified :class:`InferenceMetrics`
accumulator the user can read at the end of an inference.

Three flavors, by hardware family:

* :class:`SuperconductingMetrics` — gate fidelity, T1/T2, queue time.
  Populated by ``ibm.aer``, ``ibm.heron``.
* :class:`PhotonicMetrics` — photon counts, detection efficiency, loss
  per beamsplitter. Populated by ``quandela.perceval``, ``quandela.belenos``.
* :class:`NeutralAtomMetrics` — Rabi frequency, atom rearrangement events.
  Populated by ``aws.braket.quera``.

All three are subclasses of :class:`HardwareMetrics` so callers can
iterate them generically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = [
    "HardwareMetrics",
    "InferenceMetrics",
    "NeutralAtomMetrics",
    "PhotonicMetrics",
    "SuperconductingMetrics",
]


@dataclass(frozen=True, slots=True)
class HardwareMetrics:
    """Common fields every backend can populate.

    Attributes:
        backend: Name of the backend that produced this measurement.
        latency_us: Wall-clock latency of the most recent call.
        estimated_energy_pj: Energy estimate per the cost model.
        fidelity: Quality metric vs ideal output, when available.
    """

    backend: str
    latency_us: float
    estimated_energy_pj: float
    fidelity: float | None = None


@dataclass(frozen=True, slots=True)
class SuperconductingMetrics(HardwareMetrics):
    """IBM Heron / Brisbane style metrics."""

    n_qubits_used: int = 0
    gate_count: int = 0
    avg_gate_fidelity: float | None = None
    queue_time_us: float = 0.0
    backend_class: str = "superconducting"


@dataclass(frozen=True, slots=True)
class PhotonicMetrics(HardwareMetrics):
    """Quandela Belenos / Perceval style metrics."""

    n_modes_used: int = 0
    n_photons_injected: int = 0
    n_photons_detected: int = 0
    detection_efficiency: float | None = None
    accumulated_loss: float | None = None
    backend_class: str = "photonic"

    @property
    def photons_lost(self) -> int:
        """Photons that entered the circuit but never reached the detector."""
        return max(0, self.n_photons_injected - self.n_photons_detected)


@dataclass(frozen=True, slots=True)
class NeutralAtomMetrics(HardwareMetrics):
    """QuEra Aquila style metrics."""

    n_atoms_used: int = 0
    rearrangement_success_rate: float | None = None
    rydberg_blockade_radius_um: float | None = None
    backend_class: str = "neutral_atom"


# ---------------------------------------------------------------------------
# Inference-level accumulator
# ---------------------------------------------------------------------------
@dataclass
class InferenceMetrics:
    """Aggregated metrics across one full inference (many matmuls).

    Use as a context manager wrapping ``model.generate``::

        with InferenceMetrics() as m:
            outputs = model.generate(...)
        print(m.pretty())

    Inside the context, every traced backend's measurements are merged
    here. After exit, :attr:`total_latency_us`, :attr:`total_energy_pj`,
    and :attr:`per_backend` summarize the run.
    """

    per_call: list[HardwareMetrics] = field(default_factory=list)

    def add(self, metric: HardwareMetrics) -> None:
        """Append a per-call measurement."""
        self.per_call.append(metric)

    @property
    def total_latency_us(self) -> float:
        return sum(m.latency_us for m in self.per_call)

    @property
    def total_energy_pj(self) -> float:
        return sum(m.estimated_energy_pj for m in self.per_call)

    @property
    def n_calls(self) -> int:
        return len(self.per_call)

    @property
    def per_backend(self) -> dict[str, int]:
        """Number of calls per backend."""
        counts: dict[str, int] = {}
        for m in self.per_call:
            counts[m.backend] = counts.get(m.backend, 0) + 1
        return counts

    @property
    def total_photons_detected(self) -> int:
        """Sum of photons detected across photonic backends only."""
        return sum(
            m.n_photons_detected
            for m in self.per_call
            if isinstance(m, PhotonicMetrics)
        )

    @property
    def avg_fidelity(self) -> float | None:
        """Mean fidelity across calls that reported one."""
        fids = [m.fidelity for m in self.per_call if m.fidelity is not None]
        if not fids:
            return None
        return sum(fids) / len(fids)

    def __enter__(self) -> InferenceMetrics:
        from qaithon import tracing

        self._tracing_ctx = tracing.trace()
        self._tracing_trace = self._tracing_ctx.__enter__()
        return self

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN002
        try:
            self._tracing_ctx.__exit__(*args, **kwargs)
        finally:
            # Materialize trace events as generic HardwareMetrics.
            for ev in getattr(self._tracing_trace, "events", []):
                self.add(
                    HardwareMetrics(
                        backend=ev.backend,
                        latency_us=ev.latency_us,
                        estimated_energy_pj=ev.estimated_energy_pj,
                    )
                )

    def pretty(self) -> str:
        lines = [
            "Inference metrics",
            f"  Total backend calls:  {self.n_calls}",
            f"  Total latency:        {self.total_latency_us:>10,.0f} µs",
            f"  Total energy:         {self.total_energy_pj:>10,.1f} pJ",
        ]
        if self.avg_fidelity is not None:
            lines.append(f"  Average fidelity:     {self.avg_fidelity:.4f}")
        if self.total_photons_detected:
            lines.append(f"  Photons detected:     {self.total_photons_detected:,}")
        if self.per_backend:
            lines.append("  Calls per backend:")
            for name, count in sorted(self.per_backend.items()):
                lines.append(f"    {name:25s} {count:>6}")
        return "\n".join(lines)
