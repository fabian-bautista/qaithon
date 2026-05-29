"""Side-by-side comparison of backends on a single canonical workload.

The purpose: when somebody asks "is photonic actually faster than my
GPU?" or "how does IonQ compare to Heron?", the answer should be a table,
not an opinion. :func:`compare_backends` runs the same matmul through
every available backend (including a classical GPU/CPU baseline) and
reports latency, estimated energy, and output fidelity.

Designed to be safe by default: each backend runs in its ``"profile"`` /
``"ideal"`` mode unless the user explicitly opts into real-hardware
calibration. Running the comparison against real cloud QPUs is therefore
a deliberate, single-line opt-in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends import get_backend, list_backends
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    pass

__all__ = ["BackendBenchmark", "BenchmarkResult", "compare_backends"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackendBenchmark:
    """Measured numbers for one backend on one matmul."""

    backend: str
    kind: str
    latency_us: float
    estimated_energy_pj: float
    fidelity_vs_classical: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Container for all backend measurements + the classical baseline."""

    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    classical_latency_us: float
    backends: tuple[BackendBenchmark, ...] = field(default_factory=tuple)

    def pretty(self, explain: bool = False) -> str:
        lines = [
            "Backend benchmark",
            f"  Workload: matmul {self.input_shape} × {self.weight_shape}",
            f"  Classical baseline (CPU/MPS): {self.classical_latency_us:>7.0f} µs",
            "",
            f"  {'backend':22s} {'kind':10s} {'latency µs':>11s} "
            f"{'energy pJ':>11s} {'fidelity':>10s}",
            "  " + "-" * 70,
        ]
        for b in self.backends:
            if b.error:
                lines.append(f"  {b.backend:22s}  {b.kind:10s}  ERROR: {b.error[:40]}")
                continue
            lines.append(
                f"  {b.backend:22s} {b.kind:10s} "
                f"{b.latency_us:>11.0f} {b.estimated_energy_pj:>11.2f} "
                f"{b.fidelity_vs_classical:>10.4f}"
            )

        if explain:
            lines.extend([
                "",
                "  HOW TO READ THIS TABLE:",
                "    latency µs : wall-clock time per matmul, lower = faster.",
                "    energy pJ  : estimated energy per matmul, lower = cheaper.",
                "    fidelity   : output similarity vs classical, 1.0 = identical,",
                "                 < 0.9 = likely incoherent output.",
                "",
                "    For full definitions: qaithon glossary <term>",
            ])
        return "\n".join(lines)


def compare_backends(
    in_features: int = 16,
    out_features: int = 16,
    *,
    only: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] | None = None,
    repeats: int = 3,
    seed: int = 0,
) -> BenchmarkResult:
    """Run the same matmul on every available backend; return measurements.

    Args:
        in_features: Input dim of the test matmul.
        out_features: Output dim of the test matmul.
        only: Optional whitelist of backend names. ``None`` → all available.
        exclude: Optional blacklist (e.g. ``("aws.braket.quera",)`` to
            avoid touching billed cloud QPUs).
        repeats: Number of latency measurements per backend; reported as
            the median to reject jitter.
        seed: Seed for the input + weight tensors.

    Returns:
        :class:`BenchmarkResult` with one entry per backend tested.

    Example:
        >>> import qaithon
        >>> result = qaithon.benchmarks.compare_backends(in_features=8, out_features=8)
        >>> print(result.pretty())
    """
    torch.manual_seed(seed)
    x = torch.randn(2, in_features)
    w = torch.randn(out_features, in_features)

    # Classical reference (correctness baseline).
    ref = F.linear(x, w)

    # Classical latency baseline — median of repeats.
    classical_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = F.linear(x, w)
        classical_times.append((time.perf_counter() - t0) * 1e6)
    classical_times.sort()
    classical_latency = classical_times[len(classical_times) // 2]

    benchmarks: list[BackendBenchmark] = []
    candidates = list(only) if only else list(list_backends())
    if exclude:
        candidates = [c for c in candidates if c not in exclude]

    for name in candidates:
        try:
            backend = get_backend(name)
        except Exception as exc:  # noqa: BLE001
            benchmarks.append(
                BackendBenchmark(
                    backend=name,
                    kind="?",
                    latency_us=0.0,
                    estimated_energy_pj=0.0,
                    fidelity_vs_classical=0.0,
                    error=f"instantiate: {exc}",
                )
            )
            continue
        if not _is_available(backend):
            benchmarks.append(
                BackendBenchmark(
                    backend=name,
                    kind=backend.profile.kind,
                    latency_us=0.0,
                    estimated_energy_pj=0.0,
                    fidelity_vs_classical=0.0,
                    error="not available on this machine",
                )
            )
            continue
        bench = _measure(backend, x, w, ref, repeats=repeats)
        benchmarks.append(bench)

    return BenchmarkResult(
        input_shape=tuple(x.shape),
        weight_shape=tuple(w.shape),
        classical_latency_us=classical_latency,
        backends=tuple(benchmarks),
    )


def _is_available(backend: Backend) -> bool:
    try:
        return backend.is_available()
    except Exception:  # noqa: BLE001
        return False


def _measure(
    backend: Backend,
    x: torch.Tensor,
    w: torch.Tensor,
    reference: torch.Tensor,
    repeats: int,
) -> BackendBenchmark:
    try:
        # Warm up once.
        _ = backend.matmul(x, w)
        latencies: list[float] = []
        last_out: torch.Tensor | None = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = backend.matmul(x, w)
            latencies.append((time.perf_counter() - t0) * 1e6)
            last_out = out
        latencies.sort()
        latency = latencies[len(latencies) // 2]
        macs = int(x.shape[-1]) * int(w.shape[0])
        energy = backend.profile.energy_pj_per_mac * macs
        fidelity = _cosine_fidelity(reference, last_out) if last_out is not None else 0.0
        return BackendBenchmark(
            backend=backend.profile.name,
            kind=backend.profile.kind,
            latency_us=latency,
            estimated_energy_pj=energy,
            fidelity_vs_classical=fidelity,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return BackendBenchmark(
            backend=backend.profile.name,
            kind=backend.profile.kind,
            latency_us=0.0,
            estimated_energy_pj=0.0,
            fidelity_vs_classical=0.0,
            error=f"{type(exc).__name__}: {str(exc)[:60]}",
        )


def _cosine_fidelity(ref: torch.Tensor, candidate: torch.Tensor) -> float:
    """Cosine similarity flattened over both tensors; 1.0 means identical direction."""
    a = ref.flatten().to(torch.float64)
    b = candidate.flatten().to(torch.float64)
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)
