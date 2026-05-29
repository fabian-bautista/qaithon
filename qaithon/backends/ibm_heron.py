"""IBM Quantum (Heron QPU) backend with real cloud dispatch in calibrate mode.

This backend uses the credentials loaded from ``.env`` via
:mod:`qaithon.config` and submits real circuits to one of IBM's Heron QPUs
when ``mode="calibrate"``. In ``mode="profile"`` (default) it costs nothing
and behaves identically to the classical baseline — that is what
``qaithon.compile`` uses for routine inference. The user opts into
``calibrate`` explicitly when they want a real-hardware reading.

Quota awareness: the Open Plan grants 10 minutes of QPU time per month.
Each calibrate call dispatches one tiny circuit (~3 qubits, 32 shots,
sub-second). Practical budget: ~600 calibrate calls per month before the
cap. The library refuses to dispatch full-execute mode for now to avoid
accidentally burning the quota.
"""

from __future__ import annotations

import importlib.util
import math
import time
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends._realhw_common import BackendMode, RealHardwareBackendBase
from qaithon.backends.base import BackendProfile, register_backend
from qaithon.config import get_ibm_quantum_credentials

if TYPE_CHECKING:
    pass

__all__ = ["IBMHeronBackend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


class IBMHeronBackend(RealHardwareBackendBase):
    """IBM Heron QPU backend with real cloud calibration on demand.

    Args:
        mode: Operation mode. ``"profile"`` (default) costs nothing.
            ``"calibrate"`` dispatches one small circuit per forward call.
            ``"execute"`` is not implemented for v0.1 — too expensive to
            be useful without optimization passes.
        shots: Number of shots per calibration circuit. Defaults to 32 —
            cheap and enough for a noise-scale estimate.
        backend_preference: Optional ordered tuple of QPU names to prefer
            (e.g. ``("ibm_kingston", "ibm_marrakesh")``). When ``None``,
            the runtime picks the least-busy backend.
    """

    profile: BackendProfile = BackendProfile(
        name="ibm.heron",
        kind="quantum",
        energy_pj_per_mac=0.01,  # Heron is power-efficient compared to GPU.
        latency_us_per_op=5_000.0,  # ~5 ms per circuit including transpile.
        queue_us=30_000_000.0,  # 30 s typical queue (Open Plan).
        supports_autograd=False,  # QPUs do not expose autograd natively.
        supports_batching=False,
        max_dim=156,
        notes=(
            "IBM Heron QPU (real superconducting hardware via IBM Quantum). "
            "Dispatches real calibration circuits in mode='calibrate'. "
            "mode='execute' is intentionally disabled to protect Open Plan quota."
        ),
    )

    def __init__(
        self,
        mode: BackendMode = "profile",
        shots: int = 32,
        backend_preference: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        self._shots = shots
        self._backend_preference = backend_preference
        self._service = None  # lazy
        self._backend = None  # lazy

    def health_check(self):  # type: ignore[no-untyped-def]
        """Query the IBM Quantum service for the chosen backend's live status."""
        from qaithon.backends.base import HealthStatus
        import time as _t

        if not self.is_available():
            return HealthStatus(
                backend="ibm.heron",
                online=False,
                message="qiskit-ibm-runtime or IBM token not configured",
            )
        t0 = _t.perf_counter()
        try:
            backend = self._pick_backend()
            status = backend.status()
            latency_ms = (_t.perf_counter() - t0) * 1000.0
            return HealthStatus(
                backend="ibm.heron",
                online=bool(status.operational),
                message=status.status_msg or "operational",
                pending_jobs=int(status.pending_jobs),
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(
                backend="ibm.heron",
                online=False,
                message=f"{type(exc).__name__}: {exc}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )

    def is_available(self) -> bool:
        if not (_has("qiskit") and _has("qiskit_ibm_runtime")):
            return False
        token, _channel, _instance = get_ibm_quantum_credentials()
        return bool(token)

    def _get_service(self):  # type: ignore[no-untyped-def]
        if self._service is None:
            token, channel, instance = get_ibm_quantum_credentials()
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs: dict[str, str] = {"token": token, "channel": channel}
            if instance is not None:
                kwargs["instance"] = instance
            self._service = QiskitRuntimeService(**kwargs)
        return self._service

    def _pick_backend(self):  # type: ignore[no-untyped-def]
        if self._backend is not None:
            return self._backend
        service = self._get_service()
        if self._backend_preference:
            for name in self._backend_preference:
                try:
                    self._backend = service.backend(name)
                    return self._backend
                except Exception:  # noqa: BLE001, S110
                    continue
        # Least-busy operational backend.
        self._backend = service.least_busy(operational=True, simulator=False)
        return self._backend

    def _calibrate_once(self) -> float:
        """Run a 3-qubit GHZ circuit on Heron, measure noise vs ideal."""
        from qiskit import QuantumCircuit, transpile
        from qiskit_ibm_runtime import SamplerV2

        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure(range(3), range(3))

        backend = self._pick_backend()
        compiled = transpile(circuit, backend)

        t0 = time.perf_counter()
        sampler = SamplerV2(mode=backend)
        job = sampler.run([compiled], shots=self._shots)
        result = job.result()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._last_circuit_latency_us = elapsed_us

        # Extract bitstring counts.
        try:
            counts = result[0].data.c.get_counts()  # newer Qiskit
        except AttributeError:
            counts = result[0].data.meas.get_counts()  # fallback

        # GHZ ideal: 50% |000>, 50% |111>. Anything else is noise.
        ideal = {"000", "111"}
        total = sum(counts.values())
        unexpected = sum(c for s, c in counts.items() if s not in ideal)
        noise_fraction = unexpected / max(1, total)
        # Translate fraction-out-of-ideal into a noise scale ~ [0, ~0.5].
        return max(0.005, noise_fraction)


# Conditional registration so machines without the IBM stack don't crash.
if _has("qiskit") and _has("qiskit_ibm_runtime"):
    register_backend("ibm.heron", IBMHeronBackend, overwrite=True)
