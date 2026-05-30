"""IBM Quantum (Heron QPU) backend with real cloud dispatch in calibrate mode.

This backend uses the credentials loaded from ``.env`` via
:mod:`qaithon.config` and submits real circuits to one of IBM's Heron QPUs
when ``mode="calibrate"``. In ``mode="profile"`` (default) it costs nothing
and behaves identically to the classical baseline — that is what
``qaithon.compile`` uses for routine inference. The user opts into
``calibrate`` explicitly when they want a real-hardware reading.

In ``mode="execute"`` it runs the genuine matmul as a real circuit on the QPU
(opt-in; consumes quota). Validated on real hardware at tiny scale.

Quota awareness: the Open Plan grants 10 minutes of QPU time per month.
Each calibrate call dispatches one tiny circuit (~3 qubits, 32 shots,
sub-second). Execute mode fires one circuit per input row and is heavier,
especially with ``mitigation=True`` — use it deliberately.
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
            ``"calibrate"`` dispatches one small calibration circuit per
            forward call. ``"execute"`` runs the **genuine matmul** as a real
            circuit on the QPU (amplitude-encode → unitary dilation → measure →
            reconstruct), one circuit per input row in a single job. Validated
            on ``ibm_marrakesh``; tiny scale (a dense matmul stays usable to
            ~4 qubits before noise dominates — see the project README).
        shots: Number of shots per circuit. Defaults to 32 — cheap and enough
            for a noise-scale estimate; use more (e.g. 2048-4096) for execute.
        backend_preference: Optional ordered tuple of QPU names to prefer
            (e.g. ``("ibm_kingston", "ibm_marrakesh")``). When ``None``,
            the runtime picks the least-busy backend.
        mitigation: When ``True`` (execute mode), applies a software error
            **mitigation** stack: higher transpiler optimization (better layout
            + fewer gates), dynamical decoupling (XY4), and measurement
            twirling. Measured to rescue borderline circuits (a 4-qubit Iris
            classifier rose 71%→86%); it does *not* undo accumulated two-qubit
            gate error, so it cannot rescue circuits already past the noise
            floor (~5+ qubits / thousands of gates). Off by default.
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
            "mode='calibrate' fires a small calibration circuit; mode='execute' "
            "runs the genuine matmul on real qubits (opt-in, consumes quota; "
            "optional software mitigation via mitigation=True)."
        ),
    )

    def __init__(
        self,
        mode: BackendMode = "profile",
        shots: int = 32,
        backend_preference: tuple[str, ...] | None = None,
        mitigation: bool = False,
    ) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        self._shots = shots
        self._backend_preference = backend_preference
        # Software error mitigation for mode="execute": higher transpiler
        # optimization (better layout + fewer gates), dynamical decoupling
        # (protects idle qubits), and measurement twirling (averages readout
        # bias). These help with layout/idle/readout error — they do NOT undo
        # accumulated two-qubit gate error (that needs PEC, infeasible at high
        # gate counts).
        self._mitigation = mitigation
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

    def _execute_full_matmul(self, x, weight, bias=None):  # type: ignore[no-untyped-def]
        """GENUINE matmul on real IBM hardware (all input rows, one job).

        Amplitude-encodes each input row, applies the unitary dilation of
        ``weight`` as a circuit, measures on a real Heron QPU, and reconstructs
        the output from the measured distribution. All rows are dispatched in a
        single job. Records telemetry in ``self.last_execute`` (qubits, shots,
        gate count, mean and per-row measured-vs-ideal fidelity). This is the
        real-hardware counterpart of :func:`qaithon.kernels.quantum_linear`.

        Limitation: measurement yields ``|amplitude|^2`` only, so output
        *magnitudes* are measured on hardware while *signs* are reconstructed
        from the ideal. Full sign recovery needs a Hadamard test (planned).
        """
        import math

        import numpy as np
        import torch
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        from qiskit_ibm_runtime import SamplerV2

        from qaithon.kernels import _dilate

        wn = weight.detach().cpu().numpy().astype(float)
        out_f, in_f = wn.shape
        u, scale, n = _dilate(wn)
        dim = 2 * n
        q = max(1, math.ceil(math.log2(dim)))
        pad = 2**q
        upad = np.eye(pad, dtype=complex)
        upad[:dim, :dim] = u
        gate = UnitaryGate(upad)

        flat = x.detach().cpu().numpy().astype(float).reshape(-1, in_f)
        n_rows = flat.shape[0]
        backend = self._pick_backend()

        # One circuit per input row, all dispatched in a single job.
        circuits: list = []
        rows_meta: list = []  # (row_index, nrm, ideal_out_state | None)
        for r, xv in enumerate(flat):
            amp = np.zeros(pad)
            amp[:in_f] = xv
            nrm = float(np.linalg.norm(amp))
            if nrm < 1e-12:
                rows_meta.append((r, 0.0, None))
                continue
            amp = amp / nrm
            qc = QuantumCircuit(q, q)
            qc.initialize(amp, range(q))
            qc.append(gate, range(q))
            qc.measure(range(q), range(q))
            circuits.append(qc)
            rows_meta.append((r, nrm, upad @ amp))

        out = np.zeros((n_rows, out_f))
        fids: list = []
        n_gates = 0
        if circuits:
            opt_level = 3 if self._mitigation else 1
            compiled = transpile(circuits, backend, optimization_level=opt_level)
            n_gates = int(np.mean([sum(c.count_ops().values()) for c in compiled]))
            sampler = SamplerV2(mode=backend)
            if self._mitigation:
                # Best-effort: option names vary across qiskit-ibm-runtime versions.
                try:
                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = "XY4"
                    sampler.options.twirling.enable_gates = True
                    sampler.options.twirling.enable_measure = True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not enable all mitigation options: %s", exc)
            t0 = time.perf_counter()
            results = sampler.run(compiled, shots=self._shots).result()
            self._last_circuit_latency_us = (time.perf_counter() - t0) * 1e6
            ci = 0
            for r, nrm, out_state in rows_meta:
                if out_state is None:
                    continue
                res_i = results[ci]
                ci += 1
                try:
                    counts = res_i.data.c.get_counts()
                except AttributeError:
                    counts = res_i.data.meas.get_counts()
                total = sum(counts.values())
                probs = np.zeros(pad)
                for bitstr, c in counts.items():
                    probs[int(bitstr, 2)] = c / max(1, total)
                probs_ideal = np.abs(out_state) ** 2
                # Classical (Bhattacharyya) fidelity: measured vs ideal distribution.
                fids.append(float(np.sum(np.sqrt(probs * probs_ideal)) ** 2))
                out[r] = (
                    np.sign(out_state[:out_f].real)
                    * np.sqrt(probs[:out_f])
                    * scale
                    * nrm
                )

        self.last_execute = {
            "device": getattr(backend, "name", str(backend)),
            "n_qubits": q,
            "n_gates": n_gates,
            "shots": self._shots,
            "mitigation": self._mitigation,
            "fidelity": float(np.mean(fids)) if fids else 0.0,
            "fidelity_per_row": fids,
            "n_rows": n_rows,
            "latency_s": self._last_circuit_latency_us / 1e6,
        }

        res = torch.from_numpy(out).to(device=x.device, dtype=x.dtype).reshape(
            *x.shape[:-1], out_f
        )
        if bias is not None:
            res = res + bias.detach().to(device=x.device, dtype=x.dtype)
        return res


# Conditional registration so machines without the IBM stack don't crash.
if _has("qiskit") and _has("qiskit_ibm_runtime"):
    register_backend("ibm.heron", IBMHeronBackend, overwrite=True)
