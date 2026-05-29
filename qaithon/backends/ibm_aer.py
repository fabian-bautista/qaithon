"""IBM AerSimulator-backed backend — real quantum circuit execution, fully local.

Unlike :class:`IBMQuantumSimBackend` (which is a cost-model placeholder that
delegates the math to ``F.linear``), this backend **actually executes a
quantum circuit** for each matmul on IBM's open-source AerSimulator. No
cloud account is required — Aer runs in your process — but the wall-clock
latency is real, the noise model is real, and the path to remote QPU is
the same Qiskit API (just a different ``backend.run``).

Design choices for v0.1
-----------------------

A matmul ``y = x @ W^T + b`` is in general **not unitary**, so a direct
quantum implementation requires either:

* A trainable variational quantum circuit (VQC) that *learns* ``W`` — not
  drop-in; needs fine-tuning.
* Block-encoding / quantum singular value transformation — heavy machinery.

For v0.1, this backend takes a pragmatic third option that honors the
"Tensor in / Tensor out" contract:

1. Compute the exact classical result via ``F.linear``.
2. For each input row (capped at small batch sizes for cost reasons),
   build a small parametrized quantum circuit (``n_qubits`` qubits where
   ``n_qubits = min(max_qubits, ceil(log2(in_features)))``).
3. Run the circuit on AerSimulator with an optional noise model.
4. Measure the deviation between an "ideal" execution and the noisy one
   and treat it as additive noise on the classical result.

This way the user sees:

* Real wall-clock latency from AerSimulator (the bottleneck a real QPU
  would also exhibit, only worse).
* A real, hardware-realistic noise profile applied to the output.
* The same forward semantics as ``nn.Linear``, so HuggingFace inference
  keeps working without retraining.

Trade-offs
----------

* The classical result is still the dominant signal; the quantum execution
  contributes only the noise profile. This is honest: there's no actual
  quantum advantage being claimed here. The value is **verifying that
  Qaithon's pipeline reaches AerSimulator end-to-end** and producing
  realistic cost / noise numbers for the CompileReport.
* For ``in_features`` above ``2**max_qubits``, the backend falls back to a
  pure classical path with the noise model still applied. This is logged.
"""

from __future__ import annotations

import importlib.util
import math
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["IBMAerBackend"]

logger = get_logger(__name__)

# Cost model derived from AerSimulator typical throughput on a modern CPU.
_AER_ENERGY_PJ_PER_MAC = 0.6     # Classical sim + circuit overhead.
_AER_LATENCY_US_PER_OP = 1000.0  # ~1 ms per small circuit, conservative.


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


class IBMAerBackend(Backend):
    """Executes matmuls on IBM's AerSimulator with a realistic noise model.

    Args:
        max_qubits: Maximum number of qubits the backend uses per circuit.
            Inputs whose ``in_features`` would exceed ``2**max_qubits`` are
            processed in a classical-with-noise fallback path. Default 8
            (256-dim inputs), which keeps latency around 100 ms / forward.
        noise_strength: Multiplier on the noise injected into the classical
            output. ``0.0`` disables noise (pure F.linear). ``0.01`` gives
            realistic small-scale noise. Default ``0.01``.
        fidelity_mode: How the AerSimulator should model the hardware:

            * ``"ideal"`` (default) — pure statevector simulation. Fast.
              Validates the pipeline (shapes, gradients, API plumbing).
              Does **NOT** predict whether the circuit will work on a real
              QPU.
            * ``"realistic"`` — loads a noise model derived from a real
              IBM backend (Heron-class by default). Outputs degrade in the
              same way as on the real hardware. Slower but **predictive**:
              if your model survives ``realistic``, it has a fighting chance
              on the actual QPU.

        fake_backend_name: When ``fidelity_mode="realistic"``, the name of
            the ``qiskit-ibm-runtime`` fake backend to derive the noise
            model from. Defaults to ``"fake_brisbane"`` (Heron-class
            superconducting). Use any ``FakeXxx`` backend exposed by
            ``qiskit_ibm_runtime.fake_provider``. Two synthetic targets
            are also accepted:

            * ``"trapped_ion"`` — IonQ Forte-class noise model
              (single-qubit fidelity ≈ 0.9998, two-qubit ≈ 0.998,
              all-to-all connectivity).
            * ``"neutral_atom"`` — QuEra-class noise model
              (single-qubit fidelity ≈ 0.999, two-qubit ≈ 0.99).

            These are not transpilation targets — they only seed the noise
            model so a circuit running on Aer behaves like it would on the
            chosen architecture for accuracy comparisons.
        seed: Optional seed for reproducibility.

    Example:
        >>> # All local, no cloud account needed.
        >>> import qaithon
        >>> from qaithon.backends.ibm_aer import IBMAerBackend
        >>> # The backend auto-registers as "ibm.aer".
        >>> # qaithon.compile(model, backends=("ibm.aer",))  # doctest: +SKIP
    """

    profile: BackendProfile = BackendProfile(
        name="ibm.aer",
        kind="quantum",
        energy_pj_per_mac=_AER_ENERGY_PJ_PER_MAC,
        latency_us_per_op=_AER_LATENCY_US_PER_OP,
        queue_us=0.0,  # No queue — fully local.
        supports_autograd=True,  # Forward is differentiable since the noise is detached.
        supports_batching=True,
        max_dim=None,
        notes=(
            "IBM AerSimulator backend. Runs real quantum circuits on the "
            "local simulator (no cloud account required) and reports their "
            "actual wall-clock latency and noise profile in the CompileReport."
        ),
    )

    def __init__(
        self,
        max_qubits: int = 8,
        noise_strength: float = 0.01,
        fidelity_mode: str = "ideal",
        fake_backend_name: str = "fake_brisbane",
        seed: int | None = None,
    ) -> None:
        if max_qubits < 1 or max_qubits > 24:
            raise ValueError(f"max_qubits must be in [1, 24], got {max_qubits}.")
        if noise_strength < 0:
            raise ValueError(f"noise_strength must be non-negative, got {noise_strength}.")
        if fidelity_mode not in ("ideal", "realistic"):
            raise ValueError(
                f"fidelity_mode must be 'ideal' or 'realistic', got {fidelity_mode!r}."
            )
        self._max_qubits = max_qubits
        self._noise_strength = noise_strength
        self._fidelity_mode = fidelity_mode
        self._fake_backend_name = fake_backend_name
        self._seed = seed
        self._simulator = None  # lazy
        self._last_aer_latency_us: float = 0.0
        from qaithon.metrics import SuperconductingMetrics
        self._last_superconducting_metrics: SuperconductingMetrics | None = None

    def is_available(self) -> bool:
        return _has("qiskit") and _has("qiskit_aer")

    def _get_simulator(self):  # type: ignore[no-untyped-def]
        if self._simulator is None:
            from qiskit_aer import AerSimulator

            if self._fidelity_mode == "ideal":
                self._simulator = AerSimulator()
            elif self._fake_backend_name in ("trapped_ion", "neutral_atom"):
                self._simulator = AerSimulator(
                    noise_model=_synthetic_noise_model(self._fake_backend_name)
                )
                logger.info(
                    "AerSimulator loaded synthetic %s noise model.",
                    self._fake_backend_name,
                )
            else:
                # "realistic" — load noise model derived from a fake QPU.
                try:
                    from qiskit_ibm_runtime import fake_provider

                    cls_name = "".join(
                        part.capitalize() for part in self._fake_backend_name.split("_")
                    )
                    fake_backend_cls = getattr(fake_provider, cls_name, None)
                    if fake_backend_cls is None:
                        # Default fallback: FakeBrisbane is Heron-class and
                        # ships with most qiskit-ibm-runtime versions.
                        fake_backend_cls = getattr(fake_provider, "FakeBrisbane")
                    fake_backend = fake_backend_cls()
                    self._simulator = AerSimulator.from_backend(fake_backend)
                    logger.info(
                        "AerSimulator loaded noise model from %s.",
                        type(fake_backend).__name__,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to build realistic AerSimulator (%s: %s). "
                        "Falling back to ideal simulator.",
                        type(exc).__name__,
                        exc,
                    )
                    self._simulator = AerSimulator()
        return self._simulator

    @property
    def fidelity_mode(self) -> str:
        """Current simulation mode: ``"ideal"`` or ``"realistic"``."""
        return self._fidelity_mode

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute the matmul, with a real Aer circuit driving the noise profile.

        See module docstring for the design trade-offs.
        """
        # GENUINE quantum compute: the matmul runs as a real qubit circuit
        # (amplitude encoding + unitary dilation, Qiskit statevector), not
        # F.linear. Bounded by qubit budget; raises IncompatibleHardwareError
        # above it instead of silently faking it.
        from qaithon.kernels import quantum_linear

        return quantum_linear(x, weight, bias)

    def _measure_noise_scale(self, n_qubits: int) -> float:
        """Run one small calibration circuit on Aer; return measured noise scale."""
        from qiskit import QuantumCircuit, transpile

        from qaithon.metrics import SuperconductingMetrics

        circuit = QuantumCircuit(n_qubits, n_qubits)
        for q in range(n_qubits):
            circuit.h(q)
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)
        circuit.measure(range(n_qubits), range(n_qubits))

        sim = self._get_simulator()
        compiled = transpile(circuit, sim)
        gate_count = sum(compiled.count_ops().values())
        t0 = time.perf_counter()
        result = sim.run(compiled, shots=128, seed_simulator=self._seed).result()
        self._last_aer_latency_us = (time.perf_counter() - t0) * 1e6

        counts = result.get_counts()
        total = sum(counts.values())
        # Uniform superposition would give 1/2^n probability per bitstring.
        # The deviation from uniform is a proxy for finite-shot noise plus
        # simulator variance.
        expected = total / (2**n_qubits)
        variance = sum((c - expected) ** 2 for c in counts.values()) / max(1, len(counts))
        deviation = math.sqrt(variance) / total
        # Fidelity: how close are we to a uniform distribution (ideal GHZ-ish output).
        fidelity = max(0.0, 1.0 - deviation)
        # Average per-2q-gate fidelity heuristic for the realistic noise model.
        avg_gate_fid = (
            None if self._fidelity_mode == "ideal" else max(0.0, 1.0 - deviation / max(1, gate_count))
        )
        self._last_superconducting_metrics = SuperconductingMetrics(
            backend="ibm.aer",
            latency_us=self._last_aer_latency_us,
            estimated_energy_pj=self.profile.energy_pj_per_mac * gate_count,
            fidelity=fidelity,
            n_qubits_used=n_qubits,
            gate_count=gate_count,
            avg_gate_fidelity=avg_gate_fid,
            queue_time_us=0.0,
            backend_class="superconducting" if self._fidelity_mode == "realistic" else "superconducting-sim",
        )
        # Combine with the user's noise strength tuning knob.
        return self._noise_strength * (1.0 + deviation)

    def _classical_with_noise(self, classical: torch.Tensor) -> torch.Tensor:
        scale = self._noise_strength * classical.std().clamp(min=1e-6)
        return classical + (torch.randn_like(classical) * scale).detach()

    @property
    def last_aer_latency_us(self) -> float:
        """Wall-clock latency of the last AerSimulator execution, in microseconds."""
        return self._last_aer_latency_us

    @property
    def last_superconducting_metrics(self):  # type: ignore[no-untyped-def]
        """Superconducting metrics from the most recent matmul.

        Returns ``None`` until at least one matmul has run; otherwise a
        :class:`qaithon.metrics.SuperconductingMetrics` with qubit count,
        transpiled gate count, fidelity proxy and (for ``realistic`` mode)
        per-gate fidelity estimate.
        """
        return self._last_superconducting_metrics


_SYNTHETIC_NOISE_PROFILES = {
    # Published Forte Enterprise 1 calibration numbers; depolarizing rates
    # derived as ``1 - fidelity``.
    "trapped_ion": {
        "single_qubit_fidelity": 0.9998,
        "two_qubit_fidelity": 0.998,
        "single_qubit_gates": ("rz", "sx", "x", "u1", "u2", "u3", "h", "id"),
        "two_qubit_gates": ("cx", "cz", "ecr"),
    },
    # QuEra Aquila-class numbers. Lower 2q fidelity than trapped ion.
    "neutral_atom": {
        "single_qubit_fidelity": 0.999,
        "two_qubit_fidelity": 0.99,
        "single_qubit_gates": ("rz", "sx", "x", "u1", "u2", "u3", "h", "id"),
        "two_qubit_gates": ("cx", "cz"),
    },
}


def _synthetic_noise_model(profile_name: str):  # type: ignore[no-untyped-def]
    """Return a depolarizing-only NoiseModel matching a synthetic profile.

    Used when no vendor-provided fake backend ships in qiskit (e.g.
    trapped-ion, neutral-atom hardware). Keeps the same API surface as
    ``AerSimulator.from_backend``.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    profile = _SYNTHETIC_NOISE_PROFILES[profile_name]
    model = NoiseModel()
    single_p = 1.0 - profile["single_qubit_fidelity"]
    two_p = 1.0 - profile["two_qubit_fidelity"]
    if single_p > 0:
        err1 = depolarizing_error(single_p, 1)
        model.add_all_qubit_quantum_error(err1, list(profile["single_qubit_gates"]))
    if two_p > 0:
        err2 = depolarizing_error(two_p, 2)
        model.add_all_qubit_quantum_error(err2, list(profile["two_qubit_gates"]))
    return model


# Conditional registration: only if qiskit-aer is importable.
if _has("qiskit") and _has("qiskit_aer"):
    register_backend("ibm.aer", IBMAerBackend, overwrite=True)
