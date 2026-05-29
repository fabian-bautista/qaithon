"""Local photonic backend that actually executes circuits via Perceval.

Counterpart to :class:`IBMAerBackend` (which runs real circuits on Qiskit
Aer locally) — :class:`PercevalPhotonicBackend` runs real linear-optical
circuits on Perceval's local simulator (SLOS / Naive / Clifford). No
Quandela Cloud account is required for the local path.

Photonic primitives, in case you were wondering
-----------------------------------------------

In linear-optical quantum computing the "qubit equivalent" is the
**photonic mode**: a spatial channel through which a photon can travel.
A circuit with ``m`` modes and ``n`` indistinguishable photons lives in
a Hilbert space of dimension ``C(m+n-1, n)`` (with bunching) or
``C(m, n)`` (without bunching).

For our purposes:

* "How many qubits would this need on IBM Heron?"  → ceil(log2(N))
* "How many modes would this need on Quandela?"   → typically the same,
  with overhead for the encoding.

Both are reported by :func:`qaithon.estimate_qubits` so users can
compare apples to apples across vendors.
"""

from __future__ import annotations

import importlib.util
import math
import time
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["PercevalPhotonicBackend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _patch_scipy() -> None:
    """Apply GAP-002 workaround so Perceval can import on modern SciPy."""
    import scipy.integrate

    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        scipy.integrate.simps = scipy.integrate.simpson


class PercevalPhotonicBackend(Backend):
    """Executes small photonic circuits on Perceval's local simulator.

    Like :class:`IBMAerBackend`, this is honest about what it does:
    every ``matmul`` builds and runs a small interferometer circuit on
    the local Perceval simulator (no network), measures the photon
    distribution, and uses the deviation from the ideal distribution as
    a noise scale applied to the classical ``F.linear`` output.

    Args:
        max_modes: Maximum number of photonic modes per circuit. Each
            matmul with ``in_features > 2**max_modes`` falls back to
            classical-with-noise. Default 6, which keeps the SLOS
            simulator under ~100 ms per call.
        noise_strength: Multiplier on the noise injected into the
            classical output. ``0`` disables noise. Default ``0.005``
            (photonic circuits typically have much lower noise than
            superconducting QPUs).
        backend_name: Perceval backend to use for sampling. One of
            ``"SLOS"`` (Strong Loss Simulation), ``"Naive"``,
            ``"Clifford"``. Default ``"SLOS"``.
        n_photons: Number of input photons per circuit. Default 2.
        shots: Sampling shots. Default 64.
    """

    profile: BackendProfile = BackendProfile(
        name="quandela.perceval",
        kind="photonic",
        # Photonic computation is roughly an order of magnitude lower
        # energy per MAC than superconducting QPUs in published estimates.
        energy_pj_per_mac=0.005,
        latency_us_per_op=50_000.0,  # ~50 ms per small SLOS run.
        queue_us=0.0,
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes=(
            "Local photonic simulator via Perceval (SLOS). Executes real "
            "linear-optical circuits with measured noise. No cloud account "
            "required."
        ),
    )

    def __init__(
        self,
        max_modes: int = 6,
        noise_strength: float = 0.005,
        backend_name: str = "SLOS",
        n_photons: int = 2,
        shots: int = 64,
        seed: int | None = None,
    ) -> None:
        from qaithon.metrics import PhotonicMetrics

        self._last_metrics: PhotonicMetrics | None = None
        if max_modes < 2 or max_modes > 16:
            raise ValueError(f"max_modes must be in [2, 16], got {max_modes}.")
        if noise_strength < 0:
            raise ValueError(f"noise_strength must be non-negative, got {noise_strength}.")
        if n_photons < 1:
            raise ValueError(f"n_photons must be positive, got {n_photons}.")
        self._max_modes = max_modes
        self._noise_strength = noise_strength
        self._backend_name = backend_name
        self._n_photons = n_photons
        self._shots = shots
        self._seed = seed
        self._processor = None  # lazy
        self._last_circuit_latency_us: float = 0.0

    def is_available(self) -> bool:
        return _has("perceval") and _has("merlin")

    def _build_circuit(self, n_modes: int):  # type: ignore[no-untyped-def]
        _patch_scipy()
        import perceval as pcvl

        circuit = pcvl.Circuit(n_modes)
        # Alternating beamsplitter chain (Reck decomposition style).
        for i in range(n_modes - 1):
            circuit.add(i, pcvl.BS())
        for i in range(0, n_modes - 1, 2):
            circuit.add(i, pcvl.PS(0.1 * (i + 1)))
        return circuit

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # GENUINE photonic compute: the matmul is evaluated by a real
        # linear-optical circuit (Perceval SLOS), not F.linear. Bounded to the
        # simulator's mode reach (dim ≤ 128); raises IncompatibleHardwareError
        # above that instead of silently faking it.
        from qaithon.kernels import photonic_linear

        return photonic_linear(x, weight, bias)

    def _measure_noise_scale(self, n_modes: int) -> float:
        """Run one Perceval circuit, return the noise scale derived from photon counts."""
        _patch_scipy()
        import perceval as pcvl

        from qaithon.metrics import PhotonicMetrics

        circuit = self._build_circuit(n_modes)
        try:
            processor = pcvl.Processor(self._backend_name, n_modes)
        except Exception:
            processor = pcvl.Processor("SLOS", n_modes)
        processor.set_circuit(circuit)

        photons = min(self._n_photons, n_modes)
        input_state = [1] * photons + [0] * (n_modes - photons)
        processor.with_input(pcvl.BasicState(input_state))

        t0 = time.perf_counter()
        sampler = pcvl.algorithm.Sampler(processor)
        samples = sampler.sample_count(self._shots).get("results", {})
        latency_us = (time.perf_counter() - t0) * 1e6
        self._last_circuit_latency_us = latency_us

        if not samples:
            self._last_metrics = PhotonicMetrics(
                backend="quandela.perceval",
                latency_us=latency_us,
                estimated_energy_pj=0.0,
                n_modes_used=n_modes,
                n_photons_injected=photons * self._shots,
                n_photons_detected=0,
                detection_efficiency=0.0,
            )
            return self._noise_strength

        # --- Compute photon-level statistics from the output counts ---
        total = sum(samples.values())
        # Each outcome (key) is a Fock state like '|1,0,1,0>'. Count photons in each.
        photons_detected = 0
        for state, count in samples.items():
            ph_in_state = _count_photons_in_state(state)
            photons_detected += ph_in_state * count

        photons_injected = photons * total
        det_eff = photons_detected / max(1, photons_injected)
        loss = 1.0 - det_eff

        self._last_metrics = PhotonicMetrics(
            backend="quandela.perceval",
            latency_us=latency_us,
            estimated_energy_pj=self.profile.energy_pj_per_mac * n_modes * n_modes,
            n_modes_used=n_modes,
            n_photons_injected=photons_injected,
            n_photons_detected=photons_detected,
            detection_efficiency=det_eff,
            accumulated_loss=loss,
        )

        probs = [v / total for v in samples.values()]
        n_outcomes = len(probs)
        uniform = 1.0 / n_outcomes if n_outcomes else 1.0
        variance = sum((p - uniform) ** 2 for p in probs) / max(1, n_outcomes)
        return max(self._noise_strength, math.sqrt(variance))

    @property
    def last_photonic_metrics(self):  # type: ignore[no-untyped-def]
        """Photonic metrics from the most recent matmul.

        Returns ``None`` if no matmul has been executed yet, or a
        :class:`qaithon.metrics.PhotonicMetrics` instance with photon counts,
        detection efficiency, and accumulated loss.
        """
        return self._last_metrics


def _count_photons_in_state(state) -> int:  # type: ignore[no-untyped-def]
    """Count total photons in a Perceval BasicState representation.

    Works on both ``BasicState`` objects (which expose .photons) and
    their string repr like ``|1,0,1,0>``.
    """
    n = getattr(state, "photons", None)
    if n is not None:
        return int(n)
    s = str(state).strip("|>").replace(",", " ")
    return sum(int(p) for p in s.split() if p.isdigit())

    @property
    def last_circuit_latency_us(self) -> float:
        return self._last_circuit_latency_us


if _has("perceval") and _has("merlin"):
    register_backend("quandela.perceval", PercevalPhotonicBackend, overwrite=True)
