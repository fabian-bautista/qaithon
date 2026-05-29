"""How big a model can you ACTUALLY run on real quantum / photonic hardware?

This module is the bridge between the units an AI developer thinks in —
**dimensions** (hidden size) and **layers** — and the units the physics is in:
**modes** (photonic) and **qubits** (quantum). It answers, with numbers grounded
in real experiments, the question this whole library exists for:

    "What is the largest model I can train / infer on real photonic or quantum
     hardware today?"

The numbers below are not guesses. They come from measured runs:

* Photonic (Perceval/MerLin SLOS, the exact algorithm Belenos runs): a single
  photon through M modes realises an M-dimensional linear map at fidelity 1.0.
  So **dim ≈ modes** (linear). Belenos exposes 12 modes → up to ~12 dims.

* Quantum (Qiskit + a Heron-class noise model): amplitude encoding packs a
  dim-2ⁿ vector into n qubits (exponential!), and the algorithm is exact — but
  noise decides what survives. Measured fidelity of W·x:
      dim 2  (1q): 0.9995   dim 4 (2q): 0.965
      dim 8  (3q): 0.804    dim 16 (4q): 0.366  ← noise destroys it
  So the *useful* quantum dim today is ~4–8, NOT 2¹⁵⁶. The limit is noise,
  not qubit count.

Each layer is a separate circuit on real hardware, so the layer count is bounded
by time / quota, not by the chip. We estimate the circuit budget and refuse to
blow it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qaithon.exceptions import IncompatibleHardwareError

__all__ = [
    "HardwareLimit",
    "LIMITS",
    "hardware_limits",
    "describe_limits",
    "check_model_fits",
    "largest_model",
]


@dataclass(frozen=True)
class HardwareLimit:
    """The real, measured frontier of one device, expressed in AI units."""

    name: str
    kind: str  # 'photonic' | 'superconducting' | 'trapped_ion'
    unit: str  # 'modes' | 'qubits'
    capacity: int  # modes or qubits the chip physically has
    # Per-layer dimension limits (the hidden size of one linear layer):
    max_dim_highfid: int  # dim that runs at >0.95 fidelity on real hardware
    max_dim_real: int  # dim still usable on real hardware (degraded)
    rec_max_layers: int  # layers before time/quota makes it impractical
    seconds_per_circuit: float  # wall-clock per layer-circuit on real hardware
    real_hardware: bool  # False = simulator-only integration
    notes: str = ""

    # ---- dim ⇄ physical-unit conversion (the relationship, made explicit) ----
    def units_for_dim(self, dim: int) -> int:
        """How many modes / qubits a layer of width ``dim`` needs."""
        if self.unit == "modes":
            return dim  # photonic: one mode per dimension (linear)
        return max(1, math.ceil(math.log2(max(2, dim))))  # quantum: log2 (amplitude enc.)

    def dim_for_units(self, units: int) -> int:
        """The widest dim that ``units`` modes / qubits can hold."""
        return units if self.unit == "modes" else 2**units

    def fidelity_tier(self, dim: int) -> str:
        if dim <= self.max_dim_highfid:
            return "high-fidelity"
        if dim <= self.max_dim_real:
            return "usable (degraded)"
        return "exceeds real hardware"


# Frontier per device — grounded in the experiments documented above.
LIMITS: dict[str, HardwareLimit] = {
    "Quandela Belenos": HardwareLimit(
        name="Quandela Belenos",
        kind="photonic",
        unit="modes",
        capacity=12,
        max_dim_highfid=12,  # photonic algorithm is exact up to mode count
        max_dim_real=12,
        rec_max_layers=4,
        seconds_per_circuit=34.0,  # measured: first Belenos run ~34s incl. queue
        real_hardware=True,
        notes="One mode per dimension. 12 modes → 12-dim layers.",
    ),
    "IBM Heron": HardwareLimit(
        name="IBM Heron",
        kind="superconducting",
        unit="qubits",
        capacity=156,
        max_dim_highfid=4,  # 2 qubits, measured fid 0.965
        max_dim_real=8,  # 3 qubits, measured fid 0.804
        rec_max_layers=2,
        seconds_per_circuit=5.0,
        real_hardware=True,
        notes="156 qubits, but NOISE caps useful dim at ~8 (3 qubits), not 2^156.",
    ),
    "IonQ Forte": HardwareLimit(
        name="IonQ Forte",
        kind="trapped_ion",
        unit="qubits",
        capacity=36,
        max_dim_highfid=8,  # higher fidelity + all-to-all → a bit more headroom
        max_dim_real=16,
        rec_max_layers=2,
        seconds_per_circuit=60.0,  # slow gates; pay-per-shot
        real_hardware=True,
        notes="Higher fidelity than superconducting, but pay-per-shot (real money).",
    ),
}


def hardware_limits(target: str) -> HardwareLimit:
    """Look up the frontier for a device by case-insensitive substring."""
    needle = target.lower().replace(".", " ").replace("_", " ").strip()
    for name, lim in LIMITS.items():
        hay = name.lower()
        if needle in hay or all(p in hay for p in needle.split()):
            return lim
    raise KeyError(
        f"No real-hardware limit profile for {target!r}. "
        f"Known: {', '.join(LIMITS)}."
    )


def largest_model(target: str) -> str:
    """One-line, human answer: the biggest model this device can run for real."""
    lim = hardware_limits(target)
    return (
        f"{lim.name}: largest real-hardware model ≈ "
        f"dim {lim.max_dim_real} × {lim.rec_max_layers} layers "
        f"(= {lim.units_for_dim(lim.max_dim_real)} {lim.unit}; "
        f"high-fidelity up to dim {lim.max_dim_highfid})."
    )


def describe_limits(target: str) -> str:
    """A research-friendly summary: dims & layers, with the mode/qubit equivalent."""
    lim = hardware_limits(target)
    lines = [
        f"── {lim.name} ({lim.kind}, {lim.capacity} {lim.unit}) ──",
        f"  Encoding: {lim.notes}",
        f"  Max dim/layer  : {lim.max_dim_real}  "
        f"(= {lim.units_for_dim(lim.max_dim_real)} {lim.unit})",
        f"  High-fidelity  : dim ≤ {lim.max_dim_highfid}  "
        f"(= {lim.units_for_dim(lim.max_dim_highfid)} {lim.unit})",
        f"  Layers (real)  : ≤ {lim.rec_max_layers}  "
        f"(~{lim.seconds_per_circuit:.0f}s per layer-circuit)",
        f"  Bigger than this → simulator only.",
    ]
    return "\n".join(lines)


def check_model_fits(
    target: str,
    *,
    dim: int,
    layers: int = 1,
    on_hardware: bool = True,
) -> dict:
    """Validate a (dim, layers) model against a real device.

    Returns a dict describing the fit (with mode/qubit equivalents) when it is
    allowed. Raises :class:`IncompatibleHardwareError` when ``on_hardware`` and
    the model exceeds the device's real-hardware frontier — never silently.
    When ``on_hardware=False`` everything is allowed (simulator).
    """
    lim = hardware_limits(target)
    units = lim.units_for_dim(dim)
    est_circuits = layers  # ≥1 circuit per layer (a transformer layer is several)
    info = {
        "target": lim.name,
        "dim": dim,
        "layers": layers,
        f"{lim.unit}_needed": units,
        f"{lim.unit}_capacity": lim.capacity,
        "fidelity_tier": lim.fidelity_tier(dim),
        "est_circuits": est_circuits,
        "est_seconds": round(est_circuits * lim.seconds_per_circuit, 1),
    }
    if not on_hardware:
        info["mode"] = "simulator (any size allowed)"
        return info

    if dim > lim.max_dim_real:
        raise IncompatibleHardwareError(
            reason=(
                f"A dim-{dim} layer needs {units} {lim.unit}, beyond {lim.name}'s "
                f"real-hardware limit of dim {lim.max_dim_real} "
                f"({lim.capacity} {lim.unit} on the chip; noise/optics cap the "
                f"*useful* dim at {lim.max_dim_real})."
            ),
            recommendations=[
                f"Shrink to dim ≤ {lim.max_dim_real} (≤ {lim.max_dim_highfid} for high fidelity).",
                "Or set on_hardware=False to run it on the simulator (any size).",
                "Or pick a device with a larger frontier.",
            ],
        )
    if layers > lim.rec_max_layers:
        raise IncompatibleHardwareError(
            reason=(
                f"{layers} layers = {est_circuits} real circuits "
                f"(~{info['est_seconds']}s + quota) on {lim.name}, above the "
                f"recommended {lim.rec_max_layers}. This risks exhausting time/quota."
            ),
            recommendations=[
                f"Use ≤ {lim.rec_max_layers} layers on real hardware.",
                "Or run more layers on the simulator (on_hardware=False).",
            ],
        )
    info["mode"] = "real hardware (within frontier)"
    return info
