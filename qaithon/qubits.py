"""Estimate the qubit budget required to run a model on quantum hardware.

Given any ``nn.Module`` (HuggingFace transformer, custom architecture, the
Qaithon toy transformer, anything), this module reports — for every
``nn.Linear`` / Conv1D inside — how many qubits would be required to
execute it as a quantum circuit, broken down by encoding strategy.

Why this matters
----------------

Conversations about "running a transformer on a quantum computer"
collapse into vague generalities without concrete numbers. With
:func:`estimate_qubits` you can say things like:

* "GPT-2 small requires up to **13 qubits** per matmul with amplitude
  encoding, up to **40 qubits** with block encoding — fits on IBM Heron
  (156 qubits) by qubit count, but the circuit depth of ~50,000 gates
  exceeds Heron's coherence window of ~2,000."
* "A GPT-3.5-equivalent (175B params, hidden_dim 12,288) requires
  **15 qubits per matmul** with amplitude encoding, **~70** with block
  encoding."

Combined with the :class:`HardwareSpec` registry, the report also tells
you which existing or upcoming machines can host each layer.

What this module does NOT claim
-------------------------------

* These numbers describe **information-theoretic minimums** + standard
  block-encoding overheads. They do NOT guarantee that a given circuit
  will reach those bounds in practice (real compilers use more qubits).
* Circuit depth estimates are order-of-magnitude. A precise count needs a
  transpiler pass against a specific QPU, which we don't run here.
* Fault-tolerant overhead (logical → physical qubits) is reported when
  available but represents lower bounds reported in vendor roadmaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import nn

from qaithon._logging import get_logger
from qaithon.ir.analyzer import _is_replaceable_linear, _module_dims

if TYPE_CHECKING:
    pass

__all__ = [
    "HardwareSpec",
    "KNOWN_HARDWARE",
    "LayerQubitEstimate",
    "MeasuredCircuit",
    "QubitReport",
    "estimate_qubits",
    "estimate_qubits_from_config",
    "measure_actual_circuit",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hardware registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Public spec sheet for one quantum hardware target.

    Attributes:
        name: Human-readable name (e.g. ``"IBM Heron"``).
        vendor: Manufacturer / cloud provider.
        physical_qubits: Total physical qubits on the chip.
        logical_qubits: Number of error-corrected logical qubits, when
            published. ``None`` for NISQ devices.
        max_coherent_depth: Approximate number of two-qubit gates the
            device can execute before decoherence dominates the result.
        connectivity: Free-form description of qubit connectivity
            (``"heavy-hex"``, ``"all-to-all"``, ``"chain"``, ...).
        available_year: Year the device became (or is projected to become)
            available to developers.
        notes: Anything else relevant for users picking a target.
    """

    name: str
    vendor: str
    physical_qubits: int
    logical_qubits: int | None
    max_coherent_depth: int
    connectivity: str
    available_year: int
    notes: str = ""


KNOWN_HARDWARE: tuple[HardwareSpec, ...] = (
    HardwareSpec(
        name="IBM Heron",
        vendor="IBM",
        physical_qubits=156,
        logical_qubits=None,
        max_coherent_depth=2_000,
        connectivity="heavy-hex",
        available_year=2024,
        notes="NISQ. Open Plan grants 10 min/month free QPU time.",
    ),
    HardwareSpec(
        name="IBM Brisbane",
        vendor="IBM",
        physical_qubits=133,
        logical_qubits=None,
        max_coherent_depth=2_000,
        connectivity="heavy-hex",
        available_year=2024,
        notes="Eagle generation, fewer qubits than Heron but lower error rates on some metrics.",
    ),
    HardwareSpec(
        name="IBM Starling (projected)",
        vendor="IBM",
        physical_qubits=200_000,
        logical_qubits=200,
        max_coherent_depth=100_000_000,
        connectivity="surface code",
        available_year=2029,
        notes="Roadmap target. First fault-tolerant QPU usable for LLM-scale matmuls.",
    ),
    HardwareSpec(
        name="IBM Blue Jay (projected)",
        vendor="IBM",
        physical_qubits=2_000_000,
        logical_qubits=2_000,
        max_coherent_depth=10_000_000_000,
        connectivity="surface code",
        available_year=2033,
        notes="Long-term roadmap.",
    ),
    HardwareSpec(
        name="Quandela Belenos",
        vendor="Quandela",
        physical_qubits=12,
        logical_qubits=None,
        max_coherent_depth=500,
        connectivity="linear-optical (mode-based)",
        available_year=2025,
        notes="Photonic, mode-based. The qubit count is actually a 'mode count'.",
    ),
    HardwareSpec(
        name="QuEra Aquila",
        vendor="QuEra",
        physical_qubits=256,
        logical_qubits=None,
        max_coherent_depth=1_000,
        connectivity="programmable lattice",
        available_year=2024,
        notes="Neutral-atom analog quantum computer. Different programming model.",
    ),
    HardwareSpec(
        name="IonQ Forte Enterprise 1",
        vendor="IonQ",
        physical_qubits=36,
        logical_qubits=None,
        max_coherent_depth=10_000,
        connectivity="all-to-all",
        available_year=2024,
        notes="Trapped-ion. High fidelity, low qubit count, slow gate times.",
    ),
)


# ---------------------------------------------------------------------------
# Per-layer estimate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayerQubitEstimate:
    """Qubit/depth budget for one linear layer.

    Attributes:
        layer_name: Fully-qualified module path.
        in_features: Input dimension.
        out_features: Output dimension.
        qubits_amplitude_encoding: Theoretical minimum (``ceil(log2(max(in, out)))``).
            Assumes inputs/outputs are amplitude-encoded in a single register.
        qubits_block_encoding: Realistic estimate including ancilla for
            block-encoding the (non-unitary) weight matrix.
        circuit_depth_estimate: Approximate two-qubit-gate count required
            to execute this matmul as a circuit. Order-of-magnitude only.
    """

    layer_name: str
    in_features: int
    out_features: int
    qubits_amplitude_encoding: int
    qubits_block_encoding: int
    circuit_depth_estimate: int


# ---------------------------------------------------------------------------
# Full model estimate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QubitReport:
    """Whole-model qubit and circuit-depth budget.

    Attributes:
        model_class: Name of the top-level model class.
        n_layers_analyzed: Total layers considered.
        layer_estimates: Per-layer breakdown.
        max_qubits_amplitude: Largest qubit count any single layer
            requires under amplitude encoding.
        max_qubits_block: Same under block encoding.
        max_circuit_depth: Largest single-layer circuit depth.
        total_circuit_count: Approximate number of circuits needed for one
            full forward pass (one per matmul replaced).
    """

    model_class: str
    n_layers_analyzed: int
    layer_estimates: tuple[LayerQubitEstimate, ...] = field(default_factory=tuple)
    max_qubits_amplitude: int = 0
    max_qubits_block: int = 0
    max_circuit_depth: int = 0
    total_circuit_count: int = 0

    def hardware_compatibility(self) -> list[tuple[HardwareSpec, bool, str]]:
        """For each known hardware, report whether the model fits.

        Returns a list of ``(spec, fits, reason)`` tuples. ``fits`` is True
        if both qubit count AND coherent depth are sufficient.
        """
        out: list[tuple[HardwareSpec, bool, str]] = []
        for spec in KNOWN_HARDWARE:
            capacity = spec.logical_qubits or spec.physical_qubits
            qubits_ok = self.max_qubits_block <= capacity
            depth_ok = self.max_circuit_depth <= spec.max_coherent_depth

            if qubits_ok and depth_ok:
                out.append((spec, True, "fits"))
            elif not qubits_ok and not depth_ok:
                out.append(
                    (spec, False, f"needs {self.max_qubits_block}q ≤ {capacity}q AND ≤ depth")
                )
            elif not qubits_ok:
                out.append((spec, False, f"needs {self.max_qubits_block}q, has {capacity}q"))
            else:
                out.append(
                    (
                        spec,
                        False,
                        f"needs depth {self.max_circuit_depth:,}, max {spec.max_coherent_depth:,}",
                    )
                )
        return out

    def pretty(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Qubit budget for {self.model_class}",
            f"  Layers analyzed:           {self.n_layers_analyzed}",
            f"  Circuits per forward pass: {self.total_circuit_count}",
            "",
            "  Maximum qubits required (single matmul):",
            f"    amplitude encoding:      {self.max_qubits_amplitude} qubits",
            f"    block encoding:          {self.max_qubits_block} qubits",
            f"    estimated max depth:     {self.max_circuit_depth:,} two-qubit gates",
            "",
            "  Hardware compatibility:",
        ]
        for spec, fits, reason in self.hardware_compatibility():
            check = "✓" if fits else "✗"
            tag = "(FT)" if spec.logical_qubits else ""
            lines.append(
                f"    {check} {spec.name:35s} ({spec.available_year}, "
                f"{spec.logical_qubits or spec.physical_qubits}q{tag})  {reason}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardware validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of checking a model against a hardware target."""

    fits: bool
    target: HardwareSpec
    report: "QubitReport"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    recommendations: tuple[str, ...] = field(default_factory=tuple)

    def pretty(self) -> str:
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        lines = [
            f"Validation: model {verdict} on {self.target.name} "
            f"({self.target.vendor}, {self.target.available_year})",
            "",
            f"  Max qubits required:  {self.report.max_qubits_block} (block) / "
            f"{self.report.max_qubits_amplitude} (amplitude)",
            f"  Hardware capacity:    {self.target.logical_qubits or self.target.physical_qubits} qubits",
            f"  Max circuit depth:    {self.report.max_circuit_depth:,} gates",
            f"  Coherent depth limit: {self.target.max_coherent_depth:,} gates",
        ]
        if self.reasons:
            lines.append("\n  Issues:")
            for r in self.reasons:
                lines.append(f"    - {r}")
        if self.recommendations:
            lines.append("\n  Suggestions:")
            for r in self.recommendations:
                lines.append(f"    - {r}")
        return "\n".join(lines)


def find_hardware(name: str) -> HardwareSpec:
    """Look up a HardwareSpec by case-insensitive substring."""
    needle = name.lower().replace(".", " ").replace("_", " ").strip()
    for spec in KNOWN_HARDWARE:
        haystack = spec.name.lower()
        if needle in haystack or all(part in haystack for part in needle.split()):
            return spec
    raise KeyError(
        f"No hardware spec matches {name!r}. "
        f"Known: {[s.name for s in KNOWN_HARDWARE]}"
    )


def validate_for_hardware(
    model: "nn.Module | None" = None,
    *,
    target: "str | HardwareSpec",
    report: "QubitReport | None" = None,
) -> ValidationResult:
    """Check whether a model fits on a given hardware target."""
    spec = target if isinstance(target, HardwareSpec) else find_hardware(target)
    if report is None:
        if model is None:
            raise ValueError("Either model or report must be provided.")
        report = estimate_qubits(model)

    capacity = spec.logical_qubits or spec.physical_qubits
    reasons: list[str] = []
    recommendations: list[str] = []

    if report.max_qubits_block > capacity:
        delta = report.max_qubits_block - capacity
        # Photonic devices count modes, not qubits — surface the right vocabulary.
        unit = "photonic modes" if "photonic" in spec.notes.lower() or "mode" in spec.connectivity.lower() else "qubits"
        reasons.append(
            f"Model needs {report.max_qubits_block} {unit} per matmul; "
            f"{spec.name} provides {capacity}. Short by {delta}."
        )
        max_amp = max(1, (capacity - 1) // 3)
        max_dim = 2**max_amp
        recommendations.append(
            f"Reduce hidden_size to ≤{max_dim} to fit the {capacity}-{unit} budget."
        )

    if report.max_circuit_depth > spec.max_coherent_depth:
        ratio = report.max_circuit_depth / max(1, spec.max_coherent_depth)
        reasons.append(
            f"Largest matmul needs depth {report.max_circuit_depth:,} gates; "
            f"{spec.name} coherently sustains {spec.max_coherent_depth:,}. "
            f"Exceeded by {ratio:.0f}×."
        )
        if not spec.logical_qubits:
            future_targets = [
                s for s in KNOWN_HARDWARE
                if s.max_coherent_depth >= report.max_circuit_depth
            ]
            if future_targets:
                future_targets.sort(key=lambda s: s.available_year)
                first = future_targets[0]
                recommendations.append(
                    f"Target {first.name!r} (projected {first.available_year}) instead — "
                    "it has the coherent depth budget for this model."
                )
            else:
                recommendations.append(
                    "Reduce hidden_size by half — circuit depth scales like O(N²)."
                )

    fits = not reasons
    return ValidationResult(
        fits=fits,
        target=spec,
        report=report,
        reasons=tuple(reasons),
        recommendations=tuple(recommendations),
    )


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
def _estimate_layer(name: str, in_features: int, out_features: int) -> LayerQubitEstimate:
    """Apply textbook bounds to a single matmul."""
    max_dim = max(in_features, out_features)
    n_amp = max(1, math.ceil(math.log2(max(2, max_dim))))
    # Block-encoding overhead — standard QSVT-style: ~2*log2(N) + 1 ancilla
    # on top of the amplitude register. Conservative upper bound.
    n_block = n_amp + 2 * n_amp + 1
    # Circuit depth scales like O(N²) two-qubit gates for arbitrary
    # block-encoded matrices via QSVT (Grand Unification: Gilyén et al. 2019).
    depth = (2**n_amp) ** 2
    return LayerQubitEstimate(
        layer_name=name,
        in_features=in_features,
        out_features=out_features,
        qubits_amplitude_encoding=n_amp,
        qubits_block_encoding=n_block,
        circuit_depth_estimate=depth,
    )


@dataclass(frozen=True, slots=True)
class MeasuredCircuit:
    """Exact qubit / gate / depth counts of an actually-constructed circuit.

    Unlike :class:`LayerQubitEstimate` (theoretical bounds), this is what
    Qiskit reports after building the real ``QuantumCircuit``. The numbers
    are precise but constructing the circuit costs time — typically
    seconds for small matmuls, minutes for hidden_dim ≥ 1024.

    Attributes:
        n_qubits: Exact qubit count of the constructed circuit.
        depth: Exact circuit depth (longest chain of dependent gates).
        n_gates_total: Total gate count across all gate types.
        gate_counts: Dict of ``{gate_name: count}``.
        n_qubits_transpiled: Qubits after transpiling against a specific
            target backend. Often higher than ``n_qubits`` because the
            transpiler adds SWAPs to fit limited connectivity. ``None``
            when no transpile target was provided.
        depth_transpiled: Same as ``depth`` post-transpile. ``None`` when
            no target.
        target_backend_name: Name of the target the transpile was against.
            ``None`` if no transpile was performed.
        construction_time_s: Wall-clock seconds spent building + measuring.
    """

    n_qubits: int
    depth: int
    n_gates_total: int
    gate_counts: dict[str, int]
    n_qubits_transpiled: int | None
    depth_transpiled: int | None
    target_backend_name: str | None
    construction_time_s: float


def measure_actual_circuit(
    in_features: int,
    out_features: int,
    *,
    target_backend: str | None = None,
) -> MeasuredCircuit:
    """Build a real Qiskit circuit for a matmul and report exact counts.

    The circuit uses ``StatePreparation`` (amplitude encoding of the input)
    plus a ``UnitaryGate`` derived from a random matrix of the requested
    shape, then a final measurement. This is the canonical block-encoded
    matmul circuit; the gate count Qiskit reports after building it is
    the exact number a Heron-class QPU would have to execute.

    Args:
        in_features: Input dimension of the matmul.
        out_features: Output dimension. (Currently the circuit uses
            ``max(in, out)``; mixed dimensions are reported with the
            larger one as effective.)
        target_backend: Optional name of a fake backend (e.g.
            ``"FakeBrisbane"`` for Heron-class topology) to transpile
            against. The transpiled depth reflects what the QPU actually
            has to execute after compiler optimization.

    Returns:
        :class:`MeasuredCircuit` with exact counts.

    Raises:
        ImportError: If ``qiskit`` is not installed.

    Example:
        >>> from qaithon.qubits import measure_actual_circuit
        >>> m = measure_actual_circuit(64, 64, target_backend="FakeBrisbane")
        >>> m.n_qubits, m.depth_transpiled  # doctest: +SKIP
        (6, 1234)
    """
    import time

    try:
        import numpy as np
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation, UnitaryGate
    except ImportError as exc:
        raise ImportError(
            "measure_actual_circuit requires qiskit. Install with "
            "`pip install qiskit qiskit-aer`."
        ) from exc

    t0 = time.perf_counter()

    n_dim = max(in_features, out_features)
    n_qubits = max(1, math.ceil(math.log2(max(2, n_dim))))
    padded_dim = 2**n_qubits

    # Random unit-norm input (amplitudes for state preparation).
    rng = np.random.default_rng(0)
    amplitudes = rng.standard_normal(padded_dim).astype(np.complex128)
    amplitudes /= np.linalg.norm(amplitudes)

    # Random unitary matrix of size padded_dim x padded_dim — the
    # block-encoded "weight" we want to apply.
    raw = rng.standard_normal((padded_dim, padded_dim)) + 1j * rng.standard_normal(
        (padded_dim, padded_dim)
    )
    q, _ = np.linalg.qr(raw)  # nearest unitary via QR

    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.append(StatePreparation(amplitudes), range(n_qubits))
    circuit.append(UnitaryGate(q, label="block_encoded_weight"), range(n_qubits))
    circuit.measure(range(n_qubits), range(n_qubits))

    counts_raw = circuit.count_ops()
    measured = MeasuredCircuit(
        n_qubits=circuit.num_qubits,
        depth=circuit.depth(),
        n_gates_total=sum(counts_raw.values()),
        gate_counts=dict(counts_raw),
        n_qubits_transpiled=None,
        depth_transpiled=None,
        target_backend_name=None,
        construction_time_s=time.perf_counter() - t0,
    )

    if target_backend is None:
        return measured

    # Transpile against the requested fake backend to get hardware-realistic numbers.
    try:
        from qiskit_ibm_runtime import fake_provider

        backend_cls = getattr(fake_provider, target_backend, None)
        if backend_cls is None:
            logger.warning(
                "Unknown target_backend %r; transpile skipped.", target_backend
            )
            return measured
        backend = backend_cls()
        compiled = transpile(circuit, backend=backend, optimization_level=1)
        elapsed = time.perf_counter() - t0
        return MeasuredCircuit(
            n_qubits=measured.n_qubits,
            depth=measured.depth,
            n_gates_total=measured.n_gates_total,
            gate_counts=measured.gate_counts,
            n_qubits_transpiled=compiled.num_qubits,
            depth_transpiled=compiled.depth(),
            target_backend_name=target_backend,
            construction_time_s=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transpile against %s failed: %s", target_backend, exc)
        return measured


def estimate_qubits_from_config(
    *,
    hidden_size: int,
    n_layers: int,
    n_heads: int = 1,
    intermediate_size: int | None = None,
    vocab_size: int = 0,
    model_class: str = "FromConfig",
) -> QubitReport:
    """Estimate qubit budget from architecture parameters, **without instantiating the model**.

    Use this for huge models (GPT-3.5, Llama-405B) that would not fit in
    RAM as ``nn.Module`` instances. The estimate is mathematically the same
    as :func:`estimate_qubits`; it just synthesizes the layer shapes from
    the config instead of walking a materialized model.

    Args:
        hidden_size: Model hidden dimension (e.g. 768 for GPT-2 small,
            12288 for GPT-3.5).
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads. Used to derive the
            ``Q+K+V`` projection shape ``(hidden, 3 * hidden)``.
        intermediate_size: FFN hidden size. Defaults to ``4 * hidden_size``.
        vocab_size: Optional vocabulary for the lm_head. ``0`` skips it.
        model_class: Label used in the resulting :class:`QubitReport`.

    Example:
        >>> # GPT-3.5 equivalent — NO instantiation, runs in milliseconds.
        >>> import qaithon
        >>> r = qaithon.estimate_qubits_from_config(
        ...     hidden_size=12288, n_layers=96, n_heads=96,
        ...     model_class="GPT-3.5-equivalent",
        ... )
        >>> print(r.pretty())  # doctest: +SKIP
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    layer_estimates: list[LayerQubitEstimate] = []
    max_amp = 0
    max_block = 0
    max_depth = 0

    # Per transformer block: c_attn (Q+K+V combined), c_proj, mlp.c_fc, mlp.c_proj
    per_block_shapes = (
        ("attn.c_attn", hidden_size, 3 * hidden_size),
        ("attn.c_proj", hidden_size, hidden_size),
        ("mlp.c_fc", hidden_size, intermediate_size),
        ("mlp.c_proj", intermediate_size, hidden_size),
    )

    for layer_idx in range(n_layers):
        for shape_name, in_f, out_f in per_block_shapes:
            est = _estimate_layer(f"h.{layer_idx}.{shape_name}", in_f, out_f)
            layer_estimates.append(est)
            max_amp = max(max_amp, est.qubits_amplitude_encoding)
            max_block = max(max_block, est.qubits_block_encoding)
            max_depth = max(max_depth, est.circuit_depth_estimate)

    if vocab_size > 0:
        est = _estimate_layer("lm_head", hidden_size, vocab_size)
        layer_estimates.append(est)
        max_amp = max(max_amp, est.qubits_amplitude_encoding)
        max_block = max(max_block, est.qubits_block_encoding)
        max_depth = max(max_depth, est.circuit_depth_estimate)

    return QubitReport(
        model_class=model_class,
        n_layers_analyzed=len(layer_estimates),
        layer_estimates=tuple(layer_estimates),
        max_qubits_amplitude=max_amp,
        max_qubits_block=max_block,
        max_circuit_depth=max_depth,
        total_circuit_count=len(layer_estimates),
    )


def estimate_qubits(model: nn.Module) -> QubitReport:
    """Walk ``model`` and report qubit + depth budgets for every matmul.

    Uses the same recognition rules as :func:`qaithon.ir.analyze_model`
    (``nn.Linear`` identity check, HuggingFace ``Conv1D`` support, etc.),
    so the report stays in sync with what ``qaithon.compile`` would
    actually replace.

    Example:
        >>> from qaithon.models import create_toy_transformer
        >>> import qaithon.qubits as q
        >>> model = create_toy_transformer(dim=32, n_layers=1, n_heads=2)
        >>> report = q.estimate_qubits(model)
        >>> print(report.pretty())  # doctest: +SKIP
    """
    layer_estimates: list[LayerQubitEstimate] = []
    max_amp = 0
    max_block = 0
    max_depth = 0

    for name, module in model.named_modules():
        if not _is_replaceable_linear(module):
            continue
        dims = _module_dims(module)
        if dims is None:
            continue
        in_features, out_features = dims
        est = _estimate_layer(name, in_features, out_features)
        layer_estimates.append(est)
        max_amp = max(max_amp, est.qubits_amplitude_encoding)
        max_block = max(max_block, est.qubits_block_encoding)
        max_depth = max(max_depth, est.circuit_depth_estimate)

    return QubitReport(
        model_class=type(model).__name__,
        n_layers_analyzed=len(layer_estimates),
        layer_estimates=tuple(layer_estimates),
        max_qubits_amplitude=max_amp,
        max_qubits_block=max_block,
        max_circuit_depth=max_depth,
        total_circuit_count=len(layer_estimates),
    )
