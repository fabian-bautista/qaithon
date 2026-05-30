"""Embedded glossary that explains every term in plain AI-developer language.

The audience is **a developer who knows PyTorch and HuggingFace, not
quantum mechanics**. Every entry has:

* A short definition (one paragraph).
* An analogy to something the AI developer already knows.
* A concrete example with numbers.
* A 'rule of thumb' so the user can interpret values in their reports
  without consulting a textbook.

The glossary is intentionally opinionated rather than encyclopedic — if
you wanted Wikipedia you'd already be there. Each term is the minimum
the AI developer needs to make sense of what Qaithon reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = ["GlossaryEntry", "explain", "glossary", "list_terms"]


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """One term in the glossary.

    Attributes:
        term: Canonical name of the concept.
        short: One-line definition.
        analogy: Connection to something the AI dev already knows.
        example: A concrete numerical example.
        rule_of_thumb: How to interpret values you'll see in reports.
        also_called: Tuple of alternative names / synonyms.
    """

    term: str
    short: str
    analogy: str
    example: str
    rule_of_thumb: str
    also_called: tuple[str, ...] = ()


_GLOSSARY: dict[str, GlossaryEntry] = {}


def _add(entry: GlossaryEntry) -> None:
    _GLOSSARY[entry.term.lower()] = entry
    for alias in entry.also_called:
        _GLOSSARY[alias.lower()] = entry


# ---------------------------------------------------------------------------
# Core concepts
# ---------------------------------------------------------------------------
_add(GlossaryEntry(
    term="qubit",
    short=(
        "The fundamental unit of quantum information on a superconducting or "
        "trapped-ion QPU. Roughly equivalent to a bit but can be in a "
        "'superposition' of 0 and 1 at the same time."
    ),
    analogy=(
        "If a classical bit is a coin showing heads OR tails, a qubit is a "
        "spinning coin that is both at once until you stop it (measure it). "
        "Practical takeaway: with N qubits you can in principle represent "
        "2^N values simultaneously."
    ),
    example=(
        "IBM Heron has 156 qubits → its statevector lives in a "
        "2^156 ≈ 10^47-dimensional Hilbert space (more than there are atoms "
        "in the observable universe)."
    ),
    rule_of_thumb=(
        "For a Qaithon matmul of dimension N, you need ~log2(N) qubits "
        "for amplitude encoding. Llama-3 hidden_dim=4096 → 12 qubits."
    ),
))

_add(GlossaryEntry(
    term="photonic mode",
    short=(
        "The photonic counterpart of a qubit. A spatial channel through which "
        "a photon can travel. Computation happens via interference between "
        "photons routed through beamsplitters and phase shifters."
    ),
    analogy=(
        "Imagine N parallel optical fibres. You inject some photons in some "
        "fibres, route them through a mesh of beamsplitters, and count where "
        "they come out. The pattern of counts encodes your computation."
    ),
    example=(
        "Quandela Belenos has 12 modes. A 12-mode interferometer with 2 "
        "photons covers ~78 possible output configurations."
    ),
    rule_of_thumb=(
        "For Qaithon's purposes, treat 'photonic mode' as 'photonic qubit'. "
        "1 mode ≈ 1 qubit of capacity for our matmul circuits."
    ),
    also_called=("mode",),
))

_add(GlossaryEntry(
    term="fidelity",
    short=(
        "How close the actual output of a quantum operation is to the ideal "
        "output. 1.0 is perfect, 0.0 is unrelated noise."
    ),
    analogy=(
        "Like 'top-k accuracy' but for tensors. If you ran the same matmul "
        "classically and on the QPU, fidelity is the cosine similarity of "
        "the two outputs."
    ),
    example=(
        "Qaithon's quandela.perceval backend reports fidelity ≈ 0.987 on "
        "a small matmul — about 1.3% deviation from the classical result."
    ),
    rule_of_thumb=(
        "0.99+ = excellent; 0.95–0.99 = acceptable for noise-robust tasks "
        "(e.g. classification); < 0.9 = the model output will likely be incoherent."
    ),
))

_add(GlossaryEntry(
    term="pJ per MAC",
    short=(
        "Picojoules per multiply-accumulate operation — the energy cost of "
        "one fused multiply+add in a matmul."
    ),
    analogy=(
        "Like 'FLOPS per Watt' inverted and per-operation. Your MLPerf "
        "benchmark in joules per inference is basically (pJ/MAC) × (MACs "
        "per inference)."
    ),
    example=(
        "Modern H100 GPU: ~1.0 pJ/MAC. Lightmatter Envise estimate: "
        "~0.05 pJ/MAC. Qaithon's quandela.perceval reports ~0.005 pJ/MAC "
        "on small matmuls."
    ),
    rule_of_thumb=(
        "Lower is better. A 10× reduction at datacenter scale is "
        "millions of dollars saved per year. A 100× reduction is industry-"
        "redefining."
    ),
    also_called=("energy per MAC", "energy/MAC", "energy_pj_per_mac"),
))

_add(GlossaryEntry(
    term="coherence depth",
    short=(
        "The number of two-qubit gates a QPU can execute before its qubits "
        "lose their quantum information ('decohere'). Hard upper bound on "
        "circuit complexity."
    ),
    analogy=(
        "Like the maximum number of tokens an LLM can attend to before its "
        "KV-cache runs out — except a hard limit imposed by physics, not "
        "memory."
    ),
    example=(
        "IBM Heron: ~2,000 two-qubit gates. Past that, the output is noise. "
        "Llama-3 single matmul: ~1 million two-qubit gates required → "
        "Heron cannot execute it coherently."
    ),
    rule_of_thumb=(
        "If your model's 'estimated max depth' > QPU's coherence depth, "
        "the QPU will return garbage even if it has enough qubits. "
        "Qaithon's validator warns about this."
    ),
    also_called=("coherence", "max coherent depth"),
))

_add(GlossaryEntry(
    term="shot",
    short=(
        "One execution of a quantum circuit followed by one measurement. "
        "Because quantum measurement is probabilistic, you typically run "
        "the same circuit many times and read out a histogram."
    ),
    analogy=(
        "Like Monte Carlo sampling. One shot is one sample; with more shots "
        "you get a more accurate estimate of the underlying probability "
        "distribution."
    ),
    example=(
        "Qaithon's ibm.aer backend uses 32 shots per calibration call by "
        "default. Real-cost QPU runs may use 1,000–10,000 shots."
    ),
    rule_of_thumb=(
        "More shots = lower variance but more cost. Production cloud QPUs "
        "bill per shot: ~$0.01 per shot on IonQ, ~$0.30 per task baseline."
    ),
))

_add(GlossaryEntry(
    term="noise model",
    short=(
        "A statistical description of the errors a real QPU introduces. "
        "Includes gate errors, measurement errors, qubit decoherence, "
        "crosstalk."
    ),
    analogy=(
        "Like a dropout schedule or label noise model in training — "
        "deliberately injected randomness that the model has to be robust "
        "against to generalize from simulator to real hardware."
    ),
    example=(
        "IBM Aer's 'fake_brisbane' noise model replicates the measured "
        "noise of an actual Brisbane chip. Outputs from Aer + that noise "
        "model are predictive of what Brisbane would produce."
    ),
    rule_of_thumb=(
        "If your model works under noise_model = 'fake_brisbane' it will "
        "probably work on real Brisbane. If not, it will likely fail there too."
    ),
))

_add(GlossaryEntry(
    term="queue time",
    short=(
        "Wait time before your circuit actually starts executing on a cloud "
        "QPU. Independent of the quantum computation itself."
    ),
    analogy=(
        "Like an AWS Lambda cold-start, but minutes to hours instead of "
        "milliseconds."
    ),
    example=(
        "IBM Open Plan: typical queue 30 seconds–10 minutes for Heron. "
        "Premium plans give priority. Real-hardware Llama-3 inference "
        "would queue once per matmul → impractical."
    ),
    rule_of_thumb=(
        "Latency-sensitive workloads need queue = 0 (local simulator or "
        "on-premise QPU). Batch / offline workloads can absorb queue."
    ),
    also_called=("queue", "queue_us"),
))

_add(GlossaryEntry(
    term="amplitude encoding",
    short=(
        "A way of putting classical data into a quantum register: the N "
        "components of your vector become the amplitudes of a "
        "log2(N)-qubit superposition."
    ),
    analogy=(
        "Like one-hot encoding inverted: instead of N bits each encoding "
        "one possibility, log2(N) qubits jointly encode all N possibilities "
        "at once."
    ),
    example=(
        "A 1024-dim input vector encodes into 10 qubits via amplitude "
        "encoding. Memory savings: 1024 → 10."
    ),
    rule_of_thumb=(
        "Amplitude encoding is the information-theoretic minimum and is "
        "the number Qaithon reports as 'qubits amplitude encoding'. "
        "Real implementations use 3–4× more due to block-encoding overhead."
    ),
))

_add(GlossaryEntry(
    term="block encoding",
    short=(
        "A way of representing a non-unitary matrix (like an arbitrary "
        "weight matrix) inside a quantum circuit using ancilla qubits."
    ),
    analogy=(
        "Like having to pad a non-square matrix into a square one before "
        "you can use a standard matmul kernel. The 'padding' here is "
        "additional helper qubits."
    ),
    example=(
        "An arbitrary 1024×1024 weight matrix needs ~10 amplitude qubits + "
        "~20 ancilla = ~30 total qubits to block-encode."
    ),
    rule_of_thumb=(
        "Real QPU resource cost ≈ 3× the amplitude encoding count. Qaithon "
        "reports both numbers."
    ),
    also_called=("block encoding", "QSVT"),
))

_add(GlossaryEntry(
    term="variational quantum circuit",
    short=(
        "A parameterized quantum circuit whose parameters are trained "
        "classically (via gradient descent or parameter-shift) to perform "
        "a specific task."
    ),
    analogy=(
        "Like a tiny neural network where the 'neurons' are quantum gates "
        "and the 'activations' are quantum interference patterns."
    ),
    example=(
        "PennyLane and DeepQuantum both support VQCs natively. They are the "
        "standard way to do QML experiments today."
    ),
    rule_of_thumb=(
        "VQCs are useful research targets up to ~10–20 qubits. Beyond that "
        "training time is prohibitive."
    ),
    also_called=("VQC", "variational circuit"),
))

_add(GlossaryEntry(
    term="parameter-shift rule",
    short=(
        "A trick that lets you compute exact gradients of a quantum "
        "circuit's output without using backpropagation: evaluate the "
        "circuit twice (with parameter ± π/2)."
    ),
    analogy=(
        "Like finite-difference gradients but exact for the cosine/sine "
        "trig structure of quantum gates. Costs ~2 forward passes per "
        "parameter — gets expensive fast."
    ),
    example=(
        "Training a VQC with 1,000 parameters via parameter-shift takes "
        "~2,000 circuit executions per gradient step. At 30s/circuit on "
        "Heron → 16 hours per step."
    ),
    rule_of_thumb=(
        "Parameter-shift is only viable for tiny models. For larger ones, "
        "train classically and use the genuine kernel for inference."
    ),
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def glossary(term: str) -> GlossaryEntry:
    """Look up a single term. Case-insensitive, accepts aliases.

    Args:
        term: Term to look up. ``"qubit"``, ``"pJ per MAC"``,
            ``"fidelity"``, ``"mode"`` etc.

    Raises:
        KeyError: If the term is not in the glossary.

    Example:
        >>> from qaithon.glossary import glossary
        >>> entry = glossary("fidelity")
        >>> print(entry.short)
    """
    key = term.lower().strip()
    if key not in _GLOSSARY:
        raise KeyError(
            f"No glossary entry for {term!r}. Use qaithon.glossary.list_terms() "
            "to see what's available."
        )
    return _GLOSSARY[key]


def list_terms() -> tuple[str, ...]:
    """Return every glossary term (canonical names only, sorted)."""
    seen: set[str] = set()
    result: list[str] = []
    for value in _GLOSSARY.values():
        if value.term not in seen:
            seen.add(value.term)
            result.append(value.term)
    return tuple(sorted(result))


def explain(metric: str, value: float | None = None) -> str:
    """Plain-language explanation of a metric, optionally with the user's value.

    Args:
        metric: Metric name (``"fidelity"``, ``"pJ per MAC"``, ``"qubits"``,
            etc.). Same lookups as :func:`glossary`.
        value: Optional numeric value the user observed. When provided,
            the explanation is contextualized ("your value is X, which means…").

    Returns:
        Multi-line human-readable string suitable for printing.

    Example:
        >>> from qaithon.glossary import explain
        >>> print(explain("fidelity", 0.987))
    """
    entry = glossary(metric)
    lines = [
        f"{entry.term.upper()}",
        f"  {entry.short}",
        "",
        f"  Analogy:    {entry.analogy}",
        f"  Example:    {entry.example}",
        f"  Rule of thumb: {entry.rule_of_thumb}",
    ]
    if value is not None:
        interpretation = _interpret(entry.term.lower(), value)
        if interpretation:
            lines.extend(["", f"  Your value:  {value}", f"  Interpretation: {interpretation}"])
    return "\n".join(lines)


def _interpret(term: str, value: float) -> str:
    """Heuristic: classify the user's number against the rule-of-thumb."""
    if term == "fidelity":
        if value >= 0.99:
            return "Excellent — quantum output matches classical to within rounding."
        if value >= 0.95:
            return "Acceptable — small degradation, fine for robust tasks like classification."
        if value >= 0.9:
            return "Borderline — the output degrades noticeably at this noise level."
        return "Too low — output is likely incoherent. Try a shallower circuit or another backend."
    if term in ("pj per mac", "energy_pj_per_mac"):
        if value <= 0.01:
            return "Excellent — order of magnitude better than GPU baseline."
        if value <= 0.1:
            return "Good — measurable advantage vs ~1 pJ/MAC GPU."
        if value <= 1.0:
            return "Comparable to GPU baseline."
        return "Worse than classical GPU — does not justify offload purely for energy."
    if term == "qubit":
        if value <= 30:
            return "Fits on any modern QPU (Heron 156, Brisbane 133)."
        if value <= 156:
            return "Fits on IBM Heron and similar."
        return "Exceeds NISQ-era hardware; needs fault-tolerant QPUs (Starling 2029)."
    return ""
