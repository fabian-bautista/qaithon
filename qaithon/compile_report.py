"""Audit trail of a single :func:`qaithon.compile` invocation.

Even though Qaithon makes the optimization decisions automatically (the user
just calls ``compile(model)``), every decision is recorded in a
:class:`CompileReport`. The user never has to read it, but if they want to
know *what was done and why*, the answer is one method call away.

This satisfies the project's technical honesty rule: anything Qaithon claims
about energy or speed savings must be backed by a reproducible record.

Design rules
------------

* :class:`CompileReport` is immutable. Callers cannot mutate it after the
  fact — that would let the report drift from what actually happened.
* The report uses **AI vocabulary** (layers, models, speedup, energy
  savings), not quantum vocabulary. The user reads this and understands
  without any physics background.
* ``__repr__`` is a short one-liner suitable for logs.
  :meth:`CompileReport.pretty` returns a human-readable multi-line summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__ = ["CompileReport", "LayerDecision"]


@dataclass(frozen=True, slots=True)
class LayerDecision:
    """A single decision Qaithon made about one layer.

    Attributes:
        layer_name: Fully-qualified module path
            (``"model.layers.0.mlp.gate_proj"``).
        in_features: Input dimension of the original linear layer.
        out_features: Output dimension of the original linear layer.
        backend: Name of the backend selected for this layer.
        reason: One-line plain-text justification.
        estimated_energy_pj: Estimated energy per forward call (picojoules).
            Zero for classical / mock backends.
        estimated_latency_us: Estimated wall-clock latency per forward call
            (microseconds). Zero for classical / mock backends without queue.
    """

    layer_name: str
    in_features: int
    out_features: int
    backend: str
    reason: str
    estimated_energy_pj: float = 0.0
    estimated_latency_us: float = 0.0


@dataclass(frozen=True, slots=True)
class CompileReport:
    """Audit trail of one :func:`qaithon.compile` call.

    Returned (or attached) so the user can verify exactly what Qaithon decided
    without having to read the source.

    Attributes:
        model_class: Name of the model's top-level class
            (``"LlamaForCausalLM"``, ``"GPT2LMHeadModel"``, ...).
        n_parameters: Total parameter count of the model at compile time.
        decisions: Per-layer decisions actually applied.
        skipped: Per-layer (name, reason) pairs for layers considered but
            excluded (tied weights, already-quantized, too small, etc.).
        optimize_for: The objective the user requested (``"balanced"``,
            ``"speed"``, ``"energy"``).
        baseline_energy_pj: Sum of the classical baseline's estimated energy
            (picojoules) for the layers Qaithon touched.
        compiled_energy_pj: Sum of the chosen backends' estimated energy
            for the same layers.

    Example:
        >>> report.pretty()
        ... # 'Compiled GPT2LMHeadModel: replaced 48 layers, skipped 2 ...'
    """

    model_class: str
    n_parameters: int
    decisions: tuple[LayerDecision, ...] = field(default_factory=tuple)
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    optimize_for: str = "balanced"
    baseline_energy_pj: float = 0.0
    compiled_energy_pj: float = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def n_replaced(self) -> int:
        """Number of layers actually replaced."""
        return len(self.decisions)

    @property
    def n_skipped(self) -> int:
        """Number of layers considered but excluded."""
        return len(self.skipped)

    @property
    def estimated_energy_savings(self) -> float:
        """Estimated energy saved across all replaced layers (picojoules).

        Defined as ``baseline_energy_pj - compiled_energy_pj``. Positive means
        Qaithon picked backends with lower energy than the classical baseline.
        """
        return self.baseline_energy_pj - self.compiled_energy_pj

    @property
    def estimated_energy_savings_pct(self) -> float:
        """Estimated energy savings expressed as a percentage of the baseline.

        Returns ``0.0`` if the baseline is zero (nothing to compare against).
        """
        if self.baseline_energy_pj <= 0:
            return 0.0
        return 100.0 * self.estimated_energy_savings / self.baseline_energy_pj

    @property
    def backends_used(self) -> tuple[str, ...]:
        """Sorted tuple of distinct backend names used in this compile."""
        return tuple(sorted({d.backend for d in self.decisions}))

    # ------------------------------------------------------------------ rendering
    def __repr__(self) -> str:
        return (
            f"<CompileReport model={self.model_class!r} "
            f"replaced={self.n_replaced} skipped={self.n_skipped} "
            f"backends={list(self.backends_used)!r} "
            f"objective={self.optimize_for!r}>"
        )

    def pretty(self, explain: bool = False) -> str:
        """Return a human-readable multi-line summary of this report.

        Args:
            explain: When ``True``, inline plain-language explanations of each
                metric drawn from :mod:`qaithon.glossary`. Useful for developers
                new to quantum/photonic hardware.
        """
        lines = [
            f"Qaithon compile report for {self.model_class}",
            f"  Parameters:           {self.n_parameters:,}",
            f"  Layers replaced:      {self.n_replaced}",
            f"  Layers skipped:       {self.n_skipped}",
            f"  Objective:            {self.optimize_for}",
            f"  Backends used:        {', '.join(self.backends_used) or '<none>'}",
        ]
        if self.baseline_energy_pj > 0:
            lines.append(
                f"  Estimated energy:     "
                f"{self.compiled_energy_pj:,.1f} pJ "
                f"(baseline {self.baseline_energy_pj:,.1f} pJ, "
                f"save {self.estimated_energy_savings_pct:.1f}%)"
            )
            if explain:
                lines.append(
                    "    → pJ = picojoules; one MAC (multiply+add) costs ~1 pJ on a"
                    " modern GPU. Lower is better."
                )
        if self.decisions:
            lines.append("  Sample decisions:")
            for d in self.decisions[:5]:
                lines.append(
                    f"    {d.layer_name} [{d.in_features}->{d.out_features}] "
                    f"→ {d.backend} ({d.reason})"
                )
            if len(self.decisions) > 5:
                lines.append(f"    ... and {len(self.decisions) - 5} more")
        if self.skipped:
            lines.append("  Skipped layers:")
            for name, reason in self.skipped[:3]:
                lines.append(f"    {name} — {reason}")
            if len(self.skipped) > 3:
                lines.append(f"    ... and {len(self.skipped) - 3} more")
        if explain:
            lines.append("")
            lines.append("  ℹ  Run `qaithon glossary <term>` for any term.")
            lines.append("  ℹ  `qaithon glossary` lists every supported term.")
        return "\n".join(lines)
