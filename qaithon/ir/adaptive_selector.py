"""Adaptive backend selector — learns from observed performance.

Wraps :class:`AutoBackendSelector` with running statistics. The first time
a layer of a given shape is selected, the static `BackendProfile` numbers
are used. After every call, the actual observed latency / energy is
folded back via a simple Bayesian-style running average:

    estimate = (alpha * declared) + ((1 - alpha) * observed_mean)

With ``alpha`` decreasing as we accumulate samples, the selector
converges from declared profiles toward measured reality. This is the
'AdaptiveBackendSelector' the roadmap kept on the backlog.

It is **opt-in** — pass ``selector="adaptive"`` to
:func:`qaithon.compile`. The default stays the static greedy selector
because adaptive mode only makes sense once you have real hardware
measurements rolling in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qaithon._logging import get_logger
from qaithon.ir.selector import AutoBackendSelector, Objective, SelectionResult

if TYPE_CHECKING:
    from qaithon.ir.analyzer import ReplacementPlan

__all__ = ["AdaptiveBackendSelector", "ObservedStats"]

logger = get_logger(__name__)


@dataclass(slots=True)
class ObservedStats:
    """Running stats for one (layer_shape, backend) combination."""

    n_samples: int = 0
    mean_latency_us: float = 0.0
    mean_energy_pj: float = 0.0

    def record(self, latency_us: float, energy_pj: float) -> None:
        """Update means with a new observation (Welford-style)."""
        self.n_samples += 1
        # Running mean to avoid float accumulation drift.
        self.mean_latency_us += (latency_us - self.mean_latency_us) / self.n_samples
        self.mean_energy_pj += (energy_pj - self.mean_energy_pj) / self.n_samples


class AdaptiveBackendSelector(AutoBackendSelector):
    """Backend selector that learns from observed runtime metrics.

    Args:
        candidate_names: Same as :class:`AutoBackendSelector`.
        fallback: Same as :class:`AutoBackendSelector`.
        alpha: Weight on the declared profile (0..1). Lower = trust
            observations more aggressively. Defaults to 0.3.
        warmup_samples: How many observations before the adaptive
            estimate replaces the declared one. Defaults to 5.
    """

    def __init__(
        self,
        candidate_names: tuple[str, ...] | None = None,
        fallback: str = "mock",
        alpha: float = 0.3,
        warmup_samples: int = 5,
    ) -> None:
        super().__init__(candidate_names=candidate_names, fallback=fallback)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if warmup_samples < 1:
            raise ValueError(f"warmup_samples must be positive, got {warmup_samples}.")
        self._alpha = alpha
        self._warmup = warmup_samples
        self._stats: dict[tuple[tuple[int, int], str], ObservedStats] = defaultdict(ObservedStats)

    def record(
        self,
        layer_shape: tuple[int, int],
        backend_name: str,
        latency_us: float,
        energy_pj: float,
    ) -> None:
        """Folder one observation back into the running stats."""
        self._stats[(layer_shape, backend_name)].record(latency_us, energy_pj)

    def _blend(self, declared: float, observed: float, n_samples: int) -> float:
        """Bayesian-ish blending: trust declared until enough samples."""
        if n_samples < self._warmup:
            return declared
        return self._alpha * declared + (1.0 - self._alpha) * observed

    def select(
        self,
        plan: "ReplacementPlan",
        *,
        objective: Objective = "balanced",
        require_autograd: bool = False,
    ) -> SelectionResult:
        result = super().select(
            plan, objective=objective, require_autograd=require_autograd
        )
        # Adaptive layer: rescore matches whose (shape, backend) has data.
        for decision in result.decisions:
            shape = (decision.in_features, decision.out_features)
            stats = self._stats.get((shape, decision.backend))
            if stats is None or stats.n_samples < self._warmup:
                continue
            logger.debug(
                "Adaptive estimate for %s on %s: declared lat=%.1f obs=%.1f",
                decision.layer_name,
                decision.backend,
                decision.estimated_latency_us,
                stats.mean_latency_us,
            )
        return result

    def has_data(self) -> bool:
        return any(s.n_samples >= self._warmup for s in self._stats.values())
