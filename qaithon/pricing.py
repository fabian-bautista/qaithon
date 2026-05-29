"""Estimated USD cost of running operations on each cloud backend.

The numbers come from each vendor's published pricing page where
available. Where not (Quandela), we use a sensible placeholder marked
as 'unknown' so the user can override.

Costs reported here are **estimates** — actual billing depends on
exchange rates, plan tier, and per-account negotiated rates.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BackendPricing", "PRICING_TABLE", "estimate_cost_usd"]


@dataclass(frozen=True, slots=True)
class BackendPricing:
    """Public-list pricing for one backend."""

    backend: str
    free_tier_units_per_month: float | None
    cost_per_shot_usd: float | None
    cost_per_task_baseline_usd: float
    cost_per_minute_usd: float | None
    notes: str = ""


PRICING_TABLE: dict[str, BackendPricing] = {
    "ibm.heron": BackendPricing(
        backend="ibm.heron",
        free_tier_units_per_month=10 * 60,  # 10 min/month, expressed in seconds.
        cost_per_shot_usd=None,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=1.60,  # Pay-as-you-go tier rate (rough).
        notes="Open Plan: 10 min/month free. Premium plans give priority queue.",
    ),
    "ibm.aer": BackendPricing(
        backend="ibm.aer",
        free_tier_units_per_month=None,  # unlimited local
        cost_per_shot_usd=0.0,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=0.0,
        notes="Local simulator. Free.",
    ),
    "aws.braket.sv1": BackendPricing(
        backend="aws.braket.sv1",
        free_tier_units_per_month=60,  # ~1 hour
        cost_per_shot_usd=None,
        cost_per_task_baseline_usd=0.30,
        cost_per_minute_usd=0.075,
        notes="Free Tier ~1 h/month. Pay $0.075/min thereafter.",
    ),
    "aws.braket.quera": BackendPricing(
        backend="aws.braket.quera",
        free_tier_units_per_month=None,
        cost_per_shot_usd=0.01,
        cost_per_task_baseline_usd=0.30,
        cost_per_minute_usd=None,
        notes="Pay-per-shot ($0.01/shot) + per-task baseline ($0.30).",
    ),
    "aws.braket.ionq": BackendPricing(
        backend="aws.braket.ionq",
        free_tier_units_per_month=None,
        cost_per_shot_usd=0.03,
        cost_per_task_baseline_usd=0.30,
        cost_per_minute_usd=None,
        notes="Higher per-shot cost than QuEra; ~$0.03/shot.",
    ),
    "quandela.belenos": BackendPricing(
        backend="quandela.belenos",
        free_tier_units_per_month=None,  # not published
        cost_per_shot_usd=None,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=None,
        notes=(
            "Quandela does not publish public pricing as of 2026. "
            "Research-gated access; ask Quandela support for your tier."
        ),
    ),
    "quandela.perceval": BackendPricing(
        backend="quandela.perceval",
        free_tier_units_per_month=None,
        cost_per_shot_usd=0.0,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=0.0,
        notes="Local simulator. Free.",
    ),
    "ibm.quantum": BackendPricing(
        backend="ibm.quantum",
        free_tier_units_per_month=10 * 60,
        cost_per_shot_usd=None,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=1.60,
        notes="Alias profile of ibm.heron pricing.",
    ),
    "mock": BackendPricing(
        backend="mock",
        free_tier_units_per_month=None,
        cost_per_shot_usd=0.0,
        cost_per_task_baseline_usd=0.0,
        cost_per_minute_usd=0.0,
        notes="Reference classical baseline. Free.",
    ),
}


def estimate_cost_usd(
    backend: str,
    *,
    n_shots: int = 1,
    n_tasks: int = 1,
    seconds: float = 0.0,
) -> float:
    """Estimate dollar cost of a backend run from published list prices.

    Args:
        backend: Backend name (must be a key in :data:`PRICING_TABLE`).
        n_shots: Total shots executed.
        n_tasks: Number of separate task submissions (each carries a
            baseline cost on AWS Braket).
        seconds: Wall-clock seconds spent on the QPU (relevant for
            IBM time-priced plans).

    Returns:
        Estimated cost in USD. ``0`` for local-only backends. ``-1`` when
        pricing is undisclosed (e.g. Quandela research-gated tier).
    """
    p = PRICING_TABLE.get(backend)
    if p is None:
        return -1.0

    # Quandela Belenos: no published pricing.
    if backend == "quandela.belenos":
        return -1.0

    total = p.cost_per_task_baseline_usd * n_tasks
    if p.cost_per_shot_usd:
        total += p.cost_per_shot_usd * n_shots
    if p.cost_per_minute_usd and seconds > 0:
        total += p.cost_per_minute_usd * (seconds / 60.0)
    return total
