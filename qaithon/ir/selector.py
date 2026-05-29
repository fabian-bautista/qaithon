"""Auto-selection of the best backend per layer.

This is the component that makes Qaithon usable by AI developers who do not
know quantum / photonic computing. The user calls ``qaithon.compile(model)``
with no backend argument; the :class:`AutoBackendSelector` reads:

* The set of registered backends (``BackendRegistry``).
* Each backend's declared cost model (``BackendProfile``).
* Each backend's runtime availability (``Backend.is_available()``).
* The user's optimization objective (``"speed"`` | ``"energy"`` | ``"balanced"``).
* The shape of every replaceable layer in the model.

…and produces a decision per layer: which backend gets it and why.

Algorithm (MVP — intentionally simple, replaceable later)
---------------------------------------------------------

For each candidate layer:

1. Compute its MAC count = ``in_features * out_features``.
2. For each available backend, compute the per-layer score from the chosen
   objective:

       speed:    latency_us_per_op + queue_us
       energy:   energy_pj_per_mac * macs
       balanced: normalized sum of the above

3. Pick the backend with the lowest score, subject to:
   * ``backend.profile.max_dim`` is large enough.
   * ``backend.is_available()`` returned True.
4. If no backend is feasible, fall back to ``"mock"`` (which is classical and
   always available — guarantees Qaithon never breaks the user's model).

The "right" algorithm is the LightCode-style joint scheduling, but it
requires a real ILP solver and accurate cost models from each backend.
We start with this greedy per-layer baseline; it is enough to deliver
value to AI developers while the more sophisticated path lands in v0.3.

Why a class and not a function
------------------------------

Strategy pattern — the selector is a swappable component. Tests use a fake
selector that always returns "mock"; advanced users could plug in their own
custom solver without modifying ``compile()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, get_backend, list_backends
from qaithon.compile_report import LayerDecision

if TYPE_CHECKING:
    from qaithon.ir.analyzer import LayerMatch, ReplacementPlan

__all__ = ["AutoBackendSelector", "Objective", "SelectionResult"]

logger = get_logger(__name__)

Objective = Literal["speed", "energy", "balanced"]
"""User-facing objective. The only knob non-expert users ever touch."""


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """The output of :meth:`AutoBackendSelector.select`.

    Attributes:
        per_layer: Mapping ``layer_name -> backend_instance`` actually selected.
        decisions: Per-layer audit records, used to build the CompileReport.
        baseline_energy_pj: Sum of estimated baseline (classical) energy
            across all matched layers.
        compiled_energy_pj: Sum of estimated energy after Qaithon's choices.
    """

    per_layer: dict[str, Backend]
    decisions: tuple[LayerDecision, ...]
    baseline_energy_pj: float
    compiled_energy_pj: float


# A conservative baseline number for "what classical hardware roughly costs".
# Source: H100 datasheet rough averaging (peer-reviewed numbers vary; this is
# enough for the relative comparison the report does).
_CLASSICAL_BASELINE_PJ_PER_MAC = 1.0
_CLASSICAL_BASELINE_LATENCY_US = 0.1


class AutoBackendSelector:
    """Choose the best backend per layer based on the user's objective.

    The user typically never instantiates this directly — :func:`qaithon.compile`
    does it. Exposed publicly only so advanced users (and tests) can override
    the strategy.

    Args:
        candidate_names: Restrict selection to these backend names. If
            ``None``, all registered + available backends are considered.
        fallback: Name of the backend used when no other is feasible.
            Defaults to ``"mock"`` (classical, always available).

    Example:
        >>> from qaithon.ir.selector import AutoBackendSelector
        >>> selector = AutoBackendSelector()
        >>> # ...build a ReplacementPlan...
        >>> result = selector.select(plan, objective="balanced")
        >>> result.per_layer["model.layers.0.mlp.gate_proj"]  # doctest: +SKIP
        <MockBackend ...>
    """

    def __init__(
        self,
        candidate_names: tuple[str, ...] | None = None,
        fallback: str = "mock",
    ) -> None:
        self._explicit_candidates = candidate_names
        self._fallback = fallback

    # ------------------------------------------------------------------ public
    def select(
        self,
        plan: ReplacementPlan,
        *,
        objective: Objective = "balanced",
        require_autograd: bool = False,
    ) -> SelectionResult:
        """Decide a backend per replaceable layer in ``plan``.

        Args:
            plan: Output of :func:`qaithon.ir.analyze_model`.
            objective: User's optimization priority. ``"balanced"`` (default)
                blends latency and energy; ``"speed"`` minimizes latency;
                ``"energy"`` minimizes joules.
            require_autograd: When ``True``, backends whose profile declares
                ``supports_autograd=False`` are excluded from consideration.
                The compiler sets this automatically when the model is in
                ``training`` mode — gradient-free hardware would silently
                break ``loss.backward()``.

        Returns:
            :class:`SelectionResult` with the per-layer assignment and metrics.
        """
        candidates = self._discover_candidates()
        if require_autograd:
            before = len(candidates)
            candidates = [b for b in candidates if b.profile.supports_autograd]
            removed = before - len(candidates)
            if removed:
                logger.debug(
                    "Excluded %d backend(s) without autograd support "
                    "(model is in training mode).",
                    removed,
                )
        if not candidates:
            logger.warning("No backends available; falling back to %r only.", self._fallback)
            candidates = self._materialize([self._fallback])

        per_layer: dict[str, Backend] = {}
        decisions: list[LayerDecision] = []
        baseline_energy_total = 0.0
        compiled_energy_total = 0.0

        for match in plan.matches:
            backend, reason = self._pick_for_layer(match, candidates, objective)
            macs = match.in_features * match.out_features
            baseline_energy = _CLASSICAL_BASELINE_PJ_PER_MAC * macs
            compiled_energy = backend.profile.energy_pj_per_mac * macs

            per_layer[match.name] = backend
            decisions.append(
                LayerDecision(
                    layer_name=match.name,
                    in_features=match.in_features,
                    out_features=match.out_features,
                    backend=backend.profile.name,
                    reason=reason,
                    estimated_energy_pj=compiled_energy,
                    estimated_latency_us=(
                        backend.profile.latency_us_per_op + backend.profile.queue_us
                    ),
                )
            )
            baseline_energy_total += baseline_energy
            compiled_energy_total += compiled_energy

        return SelectionResult(
            per_layer=per_layer,
            decisions=tuple(decisions),
            baseline_energy_pj=baseline_energy_total,
            compiled_energy_pj=compiled_energy_total,
        )

    # ------------------------------------------------------------------ internal
    def _discover_candidates(self) -> list[Backend]:
        """Return the list of backends to consider for selection.

        Filters out backends whose ``is_available()`` returned False
        (typical reason: optional dep not installed).
        """
        if self._explicit_candidates is not None:
            names = list(self._explicit_candidates)
        else:
            # Exclude the classical reference 'mock' from auto-selection: it must
            # be requested explicitly (e.g. to compare against quantum/photonic
            # results), never silently chosen as if it were real acceleration.
            names = [n for n in list_backends() if n != "mock"]
        return self._materialize(names)

    def _materialize(self, names: list[str]) -> list[Backend]:
        instances: list[Backend] = []
        for name in names:
            try:
                backend = get_backend(name)
            except Exception as exc:  # noqa: BLE001 — we want to swallow and continue
                logger.debug("Skipping backend %r: instantiation failed (%s)", name, exc)
                continue
            try:
                available = backend.is_available()
            except Exception as exc:  # noqa: BLE001 — broken backend shouldn't crash compile
                logger.debug("Backend %r raised in is_available(): %s", name, exc)
                continue
            if not available:
                logger.debug("Backend %r not available on this machine; skipping", name)
                continue
            instances.append(backend)
        return instances

    def _pick_for_layer(
        self,
        match: LayerMatch,
        candidates: list[Backend],
        objective: Objective,
    ) -> tuple[Backend, str]:
        """Greedy per-layer choice. Returns the picked backend and a reason string."""
        macs = match.in_features * match.out_features

        def score(backend: Backend) -> float:
            p = backend.profile
            if p.max_dim is not None and (
                match.in_features > p.max_dim or match.out_features > p.max_dim
            ):
                # Use sentinel — math.inf would force float ops; very large is enough.
                return 1e30
            if objective == "speed":
                return p.latency_us_per_op + p.queue_us
            if objective == "energy":
                return p.energy_pj_per_mac * macs
            # "balanced" — naive normalization. Refine when we have real numbers.
            latency_component = p.latency_us_per_op + p.queue_us
            energy_component = p.energy_pj_per_mac * macs / 1_000_000.0
            return latency_component + energy_component

        # Sort by score ascending. Ties: prefer mock LAST (so real backends win when free).
        feasible = [(score(b), b.profile.name == "mock", b) for b in candidates]
        feasible.sort(key=lambda t: (t[0], t[1]))

        # First non-infeasible. If everything is infeasible, fall back to mock.
        for s, _is_mock, backend in feasible:
            if s < 1e29:
                reason = _human_reason(objective, backend.profile)
                return backend, reason

        fallback = self._materialize([self._fallback])
        if not fallback:
            # Truly nothing works. Build mock directly as last resort.
            from qaithon.backends.mock import MockBackend

            fallback_backend: Backend = MockBackend()
        else:
            fallback_backend = fallback[0]
        return fallback_backend, "fallback (no feasible backend for this layer)"


def _human_reason(objective: Objective, profile: BackendProfile) -> str:
    """Return a one-line plain-language reason for the user."""
    if objective == "speed":
        return f"lowest latency among available backends ({profile.kind})"
    if objective == "energy":
        return f"lowest energy per MAC among available backends ({profile.kind})"
    return f"best balanced score ({profile.kind})"
