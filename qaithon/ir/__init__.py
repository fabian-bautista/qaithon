"""Internal representation and analysis of models targeted by Qaithon.

This package contains the machinery that inspects a ``torch.nn.Module`` tree,
identifies which sub-modules are candidates for backend-accelerated execution,
and produces a plan that the compiler can execute. It also contains the
auto-selection logic that picks the best backend per layer based on the
user's objective.

The design is inspired by LightCode's Stacked Graph but starts at module
granularity (``nn.Linear`` swaps) instead of fine-grained tensor ops, which is
sufficient for the MVP and what every successful HF-wrapping library
(bitsandbytes, torchao, peft) does in practice.
"""

from __future__ import annotations

from qaithon.ir.analyzer import (
    LayerMatch,
    ReplacementPlan,
    SkipPredicate,
    analyze_model,
    default_skip_predicate,
)
from qaithon.ir.adaptive_selector import AdaptiveBackendSelector, ObservedStats
from qaithon.ir.selector import (
    AutoBackendSelector,
    Objective,
    SelectionResult,
)

__all__ = [
    "AdaptiveBackendSelector",
    "AutoBackendSelector",
    "LayerMatch",
    "Objective",
    "ObservedStats",
    "ReplacementPlan",
    "SelectionResult",
    "SkipPredicate",
    "analyze_model",
    "default_skip_predicate",
]
