"""Walk a model, identify replaceable layers, produce a replacement plan.

The walker is the entry point of Qaithon's transformation pipeline. Given any
``torch.nn.Module`` (HuggingFace transformer, custom PyTorch model, anything),
it produces a :class:`ReplacementPlan` listing which sub-modules will be
swapped for backend-accelerated equivalents.

What makes this non-trivial
---------------------------

PyTorch and HuggingFace models contain layers that **look like** matmul targets
but should be left alone:

* ``NonDynamicallyQuantizableLinear`` inside ``nn.MultiheadAttention`` —
  explicitly signals "do not quantize me". We exclude it with an identity
  check (``type(m) is nn.Linear``) instead of ``isinstance``. This is the
  same trick bitsandbytes uses, and it is *the* one-character distinction
  that saves us from corrupting attention layers.
* ``lm_head`` in models with tied embeddings (GPT-2, BERT, Llama-3.2 small,
  many others) shares its weight with the input embedding. Replacing only one
  side breaks the model. We detect ties and exclude either both or neither.
* Already-quantized layers (``Linear4bit``, ``Linear8bitLt``) — composing
  two quantization schemes silently corrupts numerics. We refuse cleanly.

Public surface
--------------

* :class:`LayerMatch` — one entry in the plan: full qualified name + the layer.
* :class:`ReplacementPlan` — immutable container of matches + diagnostics.
* :func:`analyze_model` — the function the compiler calls.
* :func:`default_skip_predicate` — default rule for "should I skip this layer?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

from qaithon._logging import get_logger
from qaithon.exceptions import IncompatibleModelError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "LayerMatch",
    "ReplacementPlan",
    "SkipPredicate",
    "analyze_model",
    "default_skip_predicate",
]

logger = get_logger(__name__)


# A predicate decides "should I skip this layer?". Signature mirrors what callers care about:
# the fully-qualified name (parent.child.grandchild) and the layer itself.
SkipPredicate = Callable[[str, nn.Module], bool]


# ---------------------------------------------------------------------------
# Data classes — immutable plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayerMatch:
    """A single layer that the walker decided to replace.

    Attributes:
        name: Fully-qualified module path (``"transformer.h.0.mlp.c_fc"``).
        layer: The ``nn.Linear`` instance found at that path.
        in_features: Convenience copy of ``layer.in_features``.
        out_features: Convenience copy of ``layer.out_features``.
        has_bias: ``True`` if the layer has a bias term.
    """

    name: str
    layer: nn.Linear
    in_features: int
    out_features: int
    has_bias: bool


@dataclass(frozen=True, slots=True)
class ReplacementPlan:
    """The output of :func:`analyze_model`.

    Attributes:
        matches: Tuple of layers slated for replacement.
        skipped: Tuple of (name, reason) pairs for layers that were considered
            but excluded. Useful for ``print(plan)`` and debugging.
        tied_groups: Tuple of tuples of names that share weights (e.g.
            ``("transformer.wte", "lm_head")``). Reported so the compiler can
            decide what to do (skip both, replace both, etc.).
    """

    matches: tuple[LayerMatch, ...] = field(default_factory=tuple)
    skipped: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    tied_groups: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    @property
    def total_replaceable(self) -> int:
        """Number of layers that will be replaced."""
        return len(self.matches)

    @property
    def total_skipped(self) -> int:
        """Number of layers considered but excluded."""
        return len(self.skipped)

    def summary(self) -> str:
        """Human-readable summary, useful for logging."""
        lines = [
            f"ReplacementPlan: {self.total_replaceable} replace, "
            f"{self.total_skipped} skip, {len(self.tied_groups)} tied group(s)."
        ]
        if self.matches:
            lines.append("  Replace:")
            for m in self.matches[:10]:
                lines.append(f"    {m.name} [{m.in_features}->{m.out_features}]")
            if len(self.matches) > 10:
                lines.append(f"    ... and {len(self.matches) - 10} more")
        if self.skipped:
            lines.append("  Skip:")
            for name, reason in self.skipped[:5]:
                lines.append(f"    {name} ({reason})")
            if len(self.skipped) > 5:
                lines.append(f"    ... and {len(self.skipped) - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tied-weight detection
# ---------------------------------------------------------------------------
def _find_tied_parameters(model: nn.Module) -> list[tuple[str, ...]]:
    """Return groups of parameter names that share the same storage.

    First tries ``accelerate.utils.find_tied_parameters`` (the canonical
    implementation used by HuggingFace). Falls back to a simple identity
    sweep if ``accelerate`` is not installed. Either way, the result is a
    list of tuples of parameter names that point to the same ``Tensor``.
    """
    try:
        from accelerate.utils import find_tied_parameters

        tied = find_tied_parameters(model)
        # accelerate returns either a list[list[str]] or a dict[str, list[str]]
        # depending on version. Normalize.
        if isinstance(tied, dict):
            return [(k, *v) for k, v in tied.items()]
        return [tuple(group) for group in tied if len(group) > 1]
    except ImportError:
        logger.debug("accelerate not installed, using identity-based tie detection")

    # Fallback: build a map data_ptr -> [param names], report groups with >1 name.
    by_ptr: dict[int, list[str]] = {}
    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue
        ptr = param.data_ptr()
        by_ptr.setdefault(ptr, []).append(name)
    return [tuple(names) for names in by_ptr.values() if len(names) > 1]


def _is_replaceable_linear(module: nn.Module) -> bool:
    """Return ``True`` if ``module`` is a linear projection we know how to swap.

    Currently recognized:

    * ``torch.nn.Linear`` (identity check — excludes subclasses like
      ``NonDynamicallyQuantizableLinear`` inside ``nn.MultiheadAttention``).
    * ``transformers.pytorch_utils.Conv1D`` (used by GPT-2, GPT-Neo, BLOOM —
      semantically a linear projection but with a transposed weight layout).

    More types land here as we encounter them in real models. Each addition
    must be exact (no ``isinstance`` recursion) to preserve the safety of
    not touching layers that explicitly signal "do not transform me".
    """
    if type(module) is nn.Linear:
        return True
    # transformers.pytorch_utils.Conv1D — check by qualified class name to
    # avoid an import dependency on transformers in the core walker.
    cls = type(module)
    qualified = f"{cls.__module__}.{cls.__name__}"
    if qualified == "transformers.pytorch_utils.Conv1D":
        return True
    return False


def _module_dims(module: nn.Module) -> tuple[int, int] | None:
    """Return ``(in_features, out_features)`` for any recognized layer."""
    if type(module) is nn.Linear:
        return module.in_features, module.out_features
    # transformers Conv1D stores weight as (in_features, out_features) — the
    # transpose of nn.Linear. The class exposes `nf` (out) and `nx` (in) as
    # attributes; we read them when available.
    nf = getattr(module, "nf", None)
    if nf is not None:
        weight = getattr(module, "weight", None)
        if weight is not None and weight.ndim == 2:
            in_features = weight.shape[0]
            return in_features, int(nf)
    return None


def _detect_quantized_layers(model: nn.Module) -> list[str]:
    """Return the names of layers that look already-quantized."""
    quantized: list[str] = []
    quantized_class_names = {
        "Linear4bit",
        "Linear8bitLt",
        "Linear8bitFp",
        "Params4bit",
        "EetqLinear",
        "HqqLinear",
        "GPTQLinear",
        "AWQLinear",
    }
    for name, module in model.named_modules():
        if type(module).__name__ in quantized_class_names:
            quantized.append(name)
    return quantized


# ---------------------------------------------------------------------------
# Default skip predicate
# ---------------------------------------------------------------------------
def default_skip_predicate(name: str, module: nn.Module) -> bool:  # noqa: ARG001
    """Default rule for "should I skip this layer when planning replacements?".

    The default behavior is to *not* skip anything beyond what the strict type
    check already excludes. Custom predicates can be supplied to:

    * Skip output projections (``"lm_head"``, ``"output"``).
    * Restrict to specific named scopes (``"mlp"`` only).
    * Implement size thresholds (skip tiny layers where overhead dominates).

    Args:
        name: Fully-qualified module path.
        module: The module itself (already known to be a candidate).

    Returns:
        ``True`` to skip this layer, ``False`` to include it.
    """
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_model(
    model: nn.Module,
    *,
    skip: SkipPredicate | None = None,
    min_in_features: int = 0,
    min_out_features: int = 0,
    strict_already_quantized: bool = True,
) -> ReplacementPlan:
    """Walk ``model``, return a plan listing every layer to replace.

    The plan is a pure data structure — applying it is the compiler's job
    (Single Responsibility). This separation means we can dry-run an
    analysis without touching the model, log it, ask the user, etc.

    Args:
        model: Any PyTorch model. HuggingFace transformers are the primary
            target but any ``nn.Module`` works.
        skip: Optional predicate to exclude specific layers by name/type.
            Defaults to :func:`default_skip_predicate` (no extra skipping).
        min_in_features: Skip layers whose ``in_features`` is below this
            threshold. Useful to avoid overhead on tiny projections.
        min_out_features: Skip layers whose ``out_features`` is below this
            threshold.
        strict_already_quantized: When ``True`` (default), if the model
            contains layers from a known quantization library (bitsandbytes,
            AWQ, GPTQ, ...) we raise :class:`IncompatibleModelError`
            instead of silently producing incorrect results.

    Returns:
        :class:`ReplacementPlan` with matches, skipped, and tied groups.

    Raises:
        IncompatibleModelError: If the model is already quantized and
            ``strict_already_quantized`` is ``True``.

    Example:
        >>> import torch.nn as nn
        >>> from qaithon.ir import analyze_model
        >>> model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        >>> plan = analyze_model(model)
        >>> plan.total_replaceable
        2
    """
    if skip is None:
        skip = default_skip_predicate

    # Bail out early on already-quantized models.
    quantized = _detect_quantized_layers(model)
    if quantized:
        if strict_already_quantized:
            raise IncompatibleModelError(
                reason=(
                    f"Model contains {len(quantized)} already-quantized layer(s) "
                    f"(e.g. {quantized[0]!r}). Composing Qaithon with another "
                    "quantization scheme is not supported and would corrupt outputs."
                ),
                hint=(
                    "Reload the model without quantization "
                    "(remove `load_in_4bit=True` / `load_in_8bit=True` from "
                    "`from_pretrained(...)`) and pass that fresh instance to "
                    "`qaithon.compile`."
                ),
            )
        logger.warning(
            "Found %d already-quantized layers but strict_already_quantized=False; "
            "they will be skipped.",
            len(quantized),
        )

    quantized_set = set(quantized)

    # Tied-parameter groups (full parameter paths, not module paths).
    tied = _find_tied_parameters(model)
    tied_module_names: set[str] = set()
    for group in tied:
        for param_name in group:
            # Drop trailing .weight / .bias to get the module path.
            module_path = param_name.rsplit(".", 1)[0]
            tied_module_names.add(module_path)

    matches: list[LayerMatch] = []
    skipped: list[tuple[str, str]] = []

    # Quantized layers are always reported as skipped (so the user sees them in
    # the plan), even if non-strict mode let us continue past the early check.
    for q_name in quantized_set:
        skipped.append((q_name, "already-quantized layer"))

    for name, module in model.named_modules():
        # Identity check, not isinstance — excludes NonDynamicallyQuantizableLinear
        # inside nn.MultiheadAttention and similar opt-out subclasses.
        # Also accept transformers' Conv1D (used by GPT-2, GPT-Neo, ...), which
        # is semantically a Linear but lives in transformers.pytorch_utils.
        if not _is_replaceable_linear(module):
            continue
        if name in quantized_set:
            # Already reported above; just skip the matching step.
            continue
        if name in tied_module_names:
            skipped.append((name, "weight is tied to another parameter"))
            continue
        dims = _module_dims(module)
        if dims is None:
            continue
        in_features, out_features = dims
        if in_features < min_in_features:
            skipped.append((name, f"in_features={in_features} < min_in_features={min_in_features}"))
            continue
        if out_features < min_out_features:
            skipped.append(
                (name, f"out_features={out_features} < min_out_features={min_out_features}")
            )
            continue
        if skip(name, module):
            skipped.append((name, "excluded by user skip predicate"))
            continue
        has_bias = getattr(module, "bias", None) is not None
        matches.append(
            LayerMatch(
                name=name,
                layer=module,  # type: ignore[arg-type] — Conv1D is structurally compatible
                in_features=in_features,
                out_features=out_features,
                has_bias=has_bias,
            )
        )

    plan = ReplacementPlan(
        matches=tuple(matches),
        skipped=tuple(skipped),
        tied_groups=tuple(tied),
    )
    logger.debug("Analyzed model:\n%s", plan.summary())
    return plan


# Re-export a torch type for typing convenience.
_ = torch  # quiets linters that flag `import torch` as unused in TYPE_CHECKING setups
