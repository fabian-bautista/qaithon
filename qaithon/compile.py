"""The public entry point: :func:`qaithon.compile`.

This is the function 99% of users call. It is designed to be:

* **Trivial to use.** ``model = qaithon.compile(model)`` — that's it. No
  backend choice required, no quantum knowledge required.
* **Optional to tune.** Power users can pass an objective, a backend
  whitelist, or a custom selector if they want to override defaults.
* **Auditable.** The returned model carries a ``qaithon_report`` attribute
  describing every decision made.

The signature follows the same shape as ``torch.compile`` deliberately —
muscle memory of any modern PyTorch developer maps directly onto Qaithon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from torch import nn

from qaithon._logging import get_logger
from qaithon.compile_report import CompileReport
from qaithon.exceptions import IncompatibleModelError
from qaithon.ir import AutoBackendSelector, analyze_model
from qaithon.layers.quantum_linear import QuantumLinear

if TYPE_CHECKING:
    from collections.abc import Callable

    from qaithon.backends.base import Backend
    from qaithon.ir import Objective, ReplacementPlan

__all__ = ["compile"]

logger = get_logger(__name__)


@overload
def compile(  # noqa: A001 — intentional shadow of builtin, matches torch.compile
    model: nn.Module,
    *,
    optimize_for: Objective = "balanced",
    backends: tuple[str, ...] | None = None,
    skip: Callable[[str, nn.Module], bool] | None = None,
    min_in_features: int = 0,
    min_out_features: int = 0,
    strict: bool = True,
    return_report: Literal[False] = False,
) -> nn.Module: ...


@overload
def compile(  # noqa: A001
    model: nn.Module,
    *,
    optimize_for: Objective = "balanced",
    backends: tuple[str, ...] | None = None,
    skip: Callable[[str, nn.Module], bool] | None = None,
    min_in_features: int = 0,
    min_out_features: int = 0,
    strict: bool = True,
    return_report: Literal[True],
) -> tuple[nn.Module, CompileReport]: ...


def compile(  # noqa: A001 — intentional shadow of builtin
    model: nn.Module,
    *,
    optimize_for: Objective = "balanced",
    backends: tuple[str, ...] | None = None,
    skip: Callable[[str, nn.Module], bool] | None = None,
    min_in_features: int = 0,
    min_out_features: int = 0,
    strict: bool = True,
    return_report: bool = False,
) -> nn.Module | tuple[nn.Module, CompileReport]:
    """Transform ``model`` so its large linear layers run on the best available backend.

    Qaithon inspects the model, identifies the linear projections worth
    accelerating, picks the optimal backend per layer based on what's
    installed on this machine and the chosen objective, and returns the
    transformed model. The caller never has to know which backend was chosen
    or how it works internally.

    The returned model is the **same object** as the input model
    (transformation happens in place), with an extra attribute
    ``qaithon_report`` holding the :class:`CompileReport`.

    Args:
        model: Any PyTorch model. HuggingFace transformers are the primary
            target but plain ``nn.Module`` instances work too.
        optimize_for: What Qaithon should minimize:

            * ``"speed"`` — pick whatever backend has lowest latency.
            * ``"energy"`` — pick whatever backend has lowest energy per MAC.
            * ``"balanced"`` (default) — blend latency and energy.
        backends: Optional whitelist of backend names to consider. ``None``
            (default) means "consider every available backend".
        skip: Optional predicate ``(name, module) -> bool`` that returns
            ``True`` to exclude specific layers from replacement.
        min_in_features: Skip layers whose input dimension is below this
            threshold. Useful to avoid overhead on tiny projections.
        min_out_features: Skip layers whose output dimension is below this
            threshold.
        strict: If ``True`` (default), reject models that are already
            quantized (e.g. by bitsandbytes) with a clear error rather than
            silently producing wrong outputs.
        return_report: If ``True``, return a tuple ``(model, report)`` instead
            of just the model. The report is also always attached to the
            model as ``model.qaithon_report``.

    Returns:
        The transformed model (in-place). If ``return_report=True``, returns
        ``(model, CompileReport)``.

    Raises:
        IncompatibleModelError: When the model is already quantized and
            ``strict=True``.

    Example:
        >>> import qaithon
        >>> from transformers import AutoModelForCausalLM  # doctest: +SKIP
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")  # doctest: +SKIP
        >>> model = qaithon.compile(model)  # doctest: +SKIP
        >>> # Use model exactly like before:
        >>> outputs = model.generate(input_ids, max_new_tokens=20)  # doctest: +SKIP
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"qaithon.compile expects a torch.nn.Module, got {type(model).__name__}."
        )

    # Detect the model family and apply its recommended defaults — but only
    # for arguments the user did NOT override. User intent always wins.
    from qaithon.handlers.architecture import detect_architecture, recommend_config

    family = detect_architecture(model)
    profile = recommend_config(family)
    logger.info("Detected architecture: %s (%s).", family, profile.description)

    if min_in_features == 0:
        min_in_features = profile.min_in_features
    if min_out_features == 0:
        min_out_features = profile.min_out_features

    # Compose user's skip predicate with the family's name patterns.
    family_skip = _build_family_skip(profile.skip_name_patterns)
    if skip is None:
        skip = family_skip
    else:
        user_skip = skip

        def composed_skip(name: str, module: nn.Module) -> bool:
            return user_skip(name, module) or family_skip(name, module)

        skip = composed_skip

    # If the family is MoE-aware, rewrite expert blocks before the generic walker.
    if profile.moe_aware:
        from qaithon.handlers.mixtral import (
            is_mixtral_model,
            transform_mixtral_experts,
        )

        if is_mixtral_model(model):
            backend_for_experts = backends[0] if backends else "mock"
            mixtral_result = transform_mixtral_experts(model, backend=backend_for_experts)
            logger.info(
                "Mixtral handler: transformed %d block(s), replaced %d expert(s).",
                mixtral_result.n_blocks_transformed,
                mixtral_result.n_experts_replaced,
            )

    plan = analyze_model(
        model,
        skip=skip,
        min_in_features=min_in_features,
        min_out_features=min_out_features,
        strict_already_quantized=strict,
    )
    logger.info(
        "Analyzed %s: %d replaceable, %d skipped.",
        type(model).__name__,
        plan.total_replaceable,
        plan.total_skipped,
    )

    selector = AutoBackendSelector(candidate_names=backends)
    # If the model is in training mode, exclude gradient-free backends so
    # `loss.backward()` keeps working transparently.
    selection = selector.select(
        plan,
        objective=optimize_for,
        require_autograd=model.training,
    )

    report = _apply_plan(model, plan, selection, optimize_for=optimize_for)
    # Attach so the user can inspect it later: `model.qaithon_report`.
    # Using object.__setattr__ in case the model overrides __setattr__.
    object.__setattr__(model, "qaithon_report", report)

    logger.info("Compile complete: %r", report)
    if return_report:
        return model, report
    return model


def _build_family_skip(
    patterns: tuple[str, ...],
) -> Callable[[str, nn.Module], bool]:
    """Return a SkipPredicate that excludes layers whose name contains any pattern."""
    if not patterns:
        return lambda _name, _module: False

    def predicate(name: str, _module: nn.Module) -> bool:
        return any(p in name for p in patterns)

    return predicate


def _apply_plan(
    model: nn.Module,
    plan: ReplacementPlan,
    selection: SelectionResult,
    *,
    optimize_for: Objective,
) -> CompileReport:
    """Swap each layer in ``plan`` with a :class:`QuantumLinear` backed by the chosen backend.

    Mutates ``model`` in place. Returns the resulting CompileReport.
    """
    n_parameters = sum(p.numel() for p in model.parameters())

    for match in plan.matches:
        backend = selection.per_layer[match.name]
        new_layer = QuantumLinear.from_linear(match.layer, backend=backend, copy_weights=True)
        _set_submodule(model, match.name, new_layer)

    return CompileReport(
        model_class=type(model).__name__,
        n_parameters=n_parameters,
        decisions=selection.decisions,
        skipped=plan.skipped,
        optimize_for=optimize_for,
        baseline_energy_pj=selection.baseline_energy_pj,
        compiled_energy_pj=selection.compiled_energy_pj,
    )


def _set_submodule(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """Replace the submodule at ``qualified_name`` inside ``root``.

    Works for arbitrarily deep paths like ``"transformer.h.0.mlp.c_fc"``.
    PyTorch 2.x has ``root.set_submodule``; we use a manual walk for
    compatibility and so the implementation is explicit.
    """
    # Prefer the built-in if available (PyTorch >= 2.0).
    set_submodule = getattr(root, "set_submodule", None)
    if callable(set_submodule):
        set_submodule(qualified_name, new_module)
        return

    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# Bring the imported types into the module namespace for the typed overloads above.
from qaithon.ir.selector import SelectionResult  # noqa: E402 — late import for typing
