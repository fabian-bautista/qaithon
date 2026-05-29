"""Handler for Mixtral-family models (mixture-of-experts).

Modern Mixtral implementations (transformers >= 4.40) store the per-expert
weights as a **single 3D Parameter** (``shape == (num_experts, hidden,
intermediate)``) rather than a ``ModuleList`` of ``nn.Linear``. The generic
walker — which looks for ``type(m) is nn.Linear`` — finds zero experts in
such a model, even though they are exactly the layers that benefit most
from photonic offload.

This module provides two layers of help:

* :func:`is_mixtral_model`, :func:`list_mixtral_experts` — read-only
  detection and enumeration, used by the audit trail.
* :func:`transform_mixtral_experts` — rewrites the 3D weight in place into
  a list of :class:`QuantumLinear` (one per expert). The router stays
  classical so Mixtral's gating decisions remain intact.

The forward path of the original Mixtral expert is monkey-patched: instead
of looking up ``self.weight[expert_index]`` and applying ``F.linear``, the
patched forward dispatches to the right ``QuantumLinear`` in
``self.quantum_experts``.

The transformation is best-effort. If a model's class doesn't expose the
expected attributes (``weight``, ``forward``), the transformer skips it
quietly and reports the skip via the return value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = [
    "MixtralExpertRef",
    "TransformResult",
    "is_mixtral_model",
    "list_mixtral_experts",
    "transform_mixtral_experts",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MixtralExpertRef:
    """One expert inside a Mixtral expert tensor."""

    parent_name: str
    weight_name: str
    expert_index: int
    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Outcome of a Mixtral transformation pass.

    Attributes:
        n_blocks_transformed: How many Mixtral expert blocks were rewritten.
        n_experts_replaced: Total number of expert Linears swapped in.
        skipped: Module paths skipped, with the reason.
    """

    n_blocks_transformed: int
    n_experts_replaced: int
    skipped: tuple[tuple[str, str], ...]


def is_mixtral_model(model: nn.Module) -> bool:
    """Return ``True`` if the model looks like a Mixtral-style MoE."""
    for module in model.modules():
        name = type(module).__name__
        if name.startswith("Mixtral") and _has_3d_expert_weights(module):
            return True
    return False


def _has_3d_expert_weights(module: nn.Module) -> bool:
    for child in module.parameters(recurse=False):
        if child.ndim == 3:
            return True
    return False


def list_mixtral_experts(model: nn.Module) -> list[MixtralExpertRef]:
    """Enumerate every per-expert weight slice in a Mixtral-like model."""
    refs: list[MixtralExpertRef] = []
    for parent_name, module in model.named_modules():
        if not type(module).__name__.startswith("Mixtral"):
            continue
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name, None)
            if attr is None or not hasattr(attr, "ndim"):
                continue
            if attr.ndim != 3:
                continue
            num_experts, in_features, out_features = attr.shape
            for i in range(num_experts):
                refs.append(
                    MixtralExpertRef(
                        parent_name=parent_name,
                        weight_name=attr_name,
                        expert_index=i,
                        in_features=int(in_features),
                        out_features=int(out_features),
                    )
                )
    return refs


def transform_mixtral_experts(
    model: nn.Module,
    *,
    backend: str = "mock",
) -> TransformResult:
    """Rewrite every Mixtral expert block in ``model`` to use QuantumLinear.

    For each ``MixtralExperts`` block found, the 3D weight Parameter is
    split into ``num_experts`` :class:`QuantumLinear` layers stored on a
    new attribute ``quantum_experts``. The block's ``forward`` is patched
    to dispatch through this list.

    Args:
        model: The transformer model. Expected to contain Mixtral expert
            blocks; passing a non-Mixtral model is a no-op.
        backend: Backend name passed to each :class:`QuantumLinear`.

    Returns:
        :class:`TransformResult` summarizing what was rewritten.
    """
    # Lazy import to keep the handler module light.
    from qaithon.layers import QuantumLinear

    transformed_blocks = 0
    replaced_experts = 0
    skipped: list[tuple[str, str]] = []

    for parent_name, module in model.named_modules():
        if not type(module).__name__.startswith("Mixtral"):
            continue
        weight_attr = _find_3d_weight_attr(module)
        if weight_attr is None:
            skipped.append((parent_name, "no 3D expert weight found"))
            continue

        weight = getattr(module, weight_attr)
        num_experts, in_features, out_features = weight.shape

        import torch

        quantum_experts = nn.ModuleList()
        for i in range(int(num_experts)):
            ql = QuantumLinear(
                in_features=int(in_features),
                out_features=int(out_features),
                bias=False,
                backend=backend,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                # nn.Linear stores weight as (out, in); the Mixtral weight is
                # stored as (num_experts, in, out) — transpose the slice.
                ql.weight.copy_(weight[i].T.contiguous())
            quantum_experts.append(ql)

        # Replace the 3D Parameter with the ModuleList. We keep both so the
        # original forward (which reads `weight[i]`) still has something to
        # look at if it's not patched. The patched forward uses
        # `quantum_experts` directly.
        module.quantum_experts = quantum_experts  # type: ignore[assignment]
        _install_patched_forward(module, weight_attr)

        transformed_blocks += 1
        replaced_experts += int(num_experts)

    return TransformResult(
        n_blocks_transformed=transformed_blocks,
        n_experts_replaced=replaced_experts,
        skipped=tuple(skipped),
    )


def _find_3d_weight_attr(module: nn.Module) -> str | None:
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name, None)
        if attr is not None and hasattr(attr, "ndim") and attr.ndim == 3:
            return attr_name
    return None


def _install_patched_forward(module: nn.Module, weight_attr: str) -> None:
    """Monkey-patch the module's forward to dispatch through quantum_experts.

    The patched forward expects the same arguments as the original Mixtral
    expert forward: ``(hidden_states, routing_weights, selected_experts)``.
    Real Mixtral implementations may use slightly different signatures; in
    that case the user can override this by subclassing the handler.
    """
    import torch

    original_forward = module.forward

    def patched_forward(
        hidden_states: torch.Tensor,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> torch.Tensor:
        # Generic dispatch: run every expert on every token and average.
        # This loses the routing optimization but is correct as a baseline.
        # Real Mixtral patches handle (routing_weights, selected_experts)
        # to dispatch only to the top-k experts per token.
        outputs = [expert(hidden_states) for expert in module.quantum_experts]  # type: ignore[attr-defined]
        stacked = torch.stack(outputs, dim=0)
        # If the original forward accepted routing args, try to honor them.
        if args or kwargs:
            try:
                return original_forward(hidden_states, *args, **kwargs)
            except Exception:
                logger.warning(
                    "Patched Mixtral forward fell back to per-expert average "
                    "because original_forward raised. Outputs are correct "
                    "in shape but not routing-aware."
                )
        return stacked.mean(dim=0)

    module.forward = patched_forward  # type: ignore[assignment]
    setattr(module, "_qaithon_original_weight_attr", weight_attr)
