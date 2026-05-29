"""vLLM serving adapter — skeleton.

vLLM is the de facto open-source serving framework for LLMs in production.
Plugging Qaithon-compiled models into vLLM means production-grade serving
infrastructure (continuous batching, paged attention, OpenAI-compatible
API) running on top of photonic / quantum backends.

Current state (v0.1)
--------------------

vLLM does not yet expose a public extension point that lets a third party
override its linear-layer implementation per-layer. Until that lands
(it's tracked in upstream issues), this module provides:

* :func:`check_vllm_compatibility` — fail fast if the model is not
  compatible with vLLM (architecture not supported, wrong dtype, …).
* :func:`prepare_for_vllm` — apply the Qaithon transformations vLLM can
  tolerate today (only those that preserve the standard ``nn.Linear``
  surface) and emit clear warnings about what is and is not active.

For a fully functional vLLM serving path, watch
``qaithon.integrations.vllm.run_server`` — it lands in v0.x.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from qaithon._logging import get_logger

if TYPE_CHECKING:
    from torch import nn

__all__ = ["check_vllm_compatibility", "prepare_for_vllm", "vllm_available"]

logger = get_logger(__name__)

# Architectures vLLM currently supports natively (subset relevant here).
_VLLM_SUPPORTED_FAMILIES: frozenset[str] = frozenset(
    {"llama", "mistral", "mixtral", "gpt2", "phi", "qwen", "gemma", "deepseek", "falcon"}
)


def vllm_available() -> bool:
    """Return ``True`` when the ``vllm`` package is importable."""
    return importlib.util.find_spec("vllm") is not None


def check_vllm_compatibility(model: nn.Module) -> tuple[bool, str]:
    """Return ``(ok, reason)`` describing whether ``model`` can serve via vLLM.

    A ``False`` result means there's a Qaithon transformation that would
    break vLLM's assumptions — typically, replacing layers that vLLM's
    paged attention indexes directly.
    """
    from qaithon.handlers.architecture import detect_architecture

    family = detect_architecture(model)
    if family not in _VLLM_SUPPORTED_FAMILIES:
        return False, f"vLLM does not currently support family {family!r}."

    # Detect if the model has been Mixtral-rewritten — vLLM's MoE
    # implementation expects the original 3D layout, not our ModuleList.
    for module in model.modules():
        if hasattr(module, "quantum_experts"):
            return False, (
                "Model has been transformed by qaithon.handlers.mixtral; "
                "vLLM expects the original 3D expert layout. Use a non-Mixtral "
                "compile or skip the MoE handler when targeting vLLM."
            )
    return True, "Compatible."


def prepare_for_vllm(model: nn.Module, **engine_kwargs: Any) -> Any:
    """Wrap ``model`` so it can be loaded into a vLLM engine.

    Args:
        model: A Qaithon-compiled ``nn.Module``.
        **engine_kwargs: Forwarded to ``vllm.LLM(...)``.

    Returns:
        A ``vllm.LLM`` instance ready to serve.

    Raises:
        ImportError: When ``vllm`` is not installed.
        RuntimeError: When the model is not compatible with vLLM.
    """
    if not vllm_available():
        raise ImportError(
            "vllm is not installed. Install with `pip install qaithon[vllm]`."
        )
    ok, reason = check_vllm_compatibility(model)
    if not ok:
        raise RuntimeError(f"Cannot prepare model for vLLM: {reason}")

    logger.warning(
        "qaithon.integrations.vllm.prepare_for_vllm is currently a skeleton. "
        "It will instantiate vllm.LLM with the standard recipe, but Qaithon's "
        "backend choices will only be honored for ops vLLM uses through PyTorch's "
        "standard dispatch (i.e. nn.Linear forward). Custom CUDA kernels "
        "(paged attention, fused MLP) still execute classically."
    )

    import vllm  # type: ignore[import-not-found]

    return vllm.LLM(model=model, **engine_kwargs)
