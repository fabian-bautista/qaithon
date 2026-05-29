"""Detection helpers for attention paths that must not be naively replaced.

Modern transformers use several flavors of optimized attention:

* ``nn.MultiheadAttention`` — wraps internal ``NonDynamicallyQuantizableLinear``
  layers that explicitly signal "do not transform me".
* ``F.scaled_dot_product_attention`` (SDPA) — a function call, not a layer,
  with optional Flash Attention / Memory-Efficient / Math backends chosen
  by PyTorch at runtime.
* Flash Attention 2 (separate ``flash-attn`` package) — wholly custom CUDA
  kernels; absolutely not a target for photonic offload.
* Vendor-specific attention (Llama's ``LlamaSdpaAttention``,
  ``LlamaFlashAttention2``, ``PhiSdpaAttention``, ...) — custom forward
  routes the user usually wants to leave intact.

The walker's identity check (``type(m) is nn.Linear``) already excludes
the internal Linear of ``nn.MultiheadAttention``. This module documents
what we detect for the audit trail and provides helpers downstream tools
can call.

Per the roadmap, full Mixtral / Phi-3 / GPT-Neo specialized rewrites land
in their own modules. This file covers the generic attention-detection
story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = ["AttentionInfo", "list_attention_modules"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AttentionInfo:
    """One attention-like submodule found in a model.

    Attributes:
        name: Fully-qualified module path.
        kind: ``"multihead"``, ``"sdpa"``, ``"flash"``, or ``"custom"``.
        class_name: Python class name (e.g. ``"LlamaSdpaAttention"``).
    """

    name: str
    kind: str
    class_name: str


_KIND_HINTS: tuple[tuple[str, str], ...] = (
    # (substring in class name, kind tag)
    ("FlashAttention2", "flash"),
    ("FlashAttention", "flash"),
    ("SdpaAttention", "sdpa"),
    ("SDPA", "sdpa"),
    ("MultiheadAttention", "multihead"),
    ("MultiHeadAttention", "multihead"),
)


def _infer_kind(module: nn.Module) -> str | None:
    cls = type(module).__name__
    for needle, kind in _KIND_HINTS:
        if needle in cls:
            return kind
    if cls.endswith("Attention"):
        return "custom"
    return None


def list_attention_modules(model: nn.Module) -> list[AttentionInfo]:
    """Enumerate every attention-shaped submodule in ``model``.

    Useful for the audit trail (``CompileReport`` extensions) and for
    downstream handlers that want to reason about attention sites.
    """
    out: list[AttentionInfo] = []
    for name, module in model.named_modules():
        kind = _infer_kind(module)
        if kind is None:
            continue
        out.append(
            AttentionInfo(
                name=name,
                kind=kind,
                class_name=type(module).__name__,
            )
        )
    return out
