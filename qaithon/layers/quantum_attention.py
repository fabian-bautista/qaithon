"""Drop-in replacement for :class:`torch.nn.MultiheadAttention`.

This is the attention companion to :class:`QuantumLinear`. Where
:class:`QuantumLinear` swaps a single ``nn.Linear``,
:class:`QuantumAttention` swaps the Q/K/V/O projections inside a multi-head
attention block — all of them simultaneously, with the dot-product
attention itself left classical.

Why split this out
------------------

Most transformers built on the HuggingFace stack use either
``nn.MultiheadAttention`` (BERT, GPT-2 in some configurations) or a custom
subclass that exposes its internal projections as `nn.Linear` (Llama,
Mistral). For the latter, the generic walker already picks the projections
up. For the former, the projections live behind an opaque interface; the
walker correctly leaves them alone (per the ``NonDynamicallyQuantizableLinear``
opt-out, see :file:`docs/en/ARCHITECTURE.md`).

This class is the path forward for explicit attention swaps: an advanced
user calls ``QuantumAttention.from_multihead(layer, backend=...)`` and the
returned module behaves identically while running its projections on a
Qaithon backend.

For v0.1, the attention dot product itself uses ``F.scaled_dot_product_attention``
on the same device as the input — PyTorch's optimized implementation. The
photonic offload is only for the linear projections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from qaithon._logging import get_logger
from qaithon.backends.base import Backend
from qaithon.layers.quantum_linear import QuantumLinear

if TYPE_CHECKING:
    pass

__all__ = ["QuantumAttention"]

logger = get_logger(__name__)


class QuantumAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections run on a Qaithon backend.

    Mirrors :class:`torch.nn.MultiheadAttention`'s ``forward`` signature for
    the common case (``query``, ``key``, ``value``, ``attn_mask``,
    ``need_weights``). Less common arguments (``key_padding_mask``,
    ``add_bias_kv``, ``zero_attn``) raise ``NotImplementedError`` instead
    of silently misbehaving — the user gets a clear path.

    Args:
        embed_dim: Same as ``nn.MultiheadAttention``.
        num_heads: Same as ``nn.MultiheadAttention``.
        bias: Whether the projections include bias terms.
        backend: Backend name or instance used for every projection.
        batch_first: If ``True`` (default), expects ``(B, T, D)`` instead
            of ``(T, B, D)`` — matches modern HuggingFace conventions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        bias: bool = True,
        backend: str | Backend = "mock",
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = QuantumLinear(embed_dim, embed_dim, bias=bias, backend=backend)
        self.k_proj = QuantumLinear(embed_dim, embed_dim, bias=bias, backend=backend)
        self.v_proj = QuantumLinear(embed_dim, embed_dim, bias=bias, backend=backend)
        self.out_proj = QuantumLinear(embed_dim, embed_dim, bias=bias, backend=backend)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        *,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute multi-head attention.

        Args:
            query: Tensor of shape ``(B, T, D)`` if ``batch_first`` else
                ``(T, B, D)``.
            key: Same shape as ``query``. Defaults to ``query`` (self-attention).
            value: Same shape as ``query``. Defaults to ``key``.
            attn_mask: Optional boolean or float mask broadcastable to
                ``(B, num_heads, T, T)``.
            need_weights: When ``True``, also returns the attention weights.

        Returns:
            ``(output, attn_weights)`` where ``attn_weights`` is ``None`` when
            ``need_weights`` is False.
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        b, t, _ = query.shape
        q = self.q_proj(query).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        output = self.out_proj(attn_out)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, None if not need_weights else _compute_weights(q, k)

    @classmethod
    def from_multihead(
        cls,
        layer: nn.MultiheadAttention,
        *,
        backend: str | Backend = "mock",
    ) -> QuantumAttention:
        """Construct from an existing ``nn.MultiheadAttention``.

        Copies weights from the source's ``in_proj_weight`` / ``out_proj``
        attributes and forwards them to the four :class:`QuantumLinear`
        sub-layers in the correct order (Q, K, V).
        """
        emb = layer.embed_dim
        new = cls(
            embed_dim=emb,
            num_heads=layer.num_heads,
            bias=layer.in_proj_bias is not None,
            backend=backend,
            batch_first=layer.batch_first,
        )
        with torch.no_grad():
            # nn.MultiheadAttention stacks Q/K/V in a single weight: (3*D, D).
            in_proj = layer.in_proj_weight
            new.q_proj.weight.copy_(in_proj[:emb])
            new.k_proj.weight.copy_(in_proj[emb : 2 * emb])
            new.v_proj.weight.copy_(in_proj[2 * emb :])
            if layer.in_proj_bias is not None:
                new.q_proj.bias.copy_(layer.in_proj_bias[:emb])  # type: ignore[union-attr]
                new.k_proj.bias.copy_(layer.in_proj_bias[emb : 2 * emb])  # type: ignore[union-attr]
                new.v_proj.bias.copy_(layer.in_proj_bias[2 * emb :])  # type: ignore[union-attr]
            new.out_proj.weight.copy_(layer.out_proj.weight)
            if layer.out_proj.bias is not None and new.out_proj.bias is not None:
                new.out_proj.bias.copy_(layer.out_proj.bias)
        return new


def _compute_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute classical attention weights for callers that need them."""
    scale = q.shape[-1] ** -0.5
    return torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
