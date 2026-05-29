"""Convenience wrappers over ``model.generate``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.models import ToyTokenizer

if TYPE_CHECKING:
    from torch import nn

__all__ = ["generate"]

logger = get_logger(__name__)


def generate(
    model: "nn.Module",
    prompt: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
    tokenizer: ToyTokenizer | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Generate text from a Qaithon-compiled toy model.

    Args:
        model: A trained ``GPT2LMHeadModel`` from :mod:`qaithon.models`.
        prompt: Starting string.
        max_new_tokens: How many new tokens to produce.
        temperature: Sampling temperature. ``0`` ≈ greedy.
        top_k: Top-k sampling cap; ``None`` disables.
        tokenizer: Defaults to a fresh :class:`ToyTokenizer`.
        device: Where to run. Inferred from the model if not specified.

    Returns:
        The generated string (prompt included).
    """
    tokenizer = tokenizer or ToyTokenizer()
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0].tolist())
