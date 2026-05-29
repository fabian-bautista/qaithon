"""Tiny transformer models sized to fit honestly in current QPU hardware.

The goal is **NOT** to produce useful language models — at these sizes the
quality is comparable to a bigram model. The goal is to give the user a
real HuggingFace-compatible architecture small enough that:

* Each ``nn.Linear`` can be implemented as a quantum circuit whose depth
  fits within Heron's coherence window (≤ ~2,000 gate operations).
* Every matmul completes in a few real shots, so the full inference can
  be run on a real QPU in minutes rather than days.
* The user can experiment with the **Qaithon → real hardware** loop
  without burning their entire Open Plan quota.

Design choices
--------------

* GPT-2 architecture (HuggingFace ``GPT2LMHeadModel``) so the model
  plugs into the existing ``qaithon.compile`` family-detection path
  (``family="gpt2"``).
* Character-level tokenizer to skip BPE training; the vocab is the
  printable ASCII range.
* Untrained by default — useful for measuring throughput and validating
  the path. ``ToyTokenizer`` makes generated text inspectable even when
  the model produces garbage; that's correct behavior for an untrained
  model.

Sizing matrix
-------------

================  ============  ==========  ==========  ====================
Factory           dim           n_layers    n_heads     qubits/matmul ~
================  ============  ==========  ==========  ====================
``micro``         16            1           1            4 (fits any QPU)
``toy``           32            1           2            5
``toy(dim=64)``   64            2           2            6 (still feasible)
================  ============  ==========  ==========  ====================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = ["ToyTokenizer", "create_micro_transformer", "create_toy_transformer"]

logger = get_logger(__name__)

# Printable ASCII range + control tokens for end-of-string.
_VOCAB_START = 32   # space
_VOCAB_END = 127    # DEL boundary
_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_SPECIAL_TOKENS = 3
_VOCAB_SIZE = (_VOCAB_END - _VOCAB_START) + _SPECIAL_TOKENS  # = 98


@dataclass(frozen=True)
class ToyTokenizer:
    """Character-level tokenizer for the toy transformers.

    Returns a small int per character. Untokenizes back to the original
    string. Suitable for end-to-end testing without dragging in a
    pretrained BPE tokenizer.
    """

    vocab_size: int = _VOCAB_SIZE
    pad_token_id: int = _PAD_ID
    bos_token_id: int = _BOS_ID
    eos_token_id: int = _EOS_ID

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        """Convert ``text`` to a list of token ids."""
        ids: list[int] = [self.bos_token_id] if add_bos else []
        for ch in text:
            code = ord(ch)
            if _VOCAB_START <= code <= _VOCAB_END:
                ids.append(_SPECIAL_TOKENS + code - _VOCAB_START)
            else:
                ids.append(_PAD_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Recover a (possibly lossy) string from token ids."""
        chars: list[str] = []
        for i in ids:
            if i < _SPECIAL_TOKENS:
                continue
            code = i - _SPECIAL_TOKENS + _VOCAB_START
            if _VOCAB_START <= code <= _VOCAB_END:
                chars.append(chr(code))
        return "".join(chars)

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        """HuggingFace-compatible call."""
        ids = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


def create_toy_transformer(
    dim: int = 32,
    n_layers: int = 1,
    n_heads: int = 2,
    n_positions: int = 64,
    vocab_size: int | None = None,
) -> "GPT2LMHeadModel":  # type: ignore[name-defined]
    """Build a tiny GPT-2 model sized for real-QPU experimentation.

    Args:
        dim: Hidden size. Defaults to 32 — keeps every matmul under
            ~5 qubits of input encoding, fits Heron comfortably.
        n_layers: Number of transformer blocks. 1 keeps the gate depth
            of the full forward small.
        n_heads: Number of attention heads. ``dim`` must be divisible.
        n_positions: Maximum sequence length.
        vocab_size: Optional override; defaults to the ToyTokenizer's vocab.

    Returns:
        A ``GPT2LMHeadModel`` instance. The weights are randomly initialized
        — pre-training is left to the user. Even untrained, the model is
        useful for measuring throughput, validating the pipeline, and
        running calibration circuits through the Qaithon backends.

    Example:
        >>> from qaithon.models import create_toy_transformer, ToyTokenizer
        >>> tokenizer = ToyTokenizer()
        >>> model = create_toy_transformer()
        >>> import qaithon
        >>> qaithon.compile(model, backends=("ibm.aer",))   # doctest: +SKIP
        >>> outputs = model.generate(**tokenizer("hello ", return_tensors="pt"),
        ...                          max_new_tokens=10)    # doctest: +SKIP
        >>> tokenizer.decode(outputs[0].tolist())          # doctest: +SKIP
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    if dim % n_heads != 0:
        raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads}).")
    if dim > 128:
        logger.warning(
            "dim=%d is larger than typical toy sizes (≤64). The model will "
            "still work but a single matmul may exceed practical QPU "
            "coherence depth.",
            dim,
        )

    config = GPT2Config(
        vocab_size=vocab_size or _VOCAB_SIZE,
        n_positions=n_positions,
        n_embd=dim,
        n_layer=n_layers,
        n_head=n_heads,
        # Aggressively shrink the FFN intermediate dimension so it also fits.
        n_inner=dim * 2,
        bos_token_id=_BOS_ID,
        eos_token_id=_EOS_ID,
    )
    model = GPT2LMHeadModel(config)
    # Initialize the LM head with reasonable scale so generation isn't degenerate.
    with torch.no_grad():
        std = 1.0 / math.sqrt(dim)
        model.lm_head.weight.normal_(mean=0.0, std=std)
    return model


def create_micro_transformer(
    dim: int = 16,
    n_positions: int = 32,
) -> "GPT2LMHeadModel":  # type: ignore[name-defined]
    """Even smaller transformer — single layer, single head, dim 16.

    Suitable as the absolute minimum smoke test: when you want to verify
    that Qaithon + your Qiskit setup + your IBM token can actually drive a
    real QPU to completion, without spending more than ~5 minutes total.

    See :func:`create_toy_transformer` for the parameters; this just locks
    them to the smallest practical values.
    """
    return create_toy_transformer(
        dim=dim,
        n_layers=1,
        n_heads=1,
        n_positions=n_positions,
    )
