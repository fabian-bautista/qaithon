"""Reference architectures sized for actual quantum hardware execution.

These models are intentionally tiny so they fit within the gate-depth and
qubit budgets of real QPUs (IBM Heron today, ~30 qubits practical depth).
They are the right harness for "does my pipeline really work on real
hardware?" experiments — without spending hours waiting on circuits that
were never going to converge.

Pre-built model factories
-------------------------

* :func:`create_toy_transformer` — single-block char-level transformer
  with dim ≤ 32. Hits the sweet spot for Heron's coherence window.
* :func:`create_micro_transformer` — even smaller (dim 16, no attention),
  useful as a smoke test for the Qaithon → real-QPU path.

Both factories return standard HuggingFace ``PreTrainedModel`` instances,
so the rest of the ecosystem (Trainer, generate, accelerate, peft) works
unchanged.
"""

from __future__ import annotations

from qaithon.models.toy_transformer import (
    ToyTokenizer,
    create_micro_transformer,
    create_toy_transformer,
)

__all__ = [
    "ToyTokenizer",
    "create_micro_transformer",
    "create_toy_transformer",
]
