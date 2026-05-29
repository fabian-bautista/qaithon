"""Hub of pre-trained Qaithon blocks.

The Hub is Qaithon's bet on the HuggingFace pattern: distribution channel,
not just code. Users will eventually be able to do::

    from qaithon.hub import load_block

    attention = load_block("qaithon/quantum-attention-llama-v1")
    # `attention` is an nn.Module honoring the standard Tensor-in/Tensor-out
    # contract — use it as a drop-in MultiheadAttention replacement.

For v0.1 the API is implemented as a thin skeleton — names and signatures
are stable, but the actual artifact catalog is empty. This is intentional:
fixing the API surface now means downstream code never has to change when
real artifacts are published.

Real-artifact loading lands in v0.x once we have:

* A naming convention for HF Hub repos (``qaithon/<family>-<variant>-<vN>``).
* A storage format (safetensors for weights, JSON for metadata).
* At least one published block to validate the round-trip.
"""

from __future__ import annotations

from qaithon.hub.loader import list_blocks, load_block, push_block

__all__ = [
    "list_blocks",
    "load_block",
    "push_block",
]
