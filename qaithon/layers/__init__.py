"""Photonic / quantum equivalents of transformer building blocks.

Drop-in replacements for ``torch.nn`` layers that delegate their forward pass
to a configured backend (simulator, cloud quantum hardware, on-premise chip).
"""

from __future__ import annotations

from qaithon.layers.quantum_attention import QuantumAttention
from qaithon.layers.quantum_linear import QuantumLinear

__all__ = ["QuantumAttention", "QuantumLinear"]
