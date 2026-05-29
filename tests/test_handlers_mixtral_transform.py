"""Tests for the Mixtral expert transformation pass."""

from __future__ import annotations

import torch
from torch import nn

from qaithon.handlers.mixtral import (
    is_mixtral_model,
    transform_mixtral_experts,
)
from qaithon.layers import QuantumLinear


class _FakeMixtralExperts(nn.Module):
    """A 3D-weight expert block, mimicking modern Mixtral implementations."""

    def __init__(self, num_experts: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Naive forward — averages outputs across experts. Real Mixtral uses
        # routing logits to pick top-k experts per token.
        outs = [x @ self.weight[i] for i in range(self.weight.shape[0])]
        return torch.stack(outs, dim=0).mean(dim=0)


_FakeMixtralExperts.__name__ = "MixtralExperts"


class TestTransform:
    def test_detects_mixtral(self):
        block = nn.Sequential(_FakeMixtralExperts(num_experts=4, in_features=8, out_features=16))
        assert is_mixtral_model(block) is True

    def test_transform_replaces_experts(self):
        block = nn.Sequential(_FakeMixtralExperts(num_experts=4, in_features=8, out_features=16))
        result = transform_mixtral_experts(block)
        assert result.n_blocks_transformed == 1
        assert result.n_experts_replaced == 4

        experts_module = block[0]  # the FakeMixtralExperts
        assert hasattr(experts_module, "quantum_experts")
        assert isinstance(experts_module.quantum_experts, nn.ModuleList)
        assert all(isinstance(e, QuantumLinear) for e in experts_module.quantum_experts)

    def test_transform_no_op_on_non_mixtral(self):
        block = nn.Sequential(nn.Linear(8, 16))
        result = transform_mixtral_experts(block)
        assert result.n_blocks_transformed == 0
        assert result.n_experts_replaced == 0
