"""Tests for the Mixtral expert-weight handler."""

from __future__ import annotations

import torch
from torch import nn

from qaithon.handlers.mixtral import is_mixtral_model, list_mixtral_experts


class _FakeMixtralExperts(nn.Module):
    """Simulates the modern Mixtral expert layout: a single 3D weight."""

    def __init__(self, num_experts: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, in_features, out_features)
        )


class _FakeMixtralBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Class name matters — handler checks startswith("Mixtral").
        self.experts = _FakeMixtralExperts(num_experts=4, in_features=8, out_features=16)


# Rename so type(module).__name__ starts with "Mixtral".
_FakeMixtralExperts.__name__ = "MixtralExperts"


class TestDetection:
    def test_detects_mixtral_block(self):
        model = nn.Sequential(_FakeMixtralBlock())
        assert is_mixtral_model(model) is True

    def test_detects_negative_on_regular_model(self):
        model = nn.Sequential(nn.Linear(8, 4))
        assert is_mixtral_model(model) is False


class TestEnumeration:
    def test_lists_all_experts(self):
        model = nn.Sequential(_FakeMixtralBlock())
        refs = list_mixtral_experts(model)
        assert len(refs) == 4
        assert {r.expert_index for r in refs} == {0, 1, 2, 3}
        for r in refs:
            assert r.in_features == 8
            assert r.out_features == 16
            assert r.weight_name == "gate_up_proj"
