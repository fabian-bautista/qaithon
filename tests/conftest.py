"""Shared pytest fixtures.

Fixtures are deliberately tiny (toy models) so the test suite stays fast
and runnable without HuggingFace, GPUs, or any heavy dep installed.
Tests that need real LLMs are marked with ``@pytest.mark.needs_huggingface``
and skipped by default.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


# Set a deterministic seed for every test session. Individual tests can still
# reset the seed if they want to.
@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    """A 2-layer MLP — minimal realistic test target for Qaithon."""

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TiedHead(nn.Module):
    """Model with tied embedding ↔ output head, mimicking GPT-2 / Llama-3.2-1B."""

    def __init__(self, vocab: int = 10, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        # Tie the head's weight to the embedding's weight (standard pattern).
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        z = self.proj(z)
        return self.head(z)


@pytest.fixture
def tiny_mlp() -> TinyMLP:
    """A small MLP with two real `nn.Linear` layers."""
    return TinyMLP()


@pytest.fixture
def tied_head_model() -> TiedHead:
    """Model with tied weights — replacement must skip the tied head."""
    return TiedHead()


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Standard input shaped to match TinyMLP."""
    return torch.randn(4, 16)
