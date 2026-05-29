"""Regression tests for save/load of Qaithon-compiled models.

Pin the contract that compiled models survive a round-trip through:
* ``torch.save`` / ``torch.load``
* ``safetensors`` (when installed)
* ``model.state_dict()`` reload

This is the foundation HuggingFace Hub portability sits on. If these break,
``from_pretrained`` of compiled models breaks too.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

import qaithon


class SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class TestTorchSerialization:
    def test_state_dict_keys_are_standard(self):
        model = SmallNet()
        qaithon.compile(model)
        keys = set(model.state_dict().keys())
        assert keys == {"a.weight", "a.bias", "b.weight", "b.bias"}

    def test_torch_save_and_load_roundtrip(self, tmp_path):
        model = SmallNet()
        x = torch.randn(2, 8)
        qaithon.compile(model)
        y_before = model(x)

        path = tmp_path / "ckpt.pt"
        torch.save(model.state_dict(), path)

        model2 = SmallNet()
        qaithon.compile(model2)
        model2.load_state_dict(torch.load(path, weights_only=True))

        y_after = model2(x)
        assert torch.allclose(y_before, y_after)


_HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


@pytest.mark.skipif(not _HAS_SAFETENSORS, reason="safetensors not installed")
class TestSafetensors:
    def test_save_and_load_roundtrip(self, tmp_path):
        from safetensors.torch import load_file, save_file

        model = SmallNet()
        x = torch.randn(2, 8)
        qaithon.compile(model)
        y_before = model(x)

        path = tmp_path / "model.safetensors"
        save_file(model.state_dict(), str(path))

        model2 = SmallNet()
        qaithon.compile(model2)
        model2.load_state_dict(load_file(str(path)))

        y_after = model2(x)
        assert torch.allclose(y_before, y_after)
