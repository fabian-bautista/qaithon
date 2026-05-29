"""Tests pinning Qaithon's device-agnostic behavior.

Qaithon does not write CUDA or Metal code directly. Everything inherits
from PyTorch's device dispatch: the same code paths run on CPU, CUDA, MPS,
XLA, or any future backend PyTorch supports. These tests pin the
contract so we notice immediately if a refactor accidentally hardcodes a
device.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import qaithon
from qaithon.layers import QuantumLinear

_MPS = torch.backends.mps.is_available()
_CUDA = torch.cuda.is_available()

DEVICES_TO_TEST: list[str] = ["cpu"]
if _MPS:
    DEVICES_TO_TEST.append("mps")
if _CUDA:
    DEVICES_TO_TEST.append("cuda")


class Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


@pytest.mark.parametrize("device", DEVICES_TO_TEST)
class TestDeviceAgnostic:
    def test_compile_in_place_preserves_device(self, device):
        model = Small().to(device)
        qaithon.compile(model)
        for p in model.parameters():
            assert p.device.type == device

    def test_compile_then_move_works(self, device):
        if device == "cpu":
            pytest.skip("no-op on CPU baseline")
        model = Small()  # CPU
        qaithon.compile(model)
        model = model.to(device)
        for p in model.parameters():
            assert p.device.type == device

    def test_forward_on_device(self, device):
        model = Small().to(device)
        qaithon.compile(model)
        x = torch.randn(2, 8, device=device)
        y = model(x)
        assert y.device.type == device

    def test_gradient_flows_on_device(self, device):
        model = Small().to(device)
        qaithon.compile(model)
        x = torch.randn(2, 8, device=device, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None
        assert x.grad.device.type == device


class TestQuantumLinearDirect:
    def test_from_linear_inherits_device(self):
        if not _MPS:
            pytest.skip("MPS not available")
        original = nn.Linear(4, 8).to("mps")
        new = QuantumLinear.from_linear(original)
        assert new.weight.device.type == "mps"
        assert new.bias.device.type == "mps"

    def test_explicit_device_argument(self):
        layer = QuantumLinear(4, 8, device="cpu")
        assert layer.weight.device.type == "cpu"
