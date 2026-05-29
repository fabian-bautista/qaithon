"""Tests for :class:`qaithon.layers.QuantumLinear`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qaithon.layers import QuantumLinear


class TestPublicSurface:
    """QuantumLinear must mirror nn.Linear's public surface exactly."""

    def test_in_out_features_attributes(self):
        layer = QuantumLinear(8, 4)
        assert layer.in_features == 8
        assert layer.out_features == 4

    def test_weight_shape(self):
        layer = QuantumLinear(8, 4)
        assert layer.weight.shape == (4, 8)
        assert isinstance(layer.weight, nn.Parameter)

    def test_bias_present_by_default(self):
        layer = QuantumLinear(8, 4)
        assert layer.bias is not None
        assert layer.bias.shape == (4,)

    def test_bias_can_be_disabled(self):
        layer = QuantumLinear(8, 4, bias=False)
        assert layer.bias is None

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="positive"):
            QuantumLinear(0, 4)
        with pytest.raises(ValueError, match="positive"):
            QuantumLinear(4, 0)


class TestForward:
    def test_output_shape(self):
        layer = QuantumLinear(8, 4)
        x = torch.randn(2, 8)
        y = layer(x)
        assert y.shape == (2, 4)

    def test_supports_extra_batch_dims(self):
        layer = QuantumLinear(8, 4)
        x = torch.randn(3, 5, 8)
        y = layer(x)
        assert y.shape == (3, 5, 4)

    def test_gradient_flows(self):
        layer = QuantumLinear(8, 4)
        x = torch.randn(2, 8, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None


class TestFromLinearConstructor:
    def test_preserves_shape_and_bias(self):
        original = nn.Linear(8, 4)
        new = QuantumLinear.from_linear(original)
        assert new.in_features == 8
        assert new.out_features == 4
        assert new.bias is not None

    def test_copies_weights(self):
        original = nn.Linear(8, 4)
        new = QuantumLinear.from_linear(original)
        assert torch.allclose(new.weight, original.weight)
        assert torch.allclose(new.bias, original.bias)

    def test_does_not_copy_when_requested(self):
        torch.manual_seed(1)
        original = nn.Linear(8, 4)
        original_weight = original.weight.clone()
        torch.manual_seed(2)
        new = QuantumLinear.from_linear(original, copy_weights=False)
        # The new layer was re-initialized differently.
        assert not torch.allclose(new.weight, original_weight)


class TestBackendSwapping:
    def test_backend_setter_replaces_runtime(self):
        from qaithon.backends.mock import MockBackend

        layer = QuantumLinear(4, 2)
        assert layer.backend.profile.name == "mock"
        layer.backend = MockBackend(noise_std=0.5, seed=0)
        assert isinstance(layer.backend, MockBackend)

    def test_invalid_backend_string_raises(self):
        with pytest.raises(Exception, match="not registered"):
            QuantumLinear(4, 2, backend="nonexistent_backend_xyz")


class TestRepr:
    def test_repr_contains_backend_name(self):
        layer = QuantumLinear(8, 4)
        text = repr(layer)
        assert "mock" in text
        assert "8" in text and "4" in text
