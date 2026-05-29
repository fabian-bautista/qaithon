"""Tests for the reference :class:`MockBackend`.

These tests pin the contract every other backend must satisfy. If MockBackend
drifts from ``F.linear``, the compile pipeline's correctness guarantees go
with it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from qaithon.backends import get_backend, list_backends
from qaithon.backends.mock import MockBackend


class TestMockBackend:
    def test_registered_by_default(self):
        """Importing the package must auto-register the mock backend."""
        assert "mock" in list_backends()

    def test_get_returns_instance(self):
        backend = get_backend("mock")
        assert isinstance(backend, MockBackend)

    def test_profile_is_mock_kind(self):
        backend = MockBackend()
        assert backend.profile.name == "mock"
        assert backend.profile.kind == "mock"
        assert backend.profile.supports_autograd is True

    def test_matmul_matches_F_linear(self):
        """Without noise, MockBackend must be numerically identical to F.linear."""
        backend = MockBackend()
        x = torch.randn(3, 8)
        w = torch.randn(4, 8)
        b = torch.randn(4)

        expected = F.linear(x, w, b)
        got = backend.matmul(x, w, b)
        assert torch.allclose(got, expected, atol=0.0, rtol=0.0)

    def test_matmul_no_bias(self):
        backend = MockBackend()
        x = torch.randn(2, 5)
        w = torch.randn(3, 5)

        expected = F.linear(x, w, None)
        got = backend.matmul(x, w, None)
        assert torch.allclose(got, expected)

    def test_matmul_preserves_gradients(self):
        backend = MockBackend()
        x = torch.randn(4, 6, requires_grad=True)
        w = torch.randn(2, 6, requires_grad=True)

        y = backend.matmul(x, w).sum()
        y.backward()

        assert x.grad is not None
        assert w.grad is not None
        # Compare against autograd through F.linear directly.
        x2 = x.detach().clone().requires_grad_(True)
        w2 = w.detach().clone().requires_grad_(True)
        F.linear(x2, w2).sum().backward()
        assert torch.allclose(x.grad, x2.grad)
        assert torch.allclose(w.grad, w2.grad)

    def test_noise_changes_output_within_tolerance(self):
        backend = MockBackend(noise_std=0.1, seed=42)
        x = torch.zeros(1, 4)
        w = torch.zeros(2, 4)
        y = backend.matmul(x, w)
        # All-zero inputs ⇒ deterministic noise pattern.
        assert y.shape == (1, 2)
        # The noise must not be exactly zero (probability of all-zeros is ~0).
        assert torch.any(y != 0)

    def test_negative_noise_std_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            MockBackend(noise_std=-0.1)
