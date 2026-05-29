"""Tests for the DeepQuantum backend (skip if not installed)."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_DEEPQUANTUM = importlib.util.find_spec("deepquantum") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_DEEPQUANTUM, reason="deepquantum not installed"
)


class TestDeepQuantum:
    def test_registered(self):
        from qaithon.backends import list_backends

        assert "deepquantum" in list_backends()

    def test_profile_quantum_kind(self):
        from qaithon.backends.deepquantum_backend import DeepQuantumBackend

        assert DeepQuantumBackend.profile.kind == "quantum"
        assert DeepQuantumBackend.profile.supports_autograd is True

    def test_matmul_matches_F_linear(self):
        import torch
        import torch.nn.functional as F  # noqa: N812

        from qaithon.backends import get_backend

        backend = get_backend("deepquantum")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w))

    def test_is_available(self):
        from qaithon.backends import get_backend

        assert get_backend("deepquantum").is_available() is True
