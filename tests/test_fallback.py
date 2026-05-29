"""Tests for the graceful runtime fallback wrapper."""

from __future__ import annotations

import pytest
import torch

from qaithon.backends import get_backend
from qaithon.backends.base import Backend, BackendProfile
from qaithon.fallback import FallbackBackend


class _BrokenBackend(Backend):
    profile = BackendProfile(name="broken", kind="mock", energy_pj_per_mac=0.0)

    def __init__(self) -> None:
        self.calls = 0

    def matmul(self, x, weight, bias=None):
        self.calls += 1
        raise RuntimeError("simulated failure")


class TestFallbackBackend:
    def test_returns_primary_when_primary_works(self):
        mock = get_backend("mock")
        backend = FallbackBackend(mock, fallbacks=[_BrokenBackend()])
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        y = backend.matmul(x, w)
        assert y.shape == (2, 3)

    def test_falls_through_to_next_backend(self):
        broken = _BrokenBackend()
        backend = FallbackBackend(broken, fallbacks=[get_backend("mock")])
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        y = backend.matmul(x, w)
        assert y.shape == (2, 3)
        assert broken.calls == 1

    def test_raises_when_all_fail(self):
        backend = FallbackBackend(_BrokenBackend(), fallbacks=[_BrokenBackend()])
        with pytest.raises(RuntimeError, match="simulated failure"):
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))

    def test_adopts_primary_profile(self):
        primary = get_backend("quandela.sim")
        backend = FallbackBackend(primary, fallbacks=[get_backend("mock")])
        assert backend.profile.name == "quandela.sim"
        assert backend.profile.kind == "photonic"
