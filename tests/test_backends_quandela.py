"""Tests for :class:`QuandelaSimBackend`."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from qaithon.backends import get_backend, list_backends
from qaithon.backends.quandela_sim import QuandelaSimBackend


class TestRegistration:
    def test_registered_at_import(self):
        assert "quandela.sim" in list_backends()

    def test_get_returns_instance(self):
        backend = get_backend("quandela.sim")
        assert isinstance(backend, QuandelaSimBackend)


class TestProfile:
    def test_kind_is_photonic(self):
        assert QuandelaSimBackend.profile.kind == "photonic"

    def test_energy_better_than_classical(self):
        # Per design, photonic should advertise a lower energy than the
        # classical baseline used by the selector (1.0 pJ/MAC).
        assert QuandelaSimBackend.profile.energy_pj_per_mac < 1.0


class TestMatmul:
    def test_shape_contract(self):
        backend = QuandelaSimBackend()
        x = torch.rand(3, 8)
        w = torch.randn(4, 8)
        y = backend.matmul(x, w)
        assert y.shape == (3, 4)

    def test_supports_extra_batch_dims(self):
        backend = QuandelaSimBackend()
        x = torch.rand(2, 5, 8)
        w = torch.randn(4, 8)
        y = backend.matmul(x, w)
        assert y.shape == (2, 5, 4)

    def test_default_matches_F_linear(self):
        """Default config (no normalization) preserves semantics exactly."""
        backend = QuandelaSimBackend()
        x = torch.randn(2, 6)
        w = torch.randn(3, 6)
        b = torch.randn(3)
        expected = F.linear(x, w, b)
        got = backend.matmul(x, w, b)
        assert torch.allclose(got, expected)

    def test_normalize_applies_sigmoid_first(self):
        """With normalization on, output equals ``F.linear(sigmoid(x), ...)``."""
        backend = QuandelaSimBackend(normalize_inputs=True)
        x = torch.randn(2, 6)
        w = torch.randn(3, 6)
        expected = F.linear(torch.sigmoid(x), w, None)
        got = backend.matmul(x, w, None)
        assert torch.allclose(got, expected)

    def test_genuine_matmul_matches_classical(self):
        # The genuine photonic kernel computes the matmul exactly (inference-only,
        # not differentiable — training uses the differentiable PhotonicLayer).
        import torch.nn.functional as F

        backend = QuandelaSimBackend()
        x = torch.rand(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w), atol=1e-4)


class TestNoise:
    def test_negative_noise_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            QuandelaSimBackend(noise_std=-0.1)

    def test_genuine_output_is_exact(self):
        # Genuine photonic compute reproduces F.linear exactly — no injected noise.
        # (Real fidelity loss only appears on physical hardware, not in simulation.)
        import torch.nn.functional as F

        be = QuandelaSimBackend(normalize_inputs=False)
        x = torch.randn(4, 8)
        w = torch.randn(3, 8)
        assert torch.allclose(be.matmul(x, w), F.linear(x, w), atol=1e-4)


class TestAvailability:
    def test_is_available_when_merlin_installed(self):
        # In our test env, perceval+merlin ARE installed, so this should be True.
        # If a user runs without [quandela] extra, it will report False — that
        # is correct behavior, but we can't easily simulate it from here.
        assert QuandelaSimBackend().is_available() is True


class TestSelectorIntegration:
    """Validate that the AutoBackendSelector picks quandela.sim for energy.

    Restrict to (mock, quandela.sim) so the test is independent of which
    other backends happen to be registered.
    """

    def test_selector_prefers_quandela_for_energy_among_two(self):
        import torch.nn as nn

        from qaithon.ir import AutoBackendSelector, analyze_model

        model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
        plan = analyze_model(model)
        selector = AutoBackendSelector(candidate_names=("mock", "quandela.sim"))
        result = selector.select(plan, objective="energy")

        # Mock declares 1.0 pJ/MAC honestly; quandela.sim declares 0.05.
        # With the energy objective, every layer must go to quandela.sim.
        for backend in result.per_layer.values():
            assert backend.profile.name == "quandela.sim"
