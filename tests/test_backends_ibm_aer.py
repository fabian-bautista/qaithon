"""Tests for the AerSimulator-backed quantum backend."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

_HAS_AER = importlib.util.find_spec("qiskit_aer") is not None

pytestmark = pytest.mark.skipif(not _HAS_AER, reason="qiskit-aer not installed")


class TestRegistration:
    def test_registered(self):
        from qaithon.backends import list_backends

        assert "ibm.aer" in list_backends()


class TestProfile:
    def test_kind_is_quantum(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        assert IBMAerBackend.profile.kind == "quantum"
        assert IBMAerBackend.profile.supports_autograd is True

    def test_is_available_true_in_test_env(self):
        from qaithon.backends import get_backend

        assert get_backend("ibm.aer").is_available() is True


class TestMatmulNoNoise:
    def test_matches_F_linear_exactly_with_zero_noise(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.0)
        x = torch.randn(2, 8)
        w = torch.randn(3, 8)
        b = torch.randn(3)
        assert torch.allclose(backend.matmul(x, w, b), F.linear(x, w, b))


class TestMatmulWithRealAerExecution:
    def test_runs_circuit_and_records_latency(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4, seed=0)
        x = torch.randn(2, 8)
        w = torch.randn(3, 8)
        y = backend.matmul(x, w)
        assert y.shape == (2, 3)
        # AerSimulator was invoked at least once, so we recorded latency.
        assert backend.last_aer_latency_us > 0

    def test_output_preserves_dtype_and_device(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4)
        x = torch.randn(1, 4, dtype=torch.float32)
        w = torch.randn(2, 4, dtype=torch.float32)
        y = backend.matmul(x, w)
        assert y.dtype == torch.float32

    def test_gradients_flow_through_classical_path(self):
        """Noise is detached so it doesn't contribute, but the classical path must."""
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4)
        x = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(3, 4, requires_grad=True)
        backend.matmul(x, w).sum().backward()
        assert x.grad is not None
        assert w.grad is not None


class TestArgumentValidation:
    def test_rejects_negative_noise(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        with pytest.raises(ValueError, match="non-negative"):
            IBMAerBackend(noise_strength=-0.01)

    def test_rejects_out_of_range_qubits(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        with pytest.raises(ValueError, match="max_qubits"):
            IBMAerBackend(max_qubits=0)
        with pytest.raises(ValueError, match="max_qubits"):
            IBMAerBackend(max_qubits=99)
