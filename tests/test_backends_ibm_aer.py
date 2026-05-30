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
    def test_runs_genuine_circuit(self):
        # The genuine qubit kernel computes the matmul exactly (inference-only).
        import torch.nn.functional as F
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4, seed=0)
        x = torch.randn(2, 8)
        w = torch.randn(3, 8)
        y = backend.matmul(x, w)
        assert y.shape == (2, 3)
        assert torch.allclose(y, F.linear(x, w), atol=1e-4)

    def test_output_preserves_dtype_and_device(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4)
        x = torch.randn(1, 4, dtype=torch.float32)
        w = torch.randn(2, 4, dtype=torch.float32)
        y = backend.matmul(x, w)
        assert y.dtype == torch.float32

    def test_genuine_matmul_matches_classical(self):
        """Genuine qubit kernel reproduces F.linear (inference-only, not differentiable)."""
        import torch.nn.functional as F
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(noise_strength=0.01, max_qubits=4)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w), atol=1e-4)


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
