"""Tests for the PennyLane-family backends (PennyLane / IBM Quantum / AWS Braket).

All three share the same matmul implementation but distinct cost profiles,
so we test the family parametrically.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from qaithon.backends import get_backend, list_backends
from qaithon.backends.pennylane_backend import (
    AWSBraketSimBackend,
    IBMQuantumSimBackend,
    PennyLaneSimBackend,
)


@pytest.fixture(
    params=[
        ("pennylane.sim", PennyLaneSimBackend, "quantum"),
        ("ibm.quantum", IBMQuantumSimBackend, "quantum"),
        ("aws.braket", AWSBraketSimBackend, "quantum"),
    ],
    ids=["pennylane", "ibm", "aws"],
)
def backend_spec(request):
    return request.param


class TestRegistration:
    def test_all_registered_at_import(self):
        registered = list_backends()
        assert "pennylane.sim" in registered
        assert "ibm.quantum" in registered
        assert "aws.braket" in registered

    def test_get_returns_correct_class(self, backend_spec):
        name, cls, _ = backend_spec
        backend = get_backend(name)
        assert isinstance(backend, cls)


class TestProfile:
    def test_kind_is_quantum(self, backend_spec):
        _, cls, expected_kind = backend_spec
        assert cls.profile.kind == expected_kind

    def test_ibm_advertises_queue_time(self):
        # IBM Quantum hardware has real queue; pure PennyLane simulator does not.
        assert IBMQuantumSimBackend.profile.queue_us > 0
        assert PennyLaneSimBackend.profile.queue_us == 0
        assert AWSBraketSimBackend.profile.queue_us > 0

    def test_real_qpus_advertise_no_autograd(self):
        # Real QPUs cannot do autograd; pennylane.sim (simulator) can.
        assert IBMQuantumSimBackend.profile.supports_autograd is False
        assert AWSBraketSimBackend.profile.supports_autograd is False
        assert PennyLaneSimBackend.profile.supports_autograd is True


class TestMatmul:
    def test_matches_F_linear(self, backend_spec):
        name, _, _ = backend_spec
        backend = get_backend(name)
        x = torch.randn(3, 8)
        w = torch.randn(4, 8)
        b = torch.randn(4)
        assert torch.allclose(backend.matmul(x, w, b), F.linear(x, w, b))

    def test_genuine_matmul_matches_classical(self):
        # Genuine qubit kernel computes the matmul exactly (inference-only).
        import torch.nn.functional as F

        backend = PennyLaneSimBackend()
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w), atol=1e-4)


class TestSelectorRespectAutograd:
    """When training mode is on, ibm.quantum / aws.braket must be excluded."""

    def test_qpu_backends_excluded_when_training(self):
        import torch.nn as nn

        from qaithon.ir import AutoBackendSelector, analyze_model

        model = nn.Sequential(nn.Linear(16, 32))
        plan = analyze_model(model)
        selector = AutoBackendSelector()
        result = selector.select(plan, objective="balanced", require_autograd=True)

        # The chosen backend must support autograd.
        for b in result.per_layer.values():
            assert b.profile.supports_autograd is True
