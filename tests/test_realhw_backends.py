"""Tests for the real-hardware-dispatch backends.

These verify the mode contract and registration without making any real
cloud calls — that's why every test runs in ``mode="profile"`` only.
Real calibration paths are exercised by the smoke script
``phase0/notebooks/03_realhw_smoke.py`` (which costs cloud quota).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F  # noqa: N812


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_ibm_runtime") is None,
    reason="qiskit-ibm-runtime not installed",
)
class TestIBMHeron:
    def test_registered(self):
        from qaithon.backends import list_backends

        assert "ibm.heron" in list_backends()

    def test_profile_mode_matches_F_linear(self):
        from qaithon.backends.ibm_heron import IBMHeronBackend

        backend = IBMHeronBackend(mode="profile")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w))

    def test_rejects_invalid_mode(self):
        from qaithon.backends.ibm_heron import IBMHeronBackend

        with pytest.raises(ValueError, match="mode must be"):
            IBMHeronBackend(mode="wrong")  # type: ignore[arg-type]

    def test_profile_kind_quantum_no_autograd(self):
        from qaithon.backends.ibm_heron import IBMHeronBackend

        assert IBMHeronBackend.profile.kind == "quantum"
        assert IBMHeronBackend.profile.supports_autograd is False


@pytest.mark.skipif(
    importlib.util.find_spec("merlin") is None,
    reason="merlin not installed",
)
class TestQuandelaBelenos:
    def test_registered(self):
        from qaithon.backends import list_backends

        assert "quandela.belenos" in list_backends()

    def test_profile_mode_matches_F_linear(self):
        from qaithon.backends.quandela_belenos import QuandelaBelenosBackend

        backend = QuandelaBelenosBackend(mode="profile")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w))

    def test_kind_photonic(self):
        from qaithon.backends.quandela_belenos import QuandelaBelenosBackend

        assert QuandelaBelenosBackend.profile.kind == "photonic"


@pytest.mark.skipif(
    importlib.util.find_spec("braket") is None,
    reason="amazon-braket-sdk not installed",
)
class TestAWSBraketSV1:
    def test_registered(self):
        from qaithon.backends import list_backends

        assert "aws.braket.sv1" in list_backends()

    def test_profile_mode_matches_F_linear(self):
        from qaithon.backends.aws_braket_sv1 import AWSBraketSV1Backend

        backend = AWSBraketSV1Backend(mode="profile")
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        assert torch.allclose(backend.matmul(x, w), F.linear(x, w))

    def test_profile_kind_quantum(self):
        from qaithon.backends.aws_braket_sv1 import AWSBraketSV1Backend

        assert AWSBraketSV1Backend.profile.kind == "quantum"


class TestModeContract:
    """Mode validation applies to the shared base class, so test once."""

    def test_invalid_mode_rejected(self):
        from qaithon.backends.ibm_heron import IBMHeronBackend

        with pytest.raises(ValueError, match="mode must be"):
            IBMHeronBackend(mode="garbage")  # type: ignore[arg-type]
