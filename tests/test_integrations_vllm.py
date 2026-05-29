"""Tests for the vLLM integration helpers."""

from __future__ import annotations

from torch import nn

from qaithon.integrations.vllm import check_vllm_compatibility, vllm_available


class TestCompatibility:
    def test_returns_tuple(self):
        ok, reason = check_vllm_compatibility(nn.Sequential())
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_known_supported_family_reports_ok(self):
        # Build a fake Llama model so detect_architecture returns "llama".
        class FakeLlama(nn.Module):
            pass

        FakeLlama.__name__ = "LlamaForCausalLM"
        model = FakeLlama()
        ok, _reason = check_vllm_compatibility(model)
        assert ok is True

    def test_unsupported_family_reports_reason(self):
        # A bare nn.Sequential maps to "generic", which is not in vLLM's set.
        ok, reason = check_vllm_compatibility(nn.Sequential())
        assert ok is False
        assert "generic" in reason or "support" in reason

    def test_mixtral_transformed_model_rejected(self):
        # Build a Mixtral-named wrapper so detect_architecture passes the first
        # check, then attach `quantum_experts` to a child so the second check fires.
        class FakeMixtral(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.Linear(4, 4)
                self.experts.quantum_experts = nn.ModuleList([nn.Linear(4, 4)])

        FakeMixtral.__name__ = "MixtralForCausalLM"
        ok, reason = check_vllm_compatibility(FakeMixtral())
        assert ok is False
        assert "Mixtral" in reason or "quantum_experts" in reason


class TestAvailability:
    def test_vllm_available_returns_bool(self):
        assert isinstance(vllm_available(), bool)
