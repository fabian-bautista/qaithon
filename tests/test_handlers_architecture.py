"""Tests for automatic architecture detection and recommended configs."""

from __future__ import annotations

from torch import nn

from qaithon.handlers.architecture import (
    ArchitectureProfile,
    detect_architecture,
    list_architectures,
    recommend_config,
    register_architecture,
)


class _FakeLlamaModel(nn.Module):
    pass


class _FakeGPT2LMHeadModel(nn.Module):
    pass


class _UnknownModel(nn.Module):
    pass


_FakeLlamaModel.__name__ = "LlamaForCausalLM"
_FakeGPT2LMHeadModel.__name__ = "GPT2LMHeadModel"


class TestRegistry:
    def test_default_registry_has_known_families(self):
        names = set(list_architectures())
        for family in ("generic", "llama", "gpt2", "mistral", "mixtral", "phi", "qwen", "gemma", "bert"):
            assert family in names

    def test_recommend_config_for_known(self):
        cfg = recommend_config("llama")
        assert cfg.family == "llama"
        assert "lm_head" in cfg.skip_name_patterns

    def test_recommend_config_falls_back_to_generic(self):
        cfg = recommend_config("nonexistent")
        assert cfg.family == "generic"

    def test_register_new_family(self):
        profile = ArchitectureProfile(
            family="testfam",
            description="x",
            skip_name_patterns=("foo",),
        )
        register_architecture(profile)
        assert recommend_config("testfam").family == "testfam"

    def test_double_register_rejected(self):
        profile = ArchitectureProfile(
            family="doublefam",
            description="x",
        )
        register_architecture(profile)
        try:
            register_architecture(profile)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestDetection:
    def test_detects_llama(self):
        assert detect_architecture(_FakeLlamaModel()) == "llama"

    def test_detects_gpt2(self):
        assert detect_architecture(_FakeGPT2LMHeadModel()) == "gpt2"

    def test_falls_back_to_generic(self):
        assert detect_architecture(_UnknownModel()) == "generic"

    def test_detects_through_child_module(self):
        outer = nn.Sequential(_FakeLlamaModel())
        # Top-level class is nn.Sequential; child reveals llama.
        assert detect_architecture(outer) == "llama"


class TestCompileIntegration:
    def test_family_skip_pattern_applied_automatically(self):
        """Llama profile says 'embed_tokens' is skipped; verify compile honors it."""
        import qaithon

        class FakeLlama(nn.Module):
            def __init__(self):
                super().__init__()
                # Use dims >= min_in_features (default 64 in Llama profile).
                self.embed_tokens = nn.Linear(128, 128)  # should be skipped by name
                self.layers_0_self_attn_q_proj = nn.Linear(128, 128)  # kept
                self.lm_head = nn.Linear(128, 128)  # should be skipped by name

            def forward(self, x):
                return self.lm_head(self.layers_0_self_attn_q_proj(self.embed_tokens(x)))

        FakeLlama.__name__ = "LlamaForCausalLM"
        model = FakeLlama()
        qaithon.compile(model)
        report = model.qaithon_report
        replaced_names = {d.layer_name for d in report.decisions}
        # Only the middle projection should have been replaced.
        assert "embed_tokens" not in replaced_names
        assert "lm_head" not in replaced_names
        assert "layers_0_self_attn_q_proj" in replaced_names
