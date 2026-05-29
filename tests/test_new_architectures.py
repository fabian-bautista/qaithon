"""Smoke tests for the 8 architectures added in this session."""

from __future__ import annotations

import pytest
from torch import nn

from qaithon.handlers.architecture import detect_architecture, recommend_config


def _make_fake(class_name: str) -> nn.Module:
    """Return an empty nn.Module whose Python class name is ``class_name``."""

    cls = type(class_name, (nn.Module,), {"__init__": lambda self: super(cls, self).__init__()})
    return cls()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("class_name", "expected_family"),
    [
        ("StarcoderForCausalLM", "starcoder"),
        ("StarCoder2ForCausalLM", "starcoder"),
        ("GPTBigCodeForCausalLM", "starcoder"),
        ("DeepSeekV2ForCausalLM", "deepseek"),
        ("DeepseekV3ForCausalLM", "deepseek"),
        ("FalconForCausalLM", "falcon"),
        ("MambaForCausalLM", "mamba"),
        ("GPTNeoXForCausalLM", "pythia"),
        ("OPTForCausalLM", "opt"),
        ("T5ForConditionalGeneration", "t5"),
        ("MT5ForConditionalGeneration", "t5"),
        ("BloomForCausalLM", "bloom"),
    ],
)
def test_detects_new_families(class_name: str, expected_family: str):
    model = _make_fake(class_name)
    assert detect_architecture(model) == expected_family


@pytest.mark.parametrize(
    "family",
    ["starcoder", "deepseek", "falcon", "mamba", "pythia", "opt", "t5", "bloom"],
)
def test_recommends_config_for_new_families(family: str):
    cfg = recommend_config(family)
    assert cfg.family == family
    assert cfg.description  # non-empty
