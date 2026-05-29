"""Tests for QaithonConfig and the HfQuantizer integration scaffolding."""

from __future__ import annotations

import torch
from torch import nn

import qaithon
from qaithon.hf_integration import QaithonConfig, QaithonHfQuantizer


class TestQaithonConfig:
    def test_default_quant_method(self):
        cfg = QaithonConfig()
        assert cfg.quant_method == "qaithon"

    def test_to_dict_roundtrip(self):
        cfg = QaithonConfig(
            backends=("mock", "quandela.sim"),
            optimize_for="energy",
            strict=False,
        )
        d = cfg.to_dict()
        cfg2 = QaithonConfig.from_dict(d)
        assert cfg2.backends == cfg.backends
        assert cfg2.optimize_for == cfg.optimize_for
        assert cfg2.strict == cfg.strict
        assert cfg2.quant_method == cfg.quant_method


class TestQuantizerApplication:
    def test_process_compiles_model(self):
        cfg = QaithonConfig(backends=("mock",))
        quantizer = QaithonHfQuantizer(cfg)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        model = Tiny()
        x = torch.randn(2, 4)
        y_before = model(x)

        compiled = quantizer.process_model_after_load(model)
        assert compiled is model
        assert hasattr(model, "qaithon_report")
        y_after = model(x)
        # Numerical identity preserved when using mock backend.
        assert torch.allclose(y_before, y_after)
