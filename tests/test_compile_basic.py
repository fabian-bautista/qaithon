"""End-to-end tests for :func:`qaithon.compile`.

These tests are the canonical specification of Qaithon's promise to AI
developers: pass a model, get a transformed model that works identically
in the no-noise case, with an audit attached.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import qaithon
from qaithon.compile_report import CompileReport
from qaithon.layers.quantum_linear import QuantumLinear


class TestSmokeCompile:
    def test_returns_same_object_in_place(self, tiny_mlp):
        # Override min_in_features so the toy MLP's layers (dim=16, 32) are
        # not filtered by the default family profile (which sets min_in=64).
        compiled = qaithon.compile(tiny_mlp, min_in_features=1, min_out_features=1)
        assert compiled is tiny_mlp

    def test_attaches_report(self, tiny_mlp):
        qaithon.compile(tiny_mlp, min_in_features=1, min_out_features=1)
        assert hasattr(tiny_mlp, "qaithon_report")
        assert isinstance(tiny_mlp.qaithon_report, CompileReport)

    def test_replaces_linear_layers(self, tiny_mlp):
        qaithon.compile(tiny_mlp, min_in_features=1, min_out_features=1)
        # Both Linear layers should now be QuantumLinear.
        assert isinstance(tiny_mlp.fc1, QuantumLinear)
        assert isinstance(tiny_mlp.fc2, QuantumLinear)

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="nn.Module"):
            qaithon.compile("not a model")  # type: ignore[arg-type]


class TestNumericalIdentityWithMockBackend:
    """With the mock backend (zero noise), output must be bitwise identical."""

    def test_output_identical_after_compile(self, tiny_mlp, sample_input):
        with torch.no_grad():
            expected = tiny_mlp(sample_input).clone()

        qaithon.compile(tiny_mlp, backends=("mock",))

        with torch.no_grad():
            got = tiny_mlp(sample_input)

        assert torch.allclose(got, expected, atol=0.0, rtol=0.0)

    def test_gradients_match_after_compile(self, tiny_mlp, sample_input):
        # Snapshot baseline gradient.
        x1 = sample_input.detach().clone().requires_grad_(True)
        baseline_model = type(tiny_mlp)()
        baseline_model.load_state_dict(tiny_mlp.state_dict())
        baseline_model(x1).sum().backward()

        qaithon.compile(tiny_mlp, backends=("mock",))

        x2 = sample_input.detach().clone().requires_grad_(True)
        tiny_mlp(x2).sum().backward()

        assert torch.allclose(x1.grad, x2.grad, atol=1e-6)


class TestReportContents:
    def test_report_lists_replaced_layers(self, tiny_mlp):
        qaithon.compile(tiny_mlp, min_in_features=1, min_out_features=1)
        report = tiny_mlp.qaithon_report
        names = {d.layer_name for d in report.decisions}
        assert names == {"fc1", "fc2"}

    def test_report_records_objective(self, tiny_mlp):
        qaithon.compile(tiny_mlp, optimize_for="energy", min_in_features=1, min_out_features=1)
        assert tiny_mlp.qaithon_report.optimize_for == "energy"

    def test_report_pretty_returns_str(self, tiny_mlp):
        qaithon.compile(tiny_mlp, min_in_features=1, min_out_features=1)
        text = tiny_mlp.qaithon_report.pretty()
        assert isinstance(text, str)
        assert "TinyMLP" in text


class TestIdempotency:
    def test_compile_twice_does_not_double_wrap(self, tiny_mlp):
        qaithon.compile(tiny_mlp)
        # After compile, the layers are QuantumLinear, not Linear.
        # The walker's identity check excludes QuantumLinear (not nn.Linear).
        qaithon.compile(tiny_mlp)
        # Should report zero new replacements on the second pass.
        assert tiny_mlp.qaithon_report.n_replaced == 0


class TestTiedWeightsAreSkipped:
    def test_tied_head_kept_intact(self, tied_head_model):
        qaithon.compile(tied_head_model, min_in_features=1, min_out_features=1)
        # The tied head must stay as a regular nn.Linear.
        assert type(tied_head_model.head) is nn.Linear
        # The non-tied proj should have been replaced.
        assert isinstance(tied_head_model.proj, QuantumLinear)


class TestExpertOverrides:
    def test_skip_predicate(self, tiny_mlp):
        qaithon.compile(
            tiny_mlp,
            skip=lambda name, _m: name == "fc1",
            min_in_features=1,
            min_out_features=1,
        )
        assert type(tiny_mlp.fc1) is nn.Linear  # untouched
        assert isinstance(tiny_mlp.fc2, QuantumLinear)

    def test_min_in_features(self, tiny_mlp):
        # Set both thresholds explicitly so the family default doesn't
        # interfere with the test's intent.
        qaithon.compile(tiny_mlp, min_in_features=20, min_out_features=1)
        # fc1 has in_features=16, should be skipped.
        assert type(tiny_mlp.fc1) is nn.Linear
        # fc2 has in_features=32, should be replaced.
        assert isinstance(tiny_mlp.fc2, QuantumLinear)
