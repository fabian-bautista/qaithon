"""Tests for the analyzer/walker.

These tests pin the rules that decide which layers Qaithon replaces:

* Identity check (``type is nn.Linear``) excludes subclasses like
  ``NonDynamicallyQuantizableLinear`` inside ``nn.MultiheadAttention``.
* Tied weights are detected and skipped.
* Already-quantized models raise cleanly.
* Min-size filters work.
"""

from __future__ import annotations

import pytest
from torch import nn

from qaithon.exceptions import IncompatibleModelError
from qaithon.ir import analyze_model


class TestBasicWalker:
    def test_finds_all_linear_in_tiny_mlp(self, tiny_mlp):
        plan = analyze_model(tiny_mlp)
        names = {m.name for m in plan.matches}
        assert names == {"fc1", "fc2"}
        assert plan.total_replaceable == 2

    def test_min_in_features_filter(self, tiny_mlp):
        """Filter out small layers."""
        plan = analyze_model(tiny_mlp, min_in_features=20)
        names = {m.name for m in plan.matches}
        # fc1 has in_features=16 (skipped); fc2 has in_features=32 (kept).
        assert names == {"fc2"}

    def test_custom_skip_predicate(self, tiny_mlp):
        plan = analyze_model(tiny_mlp, skip=lambda name, _m: name == "fc1")
        names = {m.name for m in plan.matches}
        assert names == {"fc2"}
        # And fc1 reported as skipped.
        skipped_names = {n for n, _r in plan.skipped}
        assert "fc1" in skipped_names

    def test_empty_model_returns_empty_plan(self):
        plan = analyze_model(nn.Sequential())
        assert plan.total_replaceable == 0
        assert plan.total_skipped == 0


class TestIdentityCheckLessonFromBitsAndBytes:
    """``type(m) is nn.Linear`` deliberately excludes subclasses."""

    def test_subclass_is_not_replaced(self):
        class CustomLinear(nn.Linear):
            """Subclass that signals 'I'm special, don't replace me'."""

        model = nn.Sequential(nn.Linear(4, 4), CustomLinear(4, 4))
        plan = analyze_model(model)
        names = {m.name for m in plan.matches}
        # Only the plain nn.Linear should appear.
        assert names == {"0"}


class TestTiedWeights:
    def test_tied_head_is_skipped(self, tied_head_model):
        plan = analyze_model(tied_head_model)
        names = {m.name for m in plan.matches}
        # `proj` is fine; `head` is tied to `embed.weight` and must be skipped.
        assert "proj" in names
        assert "head" not in names

    def test_tied_groups_reported(self, tied_head_model):
        plan = analyze_model(tied_head_model)
        # We don't pin exact names because accelerate's grouping may differ
        # by version; we only check that *some* tie was detected.
        assert len(plan.tied_groups) >= 1


class TestAlreadyQuantizedDetection:
    def test_refuses_simulated_quantized_layer(self):
        # Simulate a bitsandbytes layer by naming a class Linear4bit. The walker
        # inspects the class name, not the implementation, so this is enough.
        class Linear4bit(nn.Linear):
            pass

        model = nn.Sequential(nn.Linear(4, 4), Linear4bit(4, 4))
        with pytest.raises(IncompatibleModelError, match="already-quantized"):
            analyze_model(model, strict_already_quantized=True)

    def test_can_proceed_with_strict_false(self):
        class Linear8bitLt(nn.Linear):
            pass

        model = nn.Sequential(nn.Linear(4, 4), Linear8bitLt(4, 4))
        plan = analyze_model(model, strict_already_quantized=False)
        # The plain Linear got matched; the quantized one is skipped.
        names = {m.name for m in plan.matches}
        skipped = {n for n, _ in plan.skipped}
        assert "0" in names
        assert "1" in skipped
