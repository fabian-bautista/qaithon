"""Tests for the Pipeline composability primitive."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import qaithon
from qaithon.pipeline import Pipeline


class TinyA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


class TinyB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x)


class TestPipeline:
    def test_empty_pipeline_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Pipeline([])

    def test_single_stage(self):
        a = TinyA()
        pipe = Pipeline([a])
        y = pipe(torch.randn(3, 4))
        assert y.shape == (3, 8)

    def test_two_stages_chain(self):
        pipe = Pipeline([TinyA(), TinyB()])
        y = pipe(torch.randn(3, 4))
        assert y.shape == (3, 2)

    def test_callable_stages_allowed(self):
        pipe = Pipeline([TinyA(), torch.relu, TinyB()])
        y = pipe(torch.randn(3, 4))
        assert y.shape == (3, 2)

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            Pipeline([TinyA(), "not callable"])  # type: ignore[list-item]

    def test_propagates_errors_with_index(self):
        def bad(x):
            raise RuntimeError("bad stage")

        pipe = Pipeline([TinyA(), bad])
        with pytest.raises(RuntimeError, match="stage 1"):
            pipe(torch.randn(3, 4))

    def test_compiled_stages_work(self):
        a, b = TinyA(), TinyB()
        qaithon.compile(a)
        qaithon.compile(b)
        pipe = Pipeline([a, b])
        y = pipe(torch.randn(3, 4))
        assert y.shape == (3, 2)
