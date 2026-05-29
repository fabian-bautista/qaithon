"""Tests for the tracing context manager and event collection."""

from __future__ import annotations

import json

import torch

from qaithon.backends import get_backend
from qaithon.tracing import Trace, trace, traced


class TestTraceContext:
    def test_returns_fresh_trace(self):
        with trace() as t:
            assert isinstance(t, Trace)
            assert t.events == []

    def test_no_active_trace_outside_context(self):
        backend = traced(get_backend("mock"))
        # No active trace; matmul still works without crashing.
        y = backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        assert y.shape == (2, 3)


class TestEventCollection:
    def test_one_matmul_one_event(self):
        backend = traced(get_backend("mock"))
        with trace() as t:
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        assert len(t.events) == 1
        assert t.events[0].backend == "mock"
        assert t.events[0].input_shape == (2, 4)
        assert t.events[0].weight_shape == (3, 4)

    def test_latency_is_positive(self):
        backend = traced(get_backend("mock"))
        with trace() as t:
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        assert t.events[0].latency_us > 0

    def test_traces_are_thread_local(self):
        # Implicit: traces don't leak between with blocks.
        backend = traced(get_backend("mock"))
        with trace() as t1:
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        with trace() as t2:
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        assert len(t1.events) == 1
        assert len(t2.events) == 1
        assert t1 is not t2


class TestJSONExport:
    def test_to_json_roundtrips(self):
        backend = traced(get_backend("mock"))
        with trace() as t:
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
            backend.matmul(torch.randn(2, 4), torch.randn(3, 4))
        as_json = t.to_json()
        data = json.loads(as_json)
        assert data["summary"]["n_events"] == 2
        assert len(data["events"]) == 2
