"""Tests for the OpenTelemetry exporter (skip if otel not installed)."""

from __future__ import annotations

import pytest

from qaithon.observability import export_to_otel, otel_available
from qaithon.tracing import Trace, TraceEvent


class TestAvailability:
    def test_otel_available_returns_bool(self):
        assert isinstance(otel_available(), bool)

    def test_export_without_otel_raises_with_hint(self):
        if otel_available():
            pytest.skip("otel is installed, can't test the missing branch")
        t = Trace(events=[])
        with pytest.raises(RuntimeError, match="opentelemetry"):
            export_to_otel(t)


@pytest.mark.skipif(not otel_available(), reason="opentelemetry not installed")
class TestExporter:
    def test_export_zero_events(self):
        trace = Trace(events=[])
        export_to_otel(trace, service_name="test-service")

    def test_export_with_events(self):
        trace = Trace(
            events=[
                TraceEvent(
                    backend="mock",
                    kind="mock",
                    input_shape=(2, 4),
                    weight_shape=(3, 4),
                    latency_us=100.0,
                    estimated_energy_pj=5.0,
                ),
            ]
        )
        export_to_otel(trace, service_name="test-service")
