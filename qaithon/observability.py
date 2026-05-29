"""OpenTelemetry exporter for Qaithon traces.

Maps :class:`qaithon.tracing.Trace` events to OpenTelemetry spans so any
backend that speaks OTel (Jaeger, Tempo, Datadog, Honeycomb, OTLP
collector) can ingest Qaithon's runtime telemetry without extra glue.

Designed to be import-safe even when OpenTelemetry is not installed: this
module exposes lazy adapters that raise ``RuntimeError`` with an install
hint at call time rather than at import.

Typical usage::

    from qaithon.tracing import trace, traced
    from qaithon.observability import export_to_otel

    with trace() as t:
        # ...run inference with traced backends...

    export_to_otel(t, service_name="my-inference-service")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from qaithon._logging import get_logger

if TYPE_CHECKING:
    from qaithon.tracing import Trace

__all__ = ["export_to_otel", "otel_available"]

logger = get_logger(__name__)


def otel_available() -> bool:
    """Return ``True`` if ``opentelemetry`` is importable."""
    return importlib.util.find_spec("opentelemetry") is not None


def export_to_otel(
    trace_obj: Trace,
    *,
    service_name: str = "qaithon",
    resource_attributes: dict[str, Any] | None = None,
) -> None:
    """Emit every event in ``trace_obj`` as an OpenTelemetry span.

    Args:
        trace_obj: A :class:`qaithon.tracing.Trace` populated by a previous
            ``with trace() as t`` block.
        service_name: Logical service name shown in the OTel collector.
        resource_attributes: Optional dict of additional resource
            attributes (model id, backend version, environment, …).

    Raises:
        RuntimeError: If ``opentelemetry`` is not installed.
    """
    if not otel_available():
        raise RuntimeError(
            "opentelemetry is not installed. Install it with "
            "`pip install qaithon[observability]` to enable export_to_otel."
        )

    from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]
    from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]

    attrs: dict[str, Any] = {"service.name": service_name}
    if resource_attributes:
        attrs.update(resource_attributes)
    provider = TracerProvider(resource=Resource.create(attrs))
    otel_trace.set_tracer_provider(provider)

    tracer = otel_trace.get_tracer("qaithon")
    for event in trace_obj.events:
        with tracer.start_as_current_span(
            f"qaithon.{event.kind}.matmul",
            attributes={
                "qaithon.backend": event.backend,
                "qaithon.kind": event.kind,
                "qaithon.input_shape": list(event.input_shape),
                "qaithon.weight_shape": list(event.weight_shape),
                "qaithon.latency_us": event.latency_us,
                "qaithon.energy_pj": event.estimated_energy_pj,
            },
        ):
            pass  # The span's duration is set explicitly via the with block.
