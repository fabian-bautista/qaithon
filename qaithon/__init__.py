"""Qaithon — the bridge between generative AI and quantum-photonic computing.

A Python library that takes any HuggingFace LLM and runs parts of its
computation on photonic / quantum backends, without requiring the user to
learn quantum mechanics. Internally, Qaithon picks the best available
backend per layer based on the user's optimization goal.

Public surface
--------------

The whole library is designed so most users only ever touch one function::

    import qaithon
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = qaithon.compile(model)
    # …then use `model` exactly like before.

Power users can:

* Inspect the audit trail: ``model.qaithon_report.pretty()``.
* Wrap a backend with caching: ``qaithon.cache.cached(backend)``.
* Trace runtime: ``with qaithon.tracing.trace() as t: ...``.
* Compose multiple compiled models: ``qaithon.pipeline.Pipeline([a, b])``.
* Load a pre-trained block: ``qaithon.hub.load_block("…")`` (skeleton in v0.1).
"""

from __future__ import annotations

from qaithon._logging import enable_default_logging, get_logger
from qaithon.compile import compile  # noqa: A004 — intentional shadow of builtin
from qaithon.compile_report import CompileReport, LayerDecision
from qaithon.config import (
    configure,
    set_aws_credentials,
    set_huggingface_token,
    set_ibm_token,
    set_quandela_token,
)
from qaithon.glossary import GlossaryEntry, explain, glossary, list_terms
from qaithon.qubits import (
    KNOWN_HARDWARE,
    HardwareSpec,
    LayerQubitEstimate,
    MeasuredCircuit,
    QubitReport,
    ValidationResult,
    estimate_qubits,
    estimate_qubits_from_config,
    find_hardware,
    measure_actual_circuit,
    validate_for_hardware,
)
from qaithon.exceptions import (
    BackendError,
    BackendNotAvailableError,
    BackendNotRegisteredError,
    CompileError,
    IncompatibleHardwareError,
    IncompatibleModelError,
    QaithonError,
    UnsupportedOperationError,
)

# Sub-packages exposed for users who want explicit imports.
from qaithon import (  # noqa: F401
    benchmarks,
    cache,
    config,
    fallback,
    handlers,
    hardware_limits,
    hub,
    integrations,
    kernels,
    lab,
    metrics,
    models,
    observability,
    pipeline,
    plugins,
    pricing,
    streaming,
    tracing,
    training,
)

# Genuine, trainable quantum / photonic layers (differentiable).
from qaithon.layers.photonic_layer import PhotonicLayer
from qaithon.layers.quantum_layer import QuantumLayer

__version__ = "0.0.1"

__all__ = [
    # Exceptions (public so users can catch them).
    "BackendError",
    "BackendNotAvailableError",
    "BackendNotRegisteredError",
    "CompileError",
    "CompileReport",
    "IncompatibleModelError",
    "LayerDecision",
    "QaithonError",
    "UnsupportedOperationError",
    "__version__",
    # Main entry point.
    "compile",
    # Qubit estimation & hardware validation.
    "estimate_qubits",
    "estimate_qubits_from_config",
    "measure_actual_circuit",
    "validate_for_hardware",
    "find_hardware",
    "QubitReport",
    "LayerQubitEstimate",
    "MeasuredCircuit",
    "ValidationResult",
    "HardwareSpec",
    "KNOWN_HARDWARE",
    "IncompatibleHardwareError",
    # Glossary / explainability.
    "explain",
    "glossary",
    "list_terms",
    "GlossaryEntry",
    # SDK-style credential configuration.
    "configure",
    "set_aws_credentials",
    "set_huggingface_token",
    "set_ibm_token",
    "set_quandela_token",
    # Genuine, trainable quantum / photonic layers.
    "PhotonicLayer",
    "QuantumLayer",
    # Sub-packages.
    "benchmarks",
    "cache",
    "config",
    "fallback",
    "handlers",
    "hardware_limits",
    "hub",
    "integrations",
    "kernels",
    "lab",
    "metrics",
    "models",
    "observability",
    "pipeline",
    "plugins",
    "pricing",
    "streaming",
    "tracing",
    "training",
    # Logging helpers.
    "enable_default_logging",
    "get_logger",
]
