# Architecture (English)

How Qaithon is put together and why each piece exists.

## Goals that shape every decision

1. **Drop-in for AI developers.** The user types `qaithon.compile(model)`
   and never sees a quantum primitive. Everything below the public API is
   replaceable internal detail.
2. **Tensor in / tensor out.** Backends accept `torch.Tensor` and return
   `torch.Tensor`. Quantum vocabulary stays inside the backend.
3. **Honest costs.** Backends declare what they cost (`BackendProfile`)
   and the selector picks based on those numbers; the report shows the
   trade-off out loud.
4. **SOLID, in particular Open/Closed.** New hardware = new
   `Backend` subclass + one line in the registry. No edits to `compile()`.

## High-level flow

```
  user code            qaithon                          backends
  ─────────            ───────                          ────────
                                       ┌─► MockBackend
  AutoModel  ──►  compile()  ──►  Walker  ──►  Selector  ─┼─► PennyLane
                  │              │            │           ├─► AerSimulator
                  │              ▼            ▼           ├─► Perceval (SLOS)
                  │       ReplacementPlan  SelectionResult├─► MerLin (native)
                  ▼              │            │           ├─► IBM Heron (real)
            CompileReport  ◄─────┴────────────┘           ├─► Quandela Belenos
                                                          ├─► AWS Braket SV1
                                                          └─► AWS QuEra / IonQ
```

Three layers, each with a single responsibility.

### Layer 1 — Walker (`qaithon/ir/analyzer.py`)

Walks a `torch.nn.Module` and produces a `ReplacementPlan`:
which sub-modules are replaceable `nn.Linear` / `Conv1D` /
HuggingFace `Linear4bit`, which are not, which are tied (e.g.
`lm_head ↔ wte`), and why each excluded layer was excluded.

The non-trivial trick the walker does: it uses an **identity check**
(`type(m) is nn.Linear`) instead of `isinstance`, exactly like
bitsandbytes does. That one-character distinction is what stops Qaithon
from corrupting `NonDynamicallyQuantizableLinear` inside attention
blocks.

### Layer 2 — Selector (`qaithon/ir/selector.py` + `adaptive_selector.py`)

Given a `ReplacementPlan` and an `objective`
(`"speed" | "energy" | "balanced"`), the selector picks one backend per
layer. The default is greedy per-layer based on the declared
`BackendProfile` cost model. The opt-in `AdaptiveBackendSelector`
records observed latency / energy and blends them back into the cost
model via a Welford-style running mean.

### Layer 3 — Backend (`qaithon/backends/*.py`)

A `Backend` exposes one method: `matmul(x, weight, bias)`. Inside that
method it can call PennyLane, Aer, Perceval, MerLin, AWS Braket, or
anything else. The classical fallback is `mock`, which is always
available and never breaks the user's model.

## The `compile()` contract

```python
def compile(model, *, optimize_for="balanced", backends=None) -> nn.Module:
    plan = analyze_model(model)                     # walker
    result = AutoBackendSelector(...).select(plan,  # selector
                                              objective=optimize_for)
    replace_layers(model, result.per_layer)         # mutator
    model.qaithon_report = CompileReport(...)       # auditability
    return model
```

`compile()` mutates the model in place AND returns it (PyTorch
convention). After it returns, every replaced layer is a
`QuantumLinear` or `QuantumAttention` that delegates to its assigned
backend.

## Registry — how new backends plug in

```python
from qaithon.backends.base import Backend, register_backend

class MyBackend(Backend):
    profile = BackendProfile(name="vendor.mychip", ...)
    def matmul(self, x, w, b=None): ...
    def is_available(self) -> bool: ...

register_backend("vendor.mychip", MyBackend)
```

That is the complete extension surface. Nothing else changes. Tests for
the new backend live in `tests/test_backends_<name>.py`.

Third-party packages can also register via the
`qaithon.backends` entry point group — see `qaithon/plugins.py`.

## Layers that replace `nn.Linear`

- `QuantumLinear` (`qaithon/layers/quantum_linear.py`) — drop-in
  `nn.Linear` replacement that holds a backend reference and delegates
  `forward()` to `backend.matmul`.
- `QuantumAttention` (`qaithon/layers/quantum_attention.py`) — composable
  attention block that runs its Q/K/V projections on a backend while
  keeping the softmax classical.

Both classes preserve the host module's state_dict layout, so
`from_pretrained` keeps working.

## HuggingFace integration

Two paths, complementary:

1. **`qaithon.compile(model)`** — works on any `nn.Module`. The
   recommended entry point.
2. **`QaithonHfQuantizer`** (`qaithon/hf_integration.py`) — registered
   with HuggingFace transformers via `HfQuantizer`. Activates when a
   `QuantizationConfig(method="qaithon")` is passed to
   `AutoModel.from_pretrained`. Lets users opt in via the standard HF
   surface.

Architecture handlers (`qaithon/handlers/`) deal with vendor quirks:
Mixtral MoE expert routing, Flash Attention, SDPA, Conv1D-style
projections.

## Observability

Every backend call goes through `qaithon.tracing` — a per-call event
recorded with backend name, latency, estimated energy. The trace is
flushed to OpenTelemetry when configured, and aggregated into
`InferenceMetrics` for at-a-glance summaries.

## Hardware metrics

Three flavors specialize `HardwareMetrics`:

- `SuperconductingMetrics` — populated by `ibm.aer`. Includes qubit
  count, gate count, average gate fidelity (in `realistic` mode), queue
  time.
- `PhotonicMetrics` — populated by `quandela.perceval` and
  `quandela.belenos`. Includes mode count, photon injection vs
  detection, detection efficiency, accumulated loss.
- `NeutralAtomMetrics` — populated by `aws.braket.quera`. Atom count,
  rearrangement success rate, Rydberg blockade radius.

Each backend exposes them via a `last_*_metrics` property after a
matmul.

## Configuration

`qaithon.config` carries three layers, in priority order (highest
wins):

1. Programmatic setters (`qaithon.set_ibm_token(...)` etc.).
2. Environment variables (`IBM_QUANTUM_TOKEN`, `AWS_ACCESS_KEY_ID`, …).
3. A project-local `.env` file.

`qaithon.config.status()` returns a `{provider: bool}` dict so a
diagnostic UI can show which credentials are present **without ever
exposing the values**.

## Anti-goals (things Qaithon deliberately does not do)

- No "build the hardware". Qaithon is software that targets *existing*
  quantum / photonic hardware via official SDKs.
- No proprietary quantum DSL. Backends use vendor-native APIs.
- No mocks pretending to be real hardware. Backends with a `kind`
  field of `quantum` or `photonic` either run real circuits / shots
  (with explicit cost) or label themselves as simulators.
- No silent fallbacks. When a backend is unavailable, the report says so.

## Where to look in the source

| Concern | File |
|---|---|
| Public API surface | `qaithon/__init__.py` |
| `compile()` entry point | `qaithon/compile.py` |
| Walker | `qaithon/ir/analyzer.py` |
| Default selector | `qaithon/ir/selector.py` |
| Adaptive selector | `qaithon/ir/adaptive_selector.py` |
| Backend ABC + registry | `qaithon/backends/base.py` |
| Each backend | `qaithon/backends/<name>.py` |
| HuggingFace integration | `qaithon/hf_integration.py` |
| Architecture handlers | `qaithon/handlers/` |
| Layer replacements | `qaithon/layers/` |
| Credentials | `qaithon/config.py` |
| Tracing | `qaithon/tracing.py` |
| Metrics | `qaithon/metrics.py` |
| Glossary | `qaithon/glossary.py` |
| CLI | `qaithon/cli.py` |
