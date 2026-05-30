# API Reference (English)

Public Python API. Everything reachable via `import qaithon`.

## Conventions

- Public functions are listed in `qaithon.__all__`. Anything not listed
  there is internal and may change without notice.
- Tensor in / tensor out — every backend accepts `torch.Tensor` and
  returns `torch.Tensor`.
- Exceptions inherit from `qaithon.QaithonError` so they can be caught
  generically.

---

## Top-level

### `qaithon.compile(model, *, optimize_for="balanced", backends=None) -> nn.Module`

Replace replaceable layers of `model` with backend-accelerated versions
and return the same model (mutated in place).

| Param | Type | Default | Notes |
|---|---|---|---|
| `model` | `nn.Module` | — | Any PyTorch model, including HuggingFace transformers. |
| `optimize_for` | `"speed" \| "energy" \| "balanced"` | `"balanced"` | Objective passed to the selector. |
| `backends` | `tuple[str, ...] \| None` | `None` | Restrict to these backends. `None` = all available. |

Side effect: attaches `model.qaithon_report` (a `CompileReport`).

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = qaithon.compile(model, optimize_for="energy")
print(model.qaithon_report.pretty(explain=True))
```

### `qaithon.__version__: str`

Current package version.

---

## SDK-style credential configuration

All setters write to environment variables; existing exports always win.

### `qaithon.set_ibm_token(token, *, channel="ibm_quantum_platform", instance=None)`

Configure IBM Quantum credentials.

### `qaithon.set_aws_credentials(access_key_id, secret_access_key, *, region="us-east-1")`

Configure AWS Braket credentials.

### `qaithon.set_quandela_token(token)`

Configure Quandela Cloud credentials.

### `qaithon.set_huggingface_token(token)`

Configure HuggingFace Hub credentials. Writes both `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` so `huggingface_hub` picks it up natively.

### `qaithon.configure(*, ibm_token=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region="us-east-1", quandela_token=None, huggingface_token=None, ibm_channel="ibm_quantum_platform", ibm_instance=None)`

One-call configuration of every provider. Passing `None` leaves existing
credentials untouched (additive, never destructive).

```python
qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...", aws_secret_access_key="...",
    quandela_token="...",
    huggingface_token="hf_...",
)
```

### `qaithon.config.status() -> dict[str, bool]`

Returns `{"ibm": bool, "aws": bool, "quandela": bool, "huggingface": bool}`.
Does NOT expose values — safe to log.

---

## Qubit estimation & hardware validation

### `qaithon.estimate_qubits(model: nn.Module) -> QubitReport`

Walk `model` and produce per-layer qubit + circuit-depth budgets
(amplitude encoding + block encoding).

### `qaithon.estimate_qubits_from_config(*, hidden_size, n_layers, n_heads=1, intermediate_size=None, vocab_size=0, model_class="FromConfig") -> QubitReport`

Estimate from a config without instantiating the model (zero RAM). Works at any
size — feed it a 405B-class config and it shows the impossible qubit count such a
model would need today. Analysis only; it does not run the model.

```python
report = qaithon.estimate_qubits_from_config(
    hidden_size=12288, n_layers=96, n_heads=96,
    model_class="GPT-3.5-equivalent",
)
print(report.pretty())
```

### `qaithon.measure_actual_circuit(in_features, out_features, *, target_backend=None) -> MeasuredCircuit`

Build a real Qiskit circuit and report exact qubit + gate counts. With
`target_backend`, transpile against a fake backend (e.g. `"FakeBrisbane"`)
to get hardware-realistic numbers.

### `qaithon.validate_for_hardware(model=None, *, target, report=None) -> ValidationResult`

Check whether a model fits on a given hardware target. Pass either
`model` or a pre-computed `report`.

```python
result = qaithon.validate_for_hardware(model, target="IBM Heron")
if not result.fits:
    print(result.pretty())  # explains why + suggestions
```

### `qaithon.find_hardware(name: str) -> HardwareSpec`

Look up a hardware spec by case-insensitive substring.

### `qaithon.KNOWN_HARDWARE: tuple[HardwareSpec, ...]`

Built-in registry: IBM Heron, IBM Brisbane, IBM Starling (projected),
IBM Blue Jay (projected), Quandela Belenos, QuEra Aquila, IonQ Forte
Enterprise 1.

---

## Glossary / explainability

### `qaithon.explain(term: str, *, value=None) -> str`

Pretty-printed explanation of a quantum term. With `value`, contextualize
("a fidelity of 0.987 means …").

### `qaithon.glossary(term: str) -> GlossaryEntry`

Programmatic lookup. Returns an entry with `.short`, `.analogy`,
`.example`, `.rule_of_thumb`.

### `qaithon.list_terms() -> list[str]`

Every term covered by the glossary.

---

## Sub-packages

Each is reachable as `qaithon.<name>`.

- **`qaithon.backends`** — backend registry. Use
  `qaithon.backends.get_backend("name")` to instantiate.
- **`qaithon.benchmarks`** — `compare_backends(in_features, out_features)`
  returns a per-backend latency/energy table.
- **`qaithon.cache`** — `qaithon.cache.cached(backend)` wraps a backend
  with in-memory + disk caching.
- **`qaithon.config`** — credentials (setters, status, getters).
- **`qaithon.fallback`** — graceful runtime fallbacks.
- **`qaithon.handlers`** — architecture handlers (Mixtral, SDPA, …).
- **`qaithon.hub`** — `push_block(...)` / `load_block(...)` over
  HuggingFace Hub.
- **`qaithon.integrations`** — third-party adapters (vLLM skeleton).
- **`qaithon.lab`** — `train(model, dataset, ...)`, `generate(...)`,
  `load_dataset(...)` — a tiny integrated training surface.
- **`qaithon.metrics`** — `HardwareMetrics`, `SuperconductingMetrics`,
  `PhotonicMetrics`, `NeutralAtomMetrics`, `InferenceMetrics`.
- **`qaithon.models`** — `create_toy_transformer(...)` for the lab.
- **`qaithon.observability`** — counters and exporters.
- **`qaithon.pipeline`** — compose compiled models.
- **`qaithon.plugins`** — third-party backend discovery via entry points.
- **`qaithon.pricing`** — `estimate_cost_usd("backend.name", n_shots=…)`.
- **`qaithon.streaming`** — HF `TextStreamer` compatibility.
- **`qaithon.tracing`** — `with tracing.trace() as t: …`.

---

## Exceptions

All inherit from `qaithon.QaithonError`.

| Exception | When |
|---|---|
| `BackendError` | Any backend-level error. |
| `BackendNotAvailableError` | Backend's `is_available()` returned `False`. |
| `BackendNotRegisteredError` | Name not in registry. |
| `CompileError` | `compile()` could not produce a valid plan. |
| `IncompatibleModelError` | Model contains a layer Qaithon refuses to touch. |
| `IncompatibleHardwareError` | `validate_for_hardware` says no. |
| `UnsupportedOperationError` | Backend doesn't implement this op (yet). |

---

## Reports

### `CompileReport`

Attached to every compiled model as `model.qaithon_report`.

- `.summary()` — one-line summary.
- `.pretty(explain=False)` — full per-layer breakdown.
- `.decisions: tuple[LayerDecision, ...]` — raw audit log.

### `LayerDecision`

Per-layer record: layer name, dimensions, chosen backend, reason,
estimated energy / latency.

### `QubitReport`

Whole-model qubit + depth budget. Use `.pretty()` for human output,
`.hardware_compatibility()` for programmatic checks.

### `ValidationResult`

`fits: bool`, `target: HardwareSpec`, `report: QubitReport`, plus
`reasons` and `recommendations` lists.

---

## Genuine quantum & photonic layers

Trainable `nn.Module`s whose forward pass runs on a real quantum/photonic
algorithm. Use them like any PyTorch layer.

### `qaithon.PhotonicLayer(in_features, out_features, *, photons=1, target="Quandela Belenos", on_hardware=False)`

A photonic layer (Perceval/MerLin) — differentiable, trains with autograd.

### `qaithon.QuantumLayer(in_features, out_features, *, var_layers=2, target="IBM Heron", on_hardware=False, device="default.qubit")`

A qubit layer (PennyLane): amplitude-encode → variational circuit → readout. Pass
`device="qiskit.remote"` to run on a real QPU.

### `qaithon.ReuploadingClassifier(in_features, n_classes, *, n_qubits=5, layers=6, device="default.qubit")`

A shallow, NISQ-friendly quantum-native classifier (angle encoding + data
re-uploading). Verified on real IBM hardware (10-class Digits, 5 qubits, ~80%).

```python
import qaithon, torch
clf = qaithon.ReuploadingClassifier(in_features=10, n_classes=10)
logits = clf(torch.randn(32, 10))   # train like any nn.Module
```

---

## Real-hardware execution

Every real-hardware backend takes a `mode`:

- `mode="profile"` (default) — no real circuits; uses the declared cost model. Free.
- `mode="calibrate"` — fires one small calibration circuit; returns real telemetry.
- `mode="execute"` — runs the **genuine matmul** as a real circuit on the QPU.

```python
from qaithon.backends.ibm_heron import IBMHeronBackend
from qaithon.backends.quandela_belenos import QuandelaBelenosBackend

# Genuine matmul on a real IBM QPU, with software error mitigation:
ibm = IBMHeronBackend(mode="execute", shots=2048, mitigation=True)
y = ibm.matmul(x, weight)             # ibm.last_execute → fidelity, gates, latency

# Genuine photonic matmul on Quandela Belenos (or platform_name="local:slos"):
photonic = QuandelaBelenosBackend(mode="execute", platform_name="qpu:belenos")
y = photonic.matmul(x, weight)        # photonic.last_execute → modes, fidelity
```

`mitigation=True` (IBM) adds higher transpiler optimization + dynamical decoupling
+ measurement twirling. `execute` mode consumes real quota / credits.

---

## Backend interface (advanced)

```python
from qaithon.backends.base import Backend, BackendProfile, register_backend

class MyBackend(Backend):
    profile = BackendProfile(name="vendor.mychip", ...)
    def matmul(self, x, weight, bias=None) -> torch.Tensor: ...
    def is_available(self) -> bool: ...
    def health_check(self) -> HealthStatus: ...

register_backend("vendor.mychip", MyBackend)
```

`HealthStatus` has `backend`, `online`, `message`, `latency_ms`.

---

## Examples by use case

### Inference with caching

```python
import qaithon
backend = qaithon.backends.get_backend("ibm.aer")
cached = qaithon.cache.cached(backend)
```

### Cost estimation before running on real QPU

```python
cost = qaithon.pricing.estimate_cost_usd("aws.braket.quera", n_shots=1000)
```

### Health check before a real-hardware run

```python
heron = qaithon.backends.get_backend("ibm.heron")
status = heron.health_check()
assert status.online
```
