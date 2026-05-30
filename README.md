# Qaithon

Run parts of a HuggingFace LLM through **genuine quantum & photonic algorithms**
— on simulators today, real QPUs as they open up — without rewriting the model.
A small, honest, reproducible step at the frontier of AI × quantum × photonics.

[Español](README.es.md) · [Documentation](docs/en/README.md) · [Cookbook](docs/en/COOKBOOK.md) · [Configuration](docs/en/CONFIGURATION.md)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-253%20passing-brightgreen.svg)
![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)

---

## What it does

You write regular PyTorch / HuggingFace code. Qaithon walks the model,
swaps replaceable `nn.Linear` / Conv1D layers for versions whose matmuls
are computed by a **genuine quantum or photonic algorithm**, and returns
the same `nn.Module` you passed in.

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # genuine quantum

outputs = model.generate(input_ids, max_new_tokens=50)
```

The matmuls run on the real algorithm (Perceval/MerLin photonics,
Qiskit/PennyLane qubits) — **not** classical math with a quantum label.
Today they execute on **local simulators**; the same code targets a real
QPU as access opens up. It works at **small scale** — quantum/photonic
hardware is genuinely tiny today, so this is a research instrument first.
See [`docs/en/FINDINGS.md`](docs/en/FINDINGS.md) for exactly how far the
technology reaches, with measured numbers — including a **real pretrained
model generating coherent text on a genuine quantum circuit**.

## What you get

- Drop-in `qaithon.compile(model)`. No model rewrites.
- **Genuine compute** — matmuls evaluated by real quantum/photonic algorithms,
  validated exact against the classical result (not faked with noise).
- Multi-backend: Quandela photonic (Perceval/MerLin), IBM/PennyLane/DeepQuantum
  quantum, plus `mock` (the explicit classical reference). Connectors for real
  IBM/Quandela/AWS hardware (experimental; see the roadmap).
- Auto-selection of a genuine backend per layer (objective: speed/energy/balanced).
- **Trainable** photonic & quantum layers (`PhotonicLayer`, `QuantumLayer`).
- Qubit/mode-budget estimation + hardware validation — tells you what *any*
  model (even GPT-3-scale) would need on a real QPU. Analysis is free and instant.
- Toy/micro transformer lab: build, train, generate end-to-end.
- Embedded glossary: `qaithon.explain("fidelity", 0.987)`; HF Hub integration.
- Honest, measured limits per technology: [`docs/en/FINDINGS.md`](docs/en/FINDINGS.md).

## Hardware support

Today, **genuine compute runs on the local simulators** (the real algorithm,
exact). The "real hardware" column means a connector exists, but physical runs
are tiny and gated — see [`docs/en/FINDINGS.md`](docs/en/FINDINGS.md) and the
[roadmap](docs/en/ROADMAP.md) for what actually runs on a QPU today.

| Vendor | Device | Type | Backend name | Real hardware | Local simulator |
|---|---|---|---|---|---|
| IBM | Heron (156 qubits) | Superconductor | `ibm.heron` | yes | `ibm.aer` |
| IBM | Aer | Simulator | `ibm.aer` | n/a | yes (ideal + realistic) |
| Quandela | Belenos (12 modes) | Photonic | `quandela.belenos` | experimental | `quandela.perceval` |
| Quandela | Perceval SLOS | Photonic sim | `quandela.perceval` | n/a | yes |
| Quandela | MerLin | Photonic (autograd) | `quandela.merlin` | n/a | yes |
| AWS Braket | SV1 | Simulator | `aws.braket.sv1` | n/a | yes |
| AWS Braket | QuEra Aquila | Neutral atom | `aws.braket.quera` | yes | n/a (analog) |
| AWS Braket | IonQ Forte | Trapped ion | `aws.braket.ionq` | yes | via SV1 |
| Xanadu | PennyLane | Various | `pennylane.sim` | via plugins | yes |
| TuringQ | DeepQuantum | Multi | `deepquantum` | n/a | yes |

## Measured numbers

The genuine kernel computes the matmul **exactly** in simulation — the circuit
*is* doing the math, so it reproduces the classical result at machine precision:

| Backend | Kind | Compute | Fidelity vs classical (sim) |
|---|---|---|---:|
| `quandela.perceval` / `quandela.sim` / `quandela.merlin` | photonic | genuine (Perceval/MerLin) | 1.0000 |
| `ibm.aer` / `pennylane.sim` / `deepquantum` | quantum | genuine (Qiskit/PennyLane) | 1.0000 |
| `mock` | classical reference | `F.linear` | 1.0000 |

Energy / latency per backend come from declared cost profiles. Fidelity loss is
**not** something the exact simulator produces — it only appears on physical
hardware (optical loss, decoherence), which is the whole point of running there.

**Highlight — a real model on a genuine quantum circuit:** the pretrained
**TinyStories-1M** generated coherent text ("...a little girl named Lily. She
loved to play outside in the sunshine...") with its 48 linear layers computed
through genuine qubit circuits — output **identical** to classical (error 1e-6).

Full results, per-technology limits (measured on a laptop), and the plain-language
explanation of *why* it's small live in **[`docs/en/FINDINGS.md`](docs/en/FINDINGS.md)**.

> Physical-hardware runs are tiny and gated today (a single small circuit, not a
> full model). See the [roadmap](docs/en/ROADMAP.md) — when error-corrected
> hardware opens up, the same code targets it.

## Installation

```bash
pip install qaithon

# Add hardware backends as needed.
pip install qaithon[huggingface,pennylane,quandela,deepquantum]
```

## Configuration

Credentials are resolved in this order (highest wins): programmatic
setters, environment variables, a project-local `.env` file.

```python
import qaithon

qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...", aws_secret_access_key="...",
    aws_region="us-east-1",
    quandela_token="...",
    huggingface_token="hf_...",
)

# Per-provider setters work too.
qaithon.set_ibm_token("...")
qaithon.set_aws_credentials("AKIA...", "secret...")
qaithon.set_quandela_token("...")
qaithon.set_huggingface_token("hf_...")

# Diagnostic — booleans only, never the values.
qaithon.config.status()
# {'ibm': True, 'aws': True, 'quandela': True, 'huggingface': True}
```

Details in [`docs/en/CONFIGURATION.md`](docs/en/CONFIGURATION.md).

## Quickstart workflows

### Genuine inference of a small pretrained model

A tiny model runs end-to-end through genuine quantum circuits. (Larger models
like GPT-2 compile too, but their wider layers make the genuine path slow —
see [`FINDINGS.md`](docs/en/FINDINGS.md) for the measured limits.)

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

model = qaithon.compile(model, backends=("pennylane.sim",))  # genuine quantum
print(model.qaithon_report.pretty())

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

### Toy transformer: genuine quantum inference

```python
import qaithon
from qaithon.lab import generate

model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
qaithon.compile(model, backends=("pennylane.sim",))   # genuine quantum

# Each layer's computation runs on a real quantum circuit.
print(generate(model, "ROMEO:", max_new_tokens=100))
```

> Note: the genuine kernel is **inference**-only (not differentiable). To
> **train** genuine layers, use the differentiable `PhotonicLayer` /
> `QuantumLayer` (see [`docs/en/FINDINGS.md`](docs/en/FINDINGS.md)).

### Validate that a model fits a hardware target

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=128, n_layers=2)
result = qaithon.validate_for_hardware(model, target="IBM Heron")
print(result.pretty())
# Refuses to start training if the model exceeds the QPU's budget.
```

### Estimate qubit cost of any model

```python
import qaithon
# Works on huge models — uses the config, never instantiates the weights.
report = qaithon.estimate_qubits_from_config(
    hidden_size=12288, n_layers=96, n_heads=96,
    model_class="GPT-3.5 equivalent",
)
print(report.pretty())
```

### Compare backends side-by-side

```bash
qaithon benchmark --explain
```

## Command-line tools

```bash
qaithon list-backends        # registered backends + availability
qaithon doctor               # diagnose the local environment
qaithon inspect <model_id>   # what would compile() do to this model?
qaithon compile <model_id>   # compile a HF model and save it
qaithon estimate <model_id>  # qubit budget needed to run on a QPU
qaithon benchmark --explain  # cross-backend comparison
qaithon glossary <term>      # plain-language explanation of any term
qaithon trace inspect <file> # pretty-print a saved JSON trace
qaithon plugins list         # third-party backends discovered via entry points
```

## Project layout

```
qaithon/
  compile.py           qaithon.compile() — main entry point
  compile_report.py    audit trail of compile decisions
  config.py            credentials and SDK-style setters
  glossary.py          plain-language explanations of every term
  qubits.py            qubit-budget estimation + hardware validation
  benchmarks.py        cross-backend performance comparison
  pricing.py           USD cost estimation per provider
  metrics.py           PhotonicMetrics, SuperconductingMetrics, etc.
  backends/            IBM, Quandela, AWS, PennyLane, DeepQuantum
  layers/              QuantumLinear, QuantumAttention
  ir/                  analyzer + AutoBackendSelector + AdaptiveBackendSelector
  handlers/            per-architecture hooks (Mixtral, attention, ...)
  models/              toy transformer factories sized for real QPUs
  lab/                 datasets + training loop + inference helpers
  hub/                 HuggingFace Hub integration
  training.py          QATConfig + prepare_for_qat
  streaming.py         HF-compatible streamers
  pipeline.py          pipeline composition
  cache.py             in-memory + on-disk caches
  tracing.py           lightweight observability
  observability.py     OpenTelemetry exporter
  fallback.py          graceful runtime fallback across backends
  plugins.py           third-party backend discovery via entry_points
  integrations/        vLLM (skeleton), more to come
```

## License

MIT — see [`LICENSE`](LICENSE).

## Author

Fabián Bautista — `fabbagar83@gmail.com`
