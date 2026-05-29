# Qaithon

Run parts of any HuggingFace LLM on photonic or quantum backends without
rewriting the model.

[Español](README.es.md) · [Documentation](docs/en/README.md) · [Cookbook](docs/en/COOKBOOK.md) · [Configuration](docs/en/CONFIGURATION.md)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-253%20passing-brightgreen.svg)
![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)

---

## What it does

You write regular PyTorch / HuggingFace code. Qaithon walks the model,
swaps replaceable `nn.Linear` / Conv1D layers for backend-accelerated
versions, and returns the same `nn.Module` you passed in — only now its
matmuls can run on a photonic or quantum target.

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = qaithon.compile(model)

outputs = model.generate(input_ids, max_new_tokens=50)
```

The selector picks the best backend per layer based on what is installed
on your machine and the objective you pass (`"speed"`, `"energy"`,
`"balanced"`). Everything below the public API is replaceable internal
detail.

## What you get

- Drop-in `qaithon.compile(model)`. No model rewrites.
- Multi-backend: IBM superconductor, Quandela photonic, AWS QuEra and
  IonQ, plus local simulators.
- Auto-selection of backends per layer, with an optional adaptive mode
  that learns from observed latency / energy.
- Works with any model on the HuggingFace Hub.
- Toy transformer lab: build, train, generate end-to-end in 4 lines.
- Real measurements: latency, energy (pJ/MAC), fidelity, photon counts.
- Hardware validation that refuses to start training if a model exceeds
  the qubit / depth budget of the target QPU.
- Health checks against cloud QPUs before submitting jobs.
- Embedded glossary: `qaithon.explain("fidelity", 0.987)`.
- HuggingFace Hub integration via `qaithon.hub`.

## Hardware support

| Vendor | Device | Type | Backend name | Real hardware | Local simulator |
|---|---|---|---|---|---|
| IBM | Heron (156 qubits) | Superconductor | `ibm.heron` | yes | `ibm.aer` |
| IBM | Aer | Simulator | `ibm.aer` | n/a | yes (ideal + realistic) |
| Quandela | Belenos (6 modes) | Photonic | `quandela.belenos` | yes | `quandela.perceval` |
| Quandela | Perceval SLOS | Photonic sim | `quandela.perceval` | n/a | yes |
| Quandela | MerLin | Photonic (autograd) | `quandela.merlin` | n/a | yes |
| AWS Braket | SV1 | Simulator | `aws.braket.sv1` | n/a | yes |
| AWS Braket | QuEra Aquila | Neutral atom | `aws.braket.quera` | yes | n/a (analog) |
| AWS Braket | IonQ Forte | Trapped ion | `aws.braket.ionq` | yes | via SV1 |
| Xanadu | PennyLane | Various | `pennylane.sim` | via plugins | yes |
| TuringQ | DeepQuantum | Multi | `deepquantum` | n/a | yes |

## Measured numbers

Cross-backend comparison on a single matmul `(2, 16) × (16, 16)`:

| Backend | Kind | Latency (µs) | Energy (pJ) | Fidelity vs classical |
|---|---|---:|---:|---:|
| `mock` (GPU baseline) | classical | 1 | 256.00 | 1.0000 |
| `quandela.perceval` | photonic, real circuit | 506 | 1.28 | 0.9845 |
| `quandela.sim` | photonic profile | 1 | 12.80 | 1.0000 |
| `ibm.aer` | quantum, real circuit | 40,437 | 153.60 | 0.9999 |
| `aws.braket.sv1` | quantum | 1 | 128.00 | 1.0000 |
| `pennylane.sim` | quantum | 1 | 204.80 | 1.0000 |

`quandela.perceval` reports about two orders of magnitude lower energy
per MAC than the classical baseline, at a 1.5% fidelity cost — within
the range QAT trains models to tolerate.

First run on **Quandela Belenos** (live photonic QPU):

```
Latency:              33.99 seconds (cloud queue + execution)
Photons injected:     200
Photons detected:     103
Detection efficiency: 51.50%
Loss measured:        48.50%
```

These numbers come from the hardware itself; the local simulator does
not produce them.

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

### Inference on any HuggingFace model

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

model = qaithon.compile(model, optimize_for="energy")
print(model.qaithon_report.pretty())

inputs = tokenizer("Quantum computing is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

### Toy transformer with training and inference

```python
import qaithon
from qaithon.lab import load_dataset, train, generate

model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
qaithon.compile(model, backends=("quandela.sim",))

dataset = load_dataset("shakespeare")
train(model, dataset, steps=2000, noise_std=0.05)  # QAT noise on

print(generate(model, "ROMEO:", max_new_tokens=100))
```

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
