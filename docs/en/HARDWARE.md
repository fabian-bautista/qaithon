# Hardware Support

Every backend Qaithon talks to, with what it does, what it costs, and how
to enable it.

## Local simulators (no account needed)

| Backend | Technology | What it does | Energy/MAC | Available |
|---|---|---|---|---|
| `mock` | Classical reference | `F.linear` baseline | 1.0 pJ | Always |
| `ibm.aer` | Superconductor sim | Real Qiskit circuits, `ideal` or `realistic` (FakeBrisbane noise) | 1.5 pJ | `qiskit-aer` installed |
| `quandela.sim` | Photonic profile | `F.linear` + photonic cost profile | 0.05 pJ | `merlin` installed |
| `quandela.perceval` | Photonic real sim | Linear-optical circuits via Perceval SLOS | 0.005 pJ | `perceval`+`merlin` |
| `pennylane.sim` | QML simulator | `F.linear` + PennyLane cost | 0.8 pJ | `pennylane` installed |
| `deepquantum` | Multi-tech sim | DeepQuantum native | 0.4 pJ | `deepquantum` installed |

## Real cloud hardware

| Backend | Technology | Vendor | Free tier | Mode |
|---|---|---|---|---|
| `ibm.heron` | Superconductor (156 qubits) | IBM | 10 min/month | `calibrate` (default) |
| `ibm.quantum` | Profile only | IBM | n/a | profile |
| `aws.braket.sv1` | Classical sim (cloud) | AWS | ~1 h/month | `calibrate` |
| `aws.braket.quera` | Neutral atom (256 atoms) | QuEra via AWS | none, pay-per-shot | `calibrate` |
| `aws.braket.ionq` | Trapped ion (36 qubits) | IonQ via AWS | none, pay-per-shot | `calibrate` |
| `quandela.belenos` | Photonic (6 modes) | Quandela | research-gated | `calibrate` |

## Modes of execution

Every real-hardware backend supports three modes:

- **`profile`** (default) — no real circuits fired. Uses the declared cost
  model. Safe to use freely for development. Zero cost.
- **`calibrate`** — fires one small calibration circuit per forward call to
  capture real latency and noise. Modest cost on cloud QPUs.
- **`execute`** — every matmul becomes a real circuit. Slow and expensive;
  enable for explicit benchmarks only.

```python
from qaithon.backends.ibm_heron import IBMHeronBackend

# Default (profile mode): zero cost.
backend = IBMHeronBackend()

# Real circuit calibration per forward call.
backend = IBMHeronBackend(mode="calibrate")
```

## What "qubit equivalent" means per technology

Hardware capacity is measured differently per technology. Qaithon's
`estimate_qubits` and `validate_for_hardware` translate transparently.

| Technology | Unit | What it represents |
|---|---|---|
| Superconductor (IBM) | qubit | Spin state on a Josephson junction |
| Photonic (Quandela) | photonic mode | Spatial channel; photons interfere through it |
| Neutral atom (QuEra) | atomic qubit | Energy level of a trapped Rydberg atom |
| Trapped ion (IonQ) | ion qubit | Energy level of an electromagnetically trapped ion |

For a Qaithon matmul, the required count is roughly `log2(N)` for amplitude
encoding (information-theoretic minimum) and `~3 × log2(N)` with block
encoding overhead. Both numbers are reported by `qaithon.estimate_qubits`.

## Hardware capability matrix

| Capability | IBM | Quandela | QuEra | IonQ |
|---|---|---|---|---|
| Cost profile declared | yes | yes | yes | yes |
| Local simulator (ideal) | yes | yes | no (analog) | via SV1 |
| Local simulator (realistic noise) | FakeBrisbane | SLOS + loss | no | partial |
| Real hardware connected | yes | yes | yes | yes |
| Health check | yes | yes | yes | yes |
| Specific metrics | qubit counts | photon counts | atom counts | ion counts |
| Validation | qubits | modes | atoms | qubits |
| Real shot fired | smoke | first run | reserved | reserved |

## Hardware metrics

Each backend exposes a kind-specific metric structure after each call:

```python
from qaithon.backends.perceval_photonic import PercevalPhotonicBackend

backend = PercevalPhotonicBackend()
backend.matmul(x, w)
m = backend.last_photonic_metrics

print(m.n_modes_used)              # photonic modes
print(m.n_photons_injected)        # photons sent in
print(m.n_photons_detected)        # photons that reached detectors
print(m.detection_efficiency)      # ratio
print(m.accumulated_loss)          # 1 - efficiency
print(m.latency_us)                # wall-clock latency
```

Equivalent dataclasses: `SuperconductingMetrics`, `NeutralAtomMetrics`,
`HardwareMetrics` (generic).

## Validating a model against a target

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=64)
result = qaithon.validate_for_hardware(model, target="IBM Heron")

print(result.pretty())
# Either FITS or DOES NOT FIT with the exact constraint breached and
# concrete suggestions.
```

## Pricing estimates

```python
from qaithon.pricing import estimate_cost_usd

# 1000 shots × 1 task × 10 seconds compute:
cost = estimate_cost_usd("aws.braket.quera", n_shots=1000, n_tasks=1)
# $10.30 (publicly-listed AWS rate)
```

Local simulators report `$0.00`. Quandela Belenos returns `-1` (undisclosed
pricing for research access).

## Smoke shot, in case you want to reproduce

The first execution of a circuit on **Quandela Belenos** (real photonic QPU
hardware, 6 modes) measured:

| Metric | Value |
|---|---|
| Wall-clock latency | 33.99 s |
| Shots requested | 100 |
| Photons injected | 200 |
| Photons detected | 103 |
| Detection efficiency | 51.50% |
| Loss measured | 48.50% |

That's the live photonic loss number you cannot get from a local simulator.
