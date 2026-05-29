# Soporte de hardware

Todos los backends a los que Qaithon habla, qué hacen, cuánto cuestan y
cómo activarlos.

## Simuladores locales (sin cuenta)

| Backend | Tecnología | Qué hace | Energía/MAC | Disponible |
|---|---|---|---|---|
| `mock` | Clásico de referencia | Baseline `F.linear` | 1.0 pJ | Siempre |
| `ibm.aer` | Sim superconductor | Circuitos reales en Qiskit, modos `ideal` o `realistic` (noise FakeBrisbane) | 1.5 pJ | `qiskit-aer` instalado |
| `quandela.sim` | Perfil fotónico | `F.linear` + perfil de costo fotónico | 0.05 pJ | `merlin` instalado |
| `quandela.perceval` | Sim fotónico real | Circuitos linear-ópticos vía Perceval SLOS | 0.005 pJ | `perceval`+`merlin` |
| `pennylane.sim` | Sim QML | `F.linear` + perfil PennyLane | 0.8 pJ | `pennylane` instalado |
| `deepquantum` | Sim multi-tech | DeepQuantum nativo | 0.4 pJ | `deepquantum` instalado |

## Hardware cloud real

| Backend | Tecnología | Proveedor | Free tier | Modo |
|---|---|---|---|---|
| `ibm.heron` | Superconductor (156 qubits) | IBM | 10 min/mes | `calibrate` (default) |
| `ibm.quantum` | Solo perfil | IBM | n/a | profile |
| `aws.braket.sv1` | Sim clásico (cloud) | AWS | ~1 h/mes | `calibrate` |
| `aws.braket.quera` | Átomo neutro (256 átomos) | QuEra vía AWS | ninguno, pay-per-shot | `calibrate` |
| `aws.braket.ionq` | Ion atrapado (36 qubits) | IonQ vía AWS | ninguno, pay-per-shot | `calibrate` |
| `quandela.belenos` | Fotónico (6 modos) | Quandela | research-gated | `calibrate` |

## Modos de ejecución

Todo backend de hardware real soporta tres modos:

- **`profile`** (default) — no se disparan circuitos reales. Usa el cost
  model declarado. Seguro para desarrollo. Costo cero.
- **`calibrate`** — dispara un circuito pequeño de calibración por forward
  para capturar latencia y ruido reales. Costo modesto en QPUs cloud.
- **`execute`** — cada matmul se convierte en un circuito real. Lento y caro;
  activar solo para benchmarks explícitos.

```python
from qaithon.backends.ibm_heron import IBMHeronBackend

# Default (modo profile): costo cero.
backend = IBMHeronBackend()

# Calibración con circuito real por forward.
backend = IBMHeronBackend(mode="calibrate")
```

## Qué significa "equivalente a qubit" por tecnología

La capacidad del hardware se mide distinto según la tecnología. Las
funciones `estimate_qubits` y `validate_for_hardware` de Qaithon traducen
transparentemente.

| Tecnología | Unidad | Qué representa |
|---|---|---|
| Superconductor (IBM) | qubit | Spin en una junción de Josephson |
| Fotónico (Quandela) | modo fotónico | Canal espacial; los fotones interfieren al pasar |
| Átomo neutro (QuEra) | qubit atómico | Nivel de energía de un átomo Rydberg atrapado |
| Ion atrapado (IonQ) | qubit iónico | Nivel de energía de un ion atrapado electromagnéticamente |

Para una matmul de Qaithon, el conteo requerido es aproximadamente
`log2(N)` con amplitude encoding (mínimo teórico) y `~3 × log2(N)` con
block encoding overhead. Los dos números los reporta
`qaithon.estimate_qubits`.

## Matriz de capacidades por hardware

| Capacidad | IBM | Quandela | QuEra | IonQ |
|---|---|---|---|---|
| Cost profile declarado | sí | sí | sí | sí |
| Simulador local (ideal) | sí | sí | no (analógico) | vía SV1 |
| Simulador local (noise realista) | FakeBrisbane | SLOS + loss | no | parcial |
| Hardware real conectado | sí | sí | sí | sí |
| Health check | sí | sí | sí | sí |
| Métricas específicas | conteos qubits | conteos fotones | conteos átomos | conteos iones |
| Validación | qubits | modos | átomos | qubits |
| Disparo real realizado | smoke | primer run | reservado | reservado |

## Métricas de hardware

Cada backend expone una estructura específica de métricas tras cada call:

```python
from qaithon.backends.perceval_photonic import PercevalPhotonicBackend

backend = PercevalPhotonicBackend()
backend.matmul(x, w)
m = backend.last_photonic_metrics

print(m.n_modes_used)              # modos fotónicos
print(m.n_photons_injected)        # fotones enviados
print(m.n_photons_detected)        # fotones que llegaron al detector
print(m.detection_efficiency)      # ratio
print(m.accumulated_loss)          # 1 - efficiency
print(m.latency_us)                # latencia wall-clock
```

Dataclasses equivalentes: `SuperconductingMetrics`, `NeutralAtomMetrics`,
`HardwareMetrics` (genérico).

## Validar un modelo contra un target

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=64)
result = qaithon.validate_for_hardware(model, target="IBM Heron")

print(result.pretty())
# Devuelve FITS o DOES NOT FIT con la restricción exacta violada y
# sugerencias concretas.
```

## Estimación de costos

```python
from qaithon.pricing import estimate_cost_usd

# 1000 shots × 1 task × 10 segundos de cómputo:
cost = estimate_cost_usd("aws.braket.quera", n_shots=1000, n_tasks=1)
# $10.30 (tarifa publicada por AWS)
```

Los simuladores locales reportan `$0.00`. Quandela Belenos devuelve `-1`
(pricing no divulgado para acceso research).

## Smoke shot, por si querés reproducir

La primera ejecución de un circuito en **Quandela Belenos** (hardware QPU
fotónico real, 6 modos) midió:

| Métrica | Valor |
|---|---|
| Latencia wall-clock | 33.99 s |
| Shots solicitados | 100 |
| Fotones inyectados | 200 |
| Fotones detectados | 103 |
| Eficiencia de detección | 51.50% |
| Pérdida medida | 48.50% |

Ese es el número de pérdida fotónica en vivo que un simulador local no
puede entregarte.
