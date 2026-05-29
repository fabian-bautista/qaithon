# Qaithon

Corré partes de cualquier LLM de HuggingFace sobre backends fotónicos o
cuánticos sin reescribir el modelo.

[English](README.md) · [Documentación](docs/es/README.md) · [Recetario](docs/es/COOKBOOK.md) · [Configuración](docs/es/CONFIGURATION.md)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-253%20passing-brightgreen.svg)
![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)

---

## Qué hace

Escribís código PyTorch / HuggingFace normal. Qaithon recorre el modelo,
reemplaza las capas `nn.Linear` / Conv1D reemplazables por versiones
aceleradas por backend, y te devuelve el mismo `nn.Module` que pasaste —
solo que ahora sus matmuls pueden correr en un backend fotónico o
cuántico.

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = qaithon.compile(model)

outputs = model.generate(input_ids, max_new_tokens=50)
```

El selector elige el mejor backend por capa según lo que esté instalado
en tu máquina y el objetivo que le pases (`"speed"`, `"energy"`,
`"balanced"`). Todo lo que está debajo del API público es detalle interno
reemplazable.

## Qué obtenés

- `qaithon.compile(model)` drop-in. Sin reescribir el modelo.
- Multi-backend: IBM superconductor, Quandela fotónico, AWS QuEra e
  IonQ, más simuladores locales.
- Auto-selección de backend por capa, con un modo adaptativo opcional
  que aprende de la latencia / energía observadas.
- Funciona con cualquier modelo del HuggingFace Hub.
- Lab de toy transformers: armar, entrenar y generar de punta a punta en
  4 líneas.
- Métricas reales: latencia, energía (pJ/MAC), fidelidad, conteo de
  fotones.
- Validación de hardware que se niega a empezar a entrenar si el modelo
  excede el presupuesto de qubits / depth de la QPU objetivo.
- Health checks contra QPUs cloud antes de enviar jobs.
- Glosario embebido: `qaithon.explain("fidelity", 0.987)`.
- Integración con HuggingFace Hub vía `qaithon.hub`.

## Soporte de hardware

| Vendor | Dispositivo | Tipo | Backend | Hardware real | Simulador local |
|---|---|---|---|---|---|
| IBM | Heron (156 qubits) | Superconductor | `ibm.heron` | sí | `ibm.aer` |
| IBM | Aer | Simulador | `ibm.aer` | n/a | sí (ideal + realista) |
| Quandela | Belenos (6 modos) | Fotónico | `quandela.belenos` | sí | `quandela.perceval` |
| Quandela | Perceval SLOS | Sim fotónico | `quandela.perceval` | n/a | sí |
| Quandela | MerLin | Fotónico (autograd) | `quandela.merlin` | n/a | sí |
| AWS Braket | SV1 | Simulador | `aws.braket.sv1` | n/a | sí |
| AWS Braket | QuEra Aquila | Átomo neutro | `aws.braket.quera` | sí | n/a (analógico) |
| AWS Braket | IonQ Forte | Ion atrapado | `aws.braket.ionq` | sí | vía SV1 |
| Xanadu | PennyLane | Varios | `pennylane.sim` | vía plugins | sí |
| TuringQ | DeepQuantum | Multi | `deepquantum` | n/a | sí |

## Números medidos

Comparación cross-backend sobre un matmul `(2, 16) × (16, 16)`:

| Backend | Tipo | Latencia (µs) | Energía (pJ) | Fidelidad vs clásico |
|---|---|---:|---:|---:|
| `mock` (baseline GPU) | clásico | 1 | 256.00 | 1.0000 |
| `quandela.perceval` | fotónico, circuito real | 506 | 1.28 | 0.9845 |
| `quandela.sim` | perfil fotónico | 1 | 12.80 | 1.0000 |
| `ibm.aer` | cuántico, circuito real | 40,437 | 153.60 | 0.9999 |
| `aws.braket.sv1` | cuántico | 1 | 128.00 | 1.0000 |
| `pennylane.sim` | cuántico | 1 | 204.80 | 1.0000 |

`quandela.perceval` reporta cerca de dos órdenes de magnitud menos
energía por MAC que el baseline clásico, a un costo de 1.5% de
fidelidad — dentro del rango que QAT entrena al modelo a tolerar.

Primera corrida en **Quandela Belenos** (QPU fotónica en vivo):

```
Latencia:                33.99 segundos (cola cloud + ejecución)
Fotones inyectados:      200
Fotones detectados:      103
Eficiencia de detección: 51.50%
Pérdida medida:          48.50%
```

Estos números vienen del hardware; el simulador local no los produce.

## Instalación

```bash
pip install qaithon

# Agregá backends de hardware según necesites.
pip install qaithon[huggingface,pennylane,quandela,deepquantum]
```

## Configuración

Las credenciales se resuelven en este orden (gana la más alta): setters
programáticos, variables de entorno, un archivo `.env` local al
proyecto.

```python
import qaithon

qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...", aws_secret_access_key="...",
    aws_region="us-east-1",
    quandela_token="...",
    huggingface_token="hf_...",
)

# Los setters por proveedor también funcionan.
qaithon.set_ibm_token("...")
qaithon.set_aws_credentials("AKIA...", "secret...")
qaithon.set_quandela_token("...")
qaithon.set_huggingface_token("hf_...")

# Diagnóstico — booleanos, nunca los valores.
qaithon.config.status()
# {'ibm': True, 'aws': True, 'quandela': True, 'huggingface': True}
```

Detalles en [`docs/es/CONFIGURATION.md`](docs/es/CONFIGURATION.md).

## Workflows de inicio rápido

### Inferencia sobre cualquier modelo de HuggingFace

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

### Toy transformer con training e inferencia

```python
import qaithon
from qaithon.lab import load_dataset, train, generate

model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
qaithon.compile(model, backends=("quandela.sim",))

dataset = load_dataset("shakespeare")
train(model, dataset, steps=2000, noise_std=0.05)  # QAT activado

print(generate(model, "ROMEO:", max_new_tokens=100))
```

### Validar que un modelo cabe en un hardware target

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=128, n_layers=2)
result = qaithon.validate_for_hardware(model, target="IBM Heron")
print(result.pretty())
# Se niega a empezar a entrenar si el modelo excede el presupuesto de la QPU.
```

### Estimar costo en qubits de cualquier modelo

```python
import qaithon
# Funciona con modelos enormes — usa la config, nunca instancia los weights.
report = qaithon.estimate_qubits_from_config(
    hidden_size=12288, n_layers=96, n_heads=96,
    model_class="GPT-3.5 equivalent",
)
print(report.pretty())
```

### Comparar backends lado a lado

```bash
qaithon benchmark --explain
```

## Herramientas de línea de comandos

```bash
qaithon list-backends        # backends registrados + disponibilidad
qaithon doctor               # diagnostica el entorno local
qaithon inspect <model_id>   # ¿qué le haría compile() a este modelo?
qaithon compile <model_id>   # compila un modelo HF y lo guarda
qaithon estimate <model_id>  # presupuesto de qubits para correr en QPU
qaithon benchmark --explain  # comparación cross-backend
qaithon glossary <term>      # explicación en lenguaje sencillo
qaithon trace inspect <file> # imprime un trace JSON guardado
qaithon plugins list         # backends de terceros vía entry points
```

## Estructura del proyecto

```
qaithon/
  compile.py           qaithon.compile() — entry point principal
  compile_report.py    audit trail de las decisiones de compile
  config.py            credenciales y setters estilo SDK
  glossary.py          explicaciones en lenguaje sencillo
  qubits.py            estimación de presupuesto de qubits + validación
  benchmarks.py        comparación de performance cross-backend
  pricing.py           estimación de costo USD por proveedor
  metrics.py           PhotonicMetrics, SuperconductingMetrics, etc.
  backends/            IBM, Quandela, AWS, PennyLane, DeepQuantum
  layers/              QuantumLinear, QuantumAttention
  ir/                  analyzer + AutoBackendSelector + AdaptiveBackendSelector
  handlers/            hooks por arquitectura (Mixtral, attention, ...)
  models/              fábricas de toy transformers dimensionadas para QPUs reales
  lab/                 datasets + loop de training + helpers de inferencia
  hub/                 integración con HuggingFace Hub
  training.py          QATConfig + prepare_for_qat
  streaming.py         streamers compatibles con HF
  pipeline.py          composición de pipelines
  cache.py             caches in-memory + en disco
  tracing.py           observabilidad liviana
  observability.py     exporter de OpenTelemetry
  fallback.py          fallback graceful en runtime entre backends
  plugins.py           descubrimiento de backends de terceros vía entry_points
  integrations/        vLLM (skeleton), más por venir
```

## Licencia

MIT — ver [`LICENSE`](LICENSE).

## Autor

Fabián Bautista — `fabbagar83@gmail.com`
