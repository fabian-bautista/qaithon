# Qaithon

Corré partes de un LLM de HuggingFace con **algoritmos cuánticos y fotónicos
genuinos** — en simuladores hoy, en QPUs reales cuando se abran — sin reescribir
el modelo. Un paso pequeño, honesto y reproducible en la frontera IA × cuántica × fotónica.

[English](README.md) · [Documentación](docs/es/README.md) · [Recetario](docs/es/COOKBOOK.md) · [Configuración](docs/es/CONFIGURATION.md)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-253%20passing-brightgreen.svg)
![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)

---

## Qué hace

Escribís código PyTorch / HuggingFace normal. Qaithon recorre el modelo,
reemplaza las capas `nn.Linear` / Conv1D por versiones cuyos matmuls los
computa un **algoritmo cuántico o fotónico genuino**, y te devuelve el mismo
`nn.Module` que pasaste.

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # cuántico genuino

outputs = model.generate(input_ids, max_new_tokens=50)
```

Los matmuls corren el algoritmo real (Perceval/MerLin fotónica,
Qiskit/PennyLane qubits) — **no** matemática clásica con etiqueta cuántica.
Hoy se ejecutan en **simuladores locales**; el mismo código apunta a una QPU
real cuando se abra el acceso. Funciona a **escala pequeña** — el hardware
cuántico/fotónico es genuinamente diminuto hoy, así que esto es primero un
instrumento de investigación. Mirá [`docs/es/FINDINGS.md`](docs/es/FINDINGS.md)
para ver hasta dónde llega la tecnología, con números medidos — incluido un
**modelo real preentrenado generando texto coherente en un circuito cuántico genuino**.

## Qué obtenés

- `qaithon.compile(model)` drop-in. Sin reescribir el modelo.
- **Cómputo genuino** — matmuls computados por algoritmos cuánticos/fotónicos
  reales, validados exactos contra el resultado clásico (no fingidos con ruido).
- Multi-backend: Quandela fotónico (Perceval/MerLin), IBM/PennyLane/DeepQuantum
  cuántico, más `mock` (la referencia clásica explícita). Conectores a hardware
  real IBM/Quandela/AWS (experimental).
- Auto-selección de un backend genuino por capa (objetivo: speed/energy/balanced).
- Capas fotónicas y cuánticas **entrenables** (`PhotonicLayer`, `QuantumLayer`).
- Estimación de presupuesto de qubits/modos + validación — te dice qué
  necesitaría *cualquier* modelo (incluso a escala GPT-3) en una QPU real. El
  análisis es gratis e instantáneo.
- Lab de toy/micro transformers: armar, entrenar y generar de punta a punta.
- Glosario embebido: `qaithon.explain("fidelity", 0.987)`; integración con HF Hub.
- Límites honestos y medidos por tecnología: [`docs/es/FINDINGS.md`](docs/es/FINDINGS.md).

## Soporte de hardware

Hoy, **el cómputo genuino corre en los simuladores locales** (el algoritmo real,
exacto). La columna "hardware real" significa que existe un conector, pero las
corridas físicas son diminutas y gateadas — mirá [`docs/es/FINDINGS.md`](docs/es/FINDINGS.md)
y el [roadmap](docs/en/ROADMAP.md) para ver qué corre de verdad en una QPU hoy.

| Vendor | Dispositivo | Tipo | Backend | Hardware real | Simulador local |
|---|---|---|---|---|---|
| IBM | Heron (156 qubits) | Superconductor | `ibm.heron` | sí | `ibm.aer` |
| IBM | Aer | Simulador | `ibm.aer` | n/a | sí (ideal + realista) |
| Quandela | Belenos (12 modos) | Fotónico | `quandela.belenos` | experimental | `quandela.perceval` |
| Quandela | Perceval SLOS | Sim fotónico | `quandela.perceval` | n/a | sí |
| Quandela | MerLin | Fotónico (autograd) | `quandela.merlin` | n/a | sí |
| AWS Braket | SV1 | Simulador | `aws.braket.sv1` | n/a | sí |
| AWS Braket | QuEra Aquila | Átomo neutro | `aws.braket.quera` | sí | n/a (analógico) |
| AWS Braket | IonQ Forte | Ion atrapado | `aws.braket.ionq` | sí | vía SV1 |
| Xanadu | PennyLane | Varios | `pennylane.sim` | vía plugins | sí |
| TuringQ | DeepQuantum | Multi | `deepquantum` | n/a | sí |

## Números medidos

El kernel genuino computa el matmul **de forma exacta** en simulación — el
circuito *es* la cuenta, así que reproduce el resultado clásico con precisión
de máquina:

| Backend | Tipo | Cómputo | Fidelidad vs clásico (sim) |
|---|---|---|---:|
| `quandela.perceval` / `quandela.sim` / `quandela.merlin` | fotónico | genuino (Perceval/MerLin) | 1.0000 |
| `ibm.aer` / `pennylane.sim` / `deepquantum` | cuántico | genuino (Qiskit/PennyLane) | 1.0000 |
| `mock` | referencia clásica | `F.linear` | 1.0000 |

La energía / latencia por backend vienen de perfiles de costo declarados. La
pérdida de fidelidad **no** la produce el simulador exacto — solo aparece en
hardware físico (pérdida óptica, decoherencia), que es justo el punto de correr ahí.

**Highlight — un modelo real en un circuito cuántico genuino:** el preentrenado
**TinyStories-1M** generó texto coherente ("...una niña llamada Lily a la que le
gustaba jugar al sol...") con sus 48 capas lineales computadas por circuitos
cuánticos genuinos — salida **idéntica** al clásico (error 1e-6).

Resultados completos, límites por tecnología (medidos en un laptop) y la
explicación en lenguaje simple de *por qué* es pequeño están en
**[`docs/es/FINDINGS.md`](docs/es/FINDINGS.md)**.

> Las corridas en hardware físico son diminutas y gateadas hoy (un solo circuito
> chico, no un modelo completo). Ver el [roadmap](docs/en/ROADMAP.md) — cuando se
> abra el hardware corregido de errores, el mismo código apunta a él.

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

### Inferencia genuina de un modelo pequeño preentrenado

Un modelo diminuto corre de punta a punta por circuitos cuánticos genuinos.
(Modelos más grandes como GPT-2 también compilan, pero sus capas más anchas
hacen lento el camino genuino — ver [`FINDINGS.md`](docs/es/FINDINGS.md) para
los límites medidos.)

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

model = qaithon.compile(model, backends=("pennylane.sim",))  # cuántico genuino
print(model.qaithon_report.pretty())

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

### Toy transformer: inferencia cuántica genuina

```python
import qaithon
from qaithon.lab import generate

model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
qaithon.compile(model, backends=("pennylane.sim",))   # cuántico genuino

# El cómputo de cada capa corre por un circuito cuántico real.
print(generate(model, "ROMEO:", max_new_tokens=100))
```

> Nota: el kernel genuino es de **inferencia** (no diferenciable). Para
> **entrenar** capas genuinas usá las capas diferenciables `PhotonicLayer` /
> `QuantumLayer` (ver [`docs/es/FINDINGS.md`](docs/es/FINDINGS.md)).

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
