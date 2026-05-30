# Referencia del API (Español)

API público de Python. Todo lo alcanzable vía `import qaithon`.

## Convenciones

- Las funciones públicas están listadas en `qaithon.__all__`. Cualquier
  cosa que no esté ahí es interna y puede cambiar sin aviso.
- Tensor in / tensor out — cada backend recibe `torch.Tensor` y
  devuelve `torch.Tensor`.
- Las excepciones heredan de `qaithon.QaithonError` para que se puedan
  capturar de manera genérica.

---

## Top-level

### `qaithon.compile(model, *, optimize_for="balanced", backends=None) -> nn.Module`

Reemplaza las capas reemplazables de `model` por versiones aceleradas
por backend y devuelve el mismo modelo (mutado in place).

| Parámetro | Tipo | Default | Notas |
|---|---|---|---|
| `model` | `nn.Module` | — | Cualquier modelo PyTorch, incluyendo transformers de HuggingFace. |
| `optimize_for` | `"speed" \| "energy" \| "balanced"` | `"balanced"` | Objetivo que se pasa al selector. |
| `backends` | `tuple[str, ...] \| None` | `None` | Restringe a estos backends. `None` = todos los disponibles. |

Efecto colateral: adjunta `model.qaithon_report` (un `CompileReport`).

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = qaithon.compile(model, optimize_for="energy")
print(model.qaithon_report.pretty(explain=True))
```

### `qaithon.__version__: str`

Versión actual del paquete.

---

## Configuración de credenciales estilo SDK

Todos los setters escriben en variables de entorno; exports existentes
siempre ganan.

### `qaithon.set_ibm_token(token, *, channel="ibm_quantum_platform", instance=None)`

Configura credenciales de IBM Quantum.

### `qaithon.set_aws_credentials(access_key_id, secret_access_key, *, region="us-east-1")`

Configura credenciales de AWS Braket.

### `qaithon.set_quandela_token(token)`

Configura credenciales de Quandela Cloud.

### `qaithon.set_huggingface_token(token)`

Configura credenciales del HuggingFace Hub. Escribe tanto `HF_TOKEN`
como `HUGGING_FACE_HUB_TOKEN` para que `huggingface_hub` lo tome de
forma nativa.

### `qaithon.configure(*, ibm_token=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region="us-east-1", quandela_token=None, huggingface_token=None, ibm_channel="ibm_quantum_platform", ibm_instance=None)`

Configuración de todos los proveedores en una sola llamada. Pasar
`None` no toca credenciales existentes (aditivo, nunca destructivo).

```python
qaithon.configure(
    ibm_token="...",
    aws_access_key_id="AKIA...", aws_secret_access_key="...",
    quandela_token="...",
    huggingface_token="hf_...",
)
```

### `qaithon.config.status() -> dict[str, bool]`

Retorna `{"ibm": bool, "aws": bool, "quandela": bool, "huggingface": bool}`.
NO expone los valores — es seguro hacerle log.

---

## Estimación de qubits y validación de hardware

### `qaithon.estimate_qubits(model: nn.Module) -> QubitReport`

Recorre `model` y produce presupuestos qubit + profundidad por capa
(amplitude encoding + block encoding).

### `qaithon.estimate_qubits_from_config(*, hidden_size, n_layers, n_heads=1, intermediate_size=None, vocab_size=0, model_class="FromConfig") -> QubitReport`

Estima desde una config sin instanciar el modelo (cero RAM). Funciona a cualquier
tamaño — pasale una config clase 405B y te muestra el conteo de qubits imposible
que necesitaría hoy. Solo análisis; no corre el modelo.

```python
report = qaithon.estimate_qubits_from_config(
    hidden_size=12288, n_layers=96, n_heads=96,
    model_class="GPT-3.5-equivalent",
)
print(report.pretty())
```

### `qaithon.measure_actual_circuit(in_features, out_features, *, target_backend=None) -> MeasuredCircuit`

Construye un circuito Qiskit real y reporta conteos exactos de qubits +
gates. Con `target_backend`, transpila contra un fake backend (p. ej.
`"FakeBrisbane"`) para obtener números realistas de hardware.

### `qaithon.validate_for_hardware(model=None, *, target, report=None) -> ValidationResult`

Chequea si un modelo cabe en un hardware target dado. Pasar o `model`
o un `report` ya computado.

```python
result = qaithon.validate_for_hardware(model, target="IBM Heron")
if not result.fits:
    print(result.pretty())  # explica por qué + sugerencias
```

### `qaithon.find_hardware(name: str) -> HardwareSpec`

Busca una hardware spec por substring case-insensitive.

### `qaithon.KNOWN_HARDWARE: tuple[HardwareSpec, ...]`

Registry incorporado: IBM Heron, IBM Brisbane, IBM Starling (proyectado),
IBM Blue Jay (proyectado), Quandela Belenos, QuEra Aquila, IonQ Forte
Enterprise 1.

---

## Glosario / explicabilidad

### `qaithon.explain(term: str, *, value=None) -> str`

Explicación pretty-printed de un término cuántico. Con `value`,
contextualiza ("una fidelity de 0.987 significa …").

### `qaithon.glossary(term: str) -> GlossaryEntry`

Lookup programático. Retorna un entry con `.short`, `.analogy`,
`.example`, `.rule_of_thumb`.

### `qaithon.list_terms() -> list[str]`

Todos los términos cubiertos por el glosario.

---

## Sub-paquetes

Cada uno es alcanzable como `qaithon.<name>`.

- **`qaithon.backends`** — registry de backends. Usar
  `qaithon.backends.get_backend("name")` para instanciar.
- **`qaithon.benchmarks`** — `compare_backends(in_features, out_features)`
  retorna una tabla latencia/energía por backend.
- **`qaithon.cache`** — `qaithon.cache.cached(backend)` envuelve un
  backend con cache in-memory + en disco.
- **`qaithon.config`** — credenciales (setters, status, getters).
- **`qaithon.fallback`** — fallbacks en runtime que no rompen nada.
- **`qaithon.handlers`** — architecture handlers (Mixtral, SDPA, …).
- **`qaithon.hub`** — `push_block(...)` / `load_block(...)` sobre
  HuggingFace Hub.
- **`qaithon.integrations`** — adapters a terceros (skeleton de vLLM).
- **`qaithon.lab`** — `train(model, dataset, ...)`, `generate(...)`,
  `load_dataset(...)` — una superficie de training mini integrada.
- **`qaithon.metrics`** — `HardwareMetrics`, `SuperconductingMetrics`,
  `PhotonicMetrics`, `NeutralAtomMetrics`, `InferenceMetrics`.
- **`qaithon.models`** — `create_toy_transformer(...)` para el lab.
- **`qaithon.observability`** — counters y exporters.
- **`qaithon.pipeline`** — composición de modelos compilados.
- **`qaithon.plugins`** — descubrimiento de backends de terceros vía
  entry points.
- **`qaithon.pricing`** — `estimate_cost_usd("backend.name", n_shots=…)`.
- **`qaithon.streaming`** — compatibilidad con HF `TextStreamer`.
- **`qaithon.tracing`** — `with tracing.trace() as t: …`.
- **`qaithon.training`** — `QATConfig`, `prepare_for_qat(...)`.

---

## Excepciones

Todas heredan de `qaithon.QaithonError`.

| Excepción | Cuándo |
|---|---|
| `BackendError` | Cualquier error a nivel de backend. |
| `BackendNotAvailableError` | El `is_available()` del backend retornó `False`. |
| `BackendNotRegisteredError` | El nombre no está en el registry. |
| `CompileError` | `compile()` no pudo producir un plan válido. |
| `IncompatibleModelError` | El modelo tiene una capa que Qaithon se niega a tocar. |
| `IncompatibleHardwareError` | `validate_for_hardware` dice que no cabe. |
| `UnsupportedOperationError` | El backend no implementa esta op (todavía). |

---

## Reportes

### `CompileReport`

Adjunto a cada modelo compilado como `model.qaithon_report`.

- `.summary()` — resumen de una línea.
- `.pretty(explain=False)` — desglose completo por capa.
- `.decisions: tuple[LayerDecision, ...]` — audit log crudo.

### `LayerDecision`

Registro por capa: nombre, dimensiones, backend elegido, razón,
energía/latencia estimadas.

### `QubitReport`

Presupuesto qubit + profundidad para todo el modelo. Usar `.pretty()`
para output humano, `.hardware_compatibility()` para checks
programáticos.

### `ValidationResult`

`fits: bool`, `target: HardwareSpec`, `report: QubitReport`, más listas
de `reasons` y `recommendations`.

---

## Interface de Backend (avanzado)

```python
from qaithon.backends.base import Backend, BackendProfile, register_backend

class MyBackend(Backend):
    profile = BackendProfile(name="vendor.mychip", ...)
    def matmul(self, x, weight, bias=None) -> torch.Tensor: ...
    def is_available(self) -> bool: ...
    def health_check(self) -> HealthStatus: ...

register_backend("vendor.mychip", MyBackend)
```

`HealthStatus` tiene `backend`, `online`, `message`, `latency_ms`.

---

## Ejemplos por caso de uso

### Inferencia con caching

```python
import qaithon
backend = qaithon.backends.get_backend("ibm.aer")
cached = qaithon.cache.cached(backend)
```

### Training con ruido QAT

```python
from qaithon.training import QATConfig, prepare_for_qat
prepare_for_qat(model, QATConfig(noise_std_training=0.05))
```

### Estimación de costo antes de correr en QPU real

```python
cost = qaithon.pricing.estimate_cost_usd("aws.braket.quera", n_shots=1000)
```

### Health check antes de un run en hardware real

```python
heron = qaithon.backends.get_backend("ibm.heron")
status = heron.health_check()
assert status.online
```
