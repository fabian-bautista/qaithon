# Arquitectura (Español)

Cómo está armado Qaithon y por qué cada pieza existe.

## Objetivos que guían toda decisión

1. **Drop-in para devs de IA.** El usuario escribe `qaithon.compile(model)`
   y nunca ve una primitiva cuántica. Todo lo que está por debajo del API
   público es detalle interno reemplazable.
2. **Tensor in / tensor out.** Los backends reciben `torch.Tensor` y
   devuelven `torch.Tensor`. El vocabulario cuántico se queda dentro del
   backend.
3. **Costos honestos.** Los backends declaran lo que cuestan
   (`BackendProfile`) y el selector elige basado en esos números; el
   reporte muestra el trade-off en voz alta.
4. **SOLID, en particular Open/Closed.** Hardware nuevo = nueva subclase
   `Backend` + una línea en el registry. Cero ediciones a `compile()`.

## Flujo de alto nivel

```
  código del user      qaithon                           backends
  ───────────────      ───────                           ────────
                                       ┌─► MockBackend
  AutoModel  ──►  compile()  ──►  Walker  ──►  Selector  ─┼─► PennyLane
                  │              │            │           ├─► AerSimulator
                  │              ▼            ▼           ├─► Perceval (SLOS)
                  │       ReplacementPlan  SelectionResult├─► MerLin (nativo)
                  ▼              │            │           ├─► IBM Heron (real)
            CompileReport  ◄─────┴────────────┘           ├─► Quandela Belenos
                                                          ├─► AWS Braket SV1
                                                          └─► AWS QuEra / IonQ
```

Tres capas, cada una con una sola responsabilidad.

### Capa 1 — Walker (`qaithon/ir/analyzer.py`)

Recorre un `torch.nn.Module` y produce un `ReplacementPlan`:
qué sub-módulos son reemplazables (`nn.Linear` / `Conv1D` /
HuggingFace `Linear4bit`), cuáles no, cuáles están atadas (p. ej.
`lm_head ↔ wte`), y por qué cada capa excluida lo fue.

El truco no-trivial del walker: usa **identity check**
(`type(m) is nn.Linear`) en vez de `isinstance`, igual que
bitsandbytes. Esa distinción de un carácter es la que evita que
Qaithon corrompa `NonDynamicallyQuantizableLinear` dentro de los
bloques de atención.

### Capa 2 — Selector (`qaithon/ir/selector.py` + `adaptive_selector.py`)

Dado un `ReplacementPlan` y un `objective`
(`"speed" | "energy" | "balanced"`), el selector elige un backend por
capa. El default es greedy por capa basado en el `BackendProfile`
declarado. El `AdaptiveBackendSelector` (opt-in) registra latencia y
energía observadas y las mezcla de vuelta en el cost model vía running
mean estilo Welford.

### Capa 3 — Backend (`qaithon/backends/*.py`)

Un `Backend` expone un método: `matmul(x, weight, bias)`. Adentro
puede llamar a PennyLane, Aer, Perceval, MerLin, AWS Braket o lo que
sea. El fallback clásico es `mock`, siempre disponible y nunca rompe
el modelo del usuario.

## El contrato de `compile()`

```python
def compile(model, *, optimize_for="balanced", backends=None) -> nn.Module:
    plan = analyze_model(model)                     # walker
    result = AutoBackendSelector(...).select(plan,  # selector
                                              objective=optimize_for)
    replace_layers(model, result.per_layer)         # mutator
    model.qaithon_report = CompileReport(...)       # auditabilidad
    return model
```

`compile()` muta el modelo in place Y lo retorna (convención de
PyTorch). Cuando termina, cada capa reemplazada es un `QuantumLinear`
o un `QuantumAttention` que delega al backend asignado.

## Registry — cómo se enchufan nuevos backends

```python
from qaithon.backends.base import Backend, register_backend

class MyBackend(Backend):
    profile = BackendProfile(name="vendor.mychip", ...)
    def matmul(self, x, w, b=None): ...
    def is_available(self) -> bool: ...

register_backend("vendor.mychip", MyBackend)
```

Esa es toda la superficie de extensión. No cambia nada más. Los tests
del nuevo backend viven en `tests/test_backends_<name>.py`.

Paquetes de terceros también pueden registrarse vía el entry point
`qaithon.backends` — ver `qaithon/plugins.py`.

## Capas que reemplazan `nn.Linear`

- `QuantumLinear` (`qaithon/layers/quantum_linear.py`) — reemplazo
  drop-in de `nn.Linear` que tiene una referencia al backend y delega
  `forward()` a `backend.matmul`.
- `QuantumAttention` (`qaithon/layers/quantum_attention.py`) — bloque
  de atención componible que corre las proyecciones Q/K/V en un backend
  manteniendo el softmax clásico.

Las dos clases preservan el layout del state_dict del módulo huésped,
así que `from_pretrained` sigue funcionando.

## Integración con HuggingFace

Dos caminos, complementarios:

1. **`qaithon.compile(model)`** — funciona sobre cualquier `nn.Module`.
   Es el entry point recomendado.
2. **`QaithonHfQuantizer`** (`qaithon/hf_integration.py`) — registrado
   con HuggingFace transformers vía `HfQuantizer`. Se activa cuando se
   pasa un `QuantizationConfig(method="qaithon")` a
   `AutoModel.from_pretrained`. Permite opt-in vía la superficie
   estándar de HF.

Los architecture handlers (`qaithon/handlers/`) lidian con las
peculiaridades por vendor: routing de experts de Mixtral MoE, Flash
Attention, SDPA, proyecciones estilo Conv1D.

## Observabilidad

Cada llamada a backend pasa por `qaithon.tracing` — un evento por
llamada con nombre del backend, latencia y energía estimada. El trace
se exporta a OpenTelemetry cuando está configurado, y se agrega en
`InferenceMetrics` para resúmenes a primera vista.

## Métricas de hardware

Tres especializaciones de `HardwareMetrics`:

- `SuperconductingMetrics` — la pobla `ibm.aer`. Incluye conteo de
  qubits, conteo de gates, fidelity promedio por gate (en modo
  `realistic`), tiempo de cola.
- `PhotonicMetrics` — la pobla `quandela.perceval` y
  `quandela.belenos`. Incluye conteo de modos, fotones inyectados vs
  detectados, eficiencia de detección, pérdidas acumuladas.
- `NeutralAtomMetrics` — la pobla `aws.braket.quera`. Conteo de átomos,
  tasa de éxito de rearrangement, radio de blockade de Rydberg.

Cada backend las expone vía una property `last_*_metrics` después de un
matmul.

## Configuración

`qaithon.config` tiene tres capas, en orden de prioridad (gana la más
alta):

1. Setters programáticos (`qaithon.set_ibm_token(...)` etc.).
2. Variables de entorno (`IBM_QUANTUM_TOKEN`, `AWS_ACCESS_KEY_ID`, …).
3. Un archivo `.env` local al proyecto.

`qaithon.config.status()` retorna un dict `{provider: bool}` para que
una UI diagnóstica pueda mostrar qué credenciales están presentes **sin
exponer nunca los valores**.

## Anti-objetivos (cosas que Qaithon explícitamente NO hace)

- No "construir el hardware". Qaithon es software que apunta a
  hardware cuántico / fotónico *existente* vía SDKs oficiales.
- No DSL cuántico propietario. Los backends usan APIs nativas de cada
  vendor.
- No mocks que finjan ser hardware real. Los backends con `kind` de
  `quantum` o `photonic` o ejecutan circuits / shots reales (con costo
  explícito) o se etiquetan como simuladores.
- No fallbacks silenciosos. Cuando un backend no está disponible, el
  reporte lo dice.

## Dónde buscar en el código

| Concepto | Archivo |
|---|---|
| Superficie pública del API | `qaithon/__init__.py` |
| Entry point de `compile()` | `qaithon/compile.py` |
| Walker | `qaithon/ir/analyzer.py` |
| Selector por defecto | `qaithon/ir/selector.py` |
| Selector adaptativo | `qaithon/ir/adaptive_selector.py` |
| ABC de Backend + registry | `qaithon/backends/base.py` |
| Cada backend | `qaithon/backends/<nombre>.py` |
| Integración HuggingFace | `qaithon/hf_integration.py` |
| Architecture handlers | `qaithon/handlers/` |
| Reemplazos de capa | `qaithon/layers/` |
| Credenciales | `qaithon/config.py` |
| Tracing | `qaithon/tracing.py` |
| Métricas | `qaithon/metrics.py` |
| Glosario | `qaithon/glossary.py` |
| CLI | `qaithon/cli.py` |
