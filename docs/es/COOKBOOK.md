# Recetario (Español)

Recetas prácticas, listas para copiar y pegar.

## Receta 1 — Inferencia con un modelo tiny de HuggingFace

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

# Solo los transformers tiny corren genuinos hoy; TinyStories-1M es el verificado.
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

model = qaithon.compile(model, backends=("pennylane.sim",))  # cuántico genuino
print(model.qaithon_report.pretty(explain=True))

outputs = model.generate(**tok("Computación cuántica", return_tensors="pt"),
                         max_new_tokens=30)
print(tok.decode(outputs[0]))
```

## Receta 2 — Escoger un objetivo

```python
qaithon.compile(model, optimize_for="speed")     # prioriza latencia
qaithon.compile(model, optimize_for="energy")    # prioriza energía
qaithon.compile(model, optimize_for="balanced")  # default
```

## Receta 3 — Restringir a backends específicos

```python
qaithon.compile(model, backends=("quandela.perceval", "mock"))
```

## Receta 4 — Lab de toy transformer (entrenar, luego inferencia genuina)

El kernel genuino es solo de inferencia, así que entrená primero (autograd clásico)
y luego compilá para inferencia genuina.

```python
import qaithon
from qaithon.lab import load_dataset, train, generate

# 1. Entrená un toy transformer chico de la forma normal (diferenciable).
model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
ds = load_dataset("shakespeare", block_size=64)
train(model, ds, steps=2000, batch_size=32, log_every=200)

# 2. Compilalo para correr inferencia en circuitos cuánticos/fotónicos genuinos.
qaithon.compile(model, backends=("quandela.sim",))
print(generate(model, "ROMEO:", max_new_tokens=100))
```

## Receta 5 — Fallar rápido cuando un modelo no entra en un QPU

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=256, n_layers=2)
qaithon.lab.train(
    model, ds, steps=1000,
    target_hardware="IBM Heron",   # valida antes del step 1
)
# Levanta IncompatibleHardwareError con sugerencias concretas si no cabe.
```

## Receta 6 — Estimar qubits sin instanciar el modelo

```python
import qaithon

# Solo análisis (cero RAM): estima el presupuesto de qubits de un modelo clase
# 70B desde su config — muestra el conteo imposible que necesitaría hoy.
report = qaithon.estimate_qubits_from_config(
    hidden_size=8192, n_layers=80, n_heads=64,
    intermediate_size=28672,
    model_class="clase 70B (solo análisis — no corre)",
)
print(report.pretty())
```

## Receta 7 — Comparar backends con métricas + costo USD

```python
import qaithon

result = qaithon.benchmarks.compare_backends(in_features=32, out_features=32)
print(result.pretty(explain=True))

cost = qaithon.pricing.estimate_cost_usd("aws.braket.quera", n_shots=1000)
print(f"Costo USD: ${cost:.2f}")
```

## Receta 8 — Health-check de un backend cloud antes de ejecutar

```python
import qaithon

heron = qaithon.backends.get_backend("ibm.heron")
status = heron.health_check()
print(f"online={status.online} pending={status.pending_jobs}")
if status.online:
    # seguro para correr
    ...
```

## Receta 10 — Métricas de inferencia para todo un generate()

```python
from qaithon.metrics import InferenceMetrics

with InferenceMetrics() as m:
    outputs = model.generate(**inputs, max_new_tokens=100)

print(m.pretty())
# Total backend calls: 4800
# Total latencia: 1,234,567 µs
# Total energía: 7,890 pJ
# Fotones detectados: 12,345
```
