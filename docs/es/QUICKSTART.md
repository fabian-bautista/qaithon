# Inicio rápido (Español)

Cinco minutos desde `pip install` hasta correr tu primer transformer **tiny** de
forma genuina.

## 1. Instalar

```bash
pip install qaithon[huggingface,pennylane]
```

## 2. Correr un modelo tiny de forma genuina

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

model = qaithon.compile(model, backends=("pennylane.sim",))  # cuántico genuino
print(model.qaithon_report.pretty())
```

Eso es todo. Las capas lineales del modelo ahora corren en circuitos cuánticos
reales (en simulación). Usalo como cualquier modelo de PyTorch / HuggingFace.

> Solo los transformers **tiny** corren genuinos. GPT-2 y más grandes compilan,
> pero emular sus capas necesita mucho más que un laptop (simular cuántica es
> exponencial; un emulador de 45–50 qubits necesita una supercomputadora).

```python
inputs = tokenizer("Hola mundo", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## 3. Inspeccionar qué pasó

```python
print(model.qaithon_report.pretty(explain=True))
```

El flag `explain=True` agrega explicaciones en lenguaje sencillo de cada
métrica (latencia, energía por MAC, fidelidad, etc.). Para cualquier término:

```python
print(qaithon.explain("fidelity", 0.987))
```

## 4. Probar otro objetivo

```python
import qaithon

model = ...  # Recargá un modelo fresco
model = qaithon.compile(model, optimize_for="energy")
# vs "speed" o "balanced" (default)
```

## 5. Benchmark de backends lado a lado

```bash
qaithon benchmark --explain
```

## Próximos pasos

- [Configuración](CONFIGURATION.md) — conectar QPUs cloud reales.
- [Recetario](COOKBOOK.md) — objetivos, selección de backend, pipelines.
- [Soporte de hardware](HARDWARE.md) — cada backend en detalle.
