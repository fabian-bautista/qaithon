# Inicio rápido (Español)

Cinco minutos desde `pip install` hasta tu primer modelo HuggingFace compilado.

## 1. Instalar

```bash
pip install qaithon[huggingface]
```

## 2. Compilar cualquier modelo de HuggingFace

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

model = qaithon.compile(model)
print(model.qaithon_report.pretty())
```

Eso es todo. El modelo ahora rutea sus proyecciones lineales a través del
selector de backends de Qaithon. Usalo como cualquier modelo de PyTorch /
HuggingFace.

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
- [Recetario](COOKBOOK.md) — fine-tuning, pipelines, QAT.
- [Soporte de hardware](HARDWARE.md) — cada backend en detalle.
