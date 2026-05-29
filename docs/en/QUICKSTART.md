# Quick Start (English)

Five minutes from `pip install` to your first compiled HuggingFace model.

## 1. Install

```bash
pip install qaithon[huggingface]
```

## 2. Compile any HuggingFace model

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

model = qaithon.compile(model)
print(model.qaithon_report.pretty())
```

That's it. The model now routes its linear projections through Qaithon's
backend selector. Use it like any other PyTorch / HuggingFace model.

```python
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

## 3. Inspect what happened

```python
print(model.qaithon_report.pretty(explain=True))
```

The `explain=True` flag adds plain-language explanations of every metric
(latency, energy per MAC, fidelity, etc.). For any term:

```python
print(qaithon.explain("fidelity", 0.987))
```

## 4. Try a different objective

```python
import qaithon

model = ...  # Reload a fresh model
model = qaithon.compile(model, optimize_for="energy")
# vs "speed" or "balanced" (default)
```

## 5. Benchmark backends side-by-side

```bash
qaithon benchmark --explain
```

## Next steps

- [Configuration](CONFIGURATION.md) — connect to real cloud QPUs.
- [Cookbook](COOKBOOK.md) — fine-tuning, pipelines, QAT.
- [Hardware Support](HARDWARE.md) — every backend in detail.
