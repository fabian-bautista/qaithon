# Quick Start (English)

Five minutes from `pip install` to running your first **tiny** transformer
genuinely.

## 1. Install

```bash
pip install qaithon[huggingface,pennylane]
```

## 2. Run a tiny model genuinely

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

model = qaithon.compile(model, backends=("pennylane.sim",))  # genuine quantum
print(model.qaithon_report.pretty())
```

That's it. The model's linear layers now run on real quantum circuits (in
simulation). Use it like any other PyTorch / HuggingFace model.

> Only **tiny** transformers run genuinely. GPT-2 and larger compile, but
> emulating their layers needs far more than a laptop (simulating quantum is
> exponential; a 45–50 qubit emulator needs a supercomputer).

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
