# Quickstart

Qaithon connects HuggingFace transformers to genuine photonic / quantum
computing — for **tiny** models. No quantum knowledge required.

## Install

```bash
pip install qaithon[huggingface,pennylane]
```

## Run a tiny model genuinely (one line to compile)

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # genuine quantum

inputs = tok("Once upon a time", return_tensors="pt")
print(tok.decode(model.generate(**inputs, max_new_tokens=20)[0]))
```

That's it. Qaithon detected the model family, picked a genuine backend per layer,
skipped tied weights / embeddings, and preserved behavior — its linear layers now
run on real quantum circuits (in simulation).

> Only **tiny** transformers run genuinely. Larger models (GPT-2 and up) compile,
> but emulating their wider layers needs far more than a laptop — simulating
> quantum is exponential (each qubit doubles the memory; a 45–50 qubit emulator
> needs a supercomputer). See [`02_huggingface_models.md`](02_huggingface_models.md)
> for what runs where, and why.

## Inspect the decisions

```python
print(model.qaithon_report.pretty())
```

## Sanity-check from the CLI

```bash
qaithon list-backends   # what's installed and available
qaithon doctor          # full environment diagnosis
qaithon inspect gpt2    # analysis: what would compile() do? (does not run it)
```

That's the whole quickstart. Everything else in this cookbook is for users who
want explicit control over a specific aspect.
