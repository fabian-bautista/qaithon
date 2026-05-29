# Quickstart

Qaithon takes any HuggingFace LLM and makes it run on photonic / quantum
backends. No quantum knowledge required.

## Install

```bash
pip install qaithon[huggingface,quandela]
```

## Compile any model in one line

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = qaithon.compile(model)

# Use the model exactly like before.
inputs = tokenizer("Photonic computing is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

That's it. Qaithon:

* Detected the model family (GPT-2).
* Picked the optimal backend per layer from what's available on your machine.
* Skipped tied weights and embeddings automatically.
* Preserved the model's behavior so your inference code keeps working.

## Inspect the decisions

```python
print(model.qaithon_report.pretty())
```

Sample output:

```
Qaithon compile report for GPT2LMHeadModel
  Parameters:           124,439,808
  Layers replaced:      48
  Layers skipped:       1
  Objective:            balanced
  Backends used:        quandela.sim
  Estimated energy:     <X> pJ (baseline <Y> pJ, save 95.0%)
```

## Sanity-check from the CLI

```bash
qaithon list-backends   # what's installed and available
qaithon doctor          # full environment diagnosis
qaithon inspect gpt2    # what would compile() do to this model?
```

That's the whole quickstart. Everything else in this cookbook is for users
who want explicit control over a specific aspect.
