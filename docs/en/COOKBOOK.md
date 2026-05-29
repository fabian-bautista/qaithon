# Cookbook (English)

Practical, copy-paste-ready recipes.

## Recipe 1 — Inference on any HuggingFace model

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

model = qaithon.compile(model, optimize_for="energy")
print(model.qaithon_report.pretty(explain=True))

outputs = model.generate(**tok("Quantum computing", return_tensors="pt"),
                         max_new_tokens=30)
print(tok.decode(outputs[0]))
```

## Recipe 2 — Choose an objective

```python
qaithon.compile(model, optimize_for="speed")     # latency-first
qaithon.compile(model, optimize_for="energy")    # joules-first
qaithon.compile(model, optimize_for="balanced")  # default
```

## Recipe 3 — Restrict to specific backends

```python
qaithon.compile(model, backends=("quandela.perceval", "mock"))
```

## Recipe 4 — Toy transformer end-to-end

```python
import qaithon
from qaithon.lab import load_dataset, train, generate

model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
qaithon.compile(model, backends=("quandela.sim",))

ds = load_dataset("shakespeare", block_size=64)
result = train(model, ds, steps=2000, batch_size=32,
               noise_std=0.05,  # enables QAT
               log_every=200)

print(generate(model, "ROMEO:", max_new_tokens=100))
```

## Recipe 5 — Fail fast when a model does not fit a QPU

```python
import qaithon

model = qaithon.models.create_toy_transformer(dim=256, n_layers=2)
qaithon.lab.train(
    model, ds, steps=1000,
    target_hardware="IBM Heron",   # validates before step 1
)
# Raises IncompatibleHardwareError with concrete suggestions if too big.
```

## Recipe 6 — Estimate qubits without instantiating the model

```python
import qaithon

# Llama-3 70B equivalent, in milliseconds, zero RAM.
report = qaithon.estimate_qubits_from_config(
    hidden_size=8192, n_layers=80, n_heads=64,
    intermediate_size=28672,
    model_class="Llama-3 70B equivalent",
)
print(report.pretty())
```

## Recipe 7 — Compare backends with metrics + USD cost

```python
import qaithon

result = qaithon.benchmarks.compare_backends(in_features=32, out_features=32)
print(result.pretty(explain=True))

cost = qaithon.pricing.estimate_cost_usd("aws.braket.quera", n_shots=1000)
print(f"USD cost: ${cost:.2f}")
```

## Recipe 8 — Health-check a cloud backend before running

```python
import qaithon

heron = qaithon.backends.get_backend("ibm.heron")
status = heron.health_check()
print(f"online={status.online} pending={status.pending_jobs}")
if status.online:
    # safe to run
    ...
```

## Recipe 9 — Fine-tune with HuggingFace Trainer

```python
import qaithon
from qaithon.training import QATConfig, prepare_for_qat
from transformers import Trainer, TrainingArguments

model = ...   # your model
qaithon.compile(model, backends=("quandela.sim",))
prepare_for_qat(model, QATConfig(noise_std_training=0.05))

trainer = Trainer(model=model, args=TrainingArguments(...), train_dataset=...)
trainer.train()
```

## Recipe 10 — Inference metrics across an entire generate() call

```python
from qaithon.metrics import InferenceMetrics

with InferenceMetrics() as m:
    outputs = model.generate(**inputs, max_new_tokens=100)

print(m.pretty())
# Total backend calls: 4800
# Total latency: 1,234,567 µs
# Total energy: 7,890 pJ
# Photons detected: 12,345
```
