# Working with HuggingFace models

Qaithon connects to the HuggingFace ecosystem and **recognizes** most transformer
families automatically — it reads the architecture and plans which `nn.Linear` /
Conv1D layers *could* run on a quantum/photonic backend. But recognition and
analysis are not the same as genuine execution: **only tiny transformers actually
run genuinely** on today's simulators (verified end-to-end on TinyStories-1M).
Larger models still load and compile, but their oversized layers fall back to the
classical path — see *Why only tiny models run* below.

## Architectures Qaithon recognizes (for analysis & planning)

Recognition lets Qaithon estimate a model's qubit/mode budget and plan a compile.
It does **not** mean the model runs genuinely on a QPU at its real size.

| Family | Recognized | Notes |
|---|---:|---|
| GPT-2 / GPT-Neo | yes | Conv1D layout handled automatically. |
| Llama / Mistral / Qwen / Gemma / Phi | yes | Standard `nn.Linear` Q/K/V/O + MLP. |
| Mixtral / MoE | yes | Expert weights detected by the MoE handler. |
| BERT / RoBERTa | yes | Encoder-only; CLS / pooler skipped. |

Unlisted families fall back to a conservative `generic` profile; the
CompileReport shows `family=generic`.

## What actually runs genuinely (tiny only)

```python
import qaithon
from transformers import AutoModelForCausalLM, AutoTokenizer

# A TINY model runs genuinely on a quantum simulator (laptop-sized).
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # genuine quantum
out = model.generate(**tok("Once upon a time", return_tensors="pt"), max_new_tokens=30)
print(tok.decode(out[0]))
```

## Analyzing a large model (without running it)

You can still point Qaithon at a large model to **see why it doesn't fit** — this
reads the config only; it does **not** execute the model on a QPU:

```bash
# Analysis only — shows the qubit budget, far beyond any real device today.
qaithon estimate mistralai/Mixtral-8x7B-v0.1
```

## Why only tiny models run

Simulating a quantum computer is **exponential**: each extra qubit doubles the
memory. ~30 qubits already fills a 16 GB laptop; a 45–50 qubit emulator needs a
supercomputer. So a normal machine can only genuinely run *small* circuits — tiny
transformers like TinyStories-1M. A GPT-2-scale layer already needs more than a
laptop; real LLMs (Llama, Mistral, …) are out of reach today. On real quantum
hardware, only ~4–5 qubits are usable before noise dominates. The hardware
improves every year, and Qaithon is built to grow with it.

## What if a model breaks?

If `qaithon.compile(model)` raises `IncompatibleModelError`, the message tells you
what to do. For other failures, file an issue with the model id, the output of
`qaithon.compile(model, return_report=True)`, and the traceback. MIT-licensed —
new family profiles welcome (`qaithon/handlers/architecture.py`).
