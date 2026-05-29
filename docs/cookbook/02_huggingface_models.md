# Working with any HuggingFace model

Qaithon supports **any** transformer distributed through the HuggingFace
ecosystem — it does not maintain a per-model whitelist. The library
detects the family automatically and applies family-specific defaults
(skip patterns, minimum layer sizes, MoE awareness) that the user never
has to think about.

## Tested architectures

| Family    | Compile? | Notes |
|-----------|---------:|-------|
| GPT-2     | yes | Conv1D layout handled automatically. |
| GPT-Neo   | yes | Same Conv1D path as GPT-2. |
| Llama 1/2/3 | yes | Standard `nn.Linear` Q/K/V/O + gate/up/down. |
| Mistral   | yes | Sliding-window attention left intact. |
| Mixtral   | yes | 3D expert weights rewritten by the Mixtral handler. |
| Phi-2 / Phi-3 | yes | Fused QKV detected as a single Linear. |
| Qwen 1.5 / 2 | yes | Aggressive GQA — Linear paths swap normally. |
| Gemma     | yes | Tied embeddings auto-detected via `accelerate`. |
| BERT / RoBERTa | yes | Encoder-only. CLS / pooler skipped. |

For families not listed, Qaithon falls back to the `generic` profile —
conservative defaults that work safely on any transformer-shaped model.
The CompileReport will indicate `family=generic` so you know.

## Example with Llama-3

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
qaithon.compile(model)
# model is now compiled. Use with .generate(), Trainer, accelerate, peft, anything.
```

## Example with Mixtral (MoE)

```python
import qaithon
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
qaithon.compile(model)
# Mixtral handler kicks in automatically:
# - Each expert's weight slice becomes a QuantumLinear.
# - The router stays classical.
# - `model.qaithon_report` shows "experts transformed: 256" (8 layers × 32 experts).
```

## What if a custom model breaks?

If `qaithon.compile(model)` raises `IncompatibleModelError`, the message
will tell you what to do (usually: load the original model without
quantization first). For other failures, file an issue with:

* Model id (`AutoConfig.from_pretrained(...).architectures`).
* The output of `qaithon.compile(model, return_report=True)`.
* The Python traceback.

The library is open under MIT — contributions of new family
profiles are welcome (`qaithon/handlers/architecture.py`).
