# Fine-tuning a Qaithon-compiled model

You can fine-tune any Qaithon-compiled model exactly like a normal PyTorch
model. Gradients flow through `QuantumLinear`, the optimizer updates the
weights, and HuggingFace `Trainer`, `peft`, `accelerate`, and custom
training loops all work without modification.

## Plain fine-tune

```python
import qaithon
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
qaithon.compile(model)

trainer = Trainer(model=model, ...)   # the rest is standard HuggingFace
trainer.train()
```

That's it. No special configuration is needed.

## Fine-tune with LoRA (peft)

```python
from peft import LoraConfig, get_peft_model

qaithon.compile(model)
model = get_peft_model(model, LoraConfig(...))
trainer = Trainer(model=model, ...)
trainer.train()
```

Qaithon and peft compose: peft wraps Qaithon's `QuantumLinear` exactly like
it would wrap a regular `nn.Linear`.

## Quantization-Aware Training (QAT)

When you intend to deploy on real photonic / quantum hardware, the
hardware will introduce noise that simulators don't. **QAT** fixes this:
during training, inject the same noise the target hardware exhibits, so
the trained model becomes robust to it.

```python
import qaithon
from qaithon.training import QATConfig, prepare_for_qat
from transformers import AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
qaithon.compile(model, backends=("quandela.sim",))

prepare_for_qat(
    model,
    QATConfig(
        target_backend="quandela.sim",
        noise_std_training=0.05,   # match measured hardware noise
        noise_std_eval=0.0,        # clean evaluation
    ),
)

trainer = Trainer(model=model, ...)
trainer.train()                    # auto-switches to noisy backend
metrics = trainer.evaluate()       # auto-switches to clean backend
```

`prepare_for_qat` patches the model's `train()` / `eval()` so the noise
level toggles automatically. You don't have to call anything special at
inference time.

## What is NOT supported (yet)

* **Training directly against a real cloud QPU.** Hardware QPUs do not
  expose autograd; parameter-shift rule is the standard alternative but
  is prohibitively slow for fine-tune at LLM scale. QAT (above) is the
  recommended substitute.
* **Distributed training across multiple Qaithon backends.** v0.x.
* **Quantization-Aware Training with anything other than the noise
  distribution declared by the target backend.** v0.x will let you plug
  in a custom noise model.

## Reading the loss curve

If QAT is working, you should see:

* Training loss may be slightly higher than a clean baseline (the noise
  is adversarial).
* Evaluation loss in clean mode should be lower than training loss.
* Evaluation loss in noisy mode (simulating hardware) should be close to
  training loss.

If evaluation in noisy mode degrades sharply, increase
`noise_std_training` to better match the hardware's true noise level.
