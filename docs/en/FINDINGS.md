# What Qaithon actually does — research findings

This document reports, honestly and reproducibly, what we built and measured:
**running pieces of real neural networks on genuine quantum and photonic
algorithms**, and exactly how far today's technology reaches. Everything here
is reproducible on a laptop — the scripts are in `scripts/`.

> **Honesty contract.** "Compute" means the matmul is evaluated by a *real*
> quantum/photonic algorithm (Perceval/MerLin for photonics, Qiskit/PennyLane
> for qubits) — never classical math wearing a quantum label. All results below
> ran on **simulators**, not on a physical QPU. The same code targets real
> hardware when it becomes accessible.

---

## In plain language (no jargon)

We asked one question: *can this library run real AI computation using quantum
(qubits) and photonic (light) algorithms — for real, not faked?*

The answer is **yes, at a tiny scale, and we proved it:**

- A **real, pretrained language model** (TinyStories-1M) wrote a **coherent
  story** — *"...a little girl named Lily. She loved to play outside in the
  sunshine..."* — with its math computed through **genuine quantum circuits**
  (simulated). The output was **identical** to the normal classical result.
- Tiny neural-network layers **train and run** on both photonic and quantum
  algorithms.
- We **measured the exact limits** of each technology on a normal laptop.

It is small — but it is **real**, **honest**, and **anyone can reproduce it**.

---

## Headline results (the numbers)

| What | Result |
|---|---|
| Genuine photonic matmul vs classical | error **~1e-7** (exact) |
| Genuine quantum matmul vs classical | error **~1e-7** (exact) |
| **TinyStories-1M, genuine quantum inference** | **coherent text**, 48 genuine layers, fidelity vs classical **1.08e-6**, argmax **100%** identical |
| Tiny photonic layer (trainable) | learns a toy task (~0.83 acc) |
| Tiny quantum layer (trainable) | learns a toy task (~0.95 acc) |

How the genuine matmul works: any weight matrix `W` is embedded in a *unitary*
(Halmos dilation), realised as a real linear-optical circuit (photonic) or a
qubit circuit (quantum), the input is amplitude-encoded, and the output
amplitudes are read back. The classical comparison is only used to *verify* the
circuit did the math.

---

## Understanding the limits (this is the most useful part)

Why does it only run *tiny* models? The reason is different — and **opposite** —
for each technology.

### Photonic — "one locker per number"

Photonics stores each number in a **lane of light (a mode)**: roughly **one lane
per number**. The simulator (Perceval) handles up to **256 lanes → ~128 numbers
wide** per layer.

Why that ceiling? A simulator exists to **imitate a real chip**, and real
photonic chips are *tiny*: Quandela Belenos has **12 modes**; research chips,
a few dozen. So 256 is already ~20× larger than anything that physically exists.
Going past it would either (a) be a plain classical matmul (single-photon optics
is mathematically just matrix×vector — no quantum value), or (b) hit the
combinatorial explosion of multi-photon simulation. The cap sits right at the
edge of what is meaningful.

### Quantum — "each qubit doubles the room" (and it's the reverse)

Qubits store **2ⁿ numbers with n qubits** — exponential. 9 qubits hold 512
numbers; 11 qubits hold 2048. So a layer of width 768 (GPT-2) needs only **11
qubits**.

Here the situation is **opposite to photonics**: the *real hardware* is huge
compared to what any classical computer can simulate.

| | Real hardware | Classical simulator |
|---|---|---|
| Photonic | ~12 modes | 256 modes → simulator **over-provisions** |
| Quantum | 156+ physical qubits (IBM); 48–94 logical (Quantinuum) | ~30 qubits (laptop) → simulator **falls short** |

A statevector simulator stores 2ⁿ numbers, and **each qubit doubles the memory**:
30 qubits ≈ 17 GB (fills a 16 GB laptop), 50 qubits is a world-record
supercomputer, and 156 qubits would need more numbers than there are atoms in
the universe. **You cannot classically simulate a large quantum computer — and
that impossibility is exactly why quantum computers are valuable.**

### Measured frontier (MacBook Air M2, 16 GB)

| Technology | Genuine + fast | Usable | Hard ceiling |
|---|---|---|---|
| Photonic (Perceval) | dim ≤ 128 (<0.1s) | — | **dim 128** (256-mode sim cap) |
| Quantum (Qiskit statevector) | dim ≤ 256 (<0.05s) | dim ≤ 1024 (~2s) | dim ~4096 (~3 min — cost of the dense unitary) |

**One line:** photonics runs out of *lanes* (linear, ~128); quantum is so
powerful it can't be *simulated* large (exponential) — but with few qubits it
packs much more, so it reaches wider layers. That's why TinyStories (an internal
layer of width 256) ran on quantum but not on photonics.

---

## What we corrected along the way (full honesty)

Building this surfaced real issues; we fixed them and recorded them:

- **Inference vs training are two genuine paths.** The dilation kernel runs
  genuine **inference** of arbitrary/pretrained weights but is *not
  differentiable*; **training** uses the differentiable MerLin/PennyLane layers.
- **Photonics cannot run a full transformer genuinely** — its internal
  projections (e.g. attention QKV = 3× width, MLP = 4× width) exceed the
  128-wide photonic ceiling. The genuine photonic path is the *trainable
  PhotonicLayer*, not compiling transformers.
- A device bug (CPU↔Metal tensors) was fixed so models on Apple Silicon work.

---

## Reproduce it yourself

Everything is packaged. On Python 3.10+:

```bash
pip install -e ".[huggingface,pennylane,quandela,deepquantum]"

python scripts/run_all_experiments.py   # regenerates the full results table
```

Or directly:

```python
import qaithon
from transformers import AutoModelForCausalLM

# Genuine quantum inference of a real pretrained model (TinyStories-1M)
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model = qaithon.compile(model, backends=("pennylane.sim",))  # quantum, genuine
# ...generate as usual; the linear layers now run on a real qubit circuit.
```

No quantum hardware account needed — the genuine algorithms run on local
simulators.

---

## Why this matters, and what comes next

Today, Qaithon is two things at once:

1. **A genuine instrument** to run, train and measure tiny neural-network layers
   on real quantum and photonic algorithms — and to understand, with measured
   numbers, exactly where the technology stands.
2. **Future-ready plumbing.** The same `qaithon.compile(model)` call that targets
   a simulator today targets a real QPU tomorrow — only a backend flag changes.

And the future is not hypothetical. Error-corrected hardware with **logical
qubits already exists** (Quantinuum's Helios reports up to 48–94 logical qubits);
the barrier is **access and cost**, not physics — and both are improving. The
day that hardware opens up, the work here pays off directly: the kernels, the
size guards, the per-technology routing are already built. Qaithon becomes far
more than a simulator wrapper.

**The invitation:** clone it, run it, push the limits, and be ready. This is a
small, honest first step on a road that the hardware is actively building toward.
