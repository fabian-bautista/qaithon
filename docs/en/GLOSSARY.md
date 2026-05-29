# Glossary

The glossary is embedded in the library itself. Every term has a
definition, an analogy to AI/PyTorch concepts, a concrete example, and a
rule of thumb to interpret values you see in reports.

## CLI

```bash
qaithon glossary                  # List every term.
qaithon glossary fidelity         # Definition + analogy + example.
qaithon glossary fidelity --value 0.987   # Contextualized for your number.
```

## Python

```python
import qaithon

# Pretty-printed explanation
print(qaithon.explain("fidelity", value=0.987))

# Programmatic lookup
entry = qaithon.glossary("photonic mode")
print(entry.short)
print(entry.analogy)
print(entry.example)
print(entry.rule_of_thumb)

# Iterate every term
for term in qaithon.list_terms():
    print(term)
```

## Terms covered

- **qubit** — fundamental quantum-information unit on superconductor/ion QPUs.
- **photonic mode** — the photonic counterpart of a qubit.
- **fidelity** — output similarity vs ideal; `1.0` is perfect.
- **pJ per MAC** — energy per multiply-accumulate.
- **coherence depth** — gates before decoherence dominates.
- **shot** — one execution + measurement of a circuit.
- **noise model** — statistical description of a QPU's errors.
- **queue time** — wait before a cloud QPU starts executing.
- **amplitude encoding** — packing N values into log₂(N) qubits.
- **block encoding** — embedding non-unitary matrices via ancilla qubits.
- **QAT** — Quantization-Aware Training with quantum noise injection.
- **variational quantum circuit** — parametrized circuit trained
  classically.
- **parameter-shift rule** — exact gradients of quantum circuits.
