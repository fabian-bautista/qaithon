# Glosario

El glosario está embebido en la propia librería. Cada término tiene una
definición, una analogía con conceptos de IA/PyTorch, un ejemplo concreto
y una regla de oro para interpretar los valores que ves en los reportes.

## CLI

```bash
qaithon glossary                  # Listar todos los términos.
qaithon glossary fidelity         # Definición + analogía + ejemplo.
qaithon glossary fidelity --value 0.987   # Contextualizado para tu número.
```

## Python

```python
import qaithon

# Explicación pretty-printed
print(qaithon.explain("fidelity", value=0.987))

# Lookup programático
entry = qaithon.glossary("photonic mode")
print(entry.short)
print(entry.analogy)
print(entry.example)
print(entry.rule_of_thumb)

# Iterar todos los términos
for term in qaithon.list_terms():
    print(term)
```

## Términos cubiertos

- **qubit** — unidad fundamental de información cuántica en QPUs
  superconductoras / de iones.
- **photonic mode** — la contraparte fotónica del qubit.
- **fidelity** — similitud del output vs ideal; `1.0` es perfecto.
- **pJ per MAC** — energía por multiply-accumulate.
- **coherence depth** — gates antes de que la decoherencia domine.
- **shot** — una ejecución + medición de un circuito.
- **noise model** — descripción estadística de los errores de un QPU.
- **queue time** — espera antes de que un QPU cloud empiece a ejecutar.
- **amplitude encoding** — empaquetar N valores en log₂(N) qubits.
- **block encoding** — embeber matrices no unitarias vía ancilla qubits.
- **QAT** — Quantization-Aware Training con inyección de ruido cuántico.
- **variational quantum circuit** — circuito parametrizado entrenado
  clásicamente.
- **parameter-shift rule** — gradientes exactos de circuitos cuánticos.
