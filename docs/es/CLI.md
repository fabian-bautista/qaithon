# Interfaz de Línea de Comandos (Español)

Qaithon trae un CLI `qaithon` instalado junto con el paquete. Todos los
comandos aceptan `--help`.

```bash
qaithon --help
qaithon <comando> --help
```

Flag global:

| Flag | Efecto |
|---|---|
| `-v` / `--verbose` | Logging nivel INFO del paquete `qaithon`. |

## Comandos

### `qaithon list-backends`

Muestra cada backend registrado y si está disponible en esta máquina
(deps instaladas + credenciales presentes).

```bash
qaithon list-backends
```

Output: `name`, `kind` (`classical` / `quantum` / `photonic`),
`available`, notas breves.

### `qaithon inspect <model_id>`

Carga un modelo de HuggingFace e imprime el `ReplacementPlan` — qué
capas Qaithon reemplazaría, cuáles saltaría, cuáles están atadas.

```bash
qaithon inspect gpt2 --json
# Para un modelo grande, usá `estimate` (lee la config — sin descarga de varios GB):
qaithon estimate mistralai/Mixtral-8x7B-v0.1   # análisis: muestra que no cabe hoy
```

| Flag | Efecto |
|---|---|
| `--json` | Output JSON en vez de texto. |
| `--allow-quantized` | No falla si el modelo ya está en 4-bit / 8-bit. |

### `qaithon compile <model_id>`

Compila un modelo de HuggingFace y (opcionalmente) guarda el resultado.

```bash
qaithon compile gpt2 --optimize-for energy
qaithon compile gpt2 --backend ibm.aer --out ./compiled-gpt2
qaithon compile gpt2 -b quandela.perceval -b mock
```

| Flag | Efecto |
|---|---|
| `-b` / `--backend` | Restringe a este backend (repetible). |
| `--optimize-for` | `balanced` (default) / `speed` / `energy`. |
| `--allow-quantized` | Permite modelos ya cuantizados. |
| `--out` | Guarda modelo compilado + reporte bajo este directorio. |

### `qaithon doctor`

Diagnostica el entorno local: versión de Python, PyTorch, CUDA, cada dep
opcional (`qiskit`, `qiskit-aer`, `perceval`, `merlin`, `braket`,
`pennylane`, `deepquantum`), credenciales presentes por proveedor,
backends registrados.

```bash
qaithon doctor
```

### `qaithon estimate <model_id>`

Estima el presupuesto de qubits necesario para correr un modelo en una
QPU real. Incluye compatibilidad con cada hardware conocido ("cabe en
IBM Heron / no cabe en Belenos porque …").

```bash
qaithon estimate gpt2
qaithon estimate gpt2 --json
```

### `qaithon glossary [término]`

Busca un término de computación cuántica en lenguaje de dev de IA.

```bash
qaithon glossary                    # lista todos los términos
qaithon glossary fidelity           # entry completa
qaithon glossary fidelity --value 0.987  # contextualizada
```

| Flag | Efecto |
|---|---|
| `--value` | Valor numérico opcional — la explicación se contextualiza. |

### `qaithon benchmark`

Corre el mismo matmul en cada backend disponible y reporta
latencia / energía / métricas. Por defecto excluye las QPU cloud
facturables.

```bash
qaithon benchmark --in-features 32 --out-features 32 --explain
qaithon benchmark --exclude aws.braket.sv1 quandela.belenos
```

| Flag | Efecto |
|---|---|
| `--in-features` | Dim de entrada. Default 16. |
| `--out-features` | Dim de salida. Default 16. |
| `--exclude` | Nombres de backends a saltar. |
| `--explain` | Explicaciones del glosario en línea. |

### `qaithon trace inspect <trace.json>`

Imprime de manera legible un trace JSON producido por
`qaithon.tracing` y muestra los top N eventos por latencia.

```bash
qaithon trace inspect trace.json
```

### `qaithon plugins list`

Lista backends de plugins de terceros descubiertos vía entry points de
Python (grupo `qaithon.backends`).

```bash
qaithon plugins list
```

---

## Sesiones típicas

### Chequeo inicial

```bash
qaithon doctor
qaithon list-backends
qaithon glossary photonic\ mode
```

### Planear antes de comprometerse

```bash
qaithon inspect gpt2
qaithon estimate gpt2
qaithon benchmark --in-features 768 --out-features 768
```

### Compile reproducible

```bash
qaithon compile gpt2 \
    --optimize-for energy \
    --backend ibm.aer --backend mock \
    --out ./artifacts/gpt2-qaithon
```
