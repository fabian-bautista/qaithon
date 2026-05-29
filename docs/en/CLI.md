# Command-Line Interface (English)

Qaithon ships with a `qaithon` command-line tool installed alongside the
package. All commands accept `--help`.

```bash
qaithon --help
qaithon <command> --help
```

Global flag:

| Flag | Effect |
|---|---|
| `-v` / `--verbose` | INFO-level logging from the `qaithon` package. |

## Commands

### `qaithon list-backends`

Show every registered backend and whether it is available on this
machine (deps installed + credentials present).

```bash
qaithon list-backends
```

Output: `name`, `kind` (`classical` / `quantum` / `photonic`),
`available`, brief notes.

### `qaithon inspect <model_id>`

Load a HuggingFace model and print the `ReplacementPlan` — which layers
Qaithon would replace, which it would skip, which are tied.

```bash
qaithon inspect gpt2
qaithon inspect mistralai/Mixtral-8x7B-v0.1 --json
```

| Flag | Effect |
|---|---|
| `--json` | JSON output instead of text. |
| `--allow-quantized` | Don't bail if the model is already 4-bit / 8-bit. |

### `qaithon compile <model_id>`

Compile a HuggingFace model and (optionally) save the result.

```bash
qaithon compile gpt2 --optimize-for energy
qaithon compile gpt2 --backend ibm.aer --out ./compiled-gpt2
qaithon compile gpt2 -b quandela.perceval -b mock
```

| Flag | Effect |
|---|---|
| `-b` / `--backend` | Restrict to this backend (repeatable). |
| `--optimize-for` | `balanced` (default) / `speed` / `energy`. |
| `--allow-quantized` | Allow already-quantized models. |
| `--out` | Save the compiled model + report under this directory. |

### `qaithon doctor`

Diagnose the local environment: Python version, PyTorch, CUDA, every
optional dep (`qiskit`, `qiskit-aer`, `perceval`, `merlin`, `braket`,
`pennylane`, `deepquantum`), credentials present per provider, registered
backends.

```bash
qaithon doctor
```

### `qaithon estimate <model_id>`

Estimate the qubit budget required to run a model on a real QPU.
Includes per-known-hardware compatibility ("fits on IBM Heron / does not
fit on Belenos because …").

```bash
qaithon estimate gpt2
qaithon estimate gpt2 --json
```

### `qaithon glossary [term]`

Look up a quantum-computing term in plain AI-developer language.

```bash
qaithon glossary                    # list all terms
qaithon glossary fidelity           # full entry
qaithon glossary fidelity --value 0.987  # contextualized
```

| Flag | Effect |
|---|---|
| `--value` | Optional numeric value — explanation is contextualized. |

### `qaithon benchmark`

Run the same matmul on every available backend and report
latency / energy / metrics. Excludes billable cloud QPUs by default.

```bash
qaithon benchmark --in-features 32 --out-features 32 --explain
qaithon benchmark --exclude aws.braket.sv1 quandela.belenos
```

| Flag | Effect |
|---|---|
| `--in-features` | Input dim. Default 16. |
| `--out-features` | Output dim. Default 16. |
| `--exclude` | Backend names to skip. |
| `--explain` | Inline glossary explanations. |

### `qaithon trace inspect <trace.json>`

Pretty-print a JSON trace produced by `qaithon.tracing` and show the top
N events by latency.

```bash
qaithon trace inspect trace.json
```

### `qaithon plugins list`

List third-party plugin backends discovered via Python entry points
(`qaithon.backends` group).

```bash
qaithon plugins list
```

---

## Typical sessions

### First-time check

```bash
qaithon doctor
qaithon list-backends
qaithon glossary photonic\ mode
```

### Plan before committing

```bash
qaithon inspect gpt2
qaithon estimate gpt2
qaithon benchmark --in-features 768 --out-features 768
```

### Reproducible compile

```bash
qaithon compile gpt2 \
    --optimize-for energy \
    --backend ibm.aer --backend mock \
    --out ./artifacts/gpt2-qaithon
```
