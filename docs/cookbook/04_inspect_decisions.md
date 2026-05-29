# Inspecting Qaithon's decisions

Even though `qaithon.compile()` automates every decision, every choice is
recorded. If you want to know why a particular backend was picked, or
what got skipped, the answer is one attribute away.

## The compile report

After `qaithon.compile(model)`:

```python
report = model.qaithon_report
```

`CompileReport` has these attributes:

| Field | Type | Meaning |
|---|---|---|
| `model_class` | str | Top-level model class name. |
| `n_parameters` | int | Total parameter count. |
| `n_replaced` | int | Layers that became `QuantumLinear`. |
| `n_skipped` | int | Layers considered but not replaced. |
| `optimize_for` | str | The objective the user requested. |
| `backends_used` | tuple[str, ...] | Distinct backends chosen. |
| `decisions` | tuple[`LayerDecision`, ...] | One entry per replaced layer. |
| `skipped` | tuple[(name, reason), ...] | Why each skipped layer was excluded. |
| `baseline_energy_pj` | float | Energy if all layers ran on the classical baseline. |
| `compiled_energy_pj` | float | Energy with Qaithon's choices. |
| `estimated_energy_savings_pct` | float | Headline number. |

## Quick summary

```python
print(report.pretty())
```

Returns a multi-line human-readable summary.

## Per-layer detail

```python
for decision in report.decisions[:5]:
    print(f"{decision.layer_name} → {decision.backend} ({decision.reason})")
```

## Why was layer X skipped?

```python
for name, reason in report.skipped:
    print(f"{name}: {reason}")
```

Common reasons:

* `weight is tied to another parameter` — embedding shared with lm_head.
* `already-quantized layer` — bitsandbytes / AWQ / GPTQ layer detected.
* `in_features=N < min_in_features=M` — too small to be worth offloading.
* `excluded by user skip predicate` — your custom rule.

## Runtime tracing

If you want to know what *actually* ran (not just what was planned):

```python
from qaithon.tracing import trace, traced

# Re-wrap the model's first layer's backend for tracing.
model.fc1.backend = traced(model.fc1.backend)

with trace() as t:
    output = model(some_input)

print(t.to_json())   # JSON-serializable trace ready to archive
print(f"Total latency: {t.total_latency_us:.0f} µs")
print(f"Total energy:  {t.total_energy_pj:.0f} pJ")
```

This is opt-in (no overhead when disabled) and useful for benchmarking
real vs estimated cost.
