# Choosing the optimization objective

`qaithon.compile()` picks the best backend per layer based on what you
want to minimize: latency, energy, or a balanced blend. This is the only
knob most users ever need to touch.

## The three options

### `optimize_for="speed"`

Pick whichever backend has the lowest expected latency (including network
queue time). Best for interactive applications: chatbots, online inference,
demos.

```python
qaithon.compile(model, optimize_for="speed")
```

In practice this rules out remote QPUs (queue time dominates) and prefers
local simulators or, eventually, on-premise photonic chips.

### `optimize_for="energy"`

Pick whichever backend has the lowest expected energy per MAC. Best for
batch / offline workloads where wall-clock time is less critical than
electricity bills or sustainability claims.

```python
qaithon.compile(model, optimize_for="energy")
```

In practice this prefers photonic / quantum backends over GPU baselines.

### `optimize_for="balanced"` (default)

Blend latency and energy with a normalized score. Sensible default when
you have no strong preference; produces broadly reasonable choices for
mixed workloads.

```python
qaithon.compile(model)   # implicitly balanced
```

## Reading the result

Whichever objective you picked, `model.qaithon_report` shows the per-layer
decision and the rationale:

```python
report = model.qaithon_report
print(report.pretty())
# >  Backends used:        quandela.sim
# >  Estimated energy:     ... pJ (baseline ... pJ, save 95.0%)
```

The `estimated_energy_savings_pct` field is the headline number for "how
much did Qaithon save vs running this entirely classically".

## When to deviate from the default

Most users should not. Two cases where switching pays off:

1. **Production deployment with hard latency SLAs** — pick `speed`.
2. **Sustainability reporting / power-constrained edge** — pick `energy`.

If your needs change between runs (one phase favors speed, another energy)
you can call `qaithon.compile` again on the same model; later compiles
are no-ops because `QuantumLinear` is not seen by the walker.
