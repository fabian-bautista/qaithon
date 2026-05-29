# Composing pipelines

Sometimes you want to chain several models or transforms together — say,
a compiled embedder feeds a compiled language model. Use `Pipeline`.

```python
import qaithon
from qaithon.pipeline import Pipeline

embedder = ...   # any nn.Module — typically compiled
generator = ...  # any nn.Module — typically compiled
qaithon.compile(embedder)
qaithon.compile(generator)

pipe = Pipeline([embedder, generator])
output = pipe(input_tensor)
```

`Pipeline` is intentionally small. It threads outputs left-to-right;
stages can be `nn.Module` instances or plain callables. If any stage
raises, the exception is re-raised with the offending stage index for
easy debugging.

## When to use Pipeline vs direct chaining

* Use **Pipeline** when stages are independent compiled units (each has
  its own `qaithon_report`, its own backend choices, possibly its own
  fallback strategy).
* Use **direct Python composition** (`generator(embedder(x))`) for
  simple two-step inference where the audit trail of each stage isn't
  important.

## Adding observability

Wrap each stage with tracing to see how time is spent across the
pipeline:

```python
from qaithon.tracing import trace, traced

# Add traced backends to each stage's QuantumLinear layers.
for stage in [embedder, generator]:
    for module in stage.modules():
        if hasattr(module, "backend"):
            module.backend = traced(module.backend)

with trace() as t:
    output = pipe(input_tensor)

print(t.to_json())
```

## What's missing in v0.1

* Dynamic per-token routing (pick backend per token at runtime).
* Parallel branches (run two backends side-by-side, pick the cheaper
  result).
* Streaming aggregation (Pipeline + StreamingPipeline interplay).

These land in later releases.
