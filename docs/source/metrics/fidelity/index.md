# Fidelity

Fidelity metrics compare generated data with unseen real observations:

- [Shape](shape.md) compares individual column distributions.
- [Trend](trend.md) compares relationships between column pairs.
- [Conditional Fidelity](conditional.md) compares distributions, relationships,
  and frequencies within real-defined subpopulations.

Use them together when good marginal scores could hide lost associations or
subpopulation differences. The [matched real-real reference](../../methodology.md)
puts each score in context. For training-record overlap, see
[memorization](../memorization/index.md).

[Prepare your data](../../reference/data.md), [choose metrics](../../reference/evaluation.md),
and request `details=True` to inspect columns, pairs, or slices. Shared
uncertainty controls belong to [uncertainty and execution](../../reference/configuration.md);
retention behavior is documented under [component tables](../../reference/reports.md#component-tables).

## Shared example

The three guides use the same customer example as the
[visual walkthrough](../../getting-started/visualization.md): 600 real holdout
rows and 300 generated rows, with numeric `visits`, `spend`, and `tenure` and
categorical `segment`. The toy generator shifts `tenure`, permutes `spend`,
and repeats one record. Run this setup once to create `data` for their snippets.

::::{dropdown} Create the example data

```{literalinclude} ../../../../examples/visualization_walkthrough.py
:language: python
:lines: 8-11,14-39
```

```python
holdout, generated = make_data()
data = EvalData(
    test=holdout,
    generated=generated,
    schema=Schema(numeric=["visits", "spend", "tenure"], categorical=["segment"]),
)
```

::::

```{toctree}
:maxdepth: 1

shape
trend
conditional
```
