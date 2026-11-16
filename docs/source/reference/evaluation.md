# Run an evaluation

`evaluate` computes the metrics you select and returns a `Report`. First,
[prepare your data](data.md) as an `EvalData` object.

## Choose metrics

Start with the measurements you need:

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["shape", "trend"])
print(report.summary())
```

Use `group` to select a registered metric group:

```python
report = evaluate(data, group=["fidelity"])  # Shape and Trend
```

A nonempty `metrics` list overrides `group`. Names in both lists must be
registered, and the selected metrics must support your data roles.

With both filters empty, `evaluate(data)` computes every metric supported by
the supplied roles. For test-based evaluation, this includes MLE and nonlinear
C2ST, so the default requires `Schema.target` for MLE. XGBoost is included in the
[base installation](../getting-started/installation.md). Select a subset to run
only the metrics you need.

## Registered names

See the [pillar-to-group mapping](../metrics/index.md#pillars-and-selector-groups)
for registered groups, metric names and required data pools.

Discover names and data requirements programmatically:

```python
from tabgen_eval import list_groups, list_metrics

print(list_groups())
print(list_metrics())
```

An explicit group selects all its members. For example, the complete
`memorization` group requires train, test, and generated data. See the
[metric guides](../metrics/index.md) to choose measurements and interpret them.

## Use test data alone

To evaluate a real-data reference without generated data:

```python
from tabgen_eval import EvalData

reference = evaluate(
    EvalData(test=holdout, schema=schema),
    metrics=["shape", "trend"],
    uncertainty={"mode": "reference", "n_replicates": 10},
)
samples = reference["shape"].reference_distribution.samples
```

This retains ten ordered real-real scores in `reference_distribution`. Its
`.mean` summarizes the finite scores; `.band_low` and `.band_high` give their
central percentile band. Both point fields are marked absent for this repeated
test-only evaluation. Omit `uncertainty` to retain one real-real score in `.value`
without a band. See [uncertainty options](configuration.md#choose-uncertainty)
and [Methodology](../methodology.md) for the reference and band interpretation.
Test-only MLE follows the same reporting convention for real-trained scores; see its
[training and scoring protocol](../metrics/utility/mle.md#test-only-evaluation).

## Read your results

Access a metric through its registered name:

```python
shape = report["shape"]
print(shape.value)
```

For a worked explanation of the output, [read your first report](../getting-started/reports.md).
Use the [Report API](reports.md) to look up methods, fields and export behavior.
For uncertainty, reproducibility, and hardware choices, see
[uncertainty and execution](configuration.md). For rejected inputs or undefined
results, see [troubleshooting](../troubleshooting.md).
