# Prepare evaluation data

Pass your tables and column declarations to `EvalData` before calling `evaluate`.
The tables being compared must be pandas DataFrames with the same complete set
of uniquely named columns. Their row counts and column order may differ.

Load files directly with pandas, choosing parsing options for your data:

```python
import pandas as pd

holdout = pd.read_csv("test.csv", dtype={"region": "string"})
generated = pd.read_parquet("generated.parquet")
```

The Python API accepts prepared DataFrames and does not provide a file loader.
`EvalData` does not recast columns using the schema.

## Describe your columns

Use `Schema` to declare numeric measurements, categorical labels, and an optional
prediction target:

```python
from tabgen_eval import EvalData, Schema

schema = Schema(
    numeric=["age", "income"],
    categorical=["region", "outcome"],
    target="outcome",
)
```

Numeric and categorical lists must be disjoint, and declared columns must exist
in the tables. A target may also have a numeric or categorical role; MLE excludes
it from its predictors. Columns omitted from both feature lists do not
participate in feature metrics.

Alternatively, start with `schema = Schema.infer(holdout, target="outcome")`.
Inspect and correct the proposed roles before using them; evaluation does not
infer roles for you.

MLE requires `schema.target`. With no metric filters, evaluation with test data
includes MLE and therefore requires a target. You can omit the target when
selecting metrics that do not use it, such as Shape and Trend.

## Assign data roles

Use `train` for the records used to fit the generator, `test` for an independent
real holdout unseen by the generator, and `generated` for the generator's output:

```python
data = EvalData(test=holdout, generated=generated, schema=schema)
```

These are the valid combinations; other combinations raise:

| Supplied tables | Available evaluation |
| --- | --- |
| `test` | Real-real reference comparisons; see [sampling rules](configuration.md#choose-uncertainty) |
| `test`, `generated` | Non-memorization metrics |
| `train`, `generated` | Near-match rate and replication ratio |
| `train`, `test`, `generated` | All metric families, including DCR ratio |

Creating `EvalData` prints the default metric selection for its roles. Follow
[Run evaluation](evaluation.md) to select and compute metrics.

Prepare values before evaluating: the package validates inputs without cleaning
or converting them. See [preparation rules](../sharp-bits.md#preparation-changes-what-is-being-compared)
and the [metric guides](../metrics/index.md) for missing-value handling.

## Missing Value Protocol

Each metric has its own protocol for handling missing values. We recommend
handling missing values explicitly before evaluation rather than relying silently
on the library's metric-specific behavior. Choose and apply a consistent policy
across your input tables so that the behavior matches your expectations.

If you leave missing values in the inputs, the built-in protocols still apply:
fidelity metrics use observed values within columns, pairs, or slices; distance
metrics exclude incomplete records; and model-based metrics use their documented
predictor and target rules. These differences can change which observations
contribute to each score. Consult the [metric guides](../metrics/index.md)
for the exact protocols, use [data diagnostics](../data-diagnostics.md) to inspect
missingness, and check `report.row_usage()` for distance and MLE exclusions.
