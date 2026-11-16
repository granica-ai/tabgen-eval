# Generated near-match rate

Near-match rate answers “What fraction of generated records lie close to the
generator's actual training data?” Its denominator is the retained generated
row count, so repeated copies each contribute. [Replication ratio](replication-ratio.md)
measures the fraction of training records reached.

The metric requires `train` and `generated` as inputs.
Numeric and categorical features are supported. This metric uses exact search
and returns a point-only result in $[0,1]$.

## Example

```python
import pandas as pd
from tabgen_eval import EvalData, Schema, evaluate

training = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
generated = pd.DataFrame({"x": [0.0, 0.0, 0.0, 1.0]})

report = evaluate(
    EvalData(train=training, generated=generated, schema=Schema(numeric=["x"])),
    metrics=["near_match_rate", "replication_ratio"],
)
print(report["near_match_rate"].value)  # 1.0
print(report["replication_ratio"].value)  # 0.5
```

All generated rows reproduce training here, but only two of four training rows
are reached. Repeating the first record increases its share of generated output
without reaching additional training records.

## Definition

For retained generated rows $G$, training rows $T$, and the fixed threshold
$\tau=10^{-3}$,

$$
\operatorname{NearMatchRate}_{\tau}(G,T)=
\frac{1}{|G|}\sum_{g\in G}
\mathbf 1\!\left\{\min_{t\in T}d_T(g,t)\leq\tau\right\}.
$$

A generated row counts as a near match when its nearest training row is at
distance $10^{-3}$ or less. Near-match rate uses the Euclidean distance as its
metric. Numeric columns are scaled using their training ranges, and categorical
columns are encoded using categories observed in training, as described in
[DCR](dcr.md#distance-space). All retained rows enter the calculation.

<div id="interpretation"></div>

The result is available as `report["near_match_rate"]` or
`report["near_match_rate/rate"]`. Higher values mean more generated rows are
near training.

<div id="row-handling-and-interpretation"></div>

## Missing values

Rows missing declared distance features are excluded independently from training
and generated data. Excluded generated rows do not enter the denominator.
See [DCR missing values](dcr.md#missing-values) and `report.row_usage()` for
source exclusions.

## Unavailable results

With no usable training or generated rows, the rate is undefined. The report
retains counts and threshold diagnostics.
