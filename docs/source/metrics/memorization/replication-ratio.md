# Replication ratio

Replication ratio answers “What fraction of the generator's training records
are reached by generated records?” Its denominator is the retained **training
row count**, reversing the query used by [near-match rate](near-match-rate.md).
Each training row contributes at most once, regardless of repeated generated copies.

The metric requires the `train` data and `generated` records as inputs.
Numeric and categorical features are supported, and the result is exact and
point-only.

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

Every generated record is near training, but only two of four training rows
are reached. Duplicate training rows remain separate denominator observations.

## Definition

For retained training rows $T$ and generated rows $G$,

$$
\operatorname{ReplicationRatio}(T,G)=\frac{1}{|T|}
\sum_{t\in T}\mathbf 1\!\left\{\min_{g\in G}d_T(t,g)\leq10^{-3}\right\}.
$$

A training row counts when its nearest generated row is at distance $10^{-3}$
or less. The distance uses the same numeric scaling and categorical encoding as
[near-match rate](near-match-rate.md#definition), both determined by the training
data. All retained rows enter the calculation.

For example, with a numeric training range of $100$, a displacement of $0.1$
in that column alone gives normalized distance $10^{-3}$. Differences across
columns combine through Euclidean distance before applying the threshold.

<div id="interpretation"></div>

The result is available as `report["replication_ratio"]` or
`report["replication_ratio/rate"]`. It lies in $[0,1]$, with larger values
indicating more training rows reached.

<div id="row-handling-and-interpretation"></div>

## Missing values

Records missing declared distance features are excluded independently from
training and generated pools. Ignored columns do not exclude records.
`report.row_usage()` shows source counts and excluded percentages.
See [DCR missing values](dcr.md#missing-values) for the shared rules.

## Empty pools

If training rows remain but all generated rows are excluded, the replication
ratio is $0$, as no training row is reached. With no retained training rows, the
denominator is zero and the replication ratio is undefined.
