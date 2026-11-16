# DCR ratio and nearest-training distances

Distance to Closest Record (DCR) measures how close records lie to the generator's
actual training data. The primary result, `dcr_ratio`, compares the median nearest
distance for generated records with that for independent real test records.

Supply `train`, `generated`, and an independent `test` pool. Test data provide
the ratio's denominator; both query pools use the same training-fitted geometry.
Numeric and categorical features are supported, and no prediction target is
needed. This evaluation is exact and point-only.

## Example

```python
import pandas as pd
from tabgen_eval import EvalData, Schema, evaluate

training = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
generated = pd.DataFrame({"x": [0.0, 0.5, 1.0, 1.5]})
holdout = pd.DataFrame({"x": [0.5, 1.5, 2.5]})

report = evaluate(
    EvalData(train=training, generated=generated, test=holdout, schema=Schema(numeric=["x"])),
    metrics=["dcr_ratio"],
)
print(f"Ratio:  {report['dcr_ratio'].value:.4f}")
print(f"Median: {report['dcr_ratio/median'].value:.4f}")
```

```text
Ratio:  0.5000
Median: 0.0833
```

Distances are scaled by the training range, $3$. The generated median distance
is $0.25/3=0.0833$, while the test median is $0.5/3=0.1667$. Their ratio is
$0.5$: generated records are half as far from training at the median.

## Definition and outputs

Let $T$, $G$, and $H$ be the retained training, generated, and independent test
rows. For nearest-training distance $\delta_T$, the ratio is

$$
\delta_T(x)=\min_{t\in T}d_T(x,t),\qquad
\operatorname{DCR}_{\mathrm{ratio}}=
\frac{\operatorname{median}_{g\in G}\delta_T(g)}
     {\operatorname{median}_{h\in H}\delta_T(h)}.
$$

At the median, a ratio below $1$ means generated records are closer to training,
near $1$ means similar proximity, and above $1$ means they are farther away.

Selecting `dcr_ratio` includes the supporting distance summaries:

| Result key | Meaning |
| --- | --- |
| `dcr_ratio` or `dcr_ratio/ratio` | Generated median divided by test median |
| `dcr_ratio/median` | Generated median; an alias for `dcr_ratio/q50` |
| `dcr_ratio/qXX` | Generated nearest-training distance percentiles |
| `dcr_ratio/holdout_qXX` | Corresponding test nearest-training distance percentiles |

Both percentile sets contain q05, q10, q25, q50, q75, q90, q95, and q99, using
linear interpolation. Raw distances and defined ratios are nonnegative.
Identical records across pools have distance zero; matching observations are
included in nearest-neighbor searches.

### Distance space

For numeric column $j$, fit its minimum $a_j$ and range $s_j$ on retained
training rows, replacing a zero range with $1$:

$$
z_j(x)=\frac{x_j-a_j}{s_j}.
$$

Generated and test values use the same transform and can lie outside $[0,1]$.
For categorical column $k$, let $e_{T,k}(x_k)$ be its full one-hot vector over
categories observed in training. An unseen query category has an all-zero
vector. The Euclidean distance is

$$
d_T(x,y)^2=
\sum_j\left(z_j(x)-z_j(y)\right)^2+
\sum_k\left\|e_{T,k}(x_k)-e_{T,k}(y_k)\right\|_2^2.
$$

Different known categories contribute $2$ to squared distance; an unseen query
category compared with a training category contributes $1$. Squared distance
sums contributions across all declared columns. Vocabulary matching distinguishes
labels such as `1` and `"1"` and preserves text exactly; see
[input preparation](../../reference/data.md).

(row-selection-and-unavailable-values)=
## Missing values

Records missing any declared distance feature are excluded independently from
train, test, and generated. Ignored columns do not exclude records.
`report.row_usage()` shows retained and excluded counts and excluded percentages
for each pool; see [source eligibility](../../reference/reports.md#source-eligibility).

## Unavailable results

An exhausted training or generated pool leaves generated distance summaries
undefined. An exhausted test pool leaves test summaries and the ratio undefined,
while valid generated summaries remain available. A zero test median makes only
the ratio undefined (`ZERO_BASELINE`); no usable test rows give
`INSUFFICIENT_HOLDOUT_ROWS`. Omitting test data raises before computation.

## Interpretation

Inspect the percentiles as well as the median: the closest records can behave
differently from the typical record.
Dense or discrete populations naturally produce close neighbors, so proximity
alone does not establish memorization.

For the fraction of generated records near training, use
[near-match rate](near-match-rate.md). For the fraction of training records
reached by generated data, use [replication ratio](replication-ratio.md).

## Execution

All retained rows are used, with no volume matching or automatic row caps.
Subsetting inputs changes the evaluated population. Subsetting training also
changes the available neighbors and fitted distance scales.

Nearest-neighbor search is exact on every supported backend. Work can grow
with training rows times query rows times feature count. See
[troubleshooting](../../troubleshooting.md#evaluation-is-slow-or-runs-out-of-memory)
for slow evaluations or memory errors.
