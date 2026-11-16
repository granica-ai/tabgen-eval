# Trend

Trend measures how closely generated data preserves pairwise relationships
between columns. It compares numeric correlations and joint categorical
frequencies, then averages the pair scores with equal weight.

Use Trend alongside [Shape](shape.md) to assess dependencies that matching
marginal distributions can conceal. These pairwise checks can still miss
differences involving three or more columns.

To examine relationships within real-defined subpopulations, use
[Conditional Fidelity](conditional.md).

## Score

| Property | Value |
| --- | --- |
| Metric selection | `metrics=["trend"]` |
| Report key | `"trend/trend"` |
| Range | $[0,1]$ |
| Direction | Higher means closer pairwise relationships |
| Detailed output | One score per column pair |
| Required columns | At least two |

Read the generated-real score alongside its matched real-real reference;
independent real samples also differ. See [Methodology](../../methodology.md)
for the shared comparison procedure.

## Example: find a broken relationship

Use the [shared customer example](index.md#shared-example) to create `data`.
The toy generator permutes `spend`, weakening its relationship with `visits`.
Trend exposes this change in the pair scores:

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["trend"], details=True, seed=23)
score = report["trend/trend"]
print(score.value, score.ideal)
print(report.details("trend").sort_values("score"))
report.fidelity.plot(kind="pairs", metric="trend").show()
```

```{raw} html
:file: ../../_generated/visualization/pairs.html
```

The heatmap highlights the weakened `visits`–`spend` relationship, which an
overall average or matching marginal distributions can conceal.

## Definition

Let $R$ and $G$ denote the matched real and generated samples. For each unordered
column pair $(j,k)$, compute the similarity appropriate to its column types.

### Numeric–numeric pairs

Let $\widehat\rho_{R,jk}$ and $\widehat\rho_{G,jk}$ be the Pearson correlations
of the two numeric columns within each sample. Their similarity is

$$
t_{jk}=1-\frac{1}{2}
\left|\widehat\rho_{R,jk}-\widehat\rho_{G,jk}\right|.
$$

Pearson correlations lie in $[-1,1]$, so their absolute difference can reach
$2$. Equal correlations score $1$, even if other aspects of the joint
distributions differ. Correlation is undefined if either column is constant
within either sample, or fewer than two usable paired rows remain.

### Pairs containing categorical columns

Categorical columns retain their category identities. A numeric column paired
with a categorical column is first divided into up to ten quantile bins fitted
on the real sample. Tied boundaries reduce the number of bins. Generated values
outside the reference range are assigned to the nearest edge bin.

Let $\widehat p_{R,jk}(a,b)$ and $\widehat p_{G,jk}(a,b)$ be the normalized
joint frequencies on this shared category or bin space. The score is one minus
total variation distance:

$$
t_{jk}=1-\frac{1}{2}\sum_{a,b}
\left|\widehat p_{R,jk}(a,b)-\widehat p_{G,jk}(a,b)\right|.
$$

The joint table captures marginal frequencies as well as association. Categories
seen on only one side contribute to the discrepancy. Reference-fitted numeric
bins make mixed-type Trend directional: exchanging the real and generated tables
can change the score.

### Aggregate

For the set $P$ of pairs with defined scores,

$$
\operatorname{Trend}(G,R)=\frac{1}{|P|}\sum_{(j,k)\in P}t_{jk}.
$$

Each usable pair has equal weight. Undefined pairs are excluded. If no pair
is usable, the aggregate is `NaN`. With fewer than two declared columns, Trend
is marked `NOT_APPLICABLE`.

## Pair details

With `details=True`, `report.details("trend")` returns `column_1`, `column_2`,
and `score` for each pair, including `NaN` for undefined pairs. These are the
components used in the generated-real aggregate. See
[component tables](../../reference/reports.md#component-tables) for shared
retention behavior.

## Missing values

Missing values are handled pairwise, separately on each side. Numeric
correlations use rows where both values are finite. Joint tables exclude rows
with a missing category or an unassigned numeric bin. Different pairs can
therefore use different numbers of observations.

Sampling uses the full row pools before these pairwise exclusions.
