# Shape

Shape measures how closely generated data reproduces each column's marginal
distribution. It compares numeric columns using their empirical cumulative
distributions and categorical columns using category frequencies, then averages
the column scores with equal weight.

Use Shape to find distribution shifts in individual columns. A generator can
reproduce every marginal while losing dependencies; use [Trend](trend.md) to
assess relationships between column pairs.

## Score

| Property | Value |
| --- | --- |
| Metric selection | `metrics=["shape"]` |
| Report key | `"shape/shape"` |
| Range | $[0,1]$ |
| Direction | Higher means closer marginal distributions |
| Detailed output | One score per declared column |

A column score of $1$ means identical empirical distributions for that column.
The aggregate is $1$ when every usable column scores $1$. Independent real
samples usually score below $1$ because of sampling variation; use the reported
real–real score as context at the stated resolution.

## Example: find a shifted column

Use the [shared customer example](index.md#shared-example) to create `data`.
The toy generator shifts `tenure` by five units and repeats some records. Shape
shows which marginal distributions changed:

```python
from tabgen_eval import evaluate

report = evaluate(data, metrics=["shape"], details=True, seed=23)
score = report["shape/shape"]
print(score.value, score.ideal)
print(report.details("shape").sort_values("score"))
report.fidelity.plot(kind="columns", metric="shape").show()
```

```{raw} html
:file: ../../_generated/visualization/columns.html
```

The shifted `tenure` column has the lowest similarity. Inspecting column scores
reveals this weakness even when other columns raise the overall average.

## Definition

Let $R$ and $G$ denote the matched real and generated samples. Each column is
compared after applying its missing-value rule below.

### Numeric columns

For numeric column $j$, let $\widehat F_{R,j}$ and $\widehat F_{G,j}$ be the
empirical cumulative distribution functions of their finite values. Its score
is one minus the two-sample Kolmogorov–Smirnov statistic:

$$
s_j = 1-\sup_x\left|\widehat F_{R,j}(x)-\widehat F_{G,j}(x)\right|.
$$

The comparison uses the empirical CDFs directly. Constant numeric columns remain
valid comparisons.

### Categorical columns

For categorical column $j$, let $\mathcal C_j$ be the union of observed category
identities and $\widehat p_{R,j}(c)$ and $\widehat p_{G,j}(c)$ their relative
frequencies. Its score is one minus total variation distance:

$$
s_j = 1-\frac{1}{2}\sum_{c\in\mathcal C_j}
\left|\widehat p_{R,j}(c)-\widehat p_{G,j}(c)\right|.
$$

Categories appearing on only one side contribute to the distance. Category
identities follow the declared [input contract](../../sharp-bits.md#preparation-changes-what-is-being-compared).

### Aggregate

Let $J$ contain the columns with a defined score. Shape is

$$
\operatorname{Shape}(G,R)=\frac{1}{|J|}\sum_{j\in J}s_j.
$$

Every usable column has equal weight, regardless of its scale or category count.
Undefined columns are excluded from the mean. If none are usable, the aggregate
is `NaN`.

## Column details

With `details=True`, `report.details("shape")` returns `column` and `score`
for each declared column, including `NaN` for an undefined score. These are the
components used in the generated-real aggregate. See
[component tables](../../reference/reports.md#component-tables) for shared
retention behavior.

## Missing values

Missing cells are excluded independently within each column and table. A column
is undefined if either side has no observed values. Shape therefore compares
observed-value distributions. Compare missingness rates separately with
[data diagnostics](../../data-diagnostics.md).

Sampling uses the full row pools before these column-specific exclusions. A
column can use fewer observations than the evaluation's reported row resolution.
See [data preparation](../../reference/data.md) for accepted input representations.
