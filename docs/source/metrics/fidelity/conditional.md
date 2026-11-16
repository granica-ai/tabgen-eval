# Conditional Fidelity

Conditional Fidelity checks whether a generator preserves subpopulations that
an overall average can hide. Each column defines slices of the real reference;
within each slice, it compares the other columns and the frequency of that
slice.

Each component summarizes all eligible slices and the rare subset.

## Scores

Select `group=["conditional_fidelity"]` for all components, or choose
`metrics=["cond_shape", "cond_trend", "recovery"]` individually. Each selected
component includes its rare counterpart:

| Report key | Meaning | Range and direction |
| --- | --- | --- |
| `cond_shape/cond_shape_all` | Marginal similarity within retained slices | $[0,1]$; higher is closer |
| `cond_shape/cond_shape_rare` | Marginal similarity within retained rare slices | $[0,1]$; higher is closer |
| `cond_trend/cond_trend_all` | Pairwise similarity within retained slices | $[0,1]$; higher is closer |
| `cond_trend/cond_trend_rare` | Pairwise similarity within retained rare slices | $[0,1]$; higher is closer |
| `recovery/recovery_all` | Generated-to-real slice-frequency ratios averaged within each conditioning column, then equally across columns | $[0,\infty)$; target $1$ |
| `recovery/recovery_rare` | The same calculation restricted to eligible rare slices | $[0,\infty)$; target $1$ |

Recovery needs one declared column, Conditional Shape two, and Conditional
Trend three: the conditioning column is excluded from Shape/Trend comparisons.
Rare statistics are `NOT_APPLICABLE` if no retained rare slice exists.

## Example: inspect a customer segment

Use the [shared customer example](index.md#shared-example) to create `data`.
Conditioning on `segment` compares `visits`, `spend`, and `tenure` separately
within each customer group:

```python
from tabgen_eval import evaluate

report = evaluate(
    data,
    metrics=["cond_shape", "cond_trend", "recovery"],
    details=True,
    seed=23,
)
print(report["cond_shape/cond_shape_all"].value)
print(report["recovery/recovery_rare"].value)

slices = report.details("recovery", level="slice")
print(slices.loc[slices["eligible"], ["column", "label", "recovery"]])
report.fidelity.plot(
    kind="slices", metric="cond_shape", column="segment", stat="shape"
).show()
```

```{raw} html
:file: ../../_generated/visualization/slices.html
```

Each bar compares the other columns within one real-defined segment. Hover for
sample counts and eligibility. To inspect another component, use
`metric="cond_trend", stat="trend"` or `metric="recovery", stat="recovery"`.

## Definition

Let $R$ and $G$ be the matched real and generated samples, with $d$ declared
columns. For each conditioning column $i$, construct shared slice labels:

- A categorical column defines one slice per observed real category identity.
- A numeric column defines up to `num_bins` equal-width intervals over its real
  range. Generated values outside that range are assigned to the nearest edge
  interval. Constant real columns use a widened interval around the constant.

These metrics use fixed definitions:

| Definition | Value | Effect |
| --- | --- | --- |
| `num_bins` | `10` | Number of equal-width numeric conditioning bins and maximum number of numeric quantile bins for within-slice contingency comparisons. Ties can reduce the bin count. |
| `min_count` | `10` | Minimum real row count required for a slice to contribute. |
| `rare_threshold` | `0.05` | Eligible slices with real frequency strictly below this fraction enter rare statistics. |

### Slice eligibility and rarity

For slice $s$ of column $i$, let $r_{is}$ and $g_{is}$ be its real and generated
row counts. Let $N_{R,i}$ and $N_{G,i}$ count all rows with a usable conditioning
label on each side, including rows in slices that do not meet the minimum count.
The reference frequency is

$$
f_{R,is}=\frac{r_{is}}{N_{R,i}}.
$$

A slice is eligible if $r_{is}\ge\texttt{min_count}$. An eligible slice is rare
if $f_{R,is}<\texttt{rare_threshold}$. Eligibility depends only on the real
sample. A category found only in generated data defines no reference slice,
but its rows still enter $N_{G,i}$ and therefore affect frequency recovery.

Small slices can produce noisy comparisons.

### Within-slice Shape and Trend

Apply the [Shape](shape.md) comparison to each column other than $i$, using the
rows assigned to slice $s$ on each side. Average its defined column scores to
obtain $S_{is}$.

Apply the [Trend](trend.md) comparison to each pair of columns that excludes $i$,
again using only rows in slice $s$. Average its defined pair scores to obtain
$T_{is}$. Any numeric contingency bins are fitted on the real rows within this
slice, using `num_bins`.

If an eligible real slice has no generated rows, its Shape and Trend scores
are $0$, provided there are enough other columns for that statistic. An existing
slice whose comparisons are all undefined instead receives `NaN`.

### Frequency recovery

For each eligible slice, compute

$$
Q_{is}=\frac{f_{G,is}}{f_{R,is}},
\qquad
f_{G,is}=\begin{cases}
g_{is}/N_{G,i},&N_{G,i}>0,\\
0,&N_{G,i}=0.
\end{cases}
$$

A slice ratio below $1$ indicates underrepresentation; a ratio above $1$
indicates overrepresentation. A slice that the generator never populates
receives recovery $0$.

For example, a 20-row group in a 1,000-row real reference sample is eligible
and rare (2%). If the group represents 1% of the generated sample, its recovery
is $0.01/0.02=0.5$.

### Aggregation

Let $A_i$ contain the eligible slices for conditioning column $i$, or only its
eligible rare slices when computing a rare statistic. For
$V\in\{S,T,Q\}$, first average within each conditioning column, then across
columns:

$$
\overline V_i=\operatorname{mean}_{s\in A_i} V_{is},
\qquad
\overline V=\operatorname{mean}_i\overline V_i.
$$

Every mean excludes undefined values. A conditioning column with no defined
slice scores is omitted from the outer mean; if none remain, the result is
`NaN`. Each contributing conditioning column has equal weight, and eligible
slices within a column have equal weight.

Recovery is an average of ratios: underrepresentation of one slice and
overrepresentation of another can cancel. Inspect slice details to assess
individual groups even when the aggregate is near $1$.

## Detailed outputs

With `details=True`, `report.details("cond_shape")` returns one row per
conditioning column with its `cond_shape_all` and `cond_shape_rare` aggregates.
Use `"cond_trend"` or `"recovery"` for the corresponding selected component's
all/rare aggregates. `report.details("cond_shape", level="slice")` returns:

| Columns | Meaning |
| --- | --- |
| `column`, `label` | Conditioning column and categorical label or numeric interval. |
| `reference_count`, `candidate_count` | Rows assigned to this slice. |
| `reference_total`, `candidate_total` | Rows with usable labels in this conditioning column. |
| `eligible`, `rare` | Minimum-count eligibility and rarity relative to the real sample. Check both to select contributing rare slices. |
| `shape` | Selected component's slice score; ineligible slices have undefined scores. The other component views retain `trend` or `recovery` instead. |
| `lower`, `upper`, `left_closed`, `right_closed` | Numeric interval boundaries and closure; categorical boundaries are undefined. |

Slice details also retain ineligible real groups for inspection. See
[component tables](../../reference/reports.md#component-tables) for shared
retention behavior.

## Missing values

A missing conditioning label excludes its row from slicing and from that
column's frequency denominator. Within a slice, Shape excludes missing cells
by column and Trend excludes them by pair.

Sampling uses the full row pools before these slice-specific exclusions. See
[data preparation](../../reference/data.md) for accepted input representations.
