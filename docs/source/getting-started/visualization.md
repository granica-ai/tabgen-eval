# Visualizing reports

A report can show where a generator differs from real data. The interactive
figures use Plotly: hover to inspect values or use the toolbar to save an image.
Where a legend is shown, click an entry to show or hide that series.

## Start with the real reference

```python
figure = report.fidelity.plot(kind="scores", metrics=["shape", "trend"])
figure.show()
```

Dots show the reported estimates: repetition means when uncertainty is enabled
for that side, or a single score otherwise. Hover text identifies whether a dot
is a point or a mean. Plotting reads retained results without resampling.

In the example below,
the generated–real points are shown alongside real–real means and bands, which
provide the yardstick. Ten repeated real-real comparisons form the reference
percentile bands.

```{raw} html
:file: ../_generated/visualization/scores.html
```

These bands describe variation in **aggregate scores** across sampled
comparisons. They are not confidence intervals for the population expected
scores. See [statistical methodology](../methodology.md) for interpretation.
Component plots below show
descriptive point scores, without column, pair or slice uncertainty estimates.

## Find a weak column

Retain components with `details=True` when evaluating, then plot the stored
Shape scores:

```python
report.fidelity.plot(kind="columns", metric="shape").show()
```

```{raw} html
:file: ../_generated/visualization/columns.html
```

The `tenure` column has the lowest marginal similarity.
This locates a discrepancy; it does not establish its cause. To inspect the data themselves, pass the input tables explicitly:

```python
from tabgen_eval.visualization import plot_column

plot_column(holdout, generated, "tenure", schema=schema).show()
```

Numeric columns use empirical cumulative distributions; categorical columns
use frequency bars.
They show the supplied tables using predefined sampling for display, so
they need not show the exact row subset used for a report's draw.

## Inspect relationships

```python
report.fidelity.plot(kind="pairs", metric="trend").show()
```

```{raw} html
:file: ../_generated/visualization/pairs.html
```

Trend compares each column pair. A heatmap makes weak pairs visible even when
the aggregate hides them. An undefined entry is distinct from a low score.
Compare the actual numeric relationship with a bounded scatter plot:

```python
from tabgen_eval.visualization import plot_column_pair

plot_column_pair(holdout, generated, "visits", "spend", schema=schema).show()
```

```{raw} html
:file: ../_generated/visualization/association.html
```

The inputs show a strong real association and a weaker generated association.
The generated point pile also reflects the repeated record in this example.
This view displays all 600 holdout and 300 generated rows; the report's metric
comparisons use matched 300-row samples instead.
Scatter plots use at most `max_points=3000` rows per table by default; set the
sampling seed to reproduce a view of larger inputs.

## Inspect a conditional slice

Choose a conditioning column and one retained slice statistic:

```python
report.fidelity.plot(kind="slices", metric="cond_shape", column="segment", stat="shape").show()
```

```{raw} html
:file: ../_generated/visualization/slices.html
```

These scores compare the other columns within each real-defined segment.
Hover for counts and eligibility. They summarize the fixed generated–real
comparison; real–real slice scores are not retained. To inspect frequency
recovery instead, use `metric="recovery", stat="recovery"`.

## Compare concentration

```python
report.fidelity.plot(kind="concentration").show()
```

```{raw} html
:file: ../_generated/visualization/concentration.html
```

The fixed small radius exposes the repeated generated record: generated local
mass is positive while real local mass is zero. The upper profile is a quantile
of local masses, **not an uncertainty bound**. Larger real-derived radii probe
other scales. This figure uses the stored point profiles and actual radii; the
scalar concentration gaps have their own real–real comparisons in the report.

PRDC's four geometric-fidelity scores use the same overview interface:

```python
report.fidelity.plot(kind="scores", metrics=["precision", "recall", "density", "coverage"]).show()
```

See the [PRDC guide](../metrics/geometric-fidelity/prdc.md#visualizing-prdc)
for this example's result.

## Report families and available views

All specialized reports inherit lookup, tables and exports from `Report`.
Select a family from a mixed report without recomputing its results:

```python
fidelity = report.fidelity
memorization = report.memorization
utility = report.utility
```

| View | Plot kinds |
| --- | --- |
| `FidelityReport` | `scores`, Shape/Conditional `columns`, Trend `pairs`, Conditional `slices`, `concentration` |
| `MemorizationReport` | `scores` for retained DCR, near-match and replication-ratio summaries |
| `UtilityReport` | `scores`, `utility` for MLE raw MAE/AUROC, generated-to-real score ratios and shared-test prediction statistics |

PRDC belongs to fidelity because it measures local geometric fidelity. Missing
component tables require a new evaluation with `details=True`; plotting cannot
recover data that were not retained. Memorization plots show available scalar
summaries, rather than reconstructing distance distributions from quantiles.

## Customize, save and reproduce

Every plotting function returns a Plotly `Figure`, so normal Plotly controls
apply:

```python
figure = report.fidelity.plot(kind="columns", metric="shape", top_n=10)
figure.update_layout(title="Marginal fidelity by column")
figure.write_html("shape.html", include_plotlyjs=True)
```

That HTML file includes its plotting library and works offline. Numerical
tables and report JSON remain available through the [report API](../reference/reports.md).

Run the complete CPU-only example from a source checkout:

```bash
python examples/visualization_walkthrough.py --output visualization-demo
```

It writes seven interactive figures and a JSON report. The files share one
local Plotly JavaScript bundle, and require no network access to open.
Download the {download}`example report <../_generated/visualization/report.json>`.

```{literalinclude} ../../../examples/visualization_walkthrough.py
:language: python
:pyobject: make_report
```
