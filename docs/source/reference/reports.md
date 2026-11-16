# Report API

`evaluate` always returns a `Report`. For a walkthrough of the printed output,
start with [Read your first report](../getting-started/reports.md).

Use qualified `"metric/stat"` keys to avoid ambiguity when different metrics
share a statistic name.

```python
shape = report["shape/shape"]
print(shape.value, shape.ideal, shape.n_eff)
```

## Inspect results

| Operation | Returns |
|---|---|
| `report["metric/stat"]` | A `StatResult` for one statistic |
| `report["metric"]` | The public metric's primary `StatResult` |
| `report.summary()` | Human-readable point/reference/uncertainty summary |
| `report.to_dataframe()` | One row per metric statistic, with counts, intervals and diagnostics |
| `report.row_usage()` | Source input, retained and excluded counts and excluded percentages for each prefiltering metric and pool |
| `report.options_dict` | Resolved keyword options, with `uncertainty=None` or a nested dictionary containing `mode` and `n_replicates` |
| `report.details(metric, level=None)` | A copy of a retained component table |
| `report.iter_stats()` | Stored `(MetricResult, StatResult)` pairs |
| `report.fidelity`, `report.memorization`, `report.utility` | Focused family reports sharing the stored results |
| `report.plot(kind="scores", **kwargs)` | A Plotly figure using retained results |

Common `StatResult` attributes:

| Attribute | Meaning |
|---|---|
| `value`, `ideal` | Finite repetition means for repeated sides, or single scores otherwise. MLE's raw score uses the real-trained benchmark as ideal |
| `value_estimate`, `ideal_estimate` | `"mean"` for a repetition mean, or `"point"` for a single comparison |
| `n_eff` | Selected rows per comparison role; point-only metrics use their own row-count conventions |
| `ci_low`, `ci_high` | Generated-score percentile band endpoints, if requested and available |
| `ideal_ci_low`, `ideal_ci_high` | Reference-score percentile band endpoints, including repeated test-only evaluation |
| `generated_distribution`, `reference_distribution` | Ordered repeated scores and their finite summaries, when available |
| `flag_codes` | Computation statuses associated with this statistic |
| `score_role` | Source role of the primary score; test-only scores use `test_test` or `real_trained` |
| `point_status`, `ideal_status` | Explicit `"absent"` for an uncomputed point, distinct from an undefined computed score |

Point-only metrics do not acquire a real-real reference or intervals merely because
an `uncertainty` dictionary was supplied. See [Methodology](../methodology.md)
for interval interpretation. Reports do not attach automatic comparison labels.

Each distribution's `.samples` contains `n_replicates` ordered comparison
scores. Rows are selected without replacement within each repetition and become
available again for later draws. `.mean`, `.median`, `.band_low` and `.band_high`
summarize the finite scores. The band fields are their 2.5th and 97.5th
percentiles with linear interpolation. `.finite_count` and `.unavailable_count`
count repeated scores.

The statistic's `ci_*` or `ideal_ci_*` fields contain the same unshifted
percentile endpoints as the corresponding distribution. Despite the `ci_`
field names, these bands describe variation across sampled comparisons; they
are not confidence intervals for the population expected score.

With test data alone, fixed evaluation stores one real-reference score in
`.value` and marks `.ideal` absent. Repeated test-only evaluation stores the
finite reference mean in `.value` and retains `reference_distribution`, with its band in
`.ideal_ci_low` and `.ideal_ci_high`. Its `.ci_*` fields remain unavailable.
For MLE, each repeated ratio is computed from that comparison's raw scores.
The distribution's mean ratio can differ from dividing the averaged raw scores.
See [MLE's repetition protocol](../metrics/utility/mle.md#repeated-evaluation).

## Source eligibility

Distance metrics exclude records missing any declared distance feature before
sampling; MLE excludes records with missing targets and retains missing predictors.
Their source accounting is always retained, without `details=True`:

```python
usage = report.row_usage()
print(usage[["metric", "pool", "input", "retained", "excluded", "excluded_percent"]])
```

`policy` is `"complete_case"` or `"observed_target"`; `columns` names the checked
columns. Percentages use each original source count as denominator. For an empty
source, the percentage is undefined. `n_res` records selected rows per comparison
role from the retained pools. The summary and Markdown display source accounting
once for shared numerical computations such as PRDC. JSON's `row_usage` subtree
records it for every public metric.

These counts describe source eligibility. A result can retain every source row
and still have an undefined score. Fidelity's
component-specific observed-value rules remain separate; they do not apply one
whole-record exclusion policy to the input pool.

## Component tables

Enable fidelity capture using `evaluate(..., details=True)`. Shape defaults to
column details, Trend to pair details, and Conditional to column details.
Choose `level="slice"` for Conditional slice scores; its Trend components are
summarized by conditioning column and slice, without a separate pair table.
Concentration always retains compact radius profiles. MLE raw scores and model
diagnostics are stored with its scalar statistics; it has no per-column utility
table. Captured fidelity tables describe the first generated-real repetition
when that side is repeated, or its single comparison otherwise. The detail
metadata records the scope, repetition index and that repetition's aggregate
scores. These tables are not averages over repetitions. They include no real-real component scores or component
confidence intervals. Requesting an existing table does not rerun the metric.

## Family views

`FidelityReport`, `MemorizationReport` and `UtilityReport` inherit `Report`.
Each family view supports the same lookup, tables and exports as its source
report. A view shares stored scores and details while its schema, sample counts,
and uncertainty metadata describe its own procedure. In a mixed report the
memorization view remains point-only, even when comparison metrics use UQ.
The top-level report schema records the caller's declarations; schemas
for all procedures are retained in `report.provenance["procedures"]`.

`report.fidelity` presents the computed Shape, Trend, Conditional Fidelity,
PRDC, detection and Concentration results; MLE belongs to utility. A family
view does not select or calculate additional metrics. Use the
[pillar-to-group mapping](../metrics/index.md#pillars-and-selector-groups)
to choose which metrics to compute.

## Execution metadata

`report.provenance["metric_execution"]` records each metric's `implementation`
and `device`, together with precision and routing details where applicable.
There is no `backend` field or separate `metric_backends` summary. XGBoost model
versions and fit settings remain in the model diagnostics.

## Visualization

`report.plot(...)` and `plot_report(report, ...)` return Plotly figures from
retained results without rerunning evaluation:

```python
from tabgen_eval.visualization import plot_report

plot_report(report, kind="scores", metrics=["shape", "trend"])
report.fidelity.plot(kind="columns", metric="shape", top_n=10)
```

| Kind | Data and controls |
| --- | --- |
| `scores` | Reported scores and requested percentile bands. Dots show means for repeated comparisons and single scores otherwise. `metrics` selects the displayed metrics; each statistic has its own scale. |
| `columns` | Retained Shape or Conditional column values. Select `metric`, optionally `stat` and `top_n`. |
| `pairs` | Retained Trend pair scores. Undefined pairs remain distinguishable from low scores. |
| `slices` | Retained Conditional slices for one `column`. Select `stat="shape"`, `"trend"` or `"recovery"`; optional `top_n`. |
| `concentration` | Stored real/generated mean and upper local-mass profiles at the actual fitted radii. |
| `utility` | MLE raw generated/real-trained MAE or AUROC and generated-to-real score ratios, with task-specific directions. |

Component plots require the corresponding [retained table](#component-tables).
See [Visualizing reports](../getting-started/visualization.md)
for interactive examples and offline HTML export.

### Input-data plots

The input-data utilities take DataFrames explicitly:

```python
from tabgen_eval.visualization import plot_column, plot_column_pair

plot_column(holdout, generated, "age", schema=schema)
plot_column_pair(holdout, generated, "age", "income", schema=schema)
```

`plot_column` uses empirical cumulative distributions for numeric values and
frequency bars for categorical values. `plot_column_pair` selects a view for
the two declared column types. Both require `schema` and accept `seed` and `max_points`;
category controls are `max_categories`. Input sampling limits the displayed
records and does not alter stored metric results. Numeric ECDFs use at most
5,000 values per table by default; numeric and mixed pair views use at most
3,000 complete rows per table. Categorical frequencies aggregate all usable
rows before grouping less frequent categories.
These are views of the supplied source tables; their display samples are
independent of the matched row subsets used when computing a report.

## Export

```python
report.to_json("evaluation.json", include_replicates=True)
report.to_markdown("evaluation.md")
report.to_dataframe().to_csv("evaluation.csv", index=False)
```

`to_json(path=None, *, include_replicates=False)` returns JSON text and optionally
writes it. It retains resolved evaluation settings under `options`, plus schema,
provenance, diagnostics and details. An enabled uncertainty dictionary includes
the resolved `n_replicates`, including its default of 10. Disabled uncertainty
is stored as `null`. The original caller dictionary is not modified.
Evaluation exports use `schema_version=2` and include public metric metadata,
source roles and explicit absent-point status. Test-only repeated results retain
reference distributions, without generated scores. Data diagnostics have their
own independent export format. `include_replicates=True` adds the distributions'
ordered `samples` vectors. Failure diagnostics remain in provenance when
available. Nonfinite numeric scores become JSON `null`, with status
information preserving their meaning.
JSON decoding returns dictionaries; a Report deserialization API and historical
format migration are not provided.

`to_markdown(path=None)` returns Markdown and optionally writes it.
See [Read your first report](../getting-started/reports.md) for actual output.
