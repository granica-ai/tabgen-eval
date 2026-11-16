# Read your first report

Continue with the `report` returned by the [quickstart example](quickstart.md).
This walkthrough explains its scores and uncertainty bands and
follows an aggregate score down to individual columns. For method and field
definitions, use the [Report API](../reference/reports.md).

## The printed summary

The quickstart produces this summary:

```{literalinclude} ../_generated/point-summary.txt
:language: text
```

The generated `mean` averages generated-real comparisons. The `real-real mean`
averages comparisons of two disjoint real samples. Both use 100 rows per side in this example,
so the reference gives context at the same sample resolution.

Shape and Trend are higher-is-better similarity scores. Here, Shape is lower
than its real-real reference, as is Trend. Read each score in the
context of what its metric measures and the available sampling variability.

The quickstart requests ten repetitions for both synthetic-vs.-real and
real-vs.-real comparisons. The percentile bands describe variation across
comparisons.

## Retrieve a result

Read the Shape result directly:

```python
shape = report["shape"]
print(shape.value)  # Mean generated-real score
print(shape.ideal)  # Mean real-real score
print(shape.n_eff)  # Rows per comparison role
```

The [result lookup and field reference](../reference/reports.md#inspect-results)
explains qualified names and the other available attributes.

## Request real-reference variability only

Reuse `eval_data` prepared in the quickstart. To repeat only the real-real
comparison, call `evaluate` with reference uncertainty:

```python
report_with_bands = evaluate(
    eval_data,
    metrics=["shape", "trend"],
    device="cpu",
    seed=42,
    uncertainty={"mode": "reference", "n_replicates": 10},
    details=True,
)
```

This requests ten real-real comparisons, sampling without replacement within each
draw. With these inputs, reference-only mode uses 200 rows per role by default,
so its reported scores can differ from the quickstart's 100-row comparisons.
See [uncertainty options](../reference/configuration.md#choose-uncertainty)
for other modes, repetition counts and sample resolution.

The same example now gives:

```{raw} html
:file: ../_generated/reference-table.html
```

`ideal_ci_low` and `ideal_ci_high` give the
2.5th and 97.5th percentiles of the finite real-real scores. This band describes
variation among sampled comparisons at this resolution. It is not a confidence
interval for the expected real-real score $\mu_n(P)$. Compare the generated
score with that reference variability; the report attaches no automatic quality
label.

The [methodology](../methodology.md) explains the real-real reference and what
the intervals mean.

## Inspect individual columns and pairs

The quickstart uses `details=True` to retain component scores. Retrieve them
from the quickstart report:

```python
columns = report.details("shape")
pairs = report.details("trend")
```

Its Shape table is:

```{literalinclude} ../_generated/shape-details.txt
:language: text
```

The `age` column has the lowest marginal similarity in this example. Component
tables help locate differences in the first generated-real repetition. They
are not averages over repetitions and have no per-column uncertainty bands;
their scope is recorded in the detail metadata.
For your own evaluations, request `details=True` when you need these tables.

## Visualize stored results

Plot the same column scores through the report's fidelity view:

```python
report.fidelity.plot(kind="columns", metric="shape").show()
```

This uses stored results without recomputing the metric. Continue with the
[visual walkthrough](visualization.md) to investigate weak columns, relationships
and subpopulations. The [family-view reference](../reference/reports.md#family-views)
explains how to select fidelity, memorization and utility results from a mixed report.

## Export and retain context

Save the report with its reference interval:

```python
report_with_bands.to_json("evaluation.json")
```

JSON keeps the scores, schema, resolved options and execution diagnostics.
Add `include_replicates=True` to retain the ordered scores used to form the
bands. The [export reference](../reference/reports.md#export) covers
other formats and exact serialization behavior.

Download the {download}`example JSON report <../_generated/example-report.json>`.
For undefined scores or unexpected diagnostics, see
[Troubleshooting](../troubleshooting.md).
