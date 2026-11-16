# Five-minute quickstart

This example creates a small real holdout and a generated table with slightly
shifted distributions. In your own evaluation, supply an unseen real holdout and the records produced by your generator.

Before running this example, follow the [installation guide](installation.md) to prepare the CPU prerequisites for [Linux](installation.md#linux-cpu-installation) or [Apple Silicon macOS](installation.md#apple-silicon-macos-cpu-installation).

## Run the example

The complete example below uses native CPU execution and needs no GPU or external
dataset. Native CPU operators compile on first use and are cached for future re-use.

```{literalinclude} ../../../examples/report_walkthrough.py
:language: python
```

From a source checkout, run it with:

```bash
python examples/report_walkthrough.py
```

## Read the first scores

This table is generated from the example when the documentation is built:

```{raw} html
:file: ../_generated/point-table.html
```

- `value` is the mean generated-real score across the repetitions.
- `ideal` is the mean real-real score at the same resolution.
- `ci_low` / `ci_high` and `ideal_ci_low` / `ideal_ci_high` are the synthetic-vs.-real
  and real-vs.-real percentile bands, respectively.
- `n_eff` is the number of rows per side used in each comparison.

Here the holdout has 400 rows and the generated table has 200. With uncertainty
on both sides, the default resolution is 100 rows per role (50% of the generated table size). Each repetition
draws two disjoint real samples and one synthetic sample of that size. Shape and
Trend are higher-is-better similarity scores; a real-real score can be below one
because finite samples differ.

The example requests 10 repeated comparisons on both sides using
`uncertainty={"mode": "both", "n_replicates": 10}`. The bands span the 2.5th to
97.5th percentiles of the repeated scores; they describe score variability.
Continue with [Read your first report](reports.md) for the full printed output.

## Plot your report

The example retains column and pair scores, so these figures use the stored
results directly:

```python
report.fidelity.plot(kind="scores").show()
report.fidelity.plot(kind="columns", metric="shape").show()
report.fidelity.plot(kind="pairs", metric="trend").show()
```

The score plot from the same quickstart report shows its repetition means and
uncertainty bands for both comparisons:

```{raw} html
:file: ../_generated/point-scores.html
```

The dots and bands summarize the same repetitions; no additional point
comparison is computed. On narrow screens, scroll the two panels horizontally.

The [visual walkthrough](visualization.md) shows interactive examples of a weak
column, a broken association and excess concentration, plus saving figures as
offline HTML.

## Use your own data

```python
import pandas as pd
from tabgen_eval import EvalData, Schema, evaluate

holdout = pd.read_parquet("real_holdout.parquet")
generated = pd.read_csv("generated.csv")
schema = Schema(numeric=["age", "income"], categorical=["region"])

report = evaluate(
    EvalData(test=holdout, generated=generated, schema=schema),
    metrics=["shape", "trend"],
    uncertainty={"mode": "both", "n_replicates": 10},
)
print(report.summary())
report.to_json("evaluation.json")
```

Both frames must have the same complete set of uniquely named columns,
including columns omitted from the schema. Their column order and row counts
may differ. A schema is required. `Schema.infer(holdout)` explicitly proposes roles; inspect or edit
them before passing the result to evaluation. See
[column roles](../reference/data.md#describe-your-columns) and the [metric guide](../metrics/index.md).

This example filters to Shape and Trend. Omitting both filters selects every
metric supported by the input roles, including MLE and nonlinear C2ST. That
default needs `schema.target` for MLE; XGBoost is included in the base
installation. Missing prerequisites
raise. Use the [registered metric table](../reference/evaluation.md#registered-names)
to choose a subset.
